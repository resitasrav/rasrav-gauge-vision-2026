"""Uçtan uca zincir hatasını ölçer: tespit + açı + kalibrasyon birlikte.

    python scripts/olc_zincir.py
    python scripts/olc_zincir.py --agirlik runs/detect/models/ip5/gercek/weights/best.pt

İP7'nin ölçümüyle **tek farkı** kadranın merkezinin nereden geldiğidir:

    olc_ip7.py   merkez ETİKETTEN  → okuma yönteminin hatası
    olc_zincir.py merkez TESPİTTEN → zincirin gerçek hatası

İkisi yan yana konduğunda tespitin bütçeden ne kadar yediği görülür. Bu, İP8'in
(gerçek görüntüde uçtan uca test) sentetik veri üzerindeki provasıdır; gerçek
fotoğrafa geçmeden önce zincirin kendi içinde ne kadar hata biriktirdiğini
gösterir.

Yatıklık düzeltmesi **uygulanmaz** (`roll_deg=0`): gerçek görüntüde kameranın
yatıklığı bilinmeyeceği için sentetik ölçümde de bilinmiyormuş gibi davranılır.
Etiketten verilseydi sayı iyimser çıkardı.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.pipeline import read_frame
from gauge_vision.read.evaluate import error_stats
from gauge_vision.synth.generate import load_labels

VARSAYILAN_VERI = "data/synthetic/v0"
VARSAYILAN_AGIRLIK = "runs/detect/models/ip5/karisik/weights/best.pt"
METRIK_YOLU = "outputs/metrics/ip8_zincir_hatasi.json"
FIGUR_YOLU = "outputs/figures/ip8_zincir_hatasi.png"

# İP7'nin aynı veri setindeki ölçümü — karşılaştırma için (04.08).
IP7_ORTALAMA_YUZDE = 0.129


def kosu(veri: Path, agirlik: Path, gauges: dict, conf: float,
         *, yatiklik_etiketten: bool = False, model=None) -> list[dict]:
    """Veri setini uçtan uca okur.

    `yatiklik_etiketten` bir **ablasyon anahtarıdır**: gerçek görüntüde kameranın
    yatıklığı bilinmez, o yüzden varsayılan kapalıdır. Açıldığında yatıklığın
    zincirdeki payı ölçülür — düzeltmenin yazılmaya değip değmeyeceği buna bakılarak
    kararlaştırılır.
    """
    if model is None:
        from ultralytics import YOLO
        model = YOLO(str(agirlik))
    kayitlar = []

    for k in load_labels(veri):
        gauge = gauges[k["gauge_id"]]
        kare = cv2.imread(str(veri / k["file"]))
        if kare is None:
            raise FileNotFoundError(veri / k["file"])

        s = read_frame(kare, model, gauge, detect_conf=conf,
                       roll_deg=k["roll_deg"] if yatiklik_etiketten else 0.0)
        aralik = gauge.scale.max - gauge.scale.min

        kayit = {
            "file": k["file"],
            "gauge_id": k["gauge_id"],
            "gercek": k["value"],
            "olculen": s.reading.value if s.ok else None,
            "status": s.reading.status if s.reading else "detect_fail",
            "sebep": s.reason,
            "tespit_guveni": round(s.detect_conf, 3),
            # Merkez sapması: tespitin verdiği merkez ile etiketteki merkez arası.
            # Zincir hatasının kaynağını ayırmak için tutuluyor.
            "merkez_sapmasi_px": (
                round(float(np.hypot(s.center_px[0] - k["center_px"][0],
                                     s.center_px[1] - k["center_px"][1])), 2)
                if s.center_px else None
            ),
            "kadran_capi_px": 2 * k["radius_px"],
        }
        if kayit["olculen"] is not None:
            kayit["hata_yuzde"] = abs(kayit["olculen"] - kayit["gercek"]) / aralik * 100
        kayitlar.append(kayit)

    return kayitlar


def ozetle(kayitlar: list[dict], gauges: dict) -> dict:
    okunan = [k for k in kayitlar if k.get("hata_yuzde") is not None]
    genel = error_stats([k["hata_yuzde"] for k in okunan])

    gosterge_bazli = {}
    for gid in sorted({k["gauge_id"] for k in kayitlar}):
        satirlar = [k for k in okunan if k["gauge_id"] == gid]
        gosterge_bazli[gid] = {
            "dogrusal": gauges[gid].scale.linear,
            "okunan": len(satirlar),
            "okunamayan": sum(1 for k in kayitlar
                              if k["gauge_id"] == gid and k.get("hata_yuzde") is None),
            **error_stats([k["hata_yuzde"] for k in satirlar]).as_dict(2),
        }

    sapmalar = [k["merkez_sapmasi_px"] for k in kayitlar if k["merkez_sapmasi_px"] is not None]
    capa_gore = [k["merkez_sapmasi_px"] / k["kadran_capi_px"] * 100
                 for k in kayitlar if k["merkez_sapmasi_px"] is not None]

    return {
        "goruntu": len(kayitlar),
        "okunan": len(okunan),
        "okunamayan": len(kayitlar) - len(okunan),
        "hata_yuzde_tam_skala": genel.as_dict(2),
        "merkez_sapmasi_px": error_stats(sapmalar).as_dict(2),
        "merkez_sapmasi_yuzde_kadran_capi": error_stats(capa_gore).as_dict(2),
        "gosterge_bazli": gosterge_bazli,
        "ip7_karsilastirma": {
            "ip7_ortalama_yuzde": IP7_ORTALAMA_YUZDE,
            "zincir_ortalama_yuzde": round(genel.mean, 3),
            "kat": round(genel.mean / IP7_ORTALAMA_YUZDE, 1),
        },
    }


def figur(kayitlar: list[dict], cikti: Path) -> Path:
    """Merkez sapması ile okuma hatası arasındaki ilişki.

    İddia "hatayı merkez belirliyor"sa saçılımda yükselen bir eğilim görülmeli.
    Görülmüyorsa iddia yanlıştır ve hata başka yerden geliyordur — grafik bu
    yüzden var, süslemek için değil.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    okunan = [k for k in kayitlar if k.get("hata_yuzde") is not None]
    x = [k["merkez_sapmasi_px"] / k["kadran_capi_px"] * 100 for k in okunan]
    y = [k["hata_yuzde"] for k in okunan]

    fig, eksen = plt.subplots(figsize=(6.4, 4.4))
    eksen.scatter(x, y, s=18, alpha=0.75)
    if len(x) > 1:
        egim, kesme = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 2)
        eksen.plot(xs, egim * xs + kesme, "r--", lw=1.4,
                   label=f"eğilim: %{egim:.2f} hata / %1 merkez sapması")
        eksen.legend()

    eksen.set_xlabel("merkez sapması (kadran çapının %'si)")
    eksen.set_ylabel("okuma hatası (% tam skala)")
    eksen.set_title("Zincir hatasını merkez mi belirliyor?")
    eksen.grid(alpha=0.3)

    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=130)
    plt.close(fig)
    return cikti


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Uçtan uca zincir hatası (İP8 provası)")
    p.add_argument("--veri", default=VARSAYILAN_VERI)
    p.add_argument("--agirlik", default=VARSAYILAN_AGIRLIK)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--figur", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    veri, agirlik = Path(args.veri), Path(args.agirlik)
    if not (veri / "labels.jsonl").exists():
        print(f"veri seti yok: {veri}")
        return 1
    if not agirlik.exists():
        print(f"ağırlık yok: {agirlik} — önce scripts/egit_ip5.py")
        return 1

    from ultralytics import YOLO
    model = YOLO(str(agirlik))   # tek kez yüklensin, ablasyon aynı modeli kullansın

    gauges = load_gauges()
    kayitlar = kosu(veri, agirlik, gauges, args.conf, model=model)
    ozet = {
        "is_paketi": "IP8-prova",
        "tarih": date.today().isoformat(),
        "veri_seti": str(veri).replace("\\", "/"),
        "agirlik": str(agirlik).replace("\\", "/"),
        "not": "merkez tespitten geliyor; yatıklık düzeltmesi uygulanmıyor",
        **ozetle(kayitlar, gauges),
    }

    h = ozet["hata_yuzde_tam_skala"]
    print(f"zincir hatası (% tam skala): ortalama {h['ortalama']}  medyan {h['medyan']}"
          f"  p95 {h['p95']}  max {h['max']}")
    print(f"okunamayan: {ozet['okunamayan']}/{ozet['goruntu']}")
    print(f"merkez sapması: {ozet['merkez_sapmasi_px']['ortalama']} px  "
          f"(%{ozet['merkez_sapmasi_yuzde_kadran_capi']['ortalama']} kadran çapı)")
    k = ozet["ip7_karsilastirma"]
    print(f"İP7 (merkez etiketten) %{k['ip7_ortalama_yuzde']} → "
          f"zincir %{k['zincir_ortalama_yuzde']}  = {k['kat']} kat")
    for gid, s in ozet["gosterge_bazli"].items():
        print(f"   {gid:8s} ort %{s['ortalama']:5.2f}  p95 %{s['p95']:5.2f}  "
              f"max %{s['max']:5.2f}  okunamayan {s['okunamayan']}")

    # Ablasyon: hatanın ne kadarı düzeltilmeyen yatıklıktan geliyor?
    # Üç satırlık tablo, zincirin bütçesini üç kaleme ayırır.
    ab = ozetle(kosu(veri, agirlik, gauges, args.conf,
                     yatiklik_etiketten=True, model=model), gauges)
    ozet["ablasyon_yatiklik_etiketten"] = ab
    yat = ozet["hata_yuzde_tam_skala"]["ortalama"] - ab["hata_yuzde_tam_skala"]["ortalama"]
    ozet["butce_dagilimi_puan"] = {
        "okuma_yontemi_ip7": IP7_ORTALAMA_YUZDE,
        "tespit_merkezi": round(ab["hata_yuzde_tam_skala"]["ortalama"] - IP7_ORTALAMA_YUZDE, 3),
        "duzeltilmeyen_yatiklik": round(yat, 3),
    }
    print("\nbütçe dağılımı (% tam skala, puan olarak):")
    for ad, deger in ozet["butce_dagilimi_puan"].items():
        print(f"   {ad:26s} {deger:6.3f}")
    print(f"   {'TOPLAM':26s} {ozet['hata_yuzde_tam_skala']['ortalama']:6.3f}")

    yol = Path(METRIK_YOLU)
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nölçüm: {yol}")

    if args.figur:
        print(f"figür: {figur(kayitlar, Path(FIGUR_YOLU))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
