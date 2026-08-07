"""Zor koşullarda uçtan uca okuma hatası — İP14'ün koşul bazlı tablosu.

    python scripts/olc_ip14.py
    python scripts/olc_ip14.py --eksen egiklik --sahne 40
    python scripts/olc_ip14.py --perspektif      # düzeltme ablasyonu

Her bozulma ekseni **tek başına** taranır. Karışık bir "zor görüntü" kümesinde
hangi etkenin bozduğu ayrılamaz; İP14'ün bitti kriteri koşul bazlı bir tablodur
ve ancak eksenler ayrıyken üretilebilir.

Zincir tam olarak sahadaki gibi koşar: merkez tespitten gelir, yatıklık
görüntüden kestirilir, etiketten hiçbir şey alınmaz. Etiketten gelen tek şey
**gerçek değerdir** — kıyaslanacak sayı odur.

Yan ürün: her karenin (güven, hata) çifti kaydedilir. İP15'in eşiği bu
dağılımdan kalibre edilir — eşik uydurulmaz, ölçülür.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.pipeline import read_frame
from gauge_vision.read.evaluate import error_stats
from gauge_vision.synth.degrade import EKSENLER, Bozulma, bozulmalar_uygula
from gauge_vision.synth.dial import DialTruth
from gauge_vision.synth.generate import load_labels

VARSAYILAN_VERI = "data/synthetic/v1"
VARSAYILAN_AGIRLIK = "runs/detect/models/ip5/karisik/weights/best.pt"
METRIK_YOLU = "outputs/metrics/ip14_zor_kosullar.json"
FIGUR_YOLU = "outputs/figures/ip14_zor_kosullar.png"
CIFT_YOLU = "outputs/metrics/ip14_guven_hata_ciftleri.json"

# Bozulma yönü kare başına rastgele seçilir: tek bir eğim yönünde ölçmek,
# kadranın o yöndeki simetrisine bağlı bir sonuç verir.
TOHUM = 14


def _truth_nesnesi(k: dict) -> DialTruth:
    """Etiket sözlüğünü `DialTruth`'a çevirir — bozulma modülü onu bekliyor."""
    return DialTruth(
        gauge_id=k["gauge_id"], value=k["value"], angle_deg=k["angle_deg"],
        roll_deg=k["roll_deg"], angle_img_deg=k["angle_img_deg"],
        center_px=tuple(k["center_px"]), tip_px=tuple(k["tip_px"]),
        radius_px=int(k["radius_px"]), bbox_xyxy=tuple(k["bbox_xyxy"]),
    )


def kosu(kayitlar, veri: Path, gauges: dict, model, bozulma: Bozulma,
         conf: float, perspektif: bool) -> list[dict]:
    rng = np.random.default_rng(TOHUM)
    cikti = []

    for k in kayitlar:
        gauge = gauges[k["gauge_id"]]
        kare = cv2.imread(str(veri / k["file"]))
        if kare is None:
            raise FileNotFoundError(veri / k["file"])

        b = bozulma
        if b.egiklik_deg:
            # Eğim yönü kare başına rastgele — tek yönde ölçmek kadranın o
            # yöndeki simetrisine bağlı bir sonuç verirdi.
            b = replace(b, egiklik_yon_deg=float(rng.uniform(0, 180)))
        if b.etkin:
            kare, _ = bozulmalar_uygula(kare, _truth_nesnesi(k), b, rng)

        # Güven eşiği BİLİNÇLİ olarak devre dışı (esik=0): İP15 eşiği bu veriden
        # kalibre edecek ve bunun için eşiğin ALTINDA kalan okumaların ne kadar
        # yanlış olduğunu görmek zorunda. Eşik burada uygulanırsa kalibrasyon
        # dairesel olur. İP14'ün kendi tablosunda "okunamayan" sütunu bu yüzden
        # yalnızca tespit/ibre başarısızlıklarını sayar, düşük güveni değil.
        s = read_frame(kare, model, gauge, detect_conf=conf,
                       perspektif=perspektif, esik=0.0)
        aralik = gauge.scale.max - gauge.scale.min

        kayit = {
            "file": k["file"], "gauge_id": k["gauge_id"], "gercek": k["value"],
            "olculen": s.reading.value if s.ok else None,
            "status": s.reading.status if s.reading else "detect_fail",
            "sebep": s.reason,
            # İP15'in kalibrasyonu bu iki alandan çıkacak.
            "conf": round(s.reading.conf, 4) if s.reading else 0.0,
            "yatiklik_kestirildi": s.roll is not None,
        }
        if kayit["olculen"] is not None:
            kayit["hata_yuzde"] = abs(kayit["olculen"] - kayit["gercek"]) / aralik * 100
        cikti.append(kayit)

    return cikti


def ozetle(kayitlar: list[dict]) -> dict:
    okunan = [k for k in kayitlar if k.get("hata_yuzde") is not None]
    return {
        "goruntu": len(kayitlar),
        "okunan": len(okunan),
        "okunamayan": len(kayitlar) - len(okunan),
        "tespit_edilemeyen": sum(1 for k in kayitlar if k["status"] == "detect_fail"),
        "yatiklik_kestirilen": sum(1 for k in kayitlar if k["yatiklik_kestirildi"]),
        "hata_yuzde_tam_skala": error_stats([k["hata_yuzde"] for k in okunan]).as_dict(2),
        "ortalama_conf": round(float(np.mean([k["conf"] for k in kayitlar])), 3),
    }


def figur(rapor: dict, cikti: Path) -> Path:
    """Her eksen için bozulma seviyesine karşı hata ve okunamayan oranı."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eksenler = [e for e in rapor["eksenler"] if rapor["eksenler"][e]]
    fig, axes = plt.subplots(1, len(eksenler), figsize=(4.0 * len(eksenler), 3.8),
                             squeeze=False)

    for eksen, ax in zip(eksenler, axes[0]):
        etiketler = list(rapor["eksenler"][eksen].keys())
        hata = [rapor["eksenler"][eksen][e]["hata_yuzde_tam_skala"]["ortalama"]
                for e in etiketler]
        okunamayan = [rapor["eksenler"][eksen][e]["okunamayan"] /
                      max(1, rapor["eksenler"][eksen][e]["goruntu"]) * 100
                      for e in etiketler]
        x = np.arange(len(etiketler))

        ax.plot(x, hata, "o-", color="#c0392b", label="hata (% tam skala)")
        ax.set_xticks(x)
        ax.set_xticklabels(etiketler, fontsize=8)
        ax.set_title(eksen, fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(5.0, color="gray", ls="--", lw=1)   # hedef

        ikinci = ax.twinx()
        ikinci.plot(x, okunamayan, "s--", color="#2980b9", lw=1,
                    label="okunamayan %")
        ikinci.set_ylim(0, 100)

    axes[0][0].set_ylabel("hata (% tam skala)")
    fig.suptitle("İP14 — koşul bazlı okuma hatası (kırmızı) ve okunamayan oranı (mavi)",
                 fontweight="bold")
    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=130)
    plt.close(fig)
    return cikti


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Zor koşullarda okuma hatası (İP14)")
    p.add_argument("--veri", default=VARSAYILAN_VERI)
    p.add_argument("--agirlik", default=VARSAYILAN_AGIRLIK)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--sahne", type=int, default=0, help="kaç görüntü (0 = hepsi)")
    p.add_argument("--eksen", default=None, choices=list(EKSENLER),
                   help="sadece bu ekseni tara")
    p.add_argument("--perspektif", action=argparse.BooleanOptionalAction, default=False,
                   help="perspektif düzeltmesini aç (ablasyon)")
    p.add_argument("--figur", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    veri, agirlik = Path(args.veri), Path(args.agirlik)
    if not (veri / "labels.jsonl").exists():
        print(f"veri seti yok: {veri}")
        return 1
    if not agirlik.exists():
        print(f"ağırlık yok: {agirlik}")
        return 1

    from ultralytics import YOLO
    model = YOLO(str(agirlik))
    gauges = load_gauges()

    kayitlar = load_labels(veri)
    if args.sahne:
        kayitlar = kayitlar[:args.sahne]

    secili = {args.eksen: EKSENLER[args.eksen]} if args.eksen else EKSENLER
    rapor = {
        "is_paketi": "IP14", "tarih": date.today().isoformat(),
        "veri_seti": str(veri).replace("\\", "/"),
        "goruntu_basina": len(kayitlar),
        "perspektif_duzeltme": args.perspektif,
        "eksenler": {},
    }
    ciftler = []

    for eksen, seviyeler in secili.items():
        print(f"\n=== {eksen.upper()} ===")
        print(f"{'seviye':>10s} {'hata%':>8s} {'p95':>8s} {'max':>8s} "
              f"{'okunamayan':>11s} {'tespit yok':>11s} {'conf':>6s}")
        rapor["eksenler"][eksen] = {}

        for etiket, bozulma in seviyeler:
            k = kosu(kayitlar, veri, gauges, model, bozulma, args.conf, args.perspektif)
            o = ozetle(k)
            rapor["eksenler"][eksen][etiket] = o
            h = o["hata_yuzde_tam_skala"]
            print(f"{etiket:>10s} {h['ortalama']:>8.2f} {h['p95']:>8.2f} {h['max']:>8.2f} "
                  f"{o['okunamayan']:>6d}/{o['goruntu']:<4d} "
                  f"{o['tespit_edilemeyen']:>6d}/{o['goruntu']:<4d} "
                  f"{o['ortalama_conf']:>6.2f}")

            for kayit in k:
                ciftler.append({"eksen": eksen, "seviye": etiket,
                                "conf": kayit["conf"],
                                "hata_yuzde": kayit.get("hata_yuzde"),
                                "status": kayit["status"]})

    yol = Path(METRIK_YOLU)
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(rapor, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nölçüm: {yol}")

    Path(CIFT_YOLU).write_text(json.dumps(ciftler, ensure_ascii=False), encoding="utf-8")
    print(f"güven-hata çiftleri (İP15 girdisi): {CIFT_YOLU}")

    if args.figur and not args.eksen:
        print(f"figür: {figur(rapor, Path(FIGUR_YOLU))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
