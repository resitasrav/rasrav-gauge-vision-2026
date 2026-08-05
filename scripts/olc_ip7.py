"""Okuma hatasını ölçer: görüntü → açı → değer (İP7).

    python scripts/olc_ip7.py
    python scripts/olc_ip7.py --veri data/synthetic/v0 --yontem polar

Hedef metriğin (analog okuma ortalama hata < %5) sentetik veri üzerindeki ilk
gerçek ölçümüdür. Zincirin İP5 dışındaki tüm adımları buradadır: kırpılmış
kadran → ibre açısı (İP6) → yatıklık düzeltmesi → açı→değer (İP7).

**Hata neyin yüzdesi?** Burada **tam skalanın** yüzdesi olarak raporlanır:
`|ölçülen − gerçek| / (max − min) × 100`. Gerekçe: gösterge doğruluğu
endüstride tam skala üzerinden tanımlanır ve okunan değerin yüzdesi kadran
başında anlamsızlaşır (0,2 bar'da %5, 0,01 bar demektir — hiçbir analog
gösterge bunu vermez). Okunan değerin yüzdesi de tabloya ayrıca yazılır ki
danışman hangi tanımı kastettiğini söylediğinde sayı yeniden koşturulmasın.

İki ayrı **ablasyon** koşulur; ikisi de "bu satır olmasa ne olurdu" sorusunun
sayısal cevabıdır:

    --ablasyon  →  yatıklık düzeltmesi kapalı  · FI-310'a doğrusal formül
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from gauge_vision.config import Gauge, load_gauges
from gauge_vision.read.calibrate import DURUM_OK, DURUM_ALARM, read_value
from gauge_vision.read.evaluate import error_stats, read_dataset
from gauge_vision.read.needle import METHODS

VARSAYILAN_VERI = "data/synthetic/v0"
METRIK_YOLU = "outputs/metrics/ip7_okuma_hatasi.json"
FIGUR_YOLU = "outputs/figures/ip7_okuma_hatasi.png"

HEDEF_YUZDE = 5.0   # İP16'nın kabul ölçütü — tablo buna göre yorumlanır


def okuma_kosusu(
    veri: Path,
    gauges: dict[str, Gauge],
    *,
    yontem: str = "polar",
    roll_duzelt: bool = True,
    dogrusal_zorla: bool = False,
    **read_kwargs,
) -> list[dict]:
    """Veri setini uçtan uca okur; her kare için gerçek ve ölçülen değeri döner."""
    kayitlar = []
    for s in read_dataset(veri, method=yontem, **read_kwargs):
        gauge = gauges[s.gauge_id]
        if dogrusal_zorla and gauge.scale is not None:
            # `linear: false` alanı koda ulaşmasaydı ne olurdu — envanterdeki
            # tek bir satırın sessizce kaybolmasının bedeli ölçülüyor.
            gauge = replace(gauge, scale=replace(gauge.scale, linear=True))

        if not s.ok:
            kayitlar.append({"gauge_id": s.gauge_id, "gercek": s.value,
                             "olculen": None, "status": "unreadable"})
            continue

        okuma = read_value(
            gauge,
            angle_img_deg=s.measured_angle_deg,
            roll_deg=s.roll_deg if roll_duzelt else 0.0,
            confidence=s.confidence,
        )
        kayitlar.append({
            "gauge_id": s.gauge_id,
            "gercek": s.value,
            "olculen": okuma.value,
            "status": okuma.status,
        })
    return kayitlar


def hata_ozeti(kayitlar: list[dict], gauges: dict[str, Gauge]) -> dict:
    """Gösterge bazlı ve genel hata tablosu — tam skala ve okunan değer yüzdesiyle."""
    gostergeler: dict[str, dict] = {}
    tum_yuzde: list[float] = []

    for gid in sorted({k["gauge_id"] for k in kayitlar}):
        g = gauges[gid]
        aralik = g.scale.max - g.scale.min
        satirlar = [k for k in kayitlar if k["gauge_id"] == gid]
        okunan = [k for k in satirlar if k["olculen"] is not None]

        birim_hatalar = [abs(k["olculen"] - k["gercek"]) for k in okunan]
        ts_yuzde = [h / aralik * 100 for h in birim_hatalar]
        # Okunan değerin yüzdesi: kadran başındaki sıfıra yakın değerlerde
        # patladığı için medyanı da yazılıyor, ortalama tek başına yanıltır.
        od_yuzde = [abs(k["olculen"] - k["gercek"]) / max(abs(k["gercek"]), 1e-9) * 100
                    for k in okunan]

        tum_yuzde += ts_yuzde
        gostergeler[gid] = {
            "birim": g.unit,
            "olcek": f"{g.scale.min:g}-{g.scale.max:g}",
            "dogrusal": g.scale.linear,
            "okunan": len(okunan),
            "okunamayan": len(satirlar) - len(okunan),
            "hata_birim": error_stats(birim_hatalar).as_dict(),
            "hata_yuzde_tam_skala": error_stats(ts_yuzde).as_dict(),
            "hata_yuzde_okunan_deger": error_stats(od_yuzde).as_dict(),
        }

    genel = error_stats(tum_yuzde)
    return {
        "goruntu": len(kayitlar),
        "okunan": sum(1 for k in kayitlar if k["olculen"] is not None),
        "hata_yuzde_tam_skala": genel.as_dict(),
        "hedef_yuzde": HEDEF_YUZDE,
        "hedef_saglandi": genel.mean < HEDEF_YUZDE,
        "durum_dagilimi": {d: sum(1 for k in kayitlar if k["status"] == d)
                           for d in sorted({k["status"] for k in kayitlar})},
        "gosterge_bazli": gostergeler,
    }


def figur(kayitlar: list[dict], gauges: dict[str, Gauge], cikti: Path) -> Path:
    """Hata kadranın neresinde büyüyor — değer eksenine karşı hata.

    Ortalama tek sayıdır; kadranın belirli bir bölgesinde toplanan hatayı
    saklar. Karekök ölçekli FI-310'da hatanın alt uçta büyümesi beklenir
    (orada aynı açı farkı daha çok değere karşılık gelir) — grafik bunu
    doğruluyor mu, gözle görülsün.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gidler = sorted({k["gauge_id"] for k in kayitlar})
    fig, eksenler = plt.subplots(1, len(gidler), figsize=(4.2 * len(gidler), 3.8),
                                 squeeze=False)

    for eksen, gid in zip(eksenler[0], gidler):
        g = gauges[gid]
        aralik = g.scale.max - g.scale.min
        okunan = [k for k in kayitlar if k["gauge_id"] == gid and k["olculen"] is not None]
        x = [k["gercek"] for k in okunan]
        y = [abs(k["olculen"] - k["gercek"]) / aralik * 100 for k in okunan]

        eksen.scatter(x, y, s=16, alpha=0.75)
        eksen.set_title(f"{gid}  ({'doğrusal' if g.scale.linear else 'karekök'})")
        eksen.set_xlabel(f"gerçek değer ({g.unit})")
        eksen.set_ylabel("hata (% tam skala)")
        eksen.grid(alpha=0.3)
        eksen.set_ylim(bottom=0)

    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=130)
    plt.close(fig)
    return cikti


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Okuma hatasını ölç (İP7)")
    p.add_argument("--veri", default=VARSAYILAN_VERI)
    p.add_argument("--yontem", choices=METHODS, default="polar")
    p.add_argument("--ablasyon", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--figur", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    veri = Path(args.veri)
    if not (veri / "labels.jsonl").exists():
        print(f"veri seti yok: {veri} — önce scripts/uret_sentetik.py çalıştır")
        return 1

    gauges = load_gauges()
    kayitlar = okuma_kosusu(veri, gauges, yontem=args.yontem)
    ozet = {
        "is_paketi": "IP7",
        "tarih": date.today().isoformat(),
        "veri_seti": str(veri).replace("\\", "/"),
        "aci_yontemi": args.yontem,
        "ana_olcum": hata_ozeti(kayitlar, gauges),
    }

    ana = ozet["ana_olcum"]
    print(f"okuma hatası (% tam skala): ortalama {ana['hata_yuzde_tam_skala']['ortalama']:.3f}"
          f"  medyan {ana['hata_yuzde_tam_skala']['medyan']:.3f}"
          f"  p95 {ana['hata_yuzde_tam_skala']['p95']:.3f}"
          f"  max {ana['hata_yuzde_tam_skala']['max']:.3f}")
    print(f"hedef <%{HEDEF_YUZDE:g}  →  {'SAĞLANDI' if ana['hedef_saglandi'] else 'SAĞLANMADI'}")
    print(f"durum dağılımı: {ana['durum_dagilimi']}")
    for gid, satir in ana["gosterge_bazli"].items():
        print(f"   {gid:8s} {satir['hata_birim']['ortalama']:8.4f} {satir['birim']:6s}"
              f"  = %{satir['hata_yuzde_tam_skala']['ortalama']:.3f} tam skala"
              f"  (okunan {satir['okunan']})")

    if args.ablasyon:
        print("\nablasyon — bu adım olmasaydı:")
        ablasyonlar = {
            "yatiklik_duzeltmesi_kapali": okuma_kosusu(veri, gauges, yontem=args.yontem,
                                                       roll_duzelt=False),
            "fi310_dogrusal_kabul": okuma_kosusu(veri, gauges, yontem=args.yontem,
                                                 dogrusal_zorla=True),
        }
        ozet["ablasyon"] = {}
        for ad, kayit in ablasyonlar.items():
            o = hata_ozeti(kayit, gauges)
            ozet["ablasyon"][ad] = o
            print(f"   {ad:32s} ortalama %{o['hata_yuzde_tam_skala']['ortalama']:.3f}"
                  f"  max %{o['hata_yuzde_tam_skala']['max']:.3f}"
                  f"  kadran dışı {o['durum_dagilimi'].get('out_of_range', 0)}")

    metrik = Path(METRIK_YOLU)
    metrik.parent.mkdir(parents=True, exist_ok=True)
    metrik.write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nölçüm: {metrik}")

    if args.figur:
        print(f"figür: {figur(kayitlar, gauges, Path(FIGUR_YOLU))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
