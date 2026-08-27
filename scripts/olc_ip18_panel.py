"""Pano tipi metrede (kare çerçeve, yay skala) okuma hatası (İP18).

    python scripts/olc_ip18_panel.py
    python scripts/olc_ip18_panel.py --n 400 --roll 8

İki soruya cevap veriyor:

**1. Mevcut ibre okuyucusu bu geometride çalışıyor mu?** Kutupsal tarama
"merkezden çıkan ışın boyunca kesintisiz koyu şerit" arıyor. Yuvarlak kadranda
tarama halkası (0,22R-0,72R) her yönde kadran yüzünün içinde kalıyor. Pano
metresinde pivot ALT kenara yakın, dolayısıyla aşağı bakan ışınlar siyah
ÇERÇEVEYE çarpıyor ve orada da kesintisiz koyu bir şerit var. Ölçülen şey tam
olarak bu: ibre mi kazanıyor, çerçeve mi?

**2. Pivotu envanterden almanın kazancı ne?** Kutu merkezini pivot sanmak
okumayı kırmaz, sistematik olarak KAYDIRIR — en sinsi hata türü. İki
yapılandırma yan yana ölçülüyor.

Ground truth üreteçten geliyor (`synth/panel.py`), yani hata yalnız okuma
yöntemine ait.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

from gauge_vision.config import load_gauges                       # noqa: E402
from gauge_vision.pipeline import dial_from_box                   # noqa: E402
from gauge_vision.read.calibrate import read_value                # noqa: E402
from gauge_vision.read.needle import (angle_difference_deg,       # noqa: E402
                                      read_needle_angle)
from gauge_vision.synth.panel import render_panel_meter           # noqa: E402

METRIK = KOK / "outputs/metrics/ip18_panel.json"


def _ozet(hatalar: list[float], kapsam: int, n: int) -> dict:
    if not hatalar:
        return {"kapsam": 0.0, "okunan": 0, "toplam": n}
    h = np.abs(np.array(hatalar))
    return {"kapsam": round(kapsam / n, 3), "okunan": kapsam, "toplam": n,
            "ort": round(float(h.mean()), 3), "medyan": round(float(np.median(h)), 3),
            "p95": round(float(np.percentile(h, 95)), 3),
            "max": round(float(h.max()), 3),
            "ters_180": int((h > 150).sum())}


def kos(gauge, n: int, roll: float, pivot_kaynak: str,
        pencere_ac: bool = True) -> tuple[dict, dict]:
    """`pivot_kaynak`: 'envanter' | 'kutu_merkezi' (yanlış varsayımın bedeli).

    `pencere_ac` taramayı envanterdeki yayla sınırlar. Kapalıyken ölçülen şey
    27.08 tabanıdır: tarama ibre yerine siyah çerçeveyi buluyordu.
    """
    rng = np.random.default_rng(18)
    aci_hatalari: list[float] = []
    deger_hatalari: list[float] = []
    okunan = 0
    for _ in range(n):
        deger = float(rng.uniform(gauge.scale.min, gauge.scale.max))
        r_deg = float(rng.uniform(-roll, roll)) if roll else 0.0
        img, truth = render_panel_meter(gauge, deger, roll_deg=r_deg)

        if pivot_kaynak == "envanter":
            merkez, yaricap = dial_from_box(truth.bbox_xyxy, gauge)
        else:
            # Bilinçli YANLIŞ: yuvarlak kadran varsayımı (kutu merkezi).
            merkez, yaricap = dial_from_box(truth.bbox_xyxy, None)

        # Pencere GÖRÜNTÜ açısında: kamera yatıksa yay da o kadar dönmüş olur.
        a0, a1 = gauge.scale.ccw_araligi
        pencere = (a0 + r_deg, a1 + r_deg) if pencere_ac else None
        okuma = read_needle_angle(img, merkez, yaricap, method="polar",
                                  aci_penceresi=pencere)
        if okuma is None:
            continue
        okunan += 1
        aci_hatalari.append(angle_difference_deg(okuma.angle_img_deg,
                                                 truth.angle_img_deg))
        sonuc = read_value(gauge, okuma.angle_img_deg, roll_deg=r_deg,
                           confidence=okuma.confidence, esik=0.0)
        if sonuc.value is not None:
            aralik = gauge.scale.max - gauge.scale.min
            deger_hatalari.append(100.0 * (float(sonuc.value) - deger) / aralik)

    aci = _ozet(aci_hatalari, okunan, n)
    deger = _ozet(deger_hatalari, len(deger_hatalari), n)
    return aci, deger


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gosterge", default="EM-501")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--roll", type=float, default=0.0, help="kamera yatikligi (+/- derece)")
    a = p.parse_args(argv)

    gauge = load_gauges()[a.gosterge]
    if gauge.face_shape != "panel":
        print(f"{a.gosterge}: face.shape 'panel' degil")
        return 1

    ozet: dict = {"is_paketi": "IP18", "gosterge": a.gosterge, "n": a.n,
                  "roll_deg": a.roll, "kosular": {}}
    kosular = [("envanter + yay penceresi", "envanter", True),
               ("envanter, pencere YOK", "envanter", False),
               ("kutu merkezi + pencere", "kutu_merkezi", True),
               ("kutu merkezi, pencere YOK", "kutu_merkezi", False)]
    for etiket, kaynak, pencere_ac in kosular:
        aci, deger = kos(gauge, a.n, a.roll, kaynak, pencere_ac)
        ozet["kosular"][etiket] = {"aci_hatasi_deg": aci, "deger_hatasi_yuzde": deger}
        print(f"--- {etiket} ---")
        print(f"    aci    kapsam {aci['kapsam']:.2f}  ort {aci.get('ort')}  "
              f"p95 {aci.get('p95')}  max {aci.get('max')}  ters180 {aci.get('ters_180')}")
        print(f"    deger  ort %{deger.get('ort')}  p95 %{deger.get('p95')}  "
              f"max %{deger.get('max')}")

    METRIK.parent.mkdir(parents=True, exist_ok=True)
    METRIK.write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{METRIK.relative_to(KOK)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
