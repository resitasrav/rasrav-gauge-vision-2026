"""Sentetik kadranı elle deneme aracı (İP3).

Kaydırıcıyı çektikçe ibre oynar; değeri ve karşılık gelen açıyı pencerede
gösterir. Amacı gözle doğrulama: "8.5 bar'da ibre gerçekten 8 ile 9 arasında
mı" sorusunun cevabı okuma zincirinin geri kalanını yazmadan önce bilinmeli.

    python scripts/kadran_onizle.py                      # PT-101, kaydırıcılı
    python scripts/kadran_onizle.py TI-205               # başka gösterge
    python scripts/kadran_onizle.py FI-310 --deger 50    # tek kare, pencere yok
    python scripts/kadran_onizle.py FI-310 --deger 50 --kaydet bak.png
    python scripts/kadran_onizle.py --liste              # analog göstergeler

Pencere açıkken:  s = kaydet   ·   q / ESC = çık
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from gauge_vision.config import load_gauges
from gauge_vision.synth.dial import render_analog

SLIDER_STEPS = 1000        # kaydırıcı tamsayı ister; değeri bu çözünürlükte böl
PENCERE = "Sentetik kadran - s: kaydet, q: cikis"
BILGI_BGR = (60, 60, 60)


def _analog_gauges():
    return {gid: g for gid, g in load_gauges().items() if g.type == "analog"}


def _bilgi_yaz(img, gauge, value: float, angle: float):
    """Değeri ve açıyı görüntünün üstüne basar — kadranla yan yana okunabilsin."""
    kare = img.copy()
    satirlar = [
        f"{gauge.id}  {value:.2f} {gauge.unit or ''}",
        f"ibre acisi: {angle:+.1f} derece",
    ]
    for i, s in enumerate(satirlar):
        cv2.putText(kare, s, (12, 26 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BILGI_BGR, 1, cv2.LINE_AA)
    return kare


def _tek_kare(gauge, value: float, kaydet: str | None) -> int:
    img, truth = render_analog(gauge, value)
    print(f"{gauge.id}  deger {value:g} {gauge.unit or ''}  ->  "
          f"aci {truth.angle_deg:+.2f} derece  ·  ibre ucu {truth.tip_px}")
    if kaydet:
        yol = Path(kaydet)
        yol.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(yol), img)
        print(f"kaydedildi: {yol}")
    return 0


def _kaydiricili(gauge) -> int:
    scale = gauge.scale
    cv2.namedWindow(PENCERE)
    # Kaydırıcı yalnızca tamsayı taşır; 0..SLIDER_STEPS aralığını değere eşliyoruz.
    cv2.createTrackbar("deger", PENCERE, SLIDER_STEPS // 2, SLIDER_STEPS, lambda _: None)

    while True:
        adim = cv2.getTrackbarPos("deger", PENCERE)
        value = scale.min + (scale.max - scale.min) * adim / SLIDER_STEPS
        img, truth = render_analog(gauge, value)
        cv2.imshow(PENCERE, _bilgi_yaz(img, gauge, value, truth.angle_deg))

        tus = cv2.waitKey(30) & 0xFF
        if tus in (ord("q"), 27):
            break
        if tus == ord("s"):
            yol = Path("outputs/figures") / f"onizleme_{gauge.id}_{value:.2f}.png"
            yol.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(yol), img)
            print(f"kaydedildi: {yol}")
        # Pencere çarpıdan kapatıldıysa döngüde takılı kalma
        if cv2.getWindowProperty(PENCERE, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    # Gösterge adları Türkçe; Windows konsolunun varsayılan kod sayfası bunları
    # bozuyor. Çıktıyı UTF-8'e sabitlemek terminalde okunur kalmasını sağlıyor.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Sentetik kadranı gözle dene")
    p.add_argument("gauge_id", nargs="?", default="PT-101", help="örn. PT-101, TI-205, FI-310")
    p.add_argument("--deger", type=float, help="verilirse pencere açılmaz, tek kare üretir")
    p.add_argument("--kaydet", help="PNG yolu (--deger ile birlikte)")
    p.add_argument("--liste", action="store_true", help="analog göstergeleri listele ve çık")
    args = p.parse_args(argv)

    gauges = _analog_gauges()

    if args.liste:
        for gid, g in gauges.items():
            s = g.scale
            print(f"{gid:8s} {g.name:28s} {s.min:g}-{s.max:g} {g.unit}  "
                  f"{s.sweep_deg:.0f} derece {s.direction}"
                  f"{'  (karekok olcek)' if not s.linear else ''}")
        return 0

    if args.gauge_id not in gauges:
        print(f"'{args.gauge_id}' analog gosterge degil veya envanterde yok. "
              f"Secenekler: {', '.join(gauges)}", file=sys.stderr)
        return 2

    gauge = gauges[args.gauge_id]
    if args.deger is not None:
        return _tek_kare(gauge, args.deger, args.kaydet)
    return _kaydiricili(gauge)


if __name__ == "__main__":
    raise SystemExit(main())
