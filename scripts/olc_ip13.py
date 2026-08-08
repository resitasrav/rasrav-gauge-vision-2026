"""Dört gösterge tipi tek zincirde — İP13'ün çekirdeği.

    python scripts/olc_ip13.py
    python scripts/olc_ip13.py --yayinla        # okumaları inspect/reading'e bas
    python scripts/olc_ip13.py --tur 5 --zor

Bir **devriye turu** simüle eder ve `pipeline.read_gauge` üzerinden okur —
yani her tip kendi fonksiyonundan değil, **tek giriş noktasından** geçer. İP13'ün
asıl işi budur: dört tip ayrı ayrı çalışıyordu ama hiçbiri zincire bağlı değildi.

Analog dalında tespit gerçek YOLO modelinden gelir. Diğer üç tipte tespit
kutusu, üretilen görüntünün kendi sınırıdır: İP5 yalnızca analog kadran üzerinde
eğitildi, dijital panel ve lambayı tanımıyor. **Bu bilinçli bir sınırdır ve
gizlenmiyor** — gerçek turda kutuyu yine İP5 verecek, dolayısıyla o modelin
diğer tipleri de kapsayacak şekilde eğitilmesi gerekiyor (bkz. Notlar).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.pipeline import read_gauge
from gauge_vision.publish.reading import ReadingPublisher, SemaHatasi
from gauge_vision.synth.degrade import Bozulma, bozulmalar_uygula
from gauge_vision.synth.dial import DialTruth, render_analog
from gauge_vision.synth.digital import render_digital
from gauge_vision.synth.state import render_lamp, render_valve

VARSAYILAN_AGIRLIK = "runs/detect/models/ip5/karisik/weights/best.pt"
METRIK_YOLU = "outputs/metrics/ip13_zincir_tum_tipler.json"
TOHUM = 13


class _SahteModel:
    """Tespit yerine tüm kareyi kutu olarak veren yer tutucu.

    İP5 yalnızca analog kadranda eğitildi; dijital panel, lamba ve vanayı
    tanımıyor. Bu sınıf o boşluğu **görünür** kılar: gerçek turda kutuyu YOLO
    verecek, burada kare sınırı veriliyor. Sessizce analog modeli çağırmak,
    tespit hiç yokmuş gibi bir sonuç üretirdi.
    """

    class _Kutular:
        def __init__(self, h, w):
            self.xyxy = np.array([[0.0, 0.0, float(w), float(h)]])
            self.conf = np.array([1.0])

        def __len__(self):
            return 1

    class _Sonuc:
        def __init__(self, h, w):
            self.boxes = _SahteModel._Kutular(h, w)

    def predict(self, image, **_):
        h, w = image.shape[:2]
        return [_SahteModel._Sonuc(h, w)]


def _sahne(gauge, rng, bozulma: Bozulma):
    """Gösterge tipine göre bir kare üretir. `(görüntü, gerçek_değer)` döner."""
    if gauge.type == "analog":
        deger = float(rng.uniform(gauge.scale.min, gauge.scale.max))
        img, truth = render_analog(gauge, deger)
        gercek = round(deger, gauge.decimals)
    elif gauge.type == "digital":
        a = gauge.raw.get("range") or {}
        deger = float(rng.uniform(float(a.get("min", 0)), float(a.get("max", 100))))
        img, _ = render_digital(gauge, deger)
        gercek = round(deger, int((gauge.digits or {}).get("decimals", 1)))
        truth = None
    else:
        gercek = gauge.state_names[int(rng.integers(0, len(gauge.state_names)))]
        if gauge.type == "lamp":
            img, _ = render_lamp(gauge, gercek)
        else:
            img, _ = render_valve(gauge, gercek, sapma_deg=float(rng.uniform(-10, 10)))
        truth = None

    if bozulma.etkin:
        h, w = img.shape[:2]
        sahte = truth or DialTruth(
            gauge_id=gauge.id, value=0.0, angle_deg=0.0, roll_deg=0.0,
            angle_img_deg=0.0, center_px=(w // 2, h // 2), tip_px=(0, 0),
            radius_px=min(h, w) // 3, bbox_xyxy=(0, 0, w, h))
        img, _ = bozulmalar_uygula(img, sahte, bozulma, rng)

    return img, gercek


def _dogru_mu(gauge, gercek, okunan) -> bool | None:
    """Okuma doğru mu? Analog/dijitalde tolerans, durumda tam eşleşme.

    None = okunamadı (doğru da yanlış da değil).
    """
    if okunan is None:
        return None
    if gauge.type in ("lamp", "valve"):
        return okunan == gercek
    aralik = ((gauge.scale.max - gauge.scale.min) if gauge.type == "analog"
              else (lambda a: float(a.get("max", 100)) - float(a.get("min", 0)))
              (gauge.raw.get("range") or {}))
    # %5 tam skala — projenin hedef toleransı.
    return abs(float(okunan) - float(gercek)) <= 0.05 * aralik


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Dört tip tek zincirde (İP13)")
    p.add_argument("--agirlik", default=VARSAYILAN_AGIRLIK)
    p.add_argument("--tur", type=int, default=20, help="kaç devriye turu")
    p.add_argument("--zor", action="store_true", help="bozulmalı koşullarda da koş")
    p.add_argument("--yayinla", action="store_true", help="inspect/reading'e bas")
    args = p.parse_args(argv)

    gauges = load_gauges()
    agirlik = Path(args.agirlik)

    analog_model = None
    if agirlik.exists():
        from ultralytics import YOLO
        analog_model = YOLO(str(agirlik))
    else:
        print(f"⚠ ağırlık yok ({agirlik}) — analog dalı da yer tutucu tespitle koşacak")

    sahte = _SahteModel()
    yayinci = None
    if args.yayinla:
        yayinci = ReadingPublisher(source="ip13", zorla_dosya=True)
        print(f"yayın modu: {yayinci.baglan()}")

    kosullar: list[tuple[str, Bozulma]] = [("temiz", Bozulma())]
    if args.zor:
        kosullar += [("bulanık 9px", Bozulma(bulaniklik_px=9)),
                     ("düşük ışık ×0.4", Bozulma(isik_kazanci=0.4)),
                     ("jpeg q25", Bozulma(jpeg_kalite=25))]

    rapor = {"is_paketi": "IP13", "tarih": date.today().isoformat(),
             "tur": args.tur, "kosullar": {}}

    for kosul_ad, bozulma in kosullar:
        print(f"\n=== {kosul_ad.upper()} ===")
        print(f"{'gösterge':>10s} {'tip':>8s} {'doğru':>7s} {'yanlış':>7s} "
              f"{'okunamayan':>11s} {'güven':>7s} {'ms':>7s}")
        rapor["kosullar"][kosul_ad] = {}
        rng = np.random.default_rng(TOHUM)

        for gauge in gauges.values():
            model = analog_model if (gauge.type == "analog" and analog_model) else sahte
            dogru = yanlis = okunamayan = 0
            guvenler, sureler = [], []

            for _ in range(args.tur):
                img, gercek = _sahne(gauge, rng, bozulma)
                t0 = time.perf_counter()
                s = read_gauge(img, model, gauge)
                sureler.append((time.perf_counter() - t0) * 1000)
                guvenler.append(s.reading.conf if s.reading else 0.0)

                sonuc = _dogru_mu(gauge, gercek,
                                  s.reading.value if s.reading else None)
                if sonuc is None:
                    okunamayan += 1
                elif sonuc:
                    dogru += 1
                else:
                    yanlis += 1

                if yayinci and s.reading is not None:
                    try:
                        yayinci.yayinla(s.reading, img_ref=f"{kosul_ad}/{gauge.id}.jpg")
                    except SemaHatasi as e:
                        print(f"   ✗ {gauge.id}: {e}")

            rapor["kosullar"][kosul_ad][gauge.id] = {
                "tip": gauge.type, "tur": args.tur,
                "dogru": dogru, "yanlis": yanlis, "okunamayan": okunamayan,
                "dogruluk": round(dogru / args.tur, 3),
                "ortalama_guven": round(float(np.mean(guvenler)), 3),
                "ms": round(float(np.mean(sureler)), 1),
            }
            print(f"{gauge.id:>10s} {gauge.type:>8s} {dogru:>7d} {yanlis:>7d} "
                  f"{okunamayan:>11d} {np.mean(guvenler):>7.2f} {np.mean(sureler):>7.1f}")

    # Toplam: sessiz yanlış okuma sayısı — asıl izlenecek sayı.
    toplam_yanlis = sum(g["yanlis"] for k in rapor["kosullar"].values()
                        for g in k.values())
    toplam_kare = sum(g["tur"] for k in rapor["kosullar"].values() for g in k.values())
    rapor["toplam"] = {"kare": toplam_kare, "sessiz_yanlis": toplam_yanlis}
    print(f"\nTOPLAM {toplam_kare} kare · sessiz yanlış okuma: {toplam_yanlis}")

    if yayinci:
        yayinci.kapat()
        print(f"yayınlanan {yayinci.gonderilen} · reddedilen {yayinci.reddedilen}")
        rapor["yayin"] = {"gonderilen": yayinci.gonderilen,
                          "reddedilen": yayinci.reddedilen}

    yol = Path(METRIK_YOLU)
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(rapor, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ölçüm: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
