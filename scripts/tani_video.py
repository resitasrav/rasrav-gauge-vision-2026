r"""Videoda zincirin ne yaptığını kare kare gösterir — iki ağırlık kıyaslanabilir.

    python scripts\tani_video.py --video demo\girdi\gosterge.mp4
    python scripts\tani_video.py --klasor demo\girdi --karsilastir <eski.pt>

**Neden bu script var.** 26.08'de dört videoda (ev çekimi manometre, araç
göstergesi, termometre masası, üretilmiş fabrika koridoru) iki kusur gözle
görüldü ve sonra sayıya döküldü:

1. **Tespit çöküyordu** — `gosterge.mp4`'ün altı örnek karesinin dördünde hiç
   tespit yoktu.
2. **Kimliği bilinmeyen her kadran `bar` okunuyordu** — 0-120 °C'lik bir
   termometre "2,2 bar · ok · güven 0,724", bir devir saati "0,8 bar" diye
   yayınlandı. Yanlış cihaz, yanlış birim, yüksek güven.

Script ikisini birden görünür kılar: kare başına tespit sayısı/tipi ve
`read_all_analog` ile HER analog kutunun açısı. Değer/birim üretilmez —
kimlik beyanla gelir (U11), görüntüden çıkarılamıyor.

**Bu bir DOĞRULUK ölçümü değildir** (`olc_uretilmis_video.py` ile aynı sınır):
videodaki kadranların ground truth'u yok. Ölçülen şey "kaç kadran bulunuyor ve
geometrisi çözülebiliyor mu"dur.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.pipeline import detect_objects, read_all_analog

# Üretim ağırlığı dört sınıflı `cok_sinif` OLARAK KALIYOR. 27.08'de gerçek
# zeminli açık setle iki eğitim denendi (HF payı %60 ve %28); ikisi de kendi
# doğrulama kümesinde `gauge` mAP50'yi 0,3925 → 0,9950 çıkardı ama **videolarda
# gerileme üretti** ve eşik düşürmek (conf 0,25 → 0,10) kurtarmadı. Ayrıntı
# `docs/devam_notu.md` §2d.
VARSAYILAN_AGIRLIK = "runs/detect/models/ip5/cok_sinif/weights/best.pt"
METRIK_YOLU = Path("outputs/metrics/video_tani.json")
FIGUR_KLASORU = Path("outputs/figures/video_tani")

RENK_ANALOG = (0, 165, 255)
RENK_DIGER = (160, 160, 160)


def _model(yol: Path):
    if not yol.exists():
        raise SystemExit(f"ağırlık yok: {yol}")
    from ultralytics import YOLO
    return YOLO(str(yol))


def _kare_ciz(kare, tespitler, okumalar):
    ciz = kare.copy()
    for t in tespitler:
        x1, y1, x2, y2 = (int(v) for v in t.box_xyxy)
        renk = RENK_ANALOG if t.tip == "analog" else RENK_DIGER
        cv2.rectangle(ciz, (x1, y1), (x2, y2), renk, 2)
        if x2 - x1 >= 90:
            cv2.putText(ciz, f"{t.tip} {t.conf:.2f}", (x1 + 4, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2, cv2.LINE_AA)
    for o in okumalar:
        if not o.ok:
            continue
        cv2.circle(ciz, o.center_px, int(o.radius_px), RENK_ANALOG, 2, cv2.LINE_AA)
        cv2.line(ciz, o.center_px, o.needle.tip_px, RENK_ANALOG, 2, cv2.LINE_AA)
        x1, y1 = int(o.box_xyxy[0]), int(o.box_xyxy[1])
        cv2.putText(ciz, f"aci {o.needle.angle_img_deg:.0f} · birim ?",
                    (x1 + 4, min(int(o.box_xyxy[3]) - 8, y1 + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RENK_ANALOG, 2, cv2.LINE_AA)
    return ciz


def video_isle(video: Path, model, kare_sayisi: int, conf: float,
               figur_klasoru: Path | None) -> dict:
    cap = cv2.VideoCapture(str(video))
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if toplam <= 0:
        cap.release()
        return {"hata": "video açılamadı ya da boş"}

    indeksler = np.linspace(0, toplam - 1, kare_sayisi, dtype=int)
    tespit_sayilari, analog_sayilari, okunan_sayilari = [], [], []
    tespitli_kare = 0
    for sira, idx in enumerate(indeksler):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, kare = cap.read()
        if not ok:
            continue
        tespitler = detect_objects(kare, model, conf=conf)
        okumalar = read_all_analog(kare, model, tespitler=tespitler)
        tespit_sayilari.append(len(tespitler))
        analog_sayilari.append(sum(1 for t in tespitler if t.tip == "analog"))
        okunan_sayilari.append(sum(1 for o in okumalar if o.ok))
        tespitli_kare += 1 if tespitler else 0
        if figur_klasoru is not None:
            figur_klasoru.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(figur_klasoru / f"{video.stem}_{idx:06d}.png"),
                        _kare_ciz(kare, tespitler, okumalar))
    cap.release()

    n = max(len(tespit_sayilari), 1)
    return {
        "kare": len(tespit_sayilari),
        "tespitli_kare_orani": round(tespitli_kare / n, 3),
        "kare_basina_tespit": round(float(np.mean(tespit_sayilari or [0])), 2),
        "kare_basina_analog": round(float(np.mean(analog_sayilari or [0])), 2),
        "kare_basina_okunan_analog": round(float(np.mean(okunan_sayilari or [0])), 2),
        "en_cok_analog": int(max(analog_sayilari or [0])),
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Videoda tespit + çoklu analog okuma")
    p.add_argument("--video", type=Path, help="tek video")
    p.add_argument("--klasor", type=Path, help="klasördeki bütün videolar")
    p.add_argument("--agirlik", type=Path, default=Path(VARSAYILAN_AGIRLIK))
    p.add_argument("--karsilastir", type=Path, default=None,
                   help="ikinci ağırlık — iki model yan yana raporlanır")
    p.add_argument("--kare", type=int, default=12, help="video başına örnek kare")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--figur", action="store_true", help="işaretli kareleri kaydet")
    args = p.parse_args(argv)

    if args.video:
        videolar = [args.video]
    elif args.klasor:
        videolar = sorted(v for v in args.klasor.iterdir()
                          if v.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
    else:
        print("--video ya da --klasor gerekli")
        return 1
    if not videolar:
        print("video bulunamadı")
        return 1

    agirliklar = {"yeni": args.agirlik}
    if args.karsilastir:
        agirliklar = {"eski": args.karsilastir, "yeni": args.agirlik}

    sonuc: dict[str, dict] = {}
    for etiket, yol in agirliklar.items():
        print(f"\n=== {etiket}: {yol}")
        model = _model(yol)
        sonuc[etiket] = {}
        for v in videolar:
            klasor = (FIGUR_KLASORU / etiket) if args.figur else None
            sonuc[etiket][v.stem] = video_isle(v, model, args.kare, args.conf, klasor)

    baslik = list(agirliklar)
    print(f"\n{'video':32s} " + "  ".join(
        f"{e:>28s}" for e in baslik))
    print(f"{'':32s} " + "  ".join(
        f"{'tespitli%  analog/kare  okunan':>28s}" for _ in baslik))
    for v in videolar:
        satir = f"{v.stem[:32]:32s} "
        for e in baslik:
            d = sonuc[e][v.stem]
            if "hata" in d:
                satir += f"{d['hata']:>28s}  "
            else:
                satir += (f"{100*d['tespitli_kare_orani']:>8.0f}%"
                          f"{d['kare_basina_analog']:>13.2f}"
                          f"{d['kare_basina_okunan_analog']:>8.2f}  ")
        print(satir)

    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    METRIK_YOLU.write_text(json.dumps(
        {"tarih": date.today().isoformat(),
         "agirliklar": {k: str(v) for k, v in agirliklar.items()},
         "kare_basina": args.kare, "conf": args.conf, "sonuc": sonuc},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nyazıldı: {METRIK_YOLU}")
    if args.figur:
        print(f"işaretli kareler: {FIGUR_KLASORU}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
