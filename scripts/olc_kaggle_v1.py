"""Kaggle'da eğitilen genelleme ağırlığını yerel tabanla kıyaslar (İP5 ek deney).

    python scripts/olc_kaggle_v1.py
    python scripts/olc_kaggle_v1.py --video-yok

**Deneyin sorusu.** İP5'in `karisik` ağırlığı, kendi test kümesinde mAP50 0,967
veriyor ama o küme eğitim dağılımına çok yakın. Modelin **hiç görmediği** bir
gösterge türünde (araç gösterge paneli, sunum videosu) ne yaptığı ayrı bir
sorudur ve tek başına mAP ile cevaplanamaz.

`kaggle_v1`, yerel GPU yerine Kaggle'da, üç ek kaynakla eğitildi: kendi
videolarımızdan elle etiketlenmiş 13 zor kare, Open Images'ın `Clock` sınıfı
(yuvarlak kadran biçimi) ve Endava'nın sentetik gösterge seti.

**İki ölçüm birden alınır çünkü ikisi zıt yönde hareket edebilir:**

1. *Dar test kümesi* — 05.08'in aynı gerçek bölümü. Genelleme için eğitilen bir
   model burada bir miktar **gerileyebilir**; bu beklenen bir bedeldir.
2. *Kendi videolarımız* — etiket yok, o yüzden mAP hesaplanamaz; ölçülen şey
   **tespit oranı** (kaç karede gösterge bulundu) ve ortalama güven. Deneyin
   asıl amacı buydu.

Birinci sayıya bakıp "kötüleşti" demek yanıltıcı olurdu; bu yüzden ikisi de aynı
dosyaya yazılıyor.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egit_ip5 import merkez_hatasi  # noqa: E402

KOK = Path(__file__).resolve().parents[1]
AGIRLIKLAR = {
    "yerel_karisik": KOK / "runs/detect/models/ip5/karisik/weights/best.pt",
    "kaggle_v1": KOK / "models/ip5/kaggle_v1/weights/best.pt",
}
ANALOG_YAML = KOK / "data/detect/karisik/gauge.yaml"
VIDEO_KOK = KOK.parent / "demo/girdi"
VIDEOLAR = ("gosterge.mp4", "araba.mp4")
METRIK_YOLU = KOK / "outputs/metrics/ip5_kaggle_v1.json"

ADIM = 5            # her 5. kare — 1300+ karelik videoda tam tarama gereksiz
ESIKLER = (0.25, 0.50)


def test_kumesinde(model, yaml_yolu: Path) -> dict:
    olcum = model.val(data=str(yaml_yolu), split="test", verbose=False, workers=0)
    return {
        "mAP50": round(float(olcum.box.map50), 4),
        "mAP50_95": round(float(olcum.box.map), 4),
        "kesinlik": round(float(olcum.box.mp), 4),
        "duyarlilik": round(float(olcum.box.mr), 4),
        **merkez_hatasi(model, yaml_yolu),
    }


def videoda(model, yol: Path) -> dict:
    """Etiketsiz videoda tespit oranı ve güven.

    Doğruluk değil **kapsama** ölçülüyor: etiket olmadığı için bulunan kutunun
    doğru yerde olduğu bu sayıdan çıkarılamaz. Yanlış yorumlanmasın diye alan
    adı `tespit_orani`, `dogruluk` değil.
    """
    import cv2

    cap = cv2.VideoCapture(str(yol))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    toplam = 0
    sayac = {e: 0 for e in ESIKLER}
    guvenler: list[float] = []

    for idx in range(0, n, ADIM):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, kare = cap.read()
        if not ok:
            continue
        toplam += 1
        sonuc = model.predict(kare, conf=0.05, verbose=False)[0]
        if len(sonuc.boxes) == 0:
            continue
        en_iyi = float(sonuc.boxes.conf.max())
        guvenler.append(en_iyi)
        for e in ESIKLER:
            if en_iyi >= e:
                sayac[e] += 1
    cap.release()

    return {
        "kare": n,
        "orneklenen": toplam,
        "tespit_orani": {f"conf>={e:.2f}": round(100 * sayac[e] / toplam, 1) if toplam else 0.0
                         for e in ESIKLER},
        "ortalama_guven": round(sum(guvenler) / len(guvenler), 3) if guvenler else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="kaggle_v1 vs yerel karisik (İP5 ek deney)")
    p.add_argument("--video-yok", action="store_true",
                   help="yalnız test kümesi ölçümü (videolar demo/ altında, git dışı)")
    args = p.parse_args(argv)

    from ultralytics import YOLO

    eksik = [ad for ad, y in AGIRLIKLAR.items() if not y.exists()]
    if eksik:
        print("ağırlık yok:", ", ".join(f"{ad} → {AGIRLIKLAR[ad]}" for ad in eksik))
        return 1

    sonuc: dict = {}
    for ad, yol in AGIRLIKLAR.items():
        model = YOLO(str(yol))
        kayit: dict = {"agirlik": str(yol.relative_to(KOK)),
                       "test_kumesi": test_kumesinde(model, ANALOG_YAML)}
        if not args.video_yok:
            kayit["videolar"] = {}
            for v in VIDEOLAR:
                yol_v = VIDEO_KOK / v
                if yol_v.exists():
                    kayit["videolar"][v] = videoda(model, yol_v)
        sonuc[ad] = kayit

    ozet = {
        "is_paketi": "IP5-ek-deney",
        "tarih": date.today().isoformat(),
        "soru": "Kaggle'da genelleme icin egitilen agirlik, kendi videolarimizda "
                "yerel tabandan iyi mi? Dar test kumesinde bedeli ne?",
        "video_notu": "Etiket yok — olculen sey dogruluk degil TESPIT ORANI (kapsama).",
        "modeller": sonuc,
    }
    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    METRIK_YOLU.write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")

    for ad, k in sonuc.items():
        t = k["test_kumesi"]
        print(f"\n=== {ad} ===")
        print(f"  test kümesi : mAP50 {t['mAP50']:.4f}  "
              f"merkez sapması %{t['merkez_sapmasi_yuzde_kadran_capi']['ortalama']}")
        for v, d in k.get("videolar", {}).items():
            print(f"  {v:14s}: " + "  ".join(f"{e} %{o}" for e, o in d["tespit_orani"].items())
                  + f"  ort_güven {d['ortalama_guven']}")
    print(f"\nölçüm: {METRIK_YOLU}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
