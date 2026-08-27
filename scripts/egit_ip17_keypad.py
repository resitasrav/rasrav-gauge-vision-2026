"""Beş sınıflı tespit modelini eğitir ve GERÇEK VİDEODA ölçer (İP17).

    python scripts/hazirla_karistiricilar.py
    python scripts/hazirla_ip17_keypad.py
    python scripts/egit_ip17_keypad.py
    python scripts/egit_ip17_keypad.py --sadece-olc

**Ölçüt doğrulama mAP'i DEĞİL.** 26.08'de öğrenilen ders: gerçek zemin setiyle
eğitim in-domain mAP'i 0,39'dan 0,995'e çıkardı ve ALAN DIŞI videoda hiçbir
kazanç vermedi, hatta gerileme oldu. Doğrulama kümesi bu scriptin ürettiği
karelerdir, yani modelin eğitildiği dağılımın aynısı; oradaki sayı yalnız
"eğitim çöktü mü" sorusuna cevap verir.

Gerçek ölçüt iki tanesi ve ikisi de 14 gerçek videoda:
  1. Kadranın OLMADIĞI videolarda üretilen `gauge` kutusu — düşmeli.
     (27.08 tabanı: 383 kutu, hepsi "başarıyla" okundu)
  2. Kadranın OLDUĞU videolarda tespit — düşmemeli.
     (27.08 tabanı: 4787 analog kutu)

Eski ağırlık yan yana ölçülüyor; kazanç ispatlanamazsa üretimde eski kalır.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

VERI_YAML = KOK / "data/detect/keypad5/gauge5.yaml"
KOSU_KOK = "models/ip5"
KOSU_ADI = "keypad5"
AGIRLIK = KOK / "runs/detect" / KOSU_KOK / KOSU_ADI / "weights/best.pt"
ESKI_AGIRLIK = KOK / "runs/detect/models/ip5/cok_sinif/weights/best.pt"
METRIK = KOK / "outputs/metrics/ip17_keypad.json"
VIDEO_KOK = KOK.parent / "demo" / "girdi" / "video"

# Bu videolarda okunabilir analog kadran YOK — burada `gauge` kutusu = yanlış.
KADRAN_YOK = ["1", "2", "10", "11", "genis", "genis2"]
# Bu videolarda gerçek manometre var — burada tespit kaybı = gerileme.
KADRAN_VAR = ["4", "5s", "6", "8"]

# 27.08 tabanı (eski ağırlık, kapı öncesi tam koşu) — kıyas için sabit.
TABAN = {"kadran_yok_kutu": 383, "kadran_var_kutu": 4787}


def _sinif_bazli(olcum, adlar: list[str]) -> dict:
    """Sınıf başına mAP.

    ⚠ `class_result(i)`'nin `i`'si sınıf kimliği DEĞİL, sonuç dizisindeki
    sıradır (`ap_class_index` eşlemesi). Sınıf kimliğiyle indekslemek sessizce
    yanlış sayı üretir — 26.08'de bir val kümesinde sıfır gauge örneği varken
    "gauge 0,995" raporlandı.
    """
    sonuc: dict = {ad: None for ad in adlar}
    for sira, sinif_id in enumerate(olcum.box.ap_class_index):
        p, r, ap50, ap = olcum.box.class_result(sira)
        sonuc[adlar[int(sinif_id)]] = {
            "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4),
            "kesinlik": round(float(p), 4), "duyarlilik": round(float(r), 4)}
    return sonuc


def video_tespit(model, adlar: list[str], conf: float, kare_sayisi: int) -> dict:
    """Her videoda sınıf başına kutu sayar. Okuma yapmaz — ölçülen şey TESPİT."""
    import cv2
    from gauge_vision.pipeline import detect_objects

    sonuc: dict = {}
    for ad in adlar:
        yol = VIDEO_KOK / f"{ad}.mp4"
        if not yol.exists():
            continue
        cap = cv2.VideoCapture(str(yol))
        toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sayim: dict[str, int] = {}
        for f in np.linspace(0, max(toplam - 1, 0), kare_sayisi).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, kare = cap.read()
            if not ok:
                continue
            for d in detect_objects(kare, model, conf=conf):
                sayim[d.sinif] = sayim.get(d.sinif, 0) + 1
        cap.release()
        sonuc[ad] = sayim
    return sonuc


def _model(yol: Path):
    from ultralytics import YOLO
    if not yol.exists():
        raise SystemExit(f"agirlik yok: {yol}")
    return YOLO(str(yol))


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # 27.08 ölçümü (RTX 4050, 6 GB): imgsz 416 / batch 16 ile GPU kullanımı %57,
    # VRAM 1,3/6,1 GB. Yani kart zamanın %43'ünü veri bekleyerek geçiriyordu ve
    # bellek boştu. imgsz 640 asıl kazanç: gerçek videolarda kadran yarıçapı
    # 27-70 px ölçüldü; 1080p kareyi 416'ya indirmek 27 px'lik kadranı ~10 px'e
    # düşürüyor ve model onu göremiyor.
    p.add_argument("--epoch", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--workers", type=int, default=4,
                   help="veri yukleyici isci sayisi; RAM darsa dusur")
    p.add_argument("--conf", type=float, default=0.25,
                   help="olcumde dusuk tutulur: KAPI degil TESPIT olculuyor")
    p.add_argument("--kare", type=int, default=100, help="video basina ornek kare")
    p.add_argument("--sadece-olc", action="store_true")
    a = p.parse_args(argv)

    from egit_ip5 import cihaz_sec  # noqa: E402
    ozet: dict = {"is_paketi": "IP17", "veri": str(VERI_YAML.relative_to(KOK))}

    if not a.sadece_olc:
        if not VERI_YAML.exists():
            print(f"veri yok ({VERI_YAML}) — once scripts/hazirla_ip17_keypad.py")
            return 1
        from ultralytics import YOLO
        cihaz = cihaz_sec()
        print(f"egitim: {a.epoch} epoch · imgsz {a.imgsz} · cihaz {cihaz}")
        # Temel model COCO agirligi: sinif sayisi 4'ten 5'e ciktigi icin eski
        # basligin agirliklari zaten yeniden kurulacakti; temiz baslamak
        # karsilastirmayi da basitlestiriyor.
        model = YOLO("yolov8n.pt")
        # `project` MUTLAKA "models/ip5" — Ultralytics'in `runs_dir` ayarı zaten
        # `runs/detect` ve project onun ALTINA ekleniyor. Başına "runs/detect"
        # yazmak yolu ikiye katlıyor (runs/detect/runs/detect/...) ve eğitim
        # sorunsuz biterken ölçüm ağırlığı bulamıyor. Kardeş scriptlerin hepsi
        # çıplak `KOSU_KOK` geçiyor; buradaki sapma 27.08'de bir koşuya mal oldu.
        model.train(data=str(VERI_YAML), epochs=a.epoch, imgsz=a.imgsz,
                    batch=a.batch, workers=a.workers, seed=0, device=cihaz,
                    project=KOSU_KOK, name=KOSU_ADI, exist_ok=True)

    if not AGIRLIK.exists():
        print(f"agirlik yok: {AGIRLIK}")
        return 1

    yeni = _model(AGIRLIK)
    adlar = [yeni.names[i] for i in sorted(yeni.names)]
    olcum = yeni.val(data=str(VERI_YAML), imgsz=a.imgsz, verbose=False)
    ozet["dogrulama"] = {"mAP50": round(float(olcum.box.map50), 4),
                         "sinif_basina": _sinif_bazli(olcum, adlar)}
    print(f"\ndogrulama mAP50: {ozet['dogrulama']['mAP50']}  "
          f"(alan ICI — olcut degil)")

    eski = _model(ESKI_AGIRLIK)
    ozet["video"] = {}
    for etiket, videolar in (("kadran_yok", KADRAN_YOK), ("kadran_var", KADRAN_VAR)):
        e = video_tespit(eski, videolar, a.conf, a.kare)
        y = video_tespit(yeni, videolar, a.conf, a.kare)
        e_gauge = sum(v.get("gauge", 0) for v in e.values())
        y_gauge = sum(v.get("gauge", 0) for v in y.values())
        ozet["video"][etiket] = {"eski": e, "yeni": y,
                                 "eski_gauge_toplam": e_gauge,
                                 "yeni_gauge_toplam": y_gauge}
        print(f"\n--- {etiket} ---")
        for v in sorted(set(e) | set(y)):
            print(f"   {v:8s} eski {e.get(v, {})}")
            print(f"   {'':8s} yeni {y.get(v, {})}")
        print(f"   gauge kutusu: eski {e_gauge} -> yeni {y_gauge}")

    METRIK.parent.mkdir(parents=True, exist_ok=True)
    METRIK.write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{METRIK.relative_to(KOK)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
