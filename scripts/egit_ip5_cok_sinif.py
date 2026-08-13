"""Dört tipli tespit modelini eğitir ve iki kümede birden ölçer (İP5 genişletmesi).

    python scripts/egit_ip5_cok_sinif.py
    python scripts/egit_ip5_cok_sinif.py --epoch 60

**İki ölçüm birden alınır ve ikisi de gereklidir:**

1. **Yeni tipler bulunuyor mu** — dört sınıfın kendi doğrulama bölümünde
   sınıf başına mAP. 17.08'de eksik olan buydu.
2. **Analog gerileme var mı** — 05.08'in *aynı* gerçek test bölümünde `gauge`
   sınıfının mAP'i. Yeni sınıflar eklenirken eskisinin bozulması, tespit
   başlığı büyüdüğü için gerçek bir risktir ve ölçülmeden bilinemez. Referans:
   mAP50 **0,9674** · merkez sapması **%4,02** kadran çapı.

İkincisi bu scriptin asıl varlık sebebidir: "yeni tipler de çalışıyor" cümlesi,
analog tarafın sessizce kötüleşmediği gösterilmeden kurulamaz.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from gauge_vision.config import load_gauges

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egit_ip5 import cihaz_sec, merkez_hatasi  # noqa: E402

VERI_YAML = Path("data/detect/cok_sinif/gauge4.yaml")
ANALOG_YAML = Path("data/detect/karisik/gauge.yaml")
KOSU_KOK = "models/ip5"
KOSU_ADI = "cok_sinif"
METRIK_YOLU = Path("outputs/metrics/ip5_cok_sinif.json")
TEMEL_MODEL = "yolov8n.pt"

# 05.08'de ölçülen, kıyas için sabitlenmiş taban.
ANALOG_TABAN = {"mAP50": 0.9674, "merkez_sapmasi_yuzde": 4.02}


def _sinif_bazli(olcum, adlar: list[str]) -> dict:
    """Sınıf başına mAP — toplam mAP tek bir sınıfın çökmesini gizleyebilir.

    ⚠ `class_result(i)`'nin `i`'si **sınıf kimliği değil**, sonuç dizisindeki
    sıradır (`ap_class_index` eşlemesi). Sınıf kimliğiyle indekslemek sessizce
    yanlış sayı üretir: ilk koşuda üç sınıf da aynı 0,9950'yi gösterdi ve dördüncü
    sınıf hiç görünmedi. Sayılar makul göründüğü için ancak "hepsi neden aynı"
    sorusuyla yakalandı.
    """
    sonuc: dict = {ad: None for ad in adlar}
    for sira, sinif_id in enumerate(olcum.box.ap_class_index):
        p, r, ap50, ap = olcum.box.class_result(sira)
        sonuc[adlar[int(sinif_id)]] = {
            "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4),
            "kesinlik": round(float(p), 4), "duyarlilik": round(float(r), 4)}
    return sonuc


def ip13_sahnelerinde(modeller: dict, tekrar: int = 20) -> dict:
    """17.08'in bulgusunun kapandığını **onun kendi girdisinde** gösterir.

    Doğrulama bölümündeki mAP yeterli değil: o küme bu scriptin ürettiği
    kareler, yani modelin eğitildiği dağılımın aynısı. Bulgu ise İP13'ün
    sahnelerinde çıkmıştı. Bu yüzden kıyas orada, **eski ağırlıkla yan yana**
    yapılıyor — tespit oranının yanında sınıfın da doğru olup olmadığı ölçülüyor;
    lambayı bulup "vana" demek, bulamamaktan daha kötüdür.
    """
    import numpy as np
    from gauge_vision.synth.degrade import Bozulma

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from olc_ip13 import _sahne  # noqa: E402  (kardeş script; sahne üretimi orada)

    envanter = load_gauges("configs/gauges.yaml")
    gauges = list(envanter.values()) if isinstance(envanter, dict) else list(envanter)
    beklenen = {"analog": "gauge", "digital": "digital", "lamp": "lamp", "valve": "valve"}

    sonuc: dict = {}
    for g in gauges:
        rng = np.random.default_rng(13)          # her model aynı kareleri görsün
        kareler = [_sahne(g, rng, Bozulma())[0] for _ in range(tekrar)]
        kayit: dict = {"tip": g.type}
        for ad, model in modeller.items():
            bulundu = dogru_sinif = 0
            for kare in kareler:
                r = model.predict(kare, conf=0.25, verbose=False)[0]
                if len(r.boxes) == 0:
                    continue
                bulundu += 1
                i = int(r.boxes.conf.argmax())
                if r.names[int(r.boxes.cls[i])] == beklenen[g.type]:
                    dogru_sinif += 1
            kayit[ad] = {"tespit_yuzde": round(100 * bulundu / tekrar, 1),
                         "dogru_sinif_yuzde": round(100 * dogru_sinif / tekrar, 1)}
        sonuc[g.id] = kayit
    return sonuc


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Dört tipli tespit eğitimi (İP5)")
    p.add_argument("--epoch", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cihaz", default=None)
    p.add_argument("--olc-yalniz", action="store_true",
                   help="eğitme, mevcut best.pt'yi yeniden ölç (ölçüm hatası düzeltilince)")
    args = p.parse_args(argv)

    if not VERI_YAML.exists():
        print(f"veri yok ({VERI_YAML}) — önce scripts/hazirla_ip5_cok_sinif.py")
        return 1

    from ultralytics import YOLO

    cihaz = cihaz_sec(args.cihaz)
    print(f"cihaz: {cihaz}")

    agirlik = Path("runs/detect") / KOSU_KOK / KOSU_ADI / "weights" / "best.pt"
    if args.olc_yalniz:
        if not agirlik.exists():
            print(f"ağırlık yok ({agirlik}) — önce --olc-yalniz olmadan koştur")
            return 1
        model = YOLO(str(agirlik))
    else:
        model = YOLO(TEMEL_MODEL)
        model.train(data=str(VERI_YAML), epochs=args.epoch, imgsz=args.imgsz,
                    batch=args.batch, seed=args.seed, device=cihaz,
                    project=KOSU_KOK, name=KOSU_ADI, exist_ok=True,
                    verbose=False, plots=False, val=True)

    # 1) Dört sınıfın kendi doğrulama bölümü.
    # workers=0: Windows'ta spawn edilen DataLoader süreçleri, script düz
    # koşturulduğunda bootstrap hatası veriyor.
    dort = model.val(data=str(VERI_YAML), split="val", verbose=False, workers=0)
    adlar = [dort.names[i] for i in sorted(dort.names)]

    # 2) 05.08'in aynı gerçek test bölümü — analog gerileme denetimi.
    analog = model.val(data=str(ANALOG_YAML), split="test", verbose=False, workers=0)
    merkez = merkez_hatasi(model, ANALOG_YAML)

    # 3) 17.08'in bulgusu kendi girdisinde kapandı mı — eski ağırlıkla yan yana.
    eski_yol = Path("runs/detect/models/ip5/karisik/weights/best.pt")
    modeller = {"yeni_dort_sinif": model}
    if eski_yol.exists():
        modeller = {"eski_tek_sinif": YOLO(str(eski_yol)), **modeller}
    ip13 = ip13_sahnelerinde(modeller)

    ozet = {
        "is_paketi": "IP5-cok-sinif",
        "tarih": date.today().isoformat(),
        "temel_model": TEMEL_MODEL,
        "egitim": {"epoch": args.epoch, "imgsz": args.imgsz, "batch": args.batch,
                   "seed": args.seed, "cihaz": cihaz},
        "dort_sinif_dogrulama": {
            "mAP50": round(float(dort.box.map50), 4),
            "mAP50_95": round(float(dort.box.map), 4),
            "sinif_bazli": _sinif_bazli(dort, adlar),
        },
        "analog_gerileme_denetimi": {
            "test_kumesi": "gerçek fotoğraflar (Roboflow-100 gauge-u2lwv test bölümü) — 05.08 ile aynı",
            "mAP50": round(float(analog.box.map50), 4),
            "mAP50_95": round(float(analog.box.map), 4),
            "taban_05_08": ANALOG_TABAN,
            **merkez,
        },
        "ip13_sahnelerinde_tespit": {
            "not": "17.08 bulgusunun kendi girdisinde kapanma denetimi; 20 kare/gösterge",
            "gostergeler": ip13,
        },
        "agirlik": str(Path(KOSU_KOK) / KOSU_ADI / "weights" / "best.pt"),
    }
    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    METRIK_YOLU.write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== dört sınıf (kendi doğrulama bölümü) ===")
    print(f"  genel mAP50 {ozet['dort_sinif_dogrulama']['mAP50']:.4f}")
    for ad, v in ozet["dort_sinif_dogrulama"]["sinif_bazli"].items():
        if v:
            print(f"  {ad:8s} mAP50 {v['mAP50']:.4f}  duyarlılık {v['duyarlilik']:.4f}")

    a = ozet["analog_gerileme_denetimi"]
    print("\n=== analog gerileme denetimi (05.08 ile aynı test kümesi) ===")
    print(f"  mAP50 {a['mAP50']:.4f}  (taban {ANALOG_TABAN['mAP50']})")
    print(f"  merkez sapması %{a['merkez_sapmasi_yuzde_kadran_capi']['ortalama']}"
          f"  (taban %{ANALOG_TABAN['merkez_sapmasi_yuzde']})")

    print("\n=== İP13 sahnelerinde tespit (17.08 bulgusu kapandı mı) ===")
    basliklar = [k for k in next(iter(ip13.values())) if k != "tip"]
    print(f"  {'gösterge':10s} {'tip':8s} " + "  ".join(f"{b:>18s}" for b in basliklar))
    for gid, kayit in ip13.items():
        hucre = [f"%{kayit[b]['tespit_yuzde']:<5.0f} (sınıf %{kayit[b]['dogru_sinif_yuzde']:.0f})"
                 for b in basliklar]
        print(f"  {gid:10s} {kayit['tip']:8s} " + "  ".join(f"{h:>18s}" for h in hucre))

    print(f"\nölçüm: {METRIK_YOLU}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
