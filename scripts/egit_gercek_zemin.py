"""Tespiti gerçek endüstriyel zeminde genelleşecek biçimde yeniden eğitir.

    python scripts/egit_gercek_zemin.py
    python scripts/egit_gercek_zemin.py --epoch 80

**Neden bu eğitim var — ölçülmüş boşluk.** 26.08'de dört videoda tespit
sistematik olarak çöktü (bir videonun altı örnek karesinin dördünde hiç
tespit yok). Sebep 27.08'de sayıya döküldü: mevcut dört sınıflı model gerçek
endüstriyel zeminli kadranlarda **`gauge` mAP50 = 0,3925** veriyor — aynı
modelin kendi sentetik test kümesindeki değeri 0,9632'ydi. Yani ilan edilen
0,96 kendi çizdiğimiz zeminlerin sayısıydı; sahne değişince tespit yarıdan
fazla kayboluyor.

**Üç ölçüm birden alınır ve üçü de gereklidir:**

1. **Gerçek zeminde kazanç** — `data/detect/gercek_zemin` val bölümünde
   `gauge` mAP50. Taban 0,3925; bu eğitimin varlık sebebi budur.
2. **Azınlık sınıflar korunuyor mu** — `digital`/`lamp`/`valve` aynı val'de.
   Yeni küme yalnız `gauge` getiriyor; hazırlama scripti dengeyi `--tekrar`
   ile ~7:1'e çekiyor ama bu bir ödündür ve ölçülmeden "gerileme yok"
   denemez. Taban: üçü de 0,9950.
3. **Analog gerileme var mı** — 05.08'in *aynı* gerçek test bölümünde
   (`data/detect/karisik`) `gauge` mAP50. Referans **0,9674** / dört sınıflı
   modelde 0,9632. Yeni sahneler öğrenilirken eski kümede kötüleşme olabilir
   ve bu ancak aynı bölümde ölçülerek görülür.

**Sıfırdan değil, mevcut ağırlıktan devam edilir.** Dört sınıflı model üç
azınlık sınıfı %0'dan %100'e çıkarmıştı (13.08); o kazanç rastgele
başlatmayla yeniden aranmaz, üstüne eklenir.

⚠ **Eğitim bitişini `results.csv` satır sayısıyla yoklama.** Ultralytics son
epoch satırını yazdıktan SONRA doğrulama yapıp `best.pt`'yi bir kez daha
yazar (optimizer ayıklanır, 24 MB → 6 MB). 21.08'de yarım yazılmış ağırlığa
saatlerce ölçüm yapıldı. Bu script sırayı kendi içinde tuttuğu için sorun
yok; elle koşuluyorsa sürecin kendi bitiş sinyali beklenmeli.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

VERI_YAML = Path("data/detect/gercek_zemin/gauge4.yaml")
ANALOG_YAML = Path("data/detect/karisik/gauge.yaml")
COK_SINIF_YAML = Path("data/detect/cok_sinif/gauge4.yaml")
BASLANGIC = Path("runs/detect/models/ip5/cok_sinif/weights/best.pt")
KOSU_KOK = "models/ip5"
KOSU_ADI = "gercek_zemin"
METRIK_YOLU = Path("outputs/metrics/ip5_gercek_zemin.json")

SINIF_ADLARI = ("gauge", "digital", "lamp", "valve")

# 27.08'de mevcut modelle ölçülen taban — kıyas için sabitlendi.
TABAN = {"gercek_zemin_gauge_mAP50": 0.3925, "gercek_zemin_mAP50": 0.8444,
         "digital": 0.9950, "lamp": 0.9950, "valve": 0.9950}
# 05.08 / 13.08 referansları (aynı analog test bölümü).
ANALOG_TABAN = {"tek_sinif_mAP50": 0.9674, "dort_sinif_mAP50": 0.9632}


def _sinif_map(sonuc) -> dict[str, float]:
    """Sınıf adı → mAP50. Kümede hiç örneği olmayan sınıf listede görünmez.

    ⚠ `box.ap50` GLOBAL sınıf kimliğiyle değil, **doğrulama kümesinde bulunan**
    sınıflarla indekslenir; eşleme `ap_class_index` üzerinden yapılmalıdır.
    Doğrudan `ap50[i]` yazmak sessizce yanlış etiket üretir: `cok_sinif` val
    bölümünde hiç `gauge` örneği yok, buna rağmen ilk sıradaki sayı (aslında
    `digital`'in) "gauge" diye raporlanıyordu.
    """
    kutu = sonuc.box
    indisler = getattr(kutu, "ap_class_index", None)
    if indisler is None:
        return {}
    cikti: dict[str, float] = {}
    for sira, sinif_id in enumerate(indisler):
        sinif_id = int(sinif_id)
        if 0 <= sinif_id < len(SINIF_ADLARI):
            cikti[SINIF_ADLARI[sinif_id]] = round(float(kutu.ap50[sira]), 4)
    return cikti


def olc(model, veri_yaml: Path, imgsz: int, ad: str) -> dict:
    if not veri_yaml.exists():
        print(f"  [atlandı] {ad}: {veri_yaml} yok")
        return {}
    r = model.val(data=str(veri_yaml), imgsz=imgsz, split="val", verbose=False,
                  plots=False, project="runs/detect/_olcum", name=ad, exist_ok=True)
    return {"mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4),
            "sinif_mAP50": _sinif_map(r)}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Gerçek zeminli tespit eğitimi (İP5+)")
    p.add_argument("--epoch", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baslangic", default=str(BASLANGIC))
    p.add_argument("--veri", default=str(VERI_YAML),
                   help="eğitim kümesi yaml'ı; ölçüm her hâlde ilk kümede de yapılır "
                        "ki farklı karışımlar aynı bölümde kıyaslanabilsin")
    p.add_argument("--ad", default=KOSU_ADI, help="koşu adı (runs/detect altında)")
    p.add_argument("--sadece-olc", action="store_true",
                   help="eğitme, yalnız mevcut ağırlığı üç kümede ölç")
    args = p.parse_args(argv)

    from ultralytics import YOLO
    import torch

    cihaz = 0 if torch.cuda.is_available() else "cpu"
    print(f"cihaz: {cihaz} · torch {torch.__version__}")
    if cihaz == "cpu":
        print("⚠ GPU görünmüyor — eğitim CPU'da çok yavaş olur. "
              "requirements.txt başındaki cu126 kurulumuna bak.")

    egitim_yaml = Path(args.veri)
    if not egitim_yaml.exists():
        print(f"küme yok: {egitim_yaml}\n"
              "önce: python scripts/hazirla_gercek_zemin.py --birlestir")
        return 1

    baslangic = Path(args.baslangic)
    if not baslangic.exists():
        print(f"başlangıç ağırlığı yok: {baslangic}")
        return 1

    if args.sadece_olc:
        model = YOLO(str(baslangic))
        agirlik = baslangic
    else:
        model = YOLO(str(baslangic))
        print(f"eğitim başlıyor: {args.epoch} epoch · imgsz {args.imgsz} · "
              f"batch {args.batch} · başlangıç {baslangic} · küme {egitim_yaml}")
        model.train(data=str(egitim_yaml), epochs=args.epoch, imgsz=args.imgsz,
                    batch=args.batch, seed=args.seed, device=cihaz,
                    project=KOSU_KOK, name=args.ad, exist_ok=True, verbose=False)
        # `runs/detect` öneki Ultralytics'in kendi kök dizininden gelir: `project`
        # göreli verildiğinde altına yazar. Önek atlanırsa eğitim biter ama
        # ölçüm var olmayan bir dosyayı arar.
        agirlik = Path("runs/detect") / KOSU_KOK / args.ad / "weights" / "best.pt"
        # Eğitim nesnesi yerine DİSKTEKİ ağırlık yeniden yükleniyor: ölçülen
        # şey kaydedilen dosya olmalı, bellekteki son durum değil.
        model = YOLO(str(agirlik))

    print("\nölçümler:")
    sonuc = {
        "tarih": date.today().isoformat(),
        "agirlik": str(agirlik),
        "egitim": None if args.sadece_olc else {
            "epoch": args.epoch, "imgsz": args.imgsz, "batch": args.batch,
            "seed": args.seed, "baslangic": str(baslangic),
            "veri": str(VERI_YAML)},
        "taban": {**TABAN, "analog": ANALOG_TABAN},
    }
    sonuc["gercek_zemin"] = olc(model, VERI_YAML, args.imgsz, "gercek_zemin")
    sonuc["analog_eski_test"] = olc(model, ANALOG_YAML, args.imgsz, "analog_eski")
    sonuc["cok_sinif_eski"] = olc(model, COK_SINIF_YAML, args.imgsz, "cok_sinif_eski")

    gz = sonuc["gercek_zemin"].get("sinif_mAP50", {})
    print(f"\n{'küme':22s} {'mAP50':>8s}  sınıf kırılımı")
    for ad in ("gercek_zemin", "analog_eski_test", "cok_sinif_eski"):
        d = sonuc.get(ad) or {}
        if not d:
            continue
        print(f"{ad:22s} {d['mAP50']:>8.4f}  {d.get('sinif_mAP50', {})}")

    yeni = gz.get("gauge")
    if yeni is not None:
        fark = yeni - TABAN["gercek_zemin_gauge_mAP50"]
        print(f"\nGERÇEK ZEMİNDE gauge: {TABAN['gercek_zemin_gauge_mAP50']:.4f} "
              f"→ {yeni:.4f}  ({fark:+.4f})")
        sonuc["kazanc_gauge_mAP50"] = round(fark, 4)

    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    # Koşu adı dosya adına giriyor: farklı karışımların sayıları birbirini
    # ezmemeli, yoksa "hangi karışım neyi verdi" sorusu cevapsız kalır.
    metrik = (METRIK_YOLU if args.ad == KOSU_ADI
              else METRIK_YOLU.with_name(f"ip5_{args.ad}.json"))
    metrik.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nyazıldı: {metrik}")
    print(f"ağırlık: {agirlik}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
