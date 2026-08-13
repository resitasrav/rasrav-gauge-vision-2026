"""karisik + data/raw/video_kareler_v1 (13 elle etiketlenmiş gerçek kare) ile
yeniden eğitir, aynı test kümesinde İP5'in karışık temeliyle kıyaslar.

Kalıcı bir İP5 yapılandırması değil — Reşit'in kendi videolarından (araba,
sunum) toplanan zor örneklerin genellemeye katkısını ölçen tek seferlik bir
deney. Sonuç iyiyse `karisik`'in yerine geçmesi ayrı bir karardır.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "src"))

from gauge_vision.detect.dataset import veri_yaml_yaz

KARISIK_TRAIN = KOK / "data" / "detect" / "karisik" / "train"
YENI_KARELER = KOK / "data" / "raw" / "video_kareler_v1"
HEDEF = KOK / "data" / "detect" / "karisik_v2" / "train"
VAL = KOK / "data" / "detect" / "_gercek" / "val" / "images"
TEST = KOK / "data" / "detect" / "_gercek" / "test" / "images"

if HEDEF.exists():
    shutil.rmtree(HEDEF)
(HEDEF / "images").mkdir(parents=True, exist_ok=True)
(HEDEF / "labels").mkdir(parents=True, exist_ok=True)

n = 0
for kaynak in (KARISIK_TRAIN, YENI_KARELER):
    for img in sorted((kaynak / "images").glob("*.png")):
        etiket = kaynak / "labels" / f"{img.stem}.txt"
        shutil.copyfile(img, HEDEF / "images" / img.name)
        shutil.copyfile(etiket, HEDEF / "labels" / f"{img.stem}.txt")
        n += 1

veri_yaml_yaz(KOK / "data" / "detect" / "karisik_v2" / "gauge.yaml",
              train=HEDEF / "images", val=VAL, test=TEST)
print(f"karisik_v2/train: {n} görüntü ({KARISIK_TRAIN.name}'dan {len(list((KARISIK_TRAIN/'images').glob('*.png')))} "
      f"+ video_kareler_v1'den {len(list((YENI_KARELER/'images').glob('*.png')))})")

# --- eğit + ölç: egit_ip5.py'nin egit()/merkez_hatasi() ile aynı tarif ---
sys.path.insert(0, str(KOK / "scripts"))
from egit_ip5 import egit, cihaz_sec  # noqa: E402

cihaz = cihaz_sec(None)
print(f"cihaz: {cihaz}")

sonuc = egit("karisik_v2", epoch=40, imgsz=416, batch=8, seed=0, cihaz=cihaz)
print(json.dumps(sonuc, indent=2, ensure_ascii=False))

yol = KOK / "outputs" / "metrics" / "ip5_tespit_karisik_v2.json"
yol.parent.mkdir(parents=True, exist_ok=True)
yol.write_text(json.dumps(sonuc, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nölçüm: {yol}")
