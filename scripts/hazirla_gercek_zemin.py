"""Gerçek endüstriyel zeminli açık kadran setini YOLO eğitim kümesine çevirir.

    python scripts/hazirla_gercek_zemin.py
    python scripts/hazirla_gercek_zemin.py --limit 2000 --birlestir

**Neden bu script var.** 26.08'de mevcut zincir dört videoda (ev yapımı çekim,
araç göstergesi, termometre masası, üretilmiş fabrika koridoru) denendi ve
tespit tarafında sistematik bir boşluk ölçüldü: `gosterge.mp4`'ün altı
örnek karesinin dördünde HİÇ tespit yok, fabrika koridorunda duvardaki
~10 manometreden 2'si bulunuyordu. Okuma katmanı suçsuz — bulunan kutularda
ibre açısı sorunsuz çözülüyor. Eksik olan, modelin **sahne** görmemiş olması:
İP5'in eğitim kümesindeki kadranlar bizim çizdiğimiz düz zeminlerin üstünde
duruyor, gerçek fabrikada ise boru, flanş, kablo, yansıma ve derinlik var.

**Bu setin ne olduğu (ve ne OLMADIĞI) — dürüstçe.** Kaynak
`Synanthropic/reading-analog-gauge` (Hugging Face, herkese açık). Kadranların
kendisi **render**, ama gerçek endüstriyel fotoğrafların üstüne gerçekçi
perspektif, ölçek ve ışıkla bindirilmiş. Yani bu set gerçek fotoğrafın yerini
TUTMAZ; kapattığı şey kadranın kendisi değil **bağlamıdır**. Sentetik zeminle
gerçek saha arasında bir basamaktır ve etiketi bedavadır (dosya adında).

**Etiket biçimi.** Dosya adı dört köşeyi mutlak pikselle taşır:
`topLeft_topRight_bottomRight_bottomLeft_pin_<tip>_<kalinlik>_..._<id>.jpg`
(1280×1280 karede, `x-y` çiftleri). Kutu bu dörtgenin sınırlayıcı
dikdörtgenidir. Dörtgenin kendisi de saklanıyor (`corners.jsonl`): kadran
düzlemi eğik olduğunda köşeler perspektifi taşır ve `detect/perspective.py`
ileride bununla ölçülebilir — bilgi bir kez okunmuşken atılmaz.

**Sınıf kimliği `gauge` = 0'da bırakıldı** (bkz. `detect/dataset.py`): kümenin
mevcut dört sınıflı etiketlerle karıştırılabilmesi için şart. `--birlestir`
bunu yapar; birleştirilmezse eğitim diğer üç tipi UNUTUR (felaketvari unutma),
bu yüzden bayrak bilinçli olarak açıkça istenir.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.detect.dataset import (IMAGES_DIR, LABELS_DIR, SINIFLAR_COK,
                                         veri_yaml_yaz, yolo_satiri)

KAYNAK_ZIP = Path("data/raw/hf_analog_gauge/corners.zip")
HEDEF_KOK = Path("data/detect/gercek_zemin")
KAYNAK_COK_SINIF = Path("data/detect/cok_sinif")
METRIK_YOLU = Path("outputs/metrics/gercek_zemin_veri.json")

KAYNAK_KARE = 1280              # setin kendi çözünürlüğü (dosya adı buna göre)
HEDEF_KARE = 640                # eğitim imgsz=416; 640 fazlasıyla yeter, disk yarılanır
VAL_ORANI = 0.2


def _kose_ayikla(ad: str) -> list[tuple[int, int]] | None:
    """Dosya adındaki `x-y` çiftlerini köşe listesine çevirir.

    Yalnız İLK dört çift alınır: addaki `pin_Long_0.09_1_7953` kuyruğunda da
    tire geçebilir ve beşinci bir "köşe" uydurulursa kutu sessizce büyür.
    """
    koseler: list[tuple[int, int]] = []
    for parca in Path(ad).stem.split("_"):
        if "-" not in parca:
            continue
        a, _, b = parca.partition("-")
        if not (a.isdigit() and b.isdigit()):
            continue
        koseler.append((int(a), int(b)))
        if len(koseler) == 4:
            return koseler
    return None


def _onizleme(hedef: Path, ornekler: list[tuple[np.ndarray, list]], n: int = 6) -> Path:
    """Kutuları çizip tek ızgaraya basar — etiket doğruluğu GÖZLE denetlensin.

    Köşelerden türetilmiş bir kutu sessizce kayabilir (yanlış sıra, yanlış
    ölçek) ve bu ancak eğitim bittikten sonra mAP olarak görünür. Beş dakikalık
    bir bakış o riski baştan kapatır.
    """
    kareler = []
    for goruntu, koseler in ornekler[:n]:
        ciz = goruntu.copy()
        pts = np.array(koseler, dtype=np.int32)
        cv2.polylines(ciz, [pts], True, (0, 165, 255), 2)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        cv2.rectangle(ciz, (x1, y1), (x2, y2), (60, 220, 60), 2)
        kareler.append(cv2.resize(ciz, (320, 320)))
    while len(kareler) < n:
        kareler.append(np.zeros((320, 320, 3), np.uint8))
    izgara = np.vstack([np.hstack(kareler[:3]), np.hstack(kareler[3:6])])
    yol = hedef / "onizleme_etiket.png"
    cv2.imwrite(str(yol), izgara)
    return yol


def kur(zip_yolu: Path, hedef: Path, limit: int | None, tohum: int) -> dict:
    if not zip_yolu.exists():
        raise SystemExit(
            f"kaynak yok: {zip_yolu}\n"
            "indir: curl -L -o data/raw/hf_analog_gauge/corners.zip "
            "https://huggingface.co/datasets/Synanthropic/reading-analog-gauge/"
            "resolve/main/corners.zip")

    z = zipfile.ZipFile(zip_yolu)
    adlar = [a for a in z.namelist() if a.lower().endswith((".jpg", ".png"))]
    rastgele = random.Random(tohum)
    rastgele.shuffle(adlar)
    if limit is not None:
        adlar = adlar[:limit]

    for bolum in ("train", "val"):
        for alt in (IMAGES_DIR, LABELS_DIR):
            (hedef / bolum / alt).mkdir(parents=True, exist_ok=True)

    olcek = HEDEF_KARE / KAYNAK_KARE
    sayim = {"train": 0, "val": 0}
    atlanan = 0
    ornekler: list[tuple[np.ndarray, list]] = []
    kose_kaydi = []

    for i, ad in enumerate(adlar):
        koseler = _kose_ayikla(ad)
        if koseler is None:
            atlanan += 1
            continue
        ham = cv2.imdecode(np.frombuffer(z.read(ad), np.uint8), cv2.IMREAD_COLOR)
        if ham is None:
            atlanan += 1
            continue
        if ham.shape[0] != KAYNAK_KARE or ham.shape[1] != KAYNAK_KARE:
            # Dosya adındaki koordinatlar 1280'e göre; kare başka boyuttaysa
            # ölçek varsayımı çöker ve kutu sessizce yanlış yere oturur.
            atlanan += 1
            continue

        goruntu = cv2.resize(ham, (HEDEF_KARE, HEDEF_KARE))
        kucuk = [(int(x * olcek), int(y * olcek)) for x, y in koseler]
        xs = [p[0] for p in kucuk]
        ys = [p[1] for p in kucuk]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
            atlanan += 1
            continue

        bolum = "val" if rastgele.random() < VAL_ORANI else "train"
        kok = Path(ad).stem[:60]
        temel = f"hf_{i:05d}_{kok}"
        cv2.imwrite(str(hedef / bolum / IMAGES_DIR / f"{temel}.jpg"), goruntu,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        satir = yolo_satiri(bbox, HEDEF_KARE, HEDEF_KARE, sinif=0)
        (hedef / bolum / LABELS_DIR / f"{temel}.txt").write_text(satir + "\n",
                                                                 encoding="utf-8")
        kose_kaydi.append({"dosya": f"{temel}.jpg", "bolum": bolum,
                           "koseler": kucuk, "bbox_xyxy": list(bbox)})
        sayim[bolum] += 1
        if len(ornekler) < 6:
            ornekler.append((goruntu, kucuk))
        if (sayim["train"] + sayim["val"]) % 500 == 0:
            print(f"  {sayim['train'] + sayim['val']} / {len(adlar)}")

    (hedef / "corners.jsonl").write_text(
        "\n".join(json.dumps(k) for k in kose_kaydi), encoding="utf-8")
    onizleme = _onizleme(hedef, ornekler)
    return {"train": sayim["train"], "val": sayim["val"], "atlanan": atlanan,
            "onizleme": str(onizleme)}


def birlestir(hedef: Path, kaynak: Path, tekrar: int = 1) -> dict:
    """Mevcut dört sınıflı kümeyi bu kümenin içine kopyalar.

    Neden kopya: eğitim yalnız gerçek zeminli analog kareleriyle yapılırsa
    model dijital panel, lamba ve vanayı **unutur** — sınıflar kümede hiç
    geçmediğinde tespit başlığı onları bastırır. 13.08'de bu üç sınıf %0'dan
    %100'e çıkarılmıştı; o kazanç korunmadan yeni eğitim ilerleme sayılmaz.

    **`tekrar` neden var.** Mevcut kümede 318 `gauge`, 141 `digital`, 137
    `lamp`, 101 `valve` örneği var; yeni set yalnız `gauge` getiriyor. Ham
    birleştirmede oran 60:1'e çıkar ve azınlık sınıflar bastırılır. Eğitim
    bölümü `tekrar` kez çoğaltılarak oran ~7:1'e çekilir. Bu bir çözüm değil
    bir ödündür ve ölçülmesi şarttır — eğitim sonrası üç azınlık sınıfın
    mAP'ine bakılmadan "gerileme yok" denemez.

    Doğrulama bölümü **çoğaltılmaz**: aynı kareyi val'de birden çok kez
    saymak mAP'i sahte biçimde sabitler ve sayıyı kıyaslanamaz kılar.
    """
    if not kaynak.exists():
        raise SystemExit(f"birleştirilecek küme yok: {kaynak}\n"
                         "önce: python scripts/hazirla_ip5_cok_sinif.py")
    eklenen = {"train": 0, "val": 0}
    for bolum in ("train", "val"):
        for alt in (IMAGES_DIR, LABELS_DIR):
            (hedef / bolum / alt).mkdir(parents=True, exist_ok=True)
        kaynak_img = kaynak / bolum / IMAGES_DIR
        if not kaynak_img.exists():
            continue
        kez = tekrar if bolum == "train" else 1
        for k in range(kez):
            onek = f"cs{k}_" if kez > 1 else "cs_"
            for img in kaynak_img.iterdir():
                etiket = kaynak / bolum / LABELS_DIR / f"{img.stem}.txt"
                if not etiket.exists():
                    continue
                shutil.copy2(img, hedef / bolum / IMAGES_DIR / f"{onek}{img.name}")
                shutil.copy2(etiket,
                             hedef / bolum / LABELS_DIR / f"{onek}{img.stem}.txt")
                eklenen[bolum] += 1
    return eklenen


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Gerçek zeminli açık kadran seti → YOLO kümesi")
    p.add_argument("--zip", default=str(KAYNAK_ZIP))
    p.add_argument("--hedef", default=str(HEDEF_KOK))
    p.add_argument("--limit", type=int, default=None,
                   help="yalnız ilk N görüntüyü kullan (deneme için)")
    p.add_argument("--tohum", type=int, default=0)
    p.add_argument("--birlestir", action="store_true",
                   help="mevcut dört sınıflı kümeyi de içine kopyala (eğitim için ŞART)")
    p.add_argument("--tekrar", type=int, default=3,
                   help="dört sınıflı eğitim bölümü kaç kez çoğaltılsın (sınıf dengesi)")
    args = p.parse_args(argv)

    hedef = Path(args.hedef)
    if hedef.exists():
        shutil.rmtree(hedef)
    print(f"kaynak: {args.zip}")
    ozet = kur(Path(args.zip), hedef, args.limit, args.tohum)
    print(f"gerçek zeminli: train {ozet['train']} · val {ozet['val']} "
          f"· atlanan {ozet['atlanan']}")

    if args.birlestir:
        eklenen = birlestir(hedef, KAYNAK_COK_SINIF, args.tekrar)
        print(f"dört sınıflı küme eklendi: train {eklenen['train']} · val {eklenen['val']}")
        ozet["birlestirilen"] = eklenen

    yaml_yolu = veri_yaml_yaz(hedef / "gauge4.yaml",
                              train=hedef / "train", val=hedef / "val",
                              siniflar=SINIFLAR_COK)
    ozet["veri_yaml"] = str(yaml_yolu)
    ozet["siniflar"] = list(SINIFLAR_COK)
    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    METRIK_YOLU.write_text(json.dumps(ozet, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"veri yaml: {yaml_yolu}")
    print(f"etiket önizlemesi (GÖZLE BAK): {ozet['onizleme']}")
    print(f"özet: {METRIK_YOLU}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
