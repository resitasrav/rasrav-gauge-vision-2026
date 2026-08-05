"""YOLO eğitim kümesini kurar: sentetik + açık veri karışımı (İP5).

    from gauge_vision.detect.dataset import sentetik_disa_aktar, kume_kur

Neden ayrı bir etiket dosyası üretiliyor: YOLO kendi biçimini ister
(sınıf + normalize edilmiş merkez/genişlik/yükseklik, görüntü başına bir .txt).
Ama **kaynak yine `labels.jsonl`**'dir; bu modül türetir, ikinci bir doğru
kaynak oluşturmaz. Sentetik veri yeniden üretilirse etiketler de yeniden türetilir.

**Karışık eğitim (K1 kararı).** İP4'te ölçülen bulgu: eğitim kümesinin %100'ü
sentetik olduğunda gerçek veride belirgin domain gap oluşuyor, sentetik oranı
%25'in altındayken kayıp gözlenmiyor. `kume_kur` bu yüzden sentetik oranını
parametre olarak alır ve kurduğu kümenin gerçek oranını `meta.json`'a yazar —
oran raporlanabilir olmadan karışık eğitim iddiası doğrulanamaz.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Tek sınıf: kadran yüzü. İP11 (dijital panel) ve İP12 (lamba/vana) kendi
# sınıflarını getirdiğinde liste büyür; şimdilik sınıf karmaşası yaratmıyoruz.
SINIFLAR = ("gauge",)
GAUGE_SINIF_ID = 0

IMAGES_DIR = "images"
LABELS_DIR = "labels"


@dataclass(frozen=True)
class KumeOzeti:
    """Kurulan kümenin sayıları — günlük rapordaki tablo buradan çıkar."""

    kok: Path
    train: int = 0
    val: int = 0
    kaynak_sayilari: dict[str, int] = field(default_factory=dict)

    @property
    def sentetik_orani(self) -> float:
        toplam = sum(self.kaynak_sayilari.values())
        return self.kaynak_sayilari.get("sentetik", 0) / toplam if toplam else 0.0


def yolo_satiri(bbox_xyxy, genislik: int, yukseklik: int, sinif: int = GAUGE_SINIF_ID) -> str:
    """xyxy piksel kutusunu YOLO satırına çevirir (normalize merkez + boyut).

    Kutu görüntü sınırlarına kırpılır: taşan bir kutu YOLO'ya "gösterge kareden
    çıkıyor" diye öğretir ve tespit merkezi sistematik olarak kayar — İP6'nın
    en duyarlı olduğu hata türü budur.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    x1, x2 = max(0.0, min(x1, genislik)), max(0.0, min(x2, genislik))
    y1, y2 = max(0.0, min(y1, yukseklik)), max(0.0, min(y2, yukseklik))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"geçersiz kutu {bbox_xyxy} ({genislik}×{yukseklik})")

    cx, cy = (x1 + x2) / 2 / genislik, (y1 + y2) / 2 / yukseklik
    w, h = (x2 - x1) / genislik, (y2 - y1) / yukseklik
    return f"{sinif} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def sentetik_disa_aktar(veri_dizini: str | Path, hedef: str | Path, *,
                        onek: str = "syn") -> int:
    """`labels.jsonl`'den YOLO etiketi türetir, görüntüleri hedefe kopyalar.

    Dosya adına önek konuyor: karışık kümede bir görüntünün hangi kaynaktan
    geldiği dosya adından görülebilsin. Hata analizinde "yanlış tespitler hangi
    kaynakta yoğunlaşıyor" sorusu ancak böyle cevaplanabilir.
    """
    veri_dizini, hedef = Path(veri_dizini), Path(hedef)
    (hedef / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    (hedef / LABELS_DIR).mkdir(parents=True, exist_ok=True)

    with (veri_dizini / "labels.jsonl").open(encoding="utf-8") as f:
        kayitlar = [json.loads(s) for s in f if s.strip()]

    boyut = json.loads((veri_dizini / "meta.json").read_text(encoding="utf-8"))["image_size"]

    for k in kayitlar:
        ad = f"{onek}_{Path(k['file']).stem}"
        shutil.copyfile(veri_dizini / k["file"], hedef / IMAGES_DIR / f"{ad}.png")
        (hedef / LABELS_DIR / f"{ad}.txt").write_text(
            yolo_satiri(k["bbox_xyxy"], boyut, boyut) + "\n", encoding="utf-8")
    return len(kayitlar)


def hf_parquet_disa_aktar(parquet_yolu: str | Path, hedef: str | Path, *,
                          onek: str = "real", kaynak_sinif: int = 1) -> int:
    """Roboflow kökenli HF parquet'ini YOLO biçimine çevirir.

    Sette üç kategori var: `0 gauge` (kullanılmıyor — dönüştürme artığı üst
    kategori), `1 gauges` (kadran yüzü kutusu) ve `2 numbers` (kadran üzerindeki
    sayılar). Bize gereken kadran yüzüdür; varsayılan `kaynak_sinif=1` odur.

    `2 numbers` bilinçli olarak atılıyor ama silinmiyor: İP11'de (dijital/OCR)
    kadran üzerindeki sayıların kutusu işe yarayabilir. O gün geldiğinde bu
    fonksiyona ikinci bir sınıf eklemek yeterli olacaktır.

    COCO kutusu `[x, y, w, h]`, sol-üst köşe + boyut biçimindedir; YOLO merkez
    ister. Dönüşüm `yolo_satiri`'na bırakılıyor ki kırpma kuralı tek yerde dursun.
    """
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    hedef = Path(hedef)
    (hedef / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    (hedef / LABELS_DIR).mkdir(parents=True, exist_ok=True)

    kayitlar = pq.read_table(parquet_yolu).to_pylist()
    yazilan = 0

    for k in kayitlar:
        goruntu = Image.open(io.BytesIO(k["image"]["bytes"])).convert("RGB")
        g, y = goruntu.size   # parquet'teki width alanı bazı satırlarda bozuk

        satirlar = [
            yolo_satiri((x, ust, x + w, ust + h), g, y)
            for (x, ust, w, h), kategori in zip(k["objects"]["bbox"], k["objects"]["category"])
            if kategori == kaynak_sinif
        ]
        if not satirlar:
            continue   # kadranı etiketlenmemiş kare eğitime "burada gösterge yok" öğretir

        ad = f"{onek}_{k['image_id']:05d}"
        goruntu.save(hedef / IMAGES_DIR / f"{ad}.png")
        (hedef / LABELS_DIR / f"{ad}.txt").write_text("\n".join(satirlar) + "\n",
                                                      encoding="utf-8")
        yazilan += 1
    return yazilan


def bol(hedef: str | Path, *, val_orani: float = 0.2, seed: int = 0) -> tuple[int, int]:
    """Düz klasörü train/val olarak ikiye ayırır (YOLO'nun beklediği düzen).

    Bölme tohumlu: eğitim tekrar edilebilir olmalı. Görüntü ve etiket birlikte
    taşınır; birinin geride kalması sessizce etiketsiz eğitim demektir.
    """
    hedef = Path(hedef)
    goruntuler = sorted((hedef / IMAGES_DIR).glob("*.*"))
    random.Random(seed).shuffle(goruntuler)

    n_val = max(1, round(len(goruntuler) * val_orani))
    dagilim = {"val": goruntuler[:n_val], "train": goruntuler[n_val:]}

    for bolum, dosyalar in dagilim.items():
        (hedef / IMAGES_DIR / bolum).mkdir(parents=True, exist_ok=True)
        (hedef / LABELS_DIR / bolum).mkdir(parents=True, exist_ok=True)
        for g in dosyalar:
            etiket = hedef / LABELS_DIR / f"{g.stem}.txt"
            if not etiket.exists():
                raise FileNotFoundError(f"{g.name} için etiket yok — bölme yarıda bırakıldı")
            g.rename(hedef / IMAGES_DIR / bolum / g.name)
            etiket.rename(hedef / LABELS_DIR / bolum / etiket.name)

    return len(dagilim["train"]), len(dagilim["val"])


def veri_yaml_yaz(yol: str | Path, *, train: Path, val: Path, test: Path | None = None) -> Path:
    """Ultralytics'in beklediği veri tanımı.

    Yollar mutlak yazılıyor: eğitim başka bir çalışma dizininden koşturulduğunda
    sessizce boş bir kümeyle eğitmesin. `val` ve `test` bilerek dışarıdan
    veriliyor — üç eğitim yapılandırması (sentetik / gerçek / karışık) **aynı**
    doğrulama ve test kümesini kullanmalı, yoksa sayılar kıyaslanamaz.
    """
    yol = Path(yol)
    satirlar = [
        f"train: {Path(train).resolve().as_posix()}",
        f"val: {Path(val).resolve().as_posix()}",
    ]
    if test is not None:
        satirlar.append(f"test: {Path(test).resolve().as_posix()}")
    satirlar += [f"nc: {len(SINIFLAR)}", f"names: {list(SINIFLAR)}"]

    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    return yol
