"""Sentetik gösterge veri seti üretir — görüntü + otomatik etiket (İP3).

    from gauge_vision.synth.generate import generate_dataset
    ozet = generate_dataset("data/synthetic/v0", count=100, seed=0)

Üretilen klasör:

    v0/
      images/0001_PT-101.png ...
      labels.jsonl              her satır bir görüntünün ground truth'u
      meta.json                 tohum, sayı, varyasyon sınırları, tarih

Neden JSONL: satır satır büyür, `git diff`'te okunur, İP5'in YOLO etiketi de
İP6/İP7'nin açı-değer etiketi de aynı satırdan türetilir. Tek doğru kaynak
ilkesi burada da geçerli — ikinci bir etiket dosyası tutulmuyor.

Tohum (`seed`) zorunlu bir tasarım tercihi: ölçüm tekrar edilemiyorsa ölçüm
değildir. Aynı tohum aynı 100 görüntüyü verir, İP6'nın hata sayısı bir hafta
sonra da doğrulanabilir.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import cv2

from gauge_vision.config import Gauge, load_gauges
from gauge_vision.synth.dial import CANVAS_PX, BEZEL_WIDTH_RATIO, DialLook, render_analog

LABELS_FILE = "labels.jsonl"
META_FILE = "meta.json"
IMAGES_DIR = "images"
DATASET_VERSION = 1


@dataclass(frozen=True)
class VariationRange:
    """Rastgele varyasyonun sınırları — v0'da bilinçli olarak dar.

    Zor koşullar (parlama, bulanıklık, düşük ışık, eğik bakış açısı) BURAYA
    girmez; onlar İP14'ün konusu. Önce yöntemin temiz görüntüde çalıştığını
    ölçmek gerekiyor ki İP14'teki düşüş yönteme mi koşula mı ait, ayrılabilsin.
    """

    radius_ratio: tuple[float, float] = (0.28, 0.42)   # kadran yarıçapı / kenar
    center_offset_ratio: float = 0.05                  # merkez kayması / kenar
    needle_w_scale: tuple[float, float] = (0.75, 1.35)
    roll_deg: float = 8.0                              # kamera yatıklığı ±
    background_gray: tuple[int, int] = (140, 225)      # zemin tonu aralığı


@dataclass(frozen=True)
class DatasetSummary:
    """Üretim sonucu — günlük rapora yazılacak sayılar."""

    out_dir: Path
    count: int
    seed: int
    per_gauge: dict[str, int] = field(default_factory=dict)

    @property
    def labels_path(self) -> Path:
        return self.out_dir / LABELS_FILE


def sample_look(rng: random.Random, size: int, ranges: VariationRange) -> DialLook:
    """Tek bir kare için rastgele görünüm seçer.

    Merkez kayması kadranın görüntü dışına taşmayacak şekilde sınırlanır:
    kutusu kırpılmış bir gösterge İP5'e yanlış etiket öğretir.
    """
    radius_ratio = rng.uniform(*ranges.radius_ratio)
    outer = size * radius_ratio * (1 + BEZEL_WIDTH_RATIO)
    max_offset = min(size * ranges.center_offset_ratio, size / 2 - outer - 1)
    max_offset = max(0.0, max_offset)

    gray = rng.randint(*ranges.background_gray)
    return DialLook(
        radius_ratio=radius_ratio,
        center_offset_px=(round(rng.uniform(-max_offset, max_offset)),
                          round(rng.uniform(-max_offset, max_offset))),
        needle_w_scale=rng.uniform(*ranges.needle_w_scale),
        roll_deg=rng.uniform(-ranges.roll_deg, ranges.roll_deg),
        background_bgr=(gray, gray, gray),
    )


def _stratified_values(gauge: Gauge, n: int, rng: random.Random) -> list[float]:
    """Değerleri kadran boyunca dengeli dağıtır.

    Düz rastgele çekilseydi bazı bölgeler boş kalır, İP6'nın hatası "kadranın
    her yerinde" değil "şansın getirdiği yerlerde" ölçülmüş olurdu. Aralık n
    dilime bölünüp her dilimden bir örnek alınıyor.
    """
    lo, hi = gauge.scale.min, gauge.scale.max
    step = (hi - lo) / n
    return [rng.uniform(lo + i * step, lo + (i + 1) * step) for i in range(n)]


def _paylastir(gauge_ids: list[str], count: int) -> dict[str, int]:
    """Toplam sayıyı göstergelere olabildiğince eşit böler (fark en fazla 1)."""
    pay, kalan = divmod(count, len(gauge_ids))
    return {gid: pay + (1 if i < kalan else 0) for i, gid in enumerate(gauge_ids)}


def generate_dataset(
    out_dir: str | Path,
    *,
    count: int = 100,
    seed: int = 0,
    gauge_ids: list[str] | None = None,
    size: int = CANVAS_PX,
    ranges: VariationRange | None = None,
    config_path: str | Path | None = None,
) -> DatasetSummary:
    """`count` adet sentetik gösterge görüntüsü ve etiketini üretir."""
    ranges = ranges or VariationRange()
    out_dir = Path(out_dir)

    gauges = {gid: g for gid, g in load_gauges(config_path).items() if g.type == "analog"}
    if gauge_ids:
        eksik = [gid for gid in gauge_ids if gid not in gauges]
        if eksik:
            raise ValueError(f"analog gösterge bulunamadı: {eksik} — "
                             f"envanterdekiler: {list(gauges)}")
        gauges = {gid: gauges[gid] for gid in gauge_ids}
    if not gauges:
        raise ValueError("envanterde analog gösterge yok")
    if count < len(gauges):
        raise ValueError(f"count ({count}) gösterge sayısından ({len(gauges)}) küçük olamaz")

    rng = random.Random(seed)
    per_gauge = _paylastir(list(gauges), count)

    # (gösterge, değer) çiftleri önce toplanıp karıştırılıyor: dosya sırası
    # göstergeye göre kümelenmesin, eğitim/doğrulama bölmesi yanlı olmasın.
    isler: list[tuple[Gauge, float]] = []
    for gid, n in per_gauge.items():
        isler += [(gauges[gid], v) for v in _stratified_values(gauges[gid], n, rng)]
    rng.shuffle(isler)

    images_dir = out_dir / IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / LABELS_FILE).open("w", encoding="utf-8", newline="\n") as f:
        for i, (gauge, value) in enumerate(isler, start=1):
            img, truth = render_analog(gauge, value, size=size,
                                       look=sample_look(rng, size, ranges))
            ad = f"{i:04d}_{gauge.id}.png"
            cv2.imwrite(str(images_dir / ad), img)

            kayit = {"file": f"{IMAGES_DIR}/{ad}", "unit": gauge.unit, **asdict(truth)}
            f.write(json.dumps(kayit, ensure_ascii=False) + "\n")

    meta = {
        "dataset_version": DATASET_VERSION,
        "created": date.today().isoformat(),
        "seed": seed,
        "count": count,
        "image_size": size,
        "per_gauge": per_gauge,
        "variation": asdict(ranges),
        "note": "Zor kosullar (parlama/bulaniklik/dusuk isik) bilincli olarak yok - IP14",
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    return DatasetSummary(out_dir=out_dir, count=count, seed=seed, per_gauge=per_gauge)


def load_labels(out_dir: str | Path) -> list[dict]:
    """Üretilmiş veri setinin etiketlerini okur (İP6/İP7 buradan beslenecek)."""
    path = Path(out_dir) / LABELS_FILE
    with path.open(encoding="utf-8") as f:
        return [json.loads(satir) for satir in f if satir.strip()]
