"""Sentetik veri setinde okuma hatasını ölçer — İP6 ve İP7'nin ortak zemini.

    from gauge_vision.read.evaluate import read_dataset, error_stats

    sonuclar = read_dataset("data/synthetic/v0", method="polar")
    hatalar = [abs(s.angle_error_deg) for s in sonuclar if s.ok]
    error_stats(hatalar).mean        # ortalama açı hatası (derece)

Ölçüm koşulunu değiştiren üç düğme var; üçü de rapordaki tabloların kaynağıdır:

    dial_diameter_px   kadranı bu çapa küçültür  → U6 (640×480 yeter mi)
    jpeg_quality       kareyi JPEG'den geçirir   → U6 (yayın q80 ile sıkıştırıyor)
    center_jitter_px   merkezi bilerek kaydırır  → İP5'in kutusu kusurlu olacak
    method             "polar" | "hough"         → K3 kıyası

**Merkez ve yarıçap etiketten geliyor.** İP6'da ölçülen şey açı yöntemidir;
kadranın yerini İP5 bulacaktır. `center_jitter_px` tam da bu varsayımın ne kadar
iyimser olduğunu sayıyla göstermek için var.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from gauge_vision.read.needle import angle_difference_deg, read_needle_angle
from gauge_vision.synth.generate import load_labels


@dataclass(frozen=True)
class ErrorStats:
    """Bir hata dizisinin özeti — rapordaki ölçüm tablosunun tek satırı.

    Ortalamanın yanında **p95 ve max** de tutuluyor: hedef metrik ortalama hata
    olsa da, sahada kabul edilemez olan tek tük büyük sapmalardır. Ortalama
    onları saklar.
    """

    n: int
    mean: float
    median: float
    p95: float
    max: float

    def as_dict(self, ondalik: int = 3) -> dict[str, float | int]:
        return {
            "n": self.n,
            "ortalama": round(self.mean, ondalik),
            "medyan": round(self.median, ondalik),
            "p95": round(self.p95, ondalik),
            "max": round(self.max, ondalik),
        }


@dataclass(frozen=True)
class NeedleResult:
    """Tek görüntünün okuma sonucu — ham kayıt, istatistik burada hesaplanmaz."""

    file: str
    gauge_id: str
    value: float
    truth_angle_img_deg: float
    truth_angle_deg: float
    roll_deg: float
    measured_angle_deg: float | None
    confidence: float
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return self.measured_angle_deg is not None

    @property
    def angle_error_deg(self) -> float:
        """Ölçülen − gerçek, daireye sarılmış. Okunamayan karede anlamsızdır."""
        if self.measured_angle_deg is None:
            raise ValueError(f"{self.file}: okuma üretilemedi, hata hesaplanamaz")
        return angle_difference_deg(self.measured_angle_deg, self.truth_angle_img_deg)


def error_stats(values) -> ErrorStats:
    """Hata dizisini özetler. Boş dizi için sıfırlarla döner (rapor bozulmasın)."""
    v = sorted(float(x) for x in values)
    if not v:
        return ErrorStats(0, 0.0, 0.0, 0.0, 0.0)
    return ErrorStats(
        n=len(v),
        mean=statistics.fmean(v),
        median=statistics.median(v),
        # En yakın sıra istatistiği: 100 örnekte p95 = 95. eleman. Enterpolasyon
        # yapılmıyor; küçük örneklemde uydurma ara değer üretmesin.
        p95=v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))],
        max=v[-1],
    )


def read_dataset(
    veri_dizini: str | Path,
    *,
    method: str = "polar",
    dial_diameter_px: int | None = None,
    jpeg_quality: int | None = None,
    center_jitter_px: float = 0.0,
    seed: int = 0,
    limit: int | None = None,
) -> list[NeedleResult]:
    """Veri setindeki her görüntüde ibre açısını ölçer, ham sonuçları döner."""
    veri_dizini = Path(veri_dizini)
    kayitlar = load_labels(veri_dizini)
    if limit:
        kayitlar = kayitlar[:limit]

    rng = random.Random(seed)   # sarsıntı da tohumlu — ölçüm tekrar edilebilsin
    sonuclar: list[NeedleResult] = []

    for k in kayitlar:
        img = cv2.imread(str(veri_dizini / k["file"]))
        if img is None:
            raise FileNotFoundError(f"görüntü okunamadı: {veri_dizini / k['file']}")

        center = (float(k["center_px"][0]), float(k["center_px"][1]))
        radius = float(k["radius_px"])

        if dial_diameter_px:
            img, center, radius = _kucult(img, center, radius, dial_diameter_px)
        # Sıra saha akışıyla aynı: yayıncı önce küçültür, SONRA JPEG'ler.
        # Ters sırada sıkıştırma artefaktları küçültmede yumuşar ve ölçüm iyimser çıkar.
        if jpeg_quality:
            img = _jpeg(img, jpeg_quality)
        if center_jitter_px:
            center = _sarsit(center, center_jitter_px, rng)

        t0 = time.perf_counter()
        okuma = read_needle_angle(img, (round(center[0]), round(center[1])), radius,
                                  method=method)
        gecen_ms = (time.perf_counter() - t0) * 1000

        sonuclar.append(NeedleResult(
            file=k["file"],
            gauge_id=k["gauge_id"],
            value=k["value"],
            truth_angle_img_deg=k["angle_img_deg"],
            truth_angle_deg=k["angle_deg"],
            roll_deg=k["roll_deg"],
            measured_angle_deg=okuma.angle_img_deg if okuma else None,
            confidence=okuma.confidence if okuma else 0.0,
            elapsed_ms=gecen_ms,
        ))
    return sonuclar


# ------------------------------------------------------------------ yardımcılar --

def _kucult(img, center: tuple[float, float], radius: float, hedef_cap: int):
    """Kadranın çapı `hedef_cap` piksel olacak şekilde tüm kareyi küçültür.

    Kadranı kırpıp yeniden boyutlandırmak yerine bütün kare ölçekleniyor:
    saha koşulu da böyledir — kamera düşük çözünürlükte yayın yapar, gösterge
    karenin içinde küçülür. Kırpma yapsaydık kadranın piksel sayısını
    korurduk ve ölçüm iyimser çıkardı.
    """
    olcek = hedef_cap / (2 * radius)
    yeni = cv2.resize(img, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    return yeni, (center[0] * olcek, center[1] * olcek), radius * olcek


def _jpeg(img, quality: int):
    """Kareyi JPEG'den geçirip geri açar — yayının sıkıştırma kaybını taklit eder.

    U6'da ileri sürülen "q80 artefaktları ince ibre kenarlarını bozar" iddiası
    tahmin olarak kalmasın diye var: iddia ölçülebilir hâle getiriliyor.
    """
    ok, tampon = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG kodlaması başarısız")
    return cv2.imdecode(tampon, cv2.IMREAD_COLOR)


def _sarsit(center: tuple[float, float], buyukluk: float, rng: random.Random):
    """Merkezi rastgele bir yönde `buyukluk` piksel kaydırır (İP5 kutusu kusurlu)."""
    aci = rng.uniform(0.0, 2 * math.pi)
    return (center[0] + buyukluk * math.cos(aci), center[1] + buyukluk * math.sin(aci))
