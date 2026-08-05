"""Uçtan uca okuma zinciri: görüntü → tespit → kırp → açı → değer.

    from gauge_vision.pipeline import read_frame
    sonuc = read_frame(kare, model, gauge)
    sonuc.reading.value      # 7.9

İP5, İP6 ve İP7'yi birbirine bağlayan tek yer burasıdır. Ölçüm scripti
(`olc_zincir.py`) ile canlı demo (`canli_oku.py`) aynı kodu çalıştırsın diye
`src/` altında duruyor: demoda gördüğün sayı ile raporda yazan sayı aynı
hattan çıkmazsa ikisi de güvenilmez olur.

**Ölçüm scriptlerinden farkı:** `olc_ip6.py` ve `olc_ip7.py` kadranın merkezini
ve yarıçapını **etiketten** alır — orada ölçülen şey okuma yöntemidir. Burada
ikisi de **tespitten** gelir. Aradaki fark zincirin gerçek hatasıdır ve 05.08
ölçümüne göre 13 kattır (%0,129 → %1,72).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gauge_vision.config import Gauge
from gauge_vision.detect.refine import refine_dial
from gauge_vision.read.calibrate import GaugeReading, read_value
from gauge_vision.read.needle import NeedleReading, read_needle_angle
from gauge_vision.read.roll import RollEstimate, estimate_roll

# Kadran yüzünün yarıçapı tespit kutusundan türetilir. Kutu bezeli de içerdiğinden
# ham yarının tamamı alınmaz: sentetik üreteçte dış yarıçap = kadran yarıçapı × 1,07
# (dial.BEZEL_WIDTH_RATIO). Fazla büyük yarıçap tarama halkasını kadranın dışına
# taşırır ve ana çizgiler ibre sanılabilir.
KUTU_YARICAP_ORANI = 1 / 1.07

# Kutu kare değilse (açılı bakış, kısmi örtme) kısa kenar esas alınır; uzun kenara
# göre alınan yarıçap kadranın dışını tarar.
MIN_YARICAP_PX = 12


@dataclass(frozen=True)
class FrameResult:
    """Tek karenin zincir çıktısı. `reading` None ise okuma üretilememiştir."""

    box_xyxy: tuple[float, float, float, float] | None
    detect_conf: float
    center_px: tuple[int, int] | None
    radius_px: float
    needle: NeedleReading | None
    reading: GaugeReading | None
    reason: str = ""
    # Merkez kadran çemberinden rafine edilebildi mi? Rafine kapılardan geçemezse
    # kutu merkezinde kalınır ve bu False olur — ölçümde ikisi ayrılabilsin.
    center_refined: bool = False
    # Uygulanan yatıklık ve nereden geldiği. `roll` None ise kestirim yapılmamış
    # ya da başarısız olmuştur; `roll_deg` o zaman dışarıdan verilen değerdir.
    roll_deg: float = 0.0
    roll: RollEstimate | None = None

    @property
    def ok(self) -> bool:
        return self.reading is not None and self.reading.value is not None


def _bos(sebep: str, kutu=None, guven: float = 0.0) -> FrameResult:
    return FrameResult(box_xyxy=kutu, detect_conf=guven, center_px=None,
                       radius_px=0.0, needle=None, reading=None, reason=sebep)


def dial_from_box(box_xyxy) -> tuple[tuple[int, int], float]:
    """Tespit kutusundan kadran merkezi ve yarıçapı."""
    x1, y1, x2, y2 = box_xyxy
    merkez = (round((x1 + x2) / 2), round((y1 + y2) / 2))
    yaricap = min(x2 - x1, y2 - y1) / 2 * KUTU_YARICAP_ORANI
    return merkez, yaricap


def read_frame(
    image: np.ndarray,
    model,
    gauge: Gauge,
    *,
    detect_conf: float = 0.25,
    method: str = "polar",
    roll_deg: float | None = None,
    refine: bool = True,
) -> FrameResult:
    """Karede göstergeyi bulur, ibresini ölçer, değere çevirir.

    Hata yükseltmez; her başarısızlık `reason` ile bildirilir. Zincir saha
    döngüsünde çalışacaktır, tek bir okunamayan kare turu düşürmemelidir.

    `roll_deg=None` (varsayılan) kamera yatıklığını kadranın çizgilerinden
    KESTİRİR (`read/roll.py`). Kestirim güvenilmezse 0 kabul edilir — yanlış bir
    yatıklık, düzeltmemekten daha kötüdür. Sayı verilirse kestirim yapılmaz;
    ölçümde ablasyon (0 = düzeltme yok, etiket = ideal düzeltme) böyle kurulur.

    `refine` kutu merkezini kadran çemberinden düzeltir (bkz. `detect/refine.py`).
    Ablasyon anahtarı olarak kapatılabilir — kazancı ölçülebilsin diye parametre.
    """
    sonuc = model.predict(image, conf=detect_conf, verbose=False)[0]
    if len(sonuc.boxes) == 0:
        return _bos("gösterge bulunamadı")

    # En güvenli kutu: karede birden çok gösterge olabilir, zincir tek gösterge okur.
    # Hangisinin okunacağı saha döngüsünde robotun durağıyla belirlenecektir (U11).
    en_iyi = int(sonuc.boxes.conf.argmax())
    kutu = tuple(float(v) for v in sonuc.boxes.xyxy[en_iyi].tolist())
    tespit_guveni = float(sonuc.boxes.conf[en_iyi])

    merkez, yaricap = dial_from_box(kutu)
    if yaricap < MIN_YARICAP_PX:
        return _bos(f"kadran çok küçük ({yaricap:.0f} px)", kutu, tespit_guveni)

    # Merkez rafinesi ibre ölçümünden ÖNCE: kutupsal tarama merkeze duyarlıdır,
    # düzeltmeyi sonradan uygulamak açıyı geriye dönük kurtarmaz.
    rafine_edildi = False
    if refine:
        daire = refine_dial(image, merkez, yaricap)
        if daire is not None:
            merkez, yaricap = daire.center_px, daire.radius_px
            rafine_edildi = True

    # Yatıklık ibreden BAĞIMSIZ ölçülür: kaynağı kadranın çizgileridir, ibre
    # tarama halkasının dışında kalır. Bu yüzden sırası önemli değil, ama
    # merkezden sonra gelmeli — çizgi halkası da merkeze göre taranıyor.
    yatiklik = None
    if roll_deg is None:
        yatiklik = estimate_roll(image, merkez, yaricap, gauge)
        roll_deg = yatiklik.roll_deg if yatiklik else 0.0

    aci = read_needle_angle(image, merkez, yaricap, method=method)
    if aci is None:
        return _bos("ibre bulunamadı", kutu, tespit_guveni)

    # Güvenler çarpılıyor: zincirin güveni en zayıf halkasından yüksek olamaz.
    # İP15'in `unreadable` eşiği bu birleşik sayıya uygulanacaktır.
    okuma = read_value(gauge, aci.angle_img_deg, roll_deg=roll_deg,
                       confidence=aci.confidence * tespit_guveni)

    return FrameResult(box_xyxy=kutu, detect_conf=tespit_guveni, center_px=merkez,
                       radius_px=yaricap, needle=aci, reading=okuma,
                       center_refined=rafine_edildi,
                       roll_deg=roll_deg, roll=yatiklik)
