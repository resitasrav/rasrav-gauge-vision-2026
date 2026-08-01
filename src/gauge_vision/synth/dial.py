"""Analog gösterge kadranını çizer — sentetik veri üretecinin çekirdeği (İP3).

    from gauge_vision.config import load_gauges
    from gauge_vision.synth.dial import render_analog

    pt101 = load_gauges()["PT-101"]
    img, truth = render_analog(pt101, value=5.0)   # truth.angle_deg == 90.0

Kadran geometrisi `Scale.angle_for_value` üzerinden gelir; bu dosya sadece
piksel koyar. Böylece ölçek kuralı (karekök kadran dahil) tek yerde durur.

Çizim OpenCV ile yapılıyor, matplotlib ile değil: okuma zinciri de OpenCV
dünyasında çalışacak ve pikselin nereye düştüğünü birebir kontrol etmek
istiyoruz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from gauge_vision.config import Gauge

# --- Yerleşim oranları (hepsi kadran yarıçapına veya görüntü kenarına oran) ---
CANVAS_PX = 512               # varsayılan kare görüntü kenarı
FACE_RADIUS_RATIO = 0.40      # kadran yarıçapı / görüntü kenarı
BEZEL_WIDTH_RATIO = 0.07      # çerçeve kalınlığı / yarıçap
MAJOR_TICK_LEN_RATIO = 0.14   # ana çizgi uzunluğu / yarıçap
MINOR_TICK_LEN_RATIO = 0.07
LABEL_RADIUS_RATIO = 0.70     # sayı etiketlerinin merkeze uzaklığı / yarıçap
NEEDLE_LEN_RATIO = 0.78       # ibre ucu / yarıçap
NEEDLE_TAIL_RATIO = 0.16      # ibrenin arka kuyruğu (karşı ağırlık) / yarıçap
HUB_RADIUS_RATIO = 0.07       # merkez göbeği / yarıçap

# --- Çizgi kalınlıkları (yarıçapa oran; ince kadranda da orantılı kalsın) ---
MAJOR_TICK_W_RATIO = 0.020
MINOR_TICK_W_RATIO = 0.010
NEEDLE_W_RATIO = 0.030

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_RATIO = 0.0055     # font ölçeği / yarıçap
TEXT_FONT_RATIO = 0.0036      # kimlik/birim yazısı — sayı etiketlerinden küçük
ID_TEXT_DY_RATIO = -0.28      # yazılar etiket halkasının (0.70) içinde kalmalı,
UNIT_TEXT_DY_RATIO = 0.46     # yoksa geniş kimlikler sayılarla çakışıyor

# --- Renkler (OpenCV BGR) ---
COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (20, 20, 20),
    "red": (40, 40, 210),
    "gray": (128, 128, 128),
}
BACKGROUND_BGR = (205, 205, 205)   # duvar/pano zemini — kadran fondan ayrılsın
BEZEL_BGR = (70, 70, 70)
TEXT_BGR = (60, 60, 60)

# Hershey fontları ASCII dışını çizemez; birim etiketi sadece GÖRÜNTÜ için
# sadeleştirilir. Gerçek birim YAML'da ve MQTT yayınında olduğu gibi kalır.
UNIT_ASCII = {"°C": "C", "m³/h": "m3/h"}


@dataclass(frozen=True)
class DialLook:
    """Tek bir karenin görünüm varyasyonu — üreteç (İP3) bunu rastgele örnekler.

    Bilinçli olarak SADECE geometrik/ışıksal olmayan çeşitlilik var: kadranın
    boyu, yeri, ibre kalınlığı, kameranın yatıklığı, zemin tonu. Parlama,
    bulanıklık ve düşük ışık yok — onlar İP14'ün konusu. Sebep: İP6'da hata
    çıkınca "yöntem mi kötü, görüntü mü zor" ayırt edilebilsin.
    """

    radius_ratio: float = FACE_RADIUS_RATIO
    center_offset_px: tuple[int, int] = (0, 0)
    needle_w_scale: float = 1.0
    roll_deg: float = 0.0                 # kameranın yatıklığı (+ = saat tersi)
    background_bgr: tuple[int, int, int] = BACKGROUND_BGR


@dataclass(frozen=True)
class DialTruth:
    """Çizilen kadranın ground truth'u — etiket dosyasına bu yazılır.

    İki ayrı açı var ve karıştırılmamalı:
      `angle_deg`     — kadranın kendi çerçevesindeki ibre açısı; İP7'nin
                        açı→değer dönüşümünün girdisi.
      `angle_img_deg` — GÖRÜNTÜDE ölçülecek açı (= angle_deg + roll_deg);
                        İP6'nın Hough ile bulacağı sayı bununla kıyaslanır.
    Kamera yatık değilse (roll 0) ikisi aynıdır. `bbox` İP5'in (YOLO) hedefi.
    """

    gauge_id: str
    value: float
    angle_deg: float
    roll_deg: float
    angle_img_deg: float
    center_px: tuple[int, int]
    tip_px: tuple[int, int]
    radius_px: int
    bbox_xyxy: tuple[int, int, int, int]   # kadran yüzünün kutusu (bezel dahil)


def render_analog(
    gauge: Gauge,
    value: float,
    *,
    size: int = CANVAS_PX,
    look: DialLook | None = None,
) -> tuple[np.ndarray, DialTruth]:
    """`gauge` kadranını `value` değerinde çizer; (görüntü, ground truth) döner.

    Değer kadran aralığı dışındaysa `Scale.angle_for_value` hata yükseltir —
    sessizce kırpmak yerine patlaması bilinçli.
    """
    if gauge.type != "analog":
        raise ValueError(f"{gauge.id}: render_analog sadece analog göstergede çalışır "
                         f"(tip: {gauge.type})")

    look = look or DialLook()
    scale = gauge.scale
    angle = scale.angle_for_value(value)   # ground truth burada doğuyor

    img = np.full((size, size, 3), look.background_bgr, dtype=np.uint8)
    center = (size // 2 + look.center_offset_px[0], size // 2 + look.center_offset_px[1])
    radius = int(size * look.radius_ratio)

    style = gauge.synthetic
    face_bgr = COLORS_BGR.get(style.get("face_color", "white"), COLORS_BGR["white"])
    needle_bgr = COLORS_BGR.get(style.get("needle_color", "black"), COLORS_BGR["black"])

    _draw_face(img, center, radius, face_bgr)
    _draw_ticks(img, center, radius, gauge)
    _draw_texts(img, center, radius, gauge)
    _draw_needle(img, center, radius, angle, needle_bgr, look.needle_w_scale)

    # Kamera yatıklığı: kadranı çizdikten sonra tüm kareyi merkez etrafında
    # döndürüyoruz — yazılar ve çizgiler de birlikte dönsün. Daire dönmeye
    # duyarsız olduğu için kutu ve merkez değişmez, sadece ibrenin görüntüdeki
    # açısı roll kadar kayar.
    angle_img = angle + look.roll_deg
    if look.roll_deg:
        M = cv2.getRotationMatrix2D(center, look.roll_deg, 1.0)
        img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=look.background_bgr)

    bezel = int(radius * BEZEL_WIDTH_RATIO)
    outer = radius + bezel
    truth = DialTruth(
        gauge_id=gauge.id,
        value=float(value),
        angle_deg=float(angle),
        roll_deg=float(look.roll_deg),
        angle_img_deg=float(angle_img),
        center_px=center,
        tip_px=_polar_px(center, radius * NEEDLE_LEN_RATIO, angle_img),
        radius_px=radius,
        bbox_xyxy=(center[0] - outer, center[1] - outer,
                   center[0] + outer, center[1] + outer),
    )
    return img, truth


# ---------------------------------------------------------------- yardımcılar --

def _polar_px(center: tuple[int, int], radius: float, angle_deg: float) -> tuple[int, int]:
    """Açı+yarıçap → piksel. y ekseni AŞAĞI arttığı için sinüs eksi işaretli.

    Konvansiyonun tek uygulama noktası burası; başka yerde el ile
    cos/sin yazılmaz (yanlış işaret bu projedeki en pahalı hata sınıfı).
    """
    rad = math.radians(angle_deg)
    return (round(center[0] + radius * math.cos(rad)),
            round(center[1] - radius * math.sin(rad)))


def _draw_face(img, center, radius, face_bgr) -> None:
    bezel = int(radius * BEZEL_WIDTH_RATIO)
    cv2.circle(img, center, radius + bezel, BEZEL_BGR, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, face_bgr, -1, cv2.LINE_AA)


def _tick_values(gauge: Gauge) -> tuple[list[float], list[float]]:
    """Ana ve ara çizgilerin DEĞERLERİ (açıları değil).

    Çizgiler değer ekseninde eşit aralıklı; açıya çevirmeyi `angle_for_value`
    yapar. Karekök ölçekli kadranda bu, çizgilerin görüntüde eşit aralıklı
    ÇIKMAMASINI sağlar — gerçek debimetreler de böyle görünür.
    """
    scale = gauge.scale
    n_major = int(gauge.synthetic.get("tick_major", 11))
    n_minor = int(gauge.synthetic.get("tick_minor", 4))

    step = (scale.max - scale.min) / (n_major - 1)
    majors = [scale.min + i * step for i in range(n_major)]

    minors: list[float] = []
    for i in range(n_major - 1):
        for j in range(1, n_minor + 1):
            minors.append(majors[i] + step * j / (n_minor + 1))
    return majors, minors


def _draw_ticks(img, center, radius, gauge: Gauge) -> None:
    scale = gauge.scale
    majors, minors = _tick_values(gauge)

    major_len = radius * MAJOR_TICK_LEN_RATIO
    minor_len = radius * MINOR_TICK_LEN_RATIO
    major_w = max(1, round(radius * MAJOR_TICK_W_RATIO))
    minor_w = max(1, round(radius * MINOR_TICK_W_RATIO))
    ink = COLORS_BGR["black"]

    for v in minors:
        a = scale.angle_for_value(v)
        cv2.line(img, _polar_px(center, radius, a),
                 _polar_px(center, radius - minor_len, a), ink, minor_w, cv2.LINE_AA)

    font_scale = radius * LABEL_FONT_RATIO
    for v in majors:
        a = scale.angle_for_value(v)
        cv2.line(img, _polar_px(center, radius, a),
                 _polar_px(center, radius - major_len, a), ink, major_w, cv2.LINE_AA)

        label = f"{v:g}"
        (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, major_w)
        lx, ly = _polar_px(center, radius * LABEL_RADIUS_RATIO, a)
        cv2.putText(img, label, (lx - tw // 2, ly + th // 2),
                    FONT, font_scale, ink, major_w, cv2.LINE_AA)


def _draw_texts(img, center, radius, gauge: Gauge) -> None:
    """Kadran üstündeki yazılar: gösterge kimliği ve birim.

    Gerçek göstergelerde de bu yazılar var; İP5'in tespiti ve İP11'in OCR'ı
    boş beyaz daire yerine gerçekçi bir yüzle karşılaşsın.
    """
    font_scale = radius * TEXT_FONT_RATIO
    thick = max(1, round(radius * MINOR_TICK_W_RATIO))

    yazilar = ((gauge.id, ID_TEXT_DY_RATIO),
               (UNIT_ASCII.get(gauge.unit, gauge.unit), UNIT_TEXT_DY_RATIO))
    for text, dy_ratio in yazilar:
        (tw, _), _ = cv2.getTextSize(text, FONT, font_scale, thick)
        cv2.putText(img, text,
                    (center[0] - tw // 2, center[1] + round(radius * dy_ratio)),
                    FONT, font_scale, TEXT_BGR, thick, cv2.LINE_AA)


def _draw_needle(img, center, radius, angle_deg, needle_bgr, w_scale: float = 1.0) -> tuple[int, int]:
    """İbreyi çizer, ucunun pikselini döner (İP6'nın ölçeceği geometri)."""
    tip = _polar_px(center, radius * NEEDLE_LEN_RATIO, angle_deg)
    tail = _polar_px(center, radius * NEEDLE_TAIL_RATIO, angle_deg + 180)
    width = max(1, round(radius * NEEDLE_W_RATIO * w_scale))

    cv2.line(img, tail, tip, needle_bgr, width, cv2.LINE_AA)
    cv2.circle(img, center, max(2, round(radius * HUB_RADIUS_RATIO)),
               needle_bgr, -1, cv2.LINE_AA)
    return tip
