"""Pano tipi metreyi çizer: kare çerçeve, yay skala, kenardan dönen ibre (İP18).

    from gauge_vision.synth.panel import render_panel_meter

    img, truth = render_panel_meter(gauge, value=0.6)

Elektrik odalarındaki ampermetre/voltmetreler (`face.shape: panel`) yuvarlak
kadrandan üç noktada ayrılır ve üçü de okuma zincirini ilgilendirir:

    çerçeve     kare — `refine_dial`'ın aradığı eş merkezli çember YOK
    skala       ~90°'lik yay — tam çember değil, tarama halkasının çoğu boş
    pivot       kutunun ortasında DEĞİL, kenara yakın

Bu yüzden ayrı bir dosya: `synth/dial.py`'ye dallanma eklemek her iki
geometriyi de okunmaz hâle getirirdi.

**Neden sentetik:** 27.08'de arandı ve erişilebilir hiçbir kamu veri setinde bu
tip yok — Pointer-10K (Baidu, CC BY-NC-SA), DialBench, Synanthropic ve UFPR-ADMR
setlerinin hepsi yuvarlak kadran. Projenin dört tipi de sentetikle başladı;
beşinci geometri de aynı yoldan gidiyor: ground truth bedava, hata kaynağı
tek değişkende izlenebilir.

Açı konvansiyonu `synth/dial.py` ile AYNI: derece, 0° = saat 3 yönü, pozitif
yön saat tersi (CCW). Farklı olması iki üretecin ölçümlerini karşılaştırılamaz
yapardı.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from gauge_vision.config import Gauge
from gauge_vision.synth.dial import (COLORS_BGR, FONT, UNIT_ASCII,
                                     BACKGROUND_BGR, BEZEL_BGR, TEXT_BGR)

CANVAS_PX = 512

# --- Yerleşim (kutu GENİŞLİĞİNE oran) ---
KUTU_ORANI = 0.62             # metrenin görüntü içindeki genişliği
# Gerçek pano metreleri tam kare değil, biraz basıktır: skala yayı yatayda
# genişler, dikeyde pivotun üstünde kalan boşluk küçüktür. Tam kare çizmek
# yayın üstünde gerçekte olmayan büyük bir boşluk bırakıyordu.
KUTU_EN_BOY = 0.82            # yükseklik / genişlik
CERCEVE_ORANI = 0.055         # siyah çerçeve kalınlığı / kutu genişliği
YUZ_PAYI = 0.04               # çerçeve ile beyaz yüz arası

# --- İbre ve skala (süpürme yarıçapına oran) ---
IBRE_UZUNLUK = 0.94           # ibre ucu / süpürme yarıçapı
IBRE_KUYRUK = 0.06            # pivot kenarda, kuyruk çok kısa
IBRE_KALINLIK = 0.020
GOBEK_ORANI = 0.055
ANA_CIZGI_UZUNLUK = 0.10
ARA_CIZGI_UZUNLUK = 0.055
ANA_CIZGI_KALINLIK = 0.014
ARA_CIZGI_KALINLIK = 0.008
ETIKET_YARICAP = 0.80         # sayı etiketleri yayın içinde
ETIKET_FONT = 0.0042
BIRIM_FONT = 0.0075           # "MW" yazısı büyük — gerçek panolarda öyle


@dataclass(frozen=True)
class PanelTruth:
    """Çizilen pano metresinin ground truth'u.

    `pivot_px` alanı `DialTruth.center_px`'in karşılığıdır ama adı bilinçle
    farklı: burada o nokta kutunun MERKEZİ DEĞİL, ibrenin döndüğü yer. İki
    üreteci aynı alan adıyla yazmak, kutu merkezini pivot sanan bir hatayı
    ölçümde görünmez kılardı.
    """

    gauge_id: str
    value: float
    angle_deg: float
    roll_deg: float
    angle_img_deg: float
    pivot_px: tuple[int, int]
    tip_px: tuple[int, int]
    sweep_radius_px: float
    bbox_xyxy: tuple[int, int, int, int]


def _nokta(pivot: tuple[int, int], r: float, aci_deg: float) -> tuple[int, int]:
    rad = math.radians(aci_deg)
    return (round(pivot[0] + r * math.cos(rad)), round(pivot[1] - r * math.sin(rad)))


def render_panel_meter(
    gauge: Gauge,
    value: float,
    *,
    size: int = CANVAS_PX,
    roll_deg: float = 0.0,
    background_bgr: tuple[int, int, int] = BACKGROUND_BGR,
) -> tuple[np.ndarray, PanelTruth]:
    """`gauge` pano metresini `value` değerinde çizer; (görüntü, truth) döner.

    `gauge.face.shape` 'panel' değilse hata yükseltir — yuvarlak kadranı bu
    üreteçle çizmek sessizce yanlış geometri üretirdi.
    """
    if gauge.type != "analog":
        raise ValueError(f"{gauge.id}: render_panel_meter sadece analogda çalışır "
                         f"(tip: {gauge.type})")
    if gauge.face_shape != "panel":
        raise ValueError(f"{gauge.id}: face.shape 'panel' değil "
                         f"('{gauge.face_shape}') — yuvarlak kadran için "
                         f"synth/dial.render_analog kullanılır")

    scale = gauge.scale
    angle = scale.angle_for_value(value)          # ground truth burada doğuyor

    img = np.full((size, size, 3), background_bgr, dtype=np.uint8)
    kutu_en = int(size * KUTU_ORANI)
    kutu_boy = int(kutu_en * KUTU_EN_BOY)
    x1 = (size - kutu_en) // 2
    y1 = (size - kutu_boy) // 2
    x2, y2 = x1 + kutu_en, y1 + kutu_boy

    cv2.rectangle(img, (x1, y1), (x2, y2), BEZEL_BGR, -1)
    ic = int(kutu_en * (CERCEVE_ORANI + YUZ_PAYI))
    yuz_bgr = COLORS_BGR.get(gauge.synthetic.get("face_color", "white"),
                             COLORS_BGR["white"])
    cv2.rectangle(img, (x1 + ic, y1 + ic), (x2 - ic, y2 - ic), yuz_bgr, -1)

    px, py = gauge.pivot_ratio
    pivot = (round(x1 + px * kutu_en), round(y1 + py * kutu_boy))
    oran = gauge.sweep_radius_ratio
    r = (oran * kutu_en) if oran is not None else kutu_en * max(
        max(px, 1.0 - px), max(py, 1.0 - py))

    # --- skala: yalnız beyan edilen yay çiziliyor, tam çember DEĞİL ---
    a0, a1 = scale.angle_min, scale.angle_max
    cizgi_bgr = COLORS_BGR.get(gauge.synthetic.get("tick_color", "black"),
                               COLORS_BGR["black"])
    ana = int(gauge.synthetic.get("tick_major", 5))
    ara = int(gauge.synthetic.get("tick_minor", 4))
    toplam = ana * max(ara, 1)
    for i in range(toplam + 1):
        t = i / toplam
        a = a0 + (a1 - a0) * t
        buyuk = i % max(ara, 1) == 0
        u = ANA_CIZGI_UZUNLUK if buyuk else ARA_CIZGI_UZUNLUK
        k = max(1, int(r * (ANA_CIZGI_KALINLIK if buyuk else ARA_CIZGI_KALINLIK)))
        cv2.line(img, _nokta(pivot, r, a), _nokta(pivot, r * (1 - u), a),
                 cizgi_bgr, k, cv2.LINE_AA)
        if buyuk:
            deger = scale.min + (scale.max - scale.min) * t
            metin = f"{deger:g}"
            font = r * ETIKET_FONT
            (tw, th), _ = cv2.getTextSize(metin, FONT, font, 2)
            ex, ey = _nokta(pivot, r * ETIKET_YARICAP, a)
            cv2.putText(img, metin, (ex - tw // 2, ey + th // 2), FONT, font,
                        cizgi_bgr, 2, cv2.LINE_AA)

    # Birim yazısı: gerçek pano metrelerinde skalanın iç boşluğunda, büyük.
    if gauge.unit:
        birim = UNIT_ASCII.get(gauge.unit, gauge.unit)
        font = r * BIRIM_FONT
        (tw, th), _ = cv2.getTextSize(birim, FONT, font, 2)
        cv2.putText(img, birim, (pivot[0] - tw // 2 - int(r * 0.30),
                                 pivot[1] - int(r * 0.55) + th // 2),
                    FONT, font, TEXT_BGR, 2, cv2.LINE_AA)

    # --- ibre ---
    ibre_bgr = COLORS_BGR.get(gauge.synthetic.get("needle_color", "black"),
                              COLORS_BGR["black"])
    uc = _nokta(pivot, r * IBRE_UZUNLUK, angle)
    kuyruk = _nokta(pivot, r * IBRE_KUYRUK, angle + 180.0)
    cv2.line(img, kuyruk, uc, ibre_bgr, max(2, int(r * IBRE_KALINLIK)), cv2.LINE_AA)
    cv2.circle(img, pivot, max(2, int(r * GOBEK_ORANI)), ibre_bgr, -1, cv2.LINE_AA)

    if roll_deg:
        M = cv2.getRotationMatrix2D((size / 2, size / 2), -roll_deg, 1.0)
        img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        pivot = tuple(np.rint(M @ np.array([pivot[0], pivot[1], 1.0])).astype(int))
        uc = tuple(np.rint(M @ np.array([uc[0], uc[1], 1.0])).astype(int))

    return img, PanelTruth(
        gauge_id=gauge.id, value=float(value), angle_deg=float(angle),
        roll_deg=float(roll_deg), angle_img_deg=float(angle + roll_deg),
        pivot_px=(int(pivot[0]), int(pivot[1])), tip_px=(int(uc[0]), int(uc[1])),
        sweep_radius_px=float(r), bbox_xyxy=(x1, y1, x2, y2))
