"""7-segment dijital panel çizer — İP11'in ground truth kaynağı.

    from gauge_vision.synth.digital import render_digital

    img, truth = render_digital(gauge, value=123.4)
    truth.text   # "123.4"

Analog tarafta olduğu gibi burada da **sentetik-önce**: hangi rakamın
gösterildiğini biz belirlediğimiz için doğruluk bedava ölçülür. Gerçek panel
fotoğrafı geldiğinde yöntem çoktan oturmuş olur.

**Neden hazır bir font yerine segmentler tek tek çiziliyor:** 7-segment ekranın
okunma zorluğu fontun şeklinden değil, **segmentlerin ayrık olmasından** gelir.
Sönük segment ile yanık segment arasındaki kontrast, sayının bütününü değil
parçalarını görünür kılar; klasik OCR bu yüzden zorlanır ve `letsgodigital`
gibi ayrı modeller gerekir. Hershey fontuyla çizilen "8", gerçek bir 7-segment
"8"in yarattığı problemi üretmez — ölçüm kolay çıkar ve yanıltır.

Sönük segmentler bilinçli olarak **çiziliyor** (koyu tonda). Gerçek LED/LCD
panellerde sönük segment görünür kalır ve okumanın asıl zorluğu budur: "1"
rakamının yanındaki sönük segmentler "8"e benzer bir hayalet bırakır.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gauge_vision.config import Gauge

# --- Yerleşim (hepsi panel yüksekliğine oran) ---
CANVAS_W, CANVAS_H = 512, 220
PANEL_KENAR_ORANI = 0.10        # panel çerçevesi
HANE_YUKSEKLIK_ORANI = 0.52     # rakam yüksekliği / panel yüksekliği
HANE_EN_ORANI = 0.52            # rakam eni / rakam yüksekliği
HANE_ARALIK_ORANI = 0.22        # haneler arası boşluk / rakam eni
SEGMENT_KALINLIK_ORANI = 0.16   # segment kalınlığı / rakam eni
EGIM_ORANI = 0.10               # rakamların sağa yatıklığı (gerçek panellerde var)

# --- Renkler (BGR) ---
PANEL_BGR = (28, 28, 30)        # koyu ekran
CERCEVE_BGR = (70, 70, 75)
YANIK_BGR = (60, 240, 60)       # yeşil LED
SONUK_BGR = (44, 60, 44)        # yanık rengin çok koyusu — görünür ama sönük
LCD_YANIK_BGR = (35, 35, 40)    # LCD'de rakam koyu, zemin açık
LCD_SONUK_BGR = (168, 172, 168)
LCD_PANEL_BGR = (178, 182, 176)

# Segment düzeni:  a üst · b sağ-üst · c sağ-alt · d alt · e sol-alt · f sol-üst · g orta
RAKAM_SEGMENTLERI: dict[str, str] = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
    "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg",
    " ": "",
}
EKSI_SEGMENTLERI = "g"


@dataclass(frozen=True)
class DigitalTruth:
    """Çizilen panelin ground truth'u."""

    gauge_id: str
    value: float
    text: str                          # ekranda görünen dizge, örn. "-12.3"
    digit_boxes: list[tuple[int, int, int, int]]   # her hanenin kutusu
    panel_bbox_xyxy: tuple[int, int, int, int]     # ekranın kutusu (İP5 hedefi)


def bicimle(value: float, count: int, decimals: int, allow_minus: bool) -> str:
    """Değeri panelin hane düzenine sığdırır.

    Gerçek paneller sığmayan değeri kırpar ya da taşma gösterir; burada sağa
    hizalayıp soldan boşlukla dolduruyoruz — dört haneli bir panelde "12.3"
    değil " 12.3" görünür ve OCR'ın boşluğu rakam sanmaması gerekir.
    """
    if not allow_minus and value < 0:
        value = abs(value)
    metin = f"{value:.{decimals}f}"
    # Nokta hane sayılmaz; ayrı çizilir. Eksi işareti bir hane kaplar.
    hane_sayisi = len(metin.replace(".", ""))
    if hane_sayisi > count:
        # Taşma: en anlamlı hanelerden kırpmak yanlış okuma üretir, panel
        # gerçekte "----" gösterir. Ölçümde bu durum ayrı ele alınmalı.
        return "-" * count
    # Panelin hane sayısı FİZİKSELDİR: dört haneli bir ekranda "7.0" iki hane
    # değil, ikisi boş dört hanedir. Boşlukla soldan doldurulmazsa üreteç iki
    # hane çizer, okuyucu dört arar ve hane kutuları kayar.
    return " " * (count - hane_sayisi) + metin


def _segment_kutulari(x: float, y: float, w: float, h: float, t: float) -> dict:
    """Bir hanenin 7 segmentinin köşe noktaları. (x,y) sol üst."""
    y_orta = y + h / 2.0
    return {
        "a": [(x + t, y), (x + w - t, y), (x + w - t * 1.5, y + t), (x + t * 1.5, y + t)],
        "g": [(x + t, y_orta - t / 2), (x + w - t, y_orta - t / 2),
              (x + w - t, y_orta + t / 2), (x + t, y_orta + t / 2)],
        "d": [(x + t * 1.5, y + h - t), (x + w - t * 1.5, y + h - t),
              (x + w - t, y + h), (x + t, y + h)],
        "f": [(x, y + t), (x + t, y + t * 1.5), (x + t, y_orta - t * 0.75),
              (x, y_orta - t / 2)],
        "b": [(x + w - t, y + t * 1.5), (x + w, y + t), (x + w, y_orta - t / 2),
              (x + w - t, y_orta - t * 0.75)],
        "e": [(x, y_orta + t / 2), (x + t, y_orta + t * 0.75),
              (x + t, y + h - t * 1.5), (x, y + h - t)],
        "c": [(x + w - t, y_orta + t * 0.75), (x + w, y_orta + t / 2),
              (x + w, y + h - t), (x + w - t, y + h - t * 1.5)],
    }


def _egim_uygula(noktalar, y0: float, h: float, egim: float):
    """Rakamları sağa yatırır — gerçek 7-segment panellerde standarttır."""
    return np.array([[px + egim * (y0 + h - py), py] for (px, py) in noktalar],
                    dtype=np.int32)


def render_digital(gauge: Gauge, value: float, *, size=(CANVAS_W, CANVAS_H),
                   sonuk_goster: bool = True) -> tuple[np.ndarray, DigitalTruth]:
    """`gauge` panelini `value` değerinde çizer.

    `sonuk_goster=False` sönük segmentleri gizler — ablasyon için: sönük
    segmentlerin okumayı ne kadar zorlaştırdığı böyle ölçülür.
    """
    if gauge.type != "digital":
        raise ValueError(f"{gauge.id}: render_digital sadece dijital panelde çalışır "
                         f"(tip: {gauge.type})")

    d = gauge.digits or {}
    count = int(d.get("count", 4))
    decimals = int(d.get("decimals", 1))
    lcd = d.get("style") == "lcd_text"
    allow_minus = bool(d.get("allow_minus", True))

    w_img, h_img = size
    panel_bgr = LCD_PANEL_BGR if lcd else PANEL_BGR
    yanik = LCD_YANIK_BGR if lcd else YANIK_BGR
    sonuk = LCD_SONUK_BGR if lcd else SONUK_BGR

    img = np.full((h_img, w_img, 3), CERCEVE_BGR, dtype=np.uint8)
    kenar = int(h_img * PANEL_KENAR_ORANI)
    panel = (kenar, kenar, w_img - kenar, h_img - kenar)
    cv2.rectangle(img, panel[:2], panel[2:], panel_bgr, -1)

    metin = bicimle(value, count, decimals, allow_minus)
    haneler = [c for c in metin if c != "."]
    nokta_sonrasi = {i for i, c in enumerate(metin) if c == "."}

    hane_h = h_img * HANE_YUKSEKLIK_ORANI
    hane_w = hane_h * HANE_EN_ORANI
    aralik = hane_w * HANE_ARALIK_ORANI
    kalinlik = hane_w * SEGMENT_KALINLIK_ORANI
    egim = EGIM_ORANI

    toplam_w = len(haneler) * hane_w + (len(haneler) - 1) * aralik
    x0 = (w_img - toplam_w) / 2.0
    y0 = (h_img - hane_h) / 2.0

    kutular = []
    metin_idx = 0
    for i, ch in enumerate(haneler):
        # Metindeki noktaları atlayarak ilerle — nokta hane değil.
        while metin_idx < len(metin) and metin[metin_idx] == ".":
            metin_idx += 1
        metin_idx += 1

        hx = x0 + i * (hane_w + aralik)
        aktif = EKSI_SEGMENTLERI if ch == "-" else RAKAM_SEGMENTLERI.get(ch, "")
        segmentler = _segment_kutulari(hx, y0, hane_w, hane_h, kalinlik)

        for ad, koseler in segmentler.items():
            acik = ad in aktif
            if not acik and not sonuk_goster:
                continue
            cv2.fillPoly(img, [_egim_uygula(koseler, y0, hane_h, egim)],
                         yanik if acik else sonuk, cv2.LINE_AA)

        kutular.append((int(hx), int(y0), int(hx + hane_w + egim * hane_h),
                        int(y0 + hane_h)))

        # Ondalık nokta: bu haneden sonra geliyorsa çiz.
        if metin_idx < len(metin) and metin[metin_idx] == ".":
            nx = int(hx + hane_w + aralik * 0.35)
            ny = int(y0 + hane_h - kalinlik * 0.6)
            cv2.circle(img, (nx, ny), max(2, int(kalinlik * 0.55)), yanik, -1, cv2.LINE_AA)

    truth = DigitalTruth(gauge_id=gauge.id, value=float(value), text=metin,
                         digit_boxes=kutular, panel_bbox_xyxy=panel)
    return img, truth
