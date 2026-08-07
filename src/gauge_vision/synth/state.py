"""İkaz lambası ve vana çizer — İP12'nin ground truth kaynağı.

Analog ve dijital tarafla aynı ilke: durumu biz belirlediğimiz için doğruluk
bedava ölçülür.

**Lamba gerçekçiliği için iki ayrıntı bilinçli:**

1. *Sönük lamba görünür kalır.* Sönmüş bir ikaz lambası siyah bir delik değil,
   koyu renkli bir mercektir — kırmızı lamba söndüğünde koyu kırmızıdır. Bu
   yüzden sönük durum, rengin çok koyusuyla çiziliyor. Ayırt edici olan renk
   değil DOYGUNLUK ve PARLAKLIKTIR; okuyucu da bunu kullanıyor.
2. *Yanan lambanın çevresine hale.* LED merceği yayar; ölçümde hale olmaması,
   yanık lambayı olduğundan kolay ayırt edilir yapardı.

**Vana:** boru hattı yatay çiziliyor, kol hattın üstünde dönüyor. `open`
durumunda kol hatta paralel (yatay), `closed` durumunda dik.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from gauge_vision.config import Gauge

CANVAS_PX = 320

# --- Lamba ---
LAMBA_YARICAP_ORANI = 0.28      # mercek yarıçapı / kare kenarı
GOVDE_YARICAP_ORANI = 0.36      # metal gövde
HALE_YARICAP_ORANI = 0.46       # yanık lambanın halesi
PANO_BGR = (105, 108, 110)      # gri pano zemini
GOVDE_BGR = (60, 62, 65)

# Durum adı → (yanık BGR, sönük BGR). Sönük renk, yanığın çok koyusu.
LAMBA_RENKLERI: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "red": ((40, 40, 235), (18, 18, 62)),
    "green": ((60, 230, 60), (16, 58, 16)),
    "yellow": ((40, 220, 235), (14, 56, 60)),
    "blue": ((235, 120, 40), (60, 32, 14)),
}
SONUK_BGR = (34, 34, 38)        # `off` durumu — mercek koyu, renksiz

# --- Vana ---
BORU_KALINLIK_ORANI = 0.14
BORU_BGR = (120, 122, 125)
KOL_UZUNLUK_ORANI = 0.62        # kol uzunluğu / kare kenarı
KOL_KALINLIK_ORANI = 0.075
KOL_BGR = (35, 35, 40)
GOBEK_ORANI = 0.07


@dataclass(frozen=True)
class StateTruth:
    """Çizilen durumun ground truth'u."""

    gauge_id: str
    state: str
    lever_angle_deg: float | None      # vana için kolun açısı, lamba için None
    bbox_xyxy: tuple[int, int, int, int]


def render_lamp(gauge: Gauge, state: str, *, size: int = CANVAS_PX,
                renk: str | None = None) -> tuple[np.ndarray, StateTruth]:
    """`state` durumundaki ikaz lambasını çizer.

    `renk` yalnızca `state == "off"` iken anlamlıdır: sönük bir kırmızı lamba
    ile sönük bir yeşil lamba farklı görünür ve ikisi de `off` durumundadır.
    """
    if gauge.type != "lamp":
        raise ValueError(f"{gauge.id}: render_lamp sadece lambada çalışır "
                         f"(tip: {gauge.type})")

    img = np.full((size, size, 3), PANO_BGR, dtype=np.uint8)
    merkez = (size // 2, size // 2)
    r_mercek = int(size * LAMBA_YARICAP_ORANI)

    cv2.circle(img, merkez, int(size * GOVDE_YARICAP_ORANI), GOVDE_BGR, -1, cv2.LINE_AA)

    if state == "off":
        # Sönük lamba: mercek koyu. Renk verilmişse o rengin koyusu.
        koyu = LAMBA_RENKLERI.get(renk, (None, SONUK_BGR))[1] if renk else SONUK_BGR
        cv2.circle(img, merkez, r_mercek, koyu, -1, cv2.LINE_AA)
    else:
        yanik, _ = LAMBA_RENKLERI.get(state, ((255, 255, 255), SONUK_BGR))
        # Hale: merkezden dışa doğru sönen ışık.
        hale = np.zeros((size, size), np.float32)
        cv2.circle(hale, merkez, int(size * HALE_YARICAP_ORANI), 1.0, -1, cv2.LINE_AA)
        hale = cv2.GaussianBlur(hale, (0, 0), size * 0.06)
        katki = (hale[..., None] * np.array(yanik, np.float32) * 0.45)
        img = np.clip(img.astype(np.float32) + katki, 0, 255).astype(np.uint8)

        cv2.circle(img, merkez, r_mercek, yanik, -1, cv2.LINE_AA)
        # Mercek parlaması: merkezde daha açık bir nokta.
        cv2.circle(img, (merkez[0] - r_mercek // 4, merkez[1] - r_mercek // 4),
                   max(2, r_mercek // 4),
                   tuple(int(min(255, c + 60)) for c in yanik), -1, cv2.LINE_AA)

    r_dis = int(size * GOVDE_YARICAP_ORANI)
    truth = StateTruth(gauge_id=gauge.id, state=state, lever_angle_deg=None,
                       bbox_xyxy=(merkez[0] - r_dis, merkez[1] - r_dis,
                                  merkez[0] + r_dis, merkez[1] + r_dis))
    return img, truth


def render_valve(gauge: Gauge, state: str, *, size: int = CANVAS_PX,
                 sapma_deg: float = 0.0) -> tuple[np.ndarray, StateTruth]:
    """`state` durumundaki kelebek vanayı çizer.

    `sapma_deg` kolu ideal açıdan kaydırır — okuyucunun ±20° toleransını ve
    "arada kalırsa okuma yok" davranışını sınamak için.
    """
    if gauge.type != "valve":
        raise ValueError(f"{gauge.id}: render_valve sadece vanada çalışır "
                         f"(tip: {gauge.type})")

    img = np.full((size, size, 3), PANO_BGR, dtype=np.uint8)
    merkez = (size // 2, size // 2)

    # Boru hattı: yatay.
    kalinlik = int(size * BORU_KALINLIK_ORANI)
    cv2.rectangle(img, (0, merkez[1] - kalinlik // 2),
                  (size, merkez[1] + kalinlik // 2), BORU_BGR, -1)

    # Kol: open → hatta paralel (0°), closed → dik (90°).
    temel = 0.0 if state == "open" else 90.0
    aci = temel + sapma_deg
    yari = size * KOL_UZUNLUK_ORANI / 2.0
    rad = math.radians(aci)
    # y ekseni aşağı arttığı için sinüs eksi işaretli (CLAUDE.md §3).
    dx, dy = yari * math.cos(rad), -yari * math.sin(rad)
    p1 = (int(merkez[0] - dx), int(merkez[1] - dy))
    p2 = (int(merkez[0] + dx), int(merkez[1] + dy))

    cv2.line(img, p1, p2, KOL_BGR, max(2, int(size * KOL_KALINLIK_ORANI)), cv2.LINE_AA)
    cv2.circle(img, merkez, max(3, int(size * GOBEK_ORANI)), KOL_BGR, -1, cv2.LINE_AA)

    truth = StateTruth(gauge_id=gauge.id, state=state, lever_angle_deg=aci % 180.0,
                       bbox_xyxy=(int(merkez[0] - yari), int(merkez[1] - yari),
                                  int(merkez[0] + yari), int(merkez[1] + yari)))
    return img, truth
