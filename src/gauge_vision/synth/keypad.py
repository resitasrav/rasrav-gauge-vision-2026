"""Buton/tuş panelini çizer — `keypad` tipinin ground truth kaynağı.

    from gauge_vision.synth.keypad import render_keypad
    img, truth = render_keypad(gauge, {"power": "green", "run": "off", ...})

**Yerleşim ÇİZİCİDEN değil ENVANTERDEN geliyor.** Buton konumları
`gauge.buttons[].center/radius` alanlarından okunuyor — okuyucunun kullandığı
alanların ta kendisi. Böylece üreteç ile okuyucunun aynı varsayımı paylaşması
tesadüf değil, yapı gereği olur. Bu ayrım vana tarafında pahalıya öğrenildi:
"yatay = açık" hem çizicide hem okuyucuda ayrı ayrı yazılıydı ve ikisi sessizce
ayrışabiliyordu (14.08).

**Buton = pano üstünde bir lamba.** Görünüm `synth/state.py`'ın lamba
çizicisiyle aynı ilkeleri izler ve renk paletini oradan alır: sönük buton siyah
bir delik değil koyu renkli bir mercektir, yanan butonun çevresinde hale vardır.
İki çizici arasında palet kopyalansaydı biri değişip öteki unutulurdu.

⚠ **Sentetik panel gerçek panonun yerini TUTMAZ.** Gerçek panoda buton kapağı
çizik, üstünde yazı, cam yansıması ve tozlanma vardır; burada yoktur. Bu üreteç
"yöntem oturuyor mu" sorusunu cevaplar, "sahada ne olur"u değil (B1 ile aynı
sınır).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gauge_vision.config import Gauge
from gauge_vision.synth.state import (
    HALE_YARICAP_ORANI,
    LAMBA_RENKLERI,
    PANO_BGR,
    SONUK_BGR,
)

CANVAS_W, CANVAS_H = 480, 320

CERCEVE_BGR = (58, 60, 63)        # pano metal çerçevesi
CERCEVE_ORANI = 0.06              # çerçeve kalınlığı / kısa kenar
BILEZIK_BGR = (38, 40, 43)        # butonun metal bileziği
BILEZIK_PAYI = 1.28               # bilezik yarıçapı / buton yarıçapı
ETIKET_BGR = (200, 202, 205)


# --- Seçici anahtar (1-0 şalteri) ---
SELECTOR_GOVDE_BGR = (150, 152, 156)   # krom/plastik gövde — pano üstünde açık
SELECTOR_KOL_BGR = (28, 28, 30)        # kol koyu: okuyucu Otsu ile onu ayırıyor
SELECTOR_KOL_UZUNLUK = 0.92            # kol yarı-boyu / buton yarıçapı
SELECTOR_KOL_KALINLIK = 0.30           # kol kalınlığı / buton yarıçapı
SELECTOR_GOBEK_ORANI = 0.22


def _selector_ciz(img: np.ndarray, merkez: tuple[int, int], r: int,
                  aci_deg: float) -> None:
    """Seçici anahtarı çizer: açık gövde üstünde koyu, uzun, ince bir kol.

    Kolun UZUN VE İNCE olması şart — okuyucu (`read/keypad._selector_durumu`)
    kanıt kapısı olarak PCA uzamasını kullanıyor ve yuvarlağa yakın bir şekil
    "kol yok" diye reddedilir. Bu, düz bir yüzeyde Otsu'nun bulduğu gürültü
    öbeğini eleyen kapının ta kendisi; üretecin onu geçebilmesi gerekiyor.
    """
    import math
    cx, cy = merkez
    cv2.circle(img, (cx, cy), r, SELECTOR_GOVDE_BGR, -1, cv2.LINE_AA)
    yari = r * SELECTOR_KOL_UZUNLUK
    rad = math.radians(aci_deg)
    # y ekseni aşağı arttığı için eksi işaret — dosya başındaki konvansiyon.
    dx, dy = yari * math.cos(rad), -yari * math.sin(rad)
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(img, p1, p2, SELECTOR_KOL_BGR,
             max(2, int(r * SELECTOR_KOL_KALINLIK)), cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, int(r * SELECTOR_GOBEK_ORANI)),
               SELECTOR_KOL_BGR, -1, cv2.LINE_AA)


def _etiket_ciz(img: np.ndarray, metin: str, cx: int, cy: int, r: int) -> None:
    olcek = max(0.3, r / 55.0)
    (tw, th), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_SIMPLEX, olcek, 1)
    cv2.putText(img, metin, (cx - tw // 2, cy + int(r * BILEZIK_PAYI) + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, olcek, ETIKET_BGR, 1, cv2.LINE_AA)


@dataclass(frozen=True)
class KeypadTruth:
    """Çizilen panelin bilinen hâli — ölçüm bunun üstünden yapılır."""

    gauge_id: str
    button_states: dict[str, str]
    machine_state: str | None          # envanter kuralı eşleşmiyorsa None
    bbox_xyxy: tuple[int, int, int, int]
    button_boxes: dict[str, tuple[int, int, int, int]]


def _makine_durumu(gauge: Gauge, durumlar: dict[str, str]) -> str | None:
    """Envanterdeki kurallara göre beklenen makine durumu.

    Okuyucudaki eşleştirmenin AYNISI; burada ground truth üretmek için
    kullanılıyor. Ortak bir yardımcıya taşınmadı çünkü okuyucunun kural
    yorumunu üreteçten bağımsız sınamak istiyoruz — ikisi tek fonksiyona
    bağlansaydı, yanlış bir yorum iki tarafta birden doğru görünürdü.
    """
    for kural in gauge.machine_states:
        if all(durumlar.get(bid) == beklenen
               for bid, beklenen in (kural.get("when") or {}).items()):
            return kural["name"]
    return None


def render_keypad(
    gauge: Gauge,
    button_states: dict[str, str],
    *,
    size: tuple[int, int] = (CANVAS_W, CANVAS_H),
    etiket_goster: bool = True,
) -> tuple[np.ndarray, KeypadTruth]:
    """`gauge` panelini verilen buton durumlarıyla çizer.

    `button_states` envanterdeki her butonu kapsamalıdır; eksik buton hata
    yükseltir. Sessizce "off" varsaymak, ölçümde okuyucunun hatasını
    üretecin varsayımıyla karıştırırdı.
    """
    if gauge.type != "keypad":
        raise ValueError(f"{gauge.id}: render_keypad sadece buton panelinde çalışır "
                         f"(tip: {gauge.type})")

    eksik = [b["id"] for b in gauge.buttons if b["id"] not in button_states]
    if eksik:
        raise ValueError(f"{gauge.id}: buton durumu verilmemiş: {eksik}")

    w, h = size
    img = np.full((h, w, 3), CERCEVE_BGR, dtype=np.uint8)
    kenar = int(min(h, w) * CERCEVE_ORANI)
    cv2.rectangle(img, (kenar, kenar), (w - kenar, h - kenar), PANO_BGR, -1)

    kutular: dict[str, tuple[int, int, int, int]] = {}
    for b in gauge.buttons:
        bid = b["id"]
        durum = button_states[bid]
        izinli = list(b.get("states") or [])
        if durum not in izinli:
            raise ValueError(f"{gauge.id}/{bid}: '{durum}' envanterde tanımlı değil "
                             f"— mevcutlar: {izinli}")

        cx = int(float(b["center"][0]) * w)
        cy = int(float(b["center"][1]) * h)
        # Yarıçap KISA kenara göre: geniş panoda uzun kenara göre alınan yarıçap
        # butonu komşusunun üstüne taşırır. Okuyucu da aynı kuralı kullanıyor.
        r = int(float(b["radius"]) * min(h, w))

        cv2.circle(img, (cx, cy), int(r * BILEZIK_PAYI), BILEZIK_BGR, -1, cv2.LINE_AA)

        if str(b.get("kind", "lamp")) == "selector":
            # Seçici anahtar: ışığı yok, durumu KOLUN AÇISI söylüyor. Açı
            # envanterden (`lever_angles`) geliyor — okuyucunun baktığı alanın
            # ta kendisi, böylece üreteçle okuyucu yapı gereği aynı varsayımı
            # paylaşıyor (vana tarafında pahalıya öğrenilen ders).
            _selector_ciz(img, (cx, cy), r,
                          float((b.get("lever_angles") or {})[durum]))
            kutular[bid] = (cx - r, cy - r, cx + r, cy + r)
            if etiket_goster and b.get("label"):
                _etiket_ciz(img, str(b["label"]), cx, cy, r)
            continue

        if durum == "off":
            # Sönük buton: o butonun kendi renginin koyusu. Hangi renk olduğu
            # `states` listesinden geliyor — sönük kırmızı ile sönük yeşil
            # farklı görünür ve ikisi de "off"tur.
            renk = next((s for s in izinli if s != "off"), None)
            koyu = LAMBA_RENKLERI.get(renk, (None, SONUK_BGR))[1] if renk else SONUK_BGR
            cv2.circle(img, (cx, cy), r, koyu, -1, cv2.LINE_AA)
        else:
            yanik, _ = LAMBA_RENKLERI.get(durum, ((255, 255, 255), SONUK_BGR))
            hale = np.zeros((h, w), np.float32)
            cv2.circle(hale, (cx, cy), int(r * (HALE_YARICAP_ORANI / 0.28)), 1.0, -1,
                       cv2.LINE_AA)
            hale = cv2.GaussianBlur(hale, (0, 0), r * 0.55)
            katki = hale[..., None] * np.array(yanik, np.float32) * 0.40
            img = np.clip(img.astype(np.float32) + katki, 0, 255).astype(np.uint8)

            cv2.circle(img, (cx, cy), r, yanik, -1, cv2.LINE_AA)
            cv2.circle(img, (cx - r // 4, cy - r // 4), max(2, r // 4),
                       tuple(int(min(255, c + 60)) for c in yanik), -1, cv2.LINE_AA)

        if etiket_goster and b.get("label"):
            metin = str(b["label"])
            olcek = max(0.3, r / 55.0)
            (tw, th), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_SIMPLEX, olcek, 1)
            cv2.putText(img, metin, (cx - tw // 2, cy + int(r * BILEZIK_PAYI) + th + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, olcek, ETIKET_BGR, 1, cv2.LINE_AA)

        kutular[bid] = (cx - r, cy - r, cx + r, cy + r)

    truth = KeypadTruth(
        gauge_id=gauge.id,
        button_states=dict(button_states),
        machine_state=_makine_durumu(gauge, button_states),
        bbox_xyxy=(0, 0, w, h),
        button_boxes=kutular,
    )
    return img, truth
