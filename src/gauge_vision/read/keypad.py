"""Buton/tuş panelini okur: hangi butonlar yanıyor → makine ne yapıyor.

    from gauge_vision.read.keypad import read_keypad
    okuma = read_keypad(kesit, gauge)
    okuma.value        # "calisiyor"

**Neden ayrı bir tip.** Analog kadran, dijital panel, lamba ve vana tek bir
büyüklük okur. Buton paneli okumaz: taşıdığı bilgi **butonların bileşimidir.**
"Şalter yanıyor" tek başına bir şey söylemez; "şalter yeşil + çalıştır yeşil +
arıza sönük" makinenin çalıştığını söyler. Bu yüzden okuyucu iki katmanlıdır —
önce her buton tek tek sınıflanır, sonra bileşim envanterdeki kurallara vurulur.

**Buton = bilinen konumdaki lamba.** Renk sınıflandırması `read/state.py`'ın
lamba mantığının AYNISIDIR ve oradan çağrılır; ikinci bir kopya yazılsaydı
biri düzeltilip öteki unutulurdu (14.08'de mutlak parlaklık eşiği tam olarak
böyle üç ayrı yerde tekrarlanmıştı). Buradaki tek fark, bölgenin karenin
ortası değil **envanterin beyan ettiği konum** olmasıdır.

**Bilinmeyen bileşime ad UYDURULMAZ.** Hiçbir kural eşleşmezse `unreadable`
basılır (3. kural). Bir buton panelinde yanlış okumanın bedeli bir sayı değil
makinenin durumudur: duran bir makineye "çalışıyor" demek, yanlış bir basınç
değerinden tehlikelidir.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from gauge_vision.config import Gauge
from gauge_vision.read.calibrate import (
    DURUM_ALARM,
    DURUM_OK,
    DURUM_OKUNAMADI,
    GaugeReading,
)
from gauge_vision.read.state import _lamba_durumu

# Buton kesiti, beyan edilen yarıçapın bu katı kadar alınır. 1'den büyük olmalı:
# `_lamba_durumu` parlaklık referansını merceğin DIŞINDAKİ halkadan alıyor
# (`_cevre_bolgesi`), dolayısıyla kesitte butonun çevresinden de pay bulunmalı.
# Tam butona kırpılırsa referans butonun kendisi olur ve yanan buton "sönük"
# çıkar — mutlak eşik hatasının kılık değiştirmiş hâli.
KESIT_PAYI = 2.2
# Bundan küçük bir kesit güvenilir sınıflandırma vermez (uzaktaki pano).
MIN_KESIT_PX = 12

# --- "Burada gerçekten bir buton var mı?" kapısı ---
# **Neden gerekli.** `_lamba_durumu` parlak piksel bulamayınca "off" der ve bunu
# yüksek güvenle söyler. Buton panelinde bu ölümcül: DÜZ GRİ bir kare, dört
# butonu da "off" okuyup `enerji_yok` durumunu **güven 1,00 ile** yayınlıyordu
# (27.08, birim testi yakaladı). Sahada bu, kamera panoyu hiç görmediğinde
# "makinede enerji yok" diye rapor etmek demektir.
#
# Bu, depoda üçüncü kez çıkan hata sınıfı: kapı "cevap makul mü" diye soruyor,
# "KANIT VAR MI" diye değil (`refine.py` ve `roll.py` ilk sürümleri rastgele
# gürültüyü kabul ediyordu — aynı gerekçe).
#
# Ayırt edici: gerçek bir buton — YANIK DA SÖNÜK DE — merceğiyle çevresi
# arasında kontrast üretir; bilezik koyu, mercek renkli, pano açıktır. Düz bir
# yüzeyde bu fark yoktur. Ölçüt bağıl olduğu için ışık kazancıyla ölçeklenmez.
#
# Eşik ölçülen iki dağılımın arasına kondu (27.08, 16 gerçek + 16 sahte kesit):
#   gerçek buton (yanık+sönük) : min 0,436 · medyan 0,524 · maks 0,577
#   kanıt yok (düz/gürültü/gradyan) : maks 0,015
# Aralık çok geniş; 0,12 sahtenin 8 katı üstünde, gerçeğin 3,6 katı altında.
#
# ⚠ Bu sayılar SENTETİK panelden. Gerçek panoda cam yansıması ve tozlanma
# kontrastı düşürecektir; eşik gerçek pano fotoğrafı gelince yeniden ölçülmeli
# (`refine.MAX_ARTIK_ORANI` ile aynı durum).
MIN_BUTON_KANITI = 0.12
# Kanıt ölçümünde kullanılan halkalar, butonun KENDİ yarıçapına oran.
KANIT_IC_ORANI = 0.80    # mercek diski
KANIT_DIS_IC, KANIT_DIS_DIS = 1.40, 1.90   # bileziğin dışındaki pano halkası


def _buton_kesiti(image: np.ndarray, merkez_oran, yaricap_oran: float) -> np.ndarray | None:
    """Butonun çevresiyle birlikte kesiti. Kare dışına taşarsa sınırlara kırpılır."""
    h, w = image.shape[:2]
    cx, cy = float(merkez_oran[0]) * w, float(merkez_oran[1]) * h
    # Yarıçap oranı kutunun KISA kenarına göre: geniş bir panoda uzun kenara
    # göre alınan yarıçap butonu aşar ve komşusunu içeri alır.
    r = yaricap_oran * min(h, w) * KESIT_PAYI
    x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
    x2, y2 = min(w, int(cx + r)), min(h, int(cy + r))
    if x2 - x1 < MIN_KESIT_PX or y2 - y1 < MIN_KESIT_PX:
        return None
    return image[y1:y2, x1:x2]


def _buton_kanidi(kesit: np.ndarray) -> float:
    """Kesitte gerçekten bir buton var mı — mercek ile pano arası bağıl kontrast.

    Ölçekler butonun KENDİ yarıçapına göre: kesit yarı-eni `KESIT_PAYI × r`
    olduğundan `r = min(kesit) / (2 · KESIT_PAYI)`. Sabit oranlar
    (`_lamba_bolgesi` gibi) kullanılamaz — onlar kesitin tamamını lamba kabul
    eder ve burada mercek kesitin yalnız bir bölümüdür.
    """
    h, w = kesit.shape[:2]
    r = min(h, w) / (2.0 * KESIT_PAYI)
    if r < 2:
        return 0.0
    cx, cy = w // 2, h // 2
    v = cv2.cvtColor(kesit, cv2.COLOR_BGR2HSV)[:, :, 2]

    ic = np.zeros((h, w), np.uint8)
    cv2.circle(ic, (cx, cy), max(1, int(r * KANIT_IC_ORANI)), 255, -1)
    dis = np.zeros((h, w), np.uint8)
    cv2.circle(dis, (cx, cy), max(2, int(r * KANIT_DIS_DIS)), 255, -1)
    cv2.circle(dis, (cx, cy), max(1, int(r * KANIT_DIS_IC)), 0, -1)

    if int(np.count_nonzero(ic)) < 16 or int(np.count_nonzero(dis)) < 16:
        return 0.0
    mercek = float(np.median(v[ic > 0]))
    pano = float(np.median(v[dis > 0]))
    return abs(mercek - pano) / max(pano, 1.0)


def _kural_eslesiyor(kural: dict[str, Any], durumlar: dict[str, str]) -> bool:
    """Kuralın `when` koşulu okunan durumlara uyuyor mu.

    Kural yalnız ADI GEÇEN butonlara bakar; geçmeyenler serbesttir. Böylece
    "arıza yanıyorsa gerisi önemsiz" gibi bir kural tek satırla yazılabilir.
    """
    return all(durumlar.get(bid) == beklenen
               for bid, beklenen in (kural.get("when") or {}).items())


def read_keypad(image: np.ndarray, gauge: Gauge) -> GaugeReading:
    """Buton panelini okur ve `inspect/reading` gövdesi üretir.

    Hata yükseltmez; sorunlar `status` ile bildirilir.
    """
    if gauge.type != "keypad":
        raise ValueError(f"{gauge.id}: read_keypad sadece buton panelinde çalışır "
                         f"(tip: {gauge.type})")

    def bos(status: str, conf: float = 0.0, detay: dict | None = None) -> GaugeReading:
        return GaugeReading(gauge_id=gauge.id, type=gauge.type, value=None,
                            unit=gauge.unit, conf=conf, status=status,
                            raw_angle=0.0, dial_angle=None,
                            extra={"buttons": detay or {}})

    durumlar: dict[str, str] = {}
    guvenler: list[float] = []
    for b in gauge.buttons:
        kesit = _buton_kesiti(image, b["center"], float(b["radius"]))
        if kesit is None:
            return bos(DURUM_OKUNAMADI, 0.0, durumlar)

        # Renk sorulmadan ÖNCE "burada buton var mı" sorulur: sönük buton ile
        # boş bir yüzey aynı cevabı verir ve ikincisine durum atamak, kameranın
        # görmediği bir makineyi raporlamaktır.
        if _buton_kanidi(kesit) < MIN_BUTON_KANITI:
            return bos(DURUM_OKUNAMADI, 0.0, durumlar)

        ad, guven = _lamba_durumu(kesit, list(b.get("states") or []))
        if ad is None:
            # Tek bir okunamayan buton tüm bileşimi geçersiz kılar: eksik bir
            # butonla kural eşleştirmek, görmediğin bir lambayı sönük saymaktır.
            return bos(DURUM_OKUNAMADI, float(guven), durumlar)
        durumlar[b["id"]] = ad
        guvenler.append(float(guven))

    # Zincirin güveni en zayıf butondan yüksek olamaz — "8" ile "0"ı ayıran tek
    # segmentin dijitalde yaptığı işi burada tek buton yapıyor.
    guven = float(min(guvenler)) if guvenler else 0.0

    eslesen = next((k for k in gauge.machine_states
                    if _kural_eslesiyor(k, durumlar)), None)
    if eslesen is None:
        # Butonlar okundu ama bu bileşimin ne demek olduğu envanterde yazmıyor.
        # Okuma başarısız değil, YORUM eksik — ikisini ayırmak için sebep
        # `extra`'da taşınıyor ki envantere hangi kuralın ekleneceği görülsün.
        return bos(DURUM_OKUNAMADI, guven, durumlar)

    if guven < gauge.conf_threshold:
        return bos(DURUM_OKUNAMADI, guven, durumlar)

    durum = DURUM_ALARM if eslesen.get("alarm") else DURUM_OK
    return GaugeReading(gauge_id=gauge.id, type=gauge.type,
                        value=eslesen["name"], unit=gauge.unit, conf=guven,
                        status=durum, raw_angle=0.0, dial_angle=None,
                        extra={"buttons": durumlar})
