"""Letterbox (siyah bant) tespiti ve kırpma.

NEDEN VAR: `0831.mp4` farklı en-boy oranlarındaki AI kliplerinin birleşimi.
Bir kısmı DİKEY üretilmiş ve 1920x1080'e siyah bantlarla gömülmüş; orada
gerçek içerik karenin yalnız **%32'si**. Zincir bu kareyi olduğu gibi alınca
YOLO 640'a küçültüyor ve içerik ~200 px'e iniyor — kadranlar, butonlar ve
uzaktaki insanlar tespit edilemeyecek kadar küçülüyor.

Kırpma bunu geri kazandırıyor: aynı içerik ~3 kat daha büyük görünüyor.

İKİ TUZAK VE ÇÖZÜMLERİ

1. **Filigran.** Videonun sol üstünde "Ai" filigranı var ve siyah bandın
   İÇİNDE duruyor. Sütunun MAKSİMUMUNA bakan bir tespit onu içerik sanıp
   bandı kırpamıyor (ölçüldü: x=47'den başlıyor sanıyordu, oysa içerik
   x=656'da başlıyor). Bu yüzden maksimum değil YÜKSEK YÜZDELİK kullanılıyor:
   birkaç parlak piksel bir sütunu "içerik" yapmaya yetmiyor.

2. **Geometri video ortasında DEĞİŞİYOR.** İlk klipler dikey, sonrakiler tam
   genişlik. Sabit bir kırpma ikisinden birini bozar. Bu yüzden kırpma her
   karede yeniden ölçülüyor, ama HİSTEREZİSLE: küçük oynamalar yok sayılıyor,
   yoksa kutu her karede birkaç piksel titrer ve MOG2 bunu hareket sanar.
   Gerçekten değiştiğinde `degisti` bayrağı kalkıyor — arka plan modeli
   sıfırlanmalı, çünkü kadraj kaydığında eski arka plan geçersizdir.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Sütun/satır "siyah" sayılır: piksellerinin bu yüzdeliği bu tonun altındaysa.
#
# ÖLÇÜLDÜ (0831.mp4 kare 50, gerçek içerik x=656'da başlıyor):
#   yüzdelik 99 -> x=50   (yanlış)      yüzdelik 95 -> x=656  (doğru)
#   yüzdelik 98 -> x=73   (yanlış)      yüzdelik 90 -> x=656  (doğru)
# Filigranın bulunduğu sütunda parlak piksel oranı yalnız %2,2; gerçek içerik
# sütununda %100. 90 seçildi çünkü filigrana karşı geniş pay bırakıyor ve
# gerçek içerik sütunları o eşiğin çok üstünde.
YUZDELIK = 90.0
TON_ESIGI = 20

# Kutu bu kadar pikselden az oynadıysa değişmemiş sayılır (histerezis).
OYNAMA_PAYI = 12
# İçerik bunun altına düşerse kırpma reddedilir — tamamen karanlık bir sahne
# (ör. duman veya kesme) yanlışlıkla "her yer bant" diye okunabilir ve kare
# yok olur. Böyle bir karede ÖNCEKİ kutu korunur.
MIN_ORAN = 0.15
# Yeni kirpim kutusu kabul edilmeden once kac kare tekrar etmeli.
ONAY_KARE = 15


@dataclass
class Kirpim:
    x1: int
    y1: int
    x2: int
    y2: int
    degisti: bool = False

    @property
    def kutu(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def tam(self) -> bool:
        return self.degisti is False and (self.x1, self.y1) == (0, 0)

    def uygula(self, kare: np.ndarray) -> np.ndarray:
        return kare[self.y1:self.y2, self.x1:self.x2]


def _sinirlar(profil: np.ndarray, uzunluk: int) -> tuple[int, int]:
    dolu = np.where(profil > TON_ESIGI)[0]
    if dolu.size == 0:
        return 0, uzunluk
    return int(dolu[0]), int(dolu[-1]) + 1


class KirpimIzleyici:
    """Kareden kareye kırpma kutusunu histerezis + zamansal onayla takip eder."""

    def __init__(self, onay_kare: int = ONAY_KARE):
        self.onceki: Kirpim | None = None
        self.aday: tuple[int, int, int, int] | None = None
        self.aday_sayaci = 0
        self.onay_kare = onay_kare

    def olc(self, kare: np.ndarray) -> Kirpim:
        gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        h, w = gri.shape
        # Yüzdelik eksen boyunca alınıyor: sütun profili için satırlar üzerinden.
        sutun = np.percentile(gri, YUZDELIK, axis=0)
        satir = np.percentile(gri, YUZDELIK, axis=1)
        x1, x2 = _sinirlar(sutun, w)
        y1, y2 = _sinirlar(satir, h)

        gecerli = (x2 - x1) >= w * MIN_ORAN and (y2 - y1) >= h * MIN_ORAN
        if not gecerli:
            # Karanlık kare: yeni ölçüme güvenilmez, öncekini sürdür.
            return self.onceki or Kirpim(0, 0, w, h)

        olculen = (x1, y1, x2, y2)
        if self.onceki is None:
            self.onceki = Kirpim(*olculen, degisti=True)
            return self.onceki

        o = self.onceki
        oynama = max(abs(x1 - o.x1), abs(y1 - o.y1), abs(x2 - o.x2), abs(y2 - o.y2))
        if oynama <= OYNAMA_PAYI:
            self.aday, self.aday_sayaci = None, 0
            return Kirpim(o.x1, o.y1, o.x2, o.y2)   # değişmedi

        # ZAMANSAL ONAY: yeni kutu ancak arka arkaya `onay_kare` kez tekrar
        # ederse kabul edilir. Ölçüldü (0831.mp4): bu şart olmadan karanlık
        # geçiş sahnelerinde kırpma 50 karede 18 kez zıplıyor ve her zıplama
        # MOG2 arka planını sıfırlatıp sahte alarm üretiyordu. Gerçek format
        # değişimi yüzlerce kare sürer, geçiş 1-3 kare.
        if self.aday is not None and max(
                abs(x1 - self.aday[0]), abs(y1 - self.aday[1]),
                abs(x2 - self.aday[2]), abs(y2 - self.aday[3])) <= OYNAMA_PAYI:
            self.aday_sayaci += 1
        else:
            self.aday, self.aday_sayaci = olculen, 1

        if self.aday_sayaci >= self.onay_kare:
            self.onceki = Kirpim(*self.aday, degisti=True)
            self.aday, self.aday_sayaci = None, 0
            return self.onceki
        return Kirpim(o.x1, o.y1, o.x2, o.y2)       # onay gelene kadar eski kutu
