"""demo/birlesik.py — ÜÇ MODÜL TEK KAREDE, BİRBİRİNDEN HABERDAR.

`run_demo.py` üç modülü yan yana ÜÇ AYRI panelde gösteriyordu: aynı kare üç kez
çiziliyor, üç modül birbirinden habersiz çalışıyordu. Bu dosya onun yerine
geçmiyor (o hâlâ "kim ne buluyor" karşılaştırması için iyi), ama sahadaki
gerçek soruyu bu cevaplıyor: **üç modül aynı kareye bakınca ortak sonuç ne?**

## Senkronizasyon

Tek yakalama döngüsü, tek kare, tek zaman damgası. Üç modül de AYNI
`Kare` nesnesini alıyor ve sonuçları tek `KareSonucu`'nda birleşiyor. Modüller
ayrı süreçlerde koşup sonradan eşleştirilmiyor — eşleştirme hatası diye bir
şey olamaz.

## "Birbirinden haberdar" ne demek — somut olarak

MOG2 (ANOMALİ) hareket eden HER ön planı işaretler. Fabrikada hareket eden iki
şey normaldir ve alarm olmamalıdır:

  1. **Yürüyen insan / geçen forklift.** ALGILAMA bunları zaten sınıflıyor.
     Bir anomali kutusu bir ALGILAMA kutusuyla örtüşüyorsa alarm değil,
     "beklenen hareket"tir.
  2. **Yanıp sönen ikaz lambası.** GÖSTERGE bunu `lamp` olarak tanıyor ve
     zaten okuyor. Kırmızı arıza lambası her karede yanıp söndüğü için MOG2'yi
     sürekli tetikler — ama o bir davetsiz nesne değil, OKUNAN bir göstergedir.

Geriye kalan, yani hiçbir modülün açıklayamadığı ön plan, gerçek alarmdır:
**TANIMSIZ NESNE** (zemindeki alet, düşmüş kutu, sıvı birikintisi).

Bu, üç modülü yan yana koymanın ötesinde bir kazanç ve ÖLÇÜLEBİLİR: aynı
videoda haberdar olmayan hâlde kaç alarm çıkıyor, haberdar hâlde kaç çıkıyor.

## Letterbox

Girdi videosu farklı en-boy oranlarındaki kliplerin birleşimi olabiliyor
(bkz. `kirp.py`). Kırpma ÜÇ MODÜLDEN ÖNCE yapılıyor ki üçü de aynı, tam
çözünürlüklü içeriği görsün. Kırpma geometrisi değiştiğinde MOG2'nin arka plan
modeli geçersizdir ve yeniden ısıtılır — yoksa kadraj kayması tüm kareyi
"hareket" diye işaretler.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import kirp

# Bir anomali kutusunun bir modül kutusuyla "aynı şey" sayılması için gereken
# örtüşme. IoU değil, ÖRTÜŞEN ALANIN ANOMALİ KUTUSUNA ORANI kullanılıyor:
# insan kutusu anomali kutusundan çok büyük olabilir (kişinin tamamı vs
# hareket eden kolu) ve IoU o durumda haksız yere düşük çıkar.
ORTUSME_ESIGI = 0.35

# ALGILAMA'nın "bu hareket normaldir" dediği COCO sınıfları. Liste dar tutuldu:
# `chair`, `bottle` gibi sınıflar zeminde DURAN nesnelerdir ve onları normal
# saymak tam da yakalamak istediğimiz anomaliyi susturur.
BEKLENEN_SINIFLAR = {"person", "forklift", "truck", "car", "bus", "bicycle",
                     "motorcycle", "train"}

RENK = {
    "gauge": (0, 165, 255), "digital": (255, 180, 0), "lamp": (0, 220, 220),
    "valve": (200, 120, 255), "keypad": (120, 255, 120),
}
RENK_ALGILAMA = (0, 255, 0)
RENK_ALARM = (0, 0, 235)
RENK_BEKLENEN = (150, 150, 150)

HUD_EN = 400
OLAY_SATIRI = 9


@dataclass
class Olay:
    kare: int
    saniye: float
    kaynak: str          # GOSTERGE | ALGILAMA | ANOMALI
    metin: str


@dataclass
class KareSonucu:
    kare: int
    saniye: float
    kirpim: tuple[int, int, int, int]
    gosterge: dict = field(default_factory=dict)
    algilama: dict = field(default_factory=dict)
    anomali: dict = field(default_factory=dict)
    olaylar: list[Olay] = field(default_factory=list)


def _ortusme_orani(a, b) -> float:
    """Kesişimin A'ya oranı (A = anomali kutusu)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    kx1, ky1 = max(ax1, bx1), max(ay1, by1)
    kx2, ky2 = min(ax2, bx2), min(ay2, by2)
    if kx2 <= kx1 or ky2 <= ky1:
        return 0.0
    kesisim = (kx2 - kx1) * (ky2 - ky1)
    alan_a = max((ax2 - ax1) * (ay2 - ay1), 1)
    return kesisim / alan_a


class BirlesikZincir:
    """Üç modülü tek karede koşturur ve sonuçlarını birbirine bağlar."""

    def __init__(self, gosterge_model, gauge, algilama_model, anomali_motoru,
                 conf: float = 0.25):
        self.gmodel = gosterge_model
        self.gauge = gauge
        self.amodel = algilama_model
        self.anomali = anomali_motoru
        self.conf = conf
        self.izleyici = kirp.KirpimIzleyici()
        self.anomali_isindi = False

    # --- modüller ---------------------------------------------------------

    def _gosterge(self, kare):
        from gauge_vision.pipeline import detect_objects, read_all_analog
        tespitler = detect_objects(kare, self.gmodel, conf=self.conf)
        okumalar = read_all_analog(kare, self.gmodel, tespitler=tespitler)
        sayim: dict[str, int] = {}
        for t in tespitler:
            sayim[t.sinif] = sayim.get(t.sinif, 0) + 1
        return tespitler, okumalar, sayim

    def _algilama(self, kare, sifirla: bool = False):
        # KADRAJ DEĞİŞTİYSE TAKİPÇİ SIFIRLANIR. Ultralytics'in takipçisi
        # (BoT-SORT) kareler arası küresel hareket telafisi için bir önceki
        # karenin optik akış piramidini saklıyor. Kırpma geometrisi değişince
        # kare boyutu da değişiyor ve piramitler uyuşmuyor:
        #   "GMC failed, falling back to identity ... prevPyr.size() == nextPyr.size()"
        # Ölçüldü: kırpma değişiminden sonra bu uyarı her karede tekrarlıyordu,
        # yani takip fiilen hareket telafisiz koşuyordu. persist=False o karede
        # takipçiyi yeniden kuruyor - iz kimlikleri baştan başlıyor, ki kadraj
        # değiştiğinde zaten doğrusu budur.
        sonuc = self.amodel.track(source=kare, conf=0.4,
                                  persist=not sifirla, verbose=False)
        kutular = []
        if sonuc and sonuc[0].boxes is not None and len(sonuc[0].boxes):
            b = sonuc[0].boxes
            for i in range(len(b)):
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[i].tolist())
                kutular.append({
                    "kutu": (x1, y1, x2, y2),
                    "sinif": self.amodel.names[int(b.cls[i])],
                    "conf": float(b.conf[i]),
                    "id": int(b.id[i]) if b.id is not None else -1,
                })
        return kutular

    # --- birleştirme ------------------------------------------------------

    def _acikla(self, anomali_nesneleri, algilama_kutulari, gosterge_tespitleri):
        """Her anomali nesnesini AÇIKLAMAYA çalışır; açıklanamayan alarmdır."""
        cikti = []
        for n in anomali_nesneleri:
            kutu = (n["x"], n["y"], n["x"] + n["w"], n["y"] + n["h"])
            aciklama, kaynak = None, None

            for a in algilama_kutulari:
                if a["sinif"] in BEKLENEN_SINIFLAR and \
                        _ortusme_orani(kutu, a["kutu"]) >= ORTUSME_ESIGI:
                    aciklama = f"{a['sinif']} #{a['id']}"
                    kaynak = "ALGILAMA"
                    break

            if aciklama is None:
                for t in gosterge_tespitleri:
                    # Yanıp sönen lamba/pano hareket üretir ama davetsiz nesne
                    # değildir - GÖSTERGE onu zaten okuyor.
                    if t.sinif in ("lamp", "keypad", "digital") and \
                            _ortusme_orani(kutu, t.box_xyxy) >= ORTUSME_ESIGI:
                        aciklama = f"{t.sinif} (okunuyor)"
                        kaynak = "GOSTERGE"
                        break

            cikti.append({**n, "kutu": kutu, "aciklama": aciklama,
                          "aciklayan": kaynak, "alarm": aciklama is None})
        return cikti

    # --- ana adım ---------------------------------------------------------

    def isle(self, ham_kare: np.ndarray, kare_no: int, saniye: float):
        kirpim = self.izleyici.olc(ham_kare)
        kare = np.ascontiguousarray(kirpim.uygula(ham_kare))

        # Kadraj kaydıysa arka plan modeli geçersiz - yeniden ısıtılır.
        if kirpim.degisti or not self.anomali_isindi:
            self.anomali.algilayici.warmup(kare)
            self.anomali.isindi = True
            self.anomali_isindi = True

        tespitler, okumalar, sinif_sayim = self._gosterge(kare)
        algilama = self._algilama(kare, sifirla=kirpim.degisti)
        ham = self.anomali.algilayici.isle(kare)
        nesneler = self._acikla(ham["nesneler"], algilama, tespitler)

        olaylar = []
        alarmlar = [n for n in nesneler if n["alarm"]]
        if alarmlar and not ham["is_rotation"]:
            olaylar.append(Olay(kare_no, saniye, "ANOMALI",
                                f"{len(alarmlar)} tanimsiz nesne"))
        for n in nesneler:
            if not n["alarm"]:
                olaylar.append(Olay(kare_no, saniye, n["aciklayan"],
                                    f"beklenen: {n['aciklama']}"))
        for s, adet in sinif_sayim.items():
            if s in ("lamp", "keypad", "digital", "gauge", "valve"):
                olaylar.append(Olay(kare_no, saniye, "GOSTERGE", f"{s} x{adet}"))

        sonuc = KareSonucu(
            kare=kare_no, saniye=saniye, kirpim=kirpim.kutu,
            gosterge={"tespit": sinif_sayim,
                      "analog_kutu": len(okumalar),
                      "analog_okunan": sum(1 for o in okumalar if o.ok)},
            algilama={"kutu": len(algilama),
                      "siniflar": sorted({a["sinif"] for a in algilama})},
            anomali={"nesne": len(nesneler),
                     "alarm": len(alarmlar),
                     "aciklanan": len(nesneler) - len(alarmlar),
                     "donme": bool(ham["is_rotation"]),
                     "fg_orani": ham["fg_ratio"]},
            olaylar=olaylar)
        return kare, tespitler, okumalar, algilama, nesneler, ham, sonuc
