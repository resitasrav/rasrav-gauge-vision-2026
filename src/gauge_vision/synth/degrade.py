"""Sentetik kadranı saha koşullarına yaklaştırır — İP14'ün zemini.

    from gauge_vision.synth.degrade import bozulmalar_uygula, Bozulma

    kare, yeni_truth = bozulmalar_uygula(kare, truth, Bozulma(egiklik_deg=25))

İP3'ün ürettiği kadranlar bilinçli olarak temizdi: hata çıktığında "yöntem mi
kötü, görüntü mü zor" ayırt edilebilsin diye. Yöntem oturduğuna göre (zincir
%0,19) sıra zorluğu eklemeye geldi.

**Her bozulma tek başına açılabilir ve seviyesi verilebilir.** Karışık bir "zor
görüntü" kümesinde hangi etkenin bozduğu ayrılamaz; koşul bazlı hata tablosu
(İP14'ün bitti kriteri) ancak eksenler ayrıyken üretilebilir.

**Ground truth bozulmayla birlikte taşınır — en kritik nokta budur.**
Perspektif uygulandığında kadranın merkezi ve ibrenin ucu görüntüde başka yere
düşer; `angle_img_deg` artık `angle_deg + roll_deg` DEĞİLDİR. Değer (`value`)
ve kadran çerçevesindeki açı (`angle_deg`) değişmez — okunması gereken şey
odur. Yeni `angle_img_deg` dönüştürülmüş noktalardan yeniden hesaplanır ve
"naif bir okuyucunun göreceği açı"yı temsil eder. Aradaki fark, perspektif
düzeltmesinin kapatması gereken hatadır.

Gerçek fotoğrafın yerini tutmaz. Bunlar hâlâ bizim çizdiğimiz kadranlardır;
cam yansıması, metal doku, tozlanma ve gerçek sanayi aydınlatması yoktur.
Sentetik ile gerçek arasında bir basamaktır — İP8'in gerçek görüntü ihtiyacını
ortadan kaldırmaz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from gauge_vision.synth.dial import DialTruth

# --- Perspektif ---
# Kamera uzaklığı, görüntü kenarının katı olarak. Küçük değer daha güçlü
# perspektif (geniş açı) demek; 2.2 pan-tilt platformunun tipik bakışına yakın.
KAMERA_UZAKLIK_ORANI = 2.2

# --- Parlama ---
PARLAMA_YARICAP_ORANI = 0.55    # kadran yarıçapına oran
PARLAMA_MERKEZ_KACIKLIK = 0.45  # parlama kadranın ortasına değil kenarına vurur

# --- Düşük ışık ---
# Kazanç düşünce sensör gürültüsü baskın hale gelir; ikisi birlikte modellenir.
DUSUK_ISIK_GURULTU = 6.0


@dataclass(frozen=True)
class Bozulma:
    """Tek bir karenin zorluk ayarı. Hepsi 0/kapalı iken görüntü değişmez.

    Seviyeler bilinçli olarak FİZİKSEL birimlerde: "seviye 3" demek yerine
    "25 derece eğik" demek, ölçüm tablosunun sahada karşılığı olmasını sağlar.
    """

    egiklik_deg: float = 0.0        # kadran düzleminin kameraya göre eğimi
    egiklik_yon_deg: float = 0.0    # eğimin ekseni (0 = yatay eksen etrafında)
    parlama: float = 0.0            # 0-1, cam yansıması şiddeti
    isik_kazanci: float = 1.0       # 1 = normal, 0.3 = karanlık
    bulaniklik_px: int = 0          # hareket bulanıklığı çekirdeği
    bulaniklik_aci: float = 30.0
    jpeg_kalite: int = 0            # 0 = sıkıştırma yok

    @property
    def etkin(self) -> bool:
        return (self.egiklik_deg != 0 or self.parlama > 0 or self.isik_kazanci != 1.0
                or self.bulaniklik_px >= 3 or self.jpeg_kalite > 0)


def perspektif_matrisi(shape, egiklik_deg: float, yon_deg: float = 0.0) -> np.ndarray:
    """Kadran düzlemini 3B'de eğip yeniden yansıtan homografi.

    Görüntü düzlemindeki noktalar z=0'a yerleştirilir, `yon_deg` açısındaki bir
    eksen etrafında `egiklik_deg` kadar döndürülür ve pinhole kamerayla yeniden
    yansıtılır. Sonuç: daire elipse dönüşür, uzak kenar küçülür.

    Basit bir afin kaydırma (shear) YETMEZ: afin dönüşümde daire yine elips olur
    ama uzak kenarın küçülmesi (foreshortening) modellenmez ve kadranın merkezi
    elipsin merkezinde kalır. Gerçek eğik bakışta ikisi ayrışır; ibre açısının
    kayması da tam buradan doğar.
    """
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0
    f = max(h, w) * KAMERA_UZAKLIK_ORANI

    t = math.radians(egiklik_deg)
    a = math.radians(yon_deg)
    # Dönme ekseni: görüntü düzleminde `yon_deg` yönünde birim vektör.
    ex, ey, ez = math.cos(a), math.sin(a), 0.0
    c, s = math.cos(t), math.sin(t)
    # Rodrigues
    R = np.array([
        [c + ex * ex * (1 - c),      ex * ey * (1 - c) - ez * s, ex * ez * (1 - c) + ey * s],
        [ey * ex * (1 - c) + ez * s, c + ey * ey * (1 - c),      ey * ez * (1 - c) - ex * s],
        [ez * ex * (1 - c) - ey * s, ez * ey * (1 - c) + ex * s, c + ez * ez * (1 - c)],
    ])

    kaynak = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    hedef = []
    for (x, y) in kaynak:
        p = R @ np.array([x - cx, y - cy, 0.0])
        z = p[2] + f
        hedef.append([f * p[0] / z + cx, f * p[1] / z + cy])
    return cv2.getPerspectiveTransform(kaynak, np.float32(hedef))


def _nokta_donustur(M: np.ndarray, nokta) -> tuple[float, float]:
    x, y = float(nokta[0]), float(nokta[1])
    v = M @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def _parlama_ekle(img: np.ndarray, merkez, yaricap: float, siddet: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Cam yansıması: eliptik, yumuşak kenarlı, doyuma giden parlak leke.

    Toplamalı uygulanıyor (çarpımsal değil): gerçek yansıma sahnenin üstüne
    ışık EKLER, mevcut kontrastı ölçeklemez. Çarpımsal modelde koyu ibre koyu
    kalır ve parlama okumayı hiç zorlaştırmaz — asıl zorluk ibrenin beyaza
    boğulmasıdır.
    """
    h, w = img.shape[:2]
    r = yaricap * PARLAMA_YARICAP_ORANI
    aci = rng.uniform(0, 2 * math.pi)
    kx = merkez[0] + PARLAMA_MERKEZ_KACIKLIK * yaricap * math.cos(aci)
    ky = merkez[1] + PARLAMA_MERKEZ_KACIKLIK * yaricap * math.sin(aci)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Eliptik: yansımalar nadiren dairesel olur.
    ax = r * float(rng.uniform(0.7, 1.4))
    ay = r * float(rng.uniform(0.5, 1.1))
    d2 = ((xx - kx) / ax) ** 2 + ((yy - ky) / ay) ** 2
    maske = np.exp(-0.5 * d2) * (255.0 * siddet)

    return np.clip(img.astype(np.float32) + maske[..., None], 0, 255).astype(np.uint8)


def _dusuk_isik(img: np.ndarray, kazanc: float, rng: np.random.Generator) -> np.ndarray:
    """Işık azalınca sinyal düşer, sensör gürültüsü sabit kalır → SNR düşer."""
    karanlik = img.astype(np.float32) * kazanc
    gurultu = rng.normal(0.0, DUSUK_ISIK_GURULTU, img.shape).astype(np.float32)
    return np.clip(karanlik + gurultu, 0, 255).astype(np.uint8)


def _hareket_bulanikligi(img: np.ndarray, boyut: int, aci: float) -> np.ndarray:
    cekirdek = np.zeros((boyut, boyut), np.float32)
    cekirdek[boyut // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((boyut / 2 - 0.5, boyut / 2 - 0.5), aci, 1.0)
    cekirdek = cv2.warpAffine(cekirdek, M, (boyut, boyut))
    toplam = cekirdek.sum()
    return img if toplam <= 0 else cv2.filter2D(img, -1, cekirdek / toplam)


def _jpeg(img: np.ndarray, kalite: int) -> np.ndarray:
    ok, tampon = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(kalite)])
    return cv2.imdecode(tampon, cv2.IMREAD_COLOR) if ok else img


def bozulmalar_uygula(image: np.ndarray, truth: DialTruth, bozulma: Bozulma,
                      rng: np.random.Generator | None = None
                      ) -> tuple[np.ndarray, DialTruth]:
    """Bozulmaları sırayla uygular ve ground truth'u birlikte taşır.

    Sıra fiziksel: önce geometri (kamera nereden bakıyor), sonra ışık ve
    yansıma (sahnede ne oluyor), sonra optik bulanıklık, en sonda sıkıştırma
    (yayın hattında ne oluyor). Ters sırada uygulanırsa örneğin JPEG artefaktı
    perspektifle birlikte esner ve gerçekte olmayan bir bozulma üretilir.
    """
    rng = rng or np.random.default_rng(0)
    img = image
    t = truth

    if bozulma.egiklik_deg:
        M = perspektif_matrisi(img.shape, bozulma.egiklik_deg, bozulma.egiklik_yon_deg)
        img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        merkez = _nokta_donustur(M, t.center_px)
        uc = _nokta_donustur(M, t.tip_px)

        # Yarıçap artık yöne bağlı (daire elips oldu). Kutunun yarı kenarını
        # temsilci yarıçap alıyoruz: tespit de kutudan yarıçap türetiyor,
        # dolayısıyla ölçüm zincirin gördüğü büyüklükle uyumlu kalıyor.
        koseler = [_nokta_donustur(M, (t.bbox_xyxy[i], t.bbox_xyxy[j]))
                   for i, j in ((0, 1), (2, 1), (2, 3), (0, 3))]
        xs = [p[0] for p in koseler]
        ys = [p[1] for p in koseler]
        kutu = (min(xs), min(ys), max(xs), max(ys))
        yaricap = min(kutu[2] - kutu[0], kutu[3] - kutu[1]) / 2.0

        # Naif okuyucunun göreceği açı. `angle_deg` (kadran çerçevesi) DEĞİŞMEZ:
        # okunması gereken değer odur ve perspektif onu değiştirmez.
        aci_img = math.degrees(math.atan2(-(uc[1] - merkez[1]), uc[0] - merkez[0]))

        t = replace(t,
                    center_px=(round(merkez[0]), round(merkez[1])),
                    tip_px=(round(uc[0]), round(uc[1])),
                    radius_px=int(round(yaricap)),
                    bbox_xyxy=tuple(int(round(v)) for v in kutu),
                    angle_img_deg=float(aci_img))

    if bozulma.isik_kazanci != 1.0:
        img = _dusuk_isik(img, bozulma.isik_kazanci, rng)

    if bozulma.parlama > 0:
        img = _parlama_ekle(img, t.center_px, float(t.radius_px), bozulma.parlama, rng)

    if bozulma.bulaniklik_px >= 3:
        img = _hareket_bulanikligi(img, bozulma.bulaniklik_px, bozulma.bulaniklik_aci)

    if bozulma.jpeg_kalite > 0:
        img = _jpeg(img, bozulma.jpeg_kalite)

    return img, t


# Ölçüm eksenleri — İP14'ün koşul bazlı hata tablosu bunlardan üretilir.
# Değerler fiziksel: "seviye 3" değil "25 derece eğik".
EKSENLER: dict[str, list[tuple[str, Bozulma]]] = {
    "egiklik": [(f"{d}°", Bozulma(egiklik_deg=d)) for d in (0, 10, 20, 30, 40, 50)],
    "parlama": [(f"%{int(p*100)}", Bozulma(parlama=p)) for p in (0.0, 0.3, 0.5, 0.7, 0.9)],
    "dusuk_isik": [(f"×{g:.2f}", Bozulma(isik_kazanci=g)) for g in (1.0, 0.6, 0.4, 0.25, 0.15)],
    "bulaniklik": [(f"{b}px", Bozulma(bulaniklik_px=b)) for b in (0, 5, 9, 15, 21)],
    "jpeg": [(f"q{q}", Bozulma(jpeg_kalite=q)) for q in (0, 60, 40, 25, 15)],
}
