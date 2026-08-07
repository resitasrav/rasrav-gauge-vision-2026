"""Eğik bakışta kadranı düzleştirir — K2 kararının uygulaması.

    from gauge_vision.detect.perspective import duzlestir

    sonuc = duzlestir(kare, merkez, yaricap)
    if sonuc:
        duz_kare, yeni_merkez, yeni_yaricap = sonuc.image, sonuc.center_px, sonuc.radius_px

**Neden gerekli (İP4 bulgusu, K2):** kadran açılı görüldüğünde daire ELİPSE
dönüşür. Çizgiler de elipsin üzerine oturduğu için açı→değer dönüşümü doğrudan
bozulur — ibre aynı değeri gösterirken görüntüdeki açısı kayar. Literatürde
düzeltme standart bir ön adımdır; İP14'e (zor koşullar) bırakmak yanlış olurdu,
çünkü sahada kameranın kadrana tam dik bakması İSTİSNADIR.

**Yöntem: kadran çemberine elips uydur, elipsi daireye geri götür.**

Bir daire perspektif altında elipse dönüşür. Elipsin ekseni oranı eğimi,
yönelimi ise eğim eksenini verir. Elipsi birim daireye götüren afin dönüşüm
kadranı düzleştirir.

**Neden tam homografi değil de afin:** tam homografi kadranın dört köşesini
gerektirir; dairesel bir kadranda "köşe" yoktur. Elipsten yalnızca 5 parametre
(merkez, iki eksen, açı) çıkar ve bunlar afin bir düzeltmeye yeter. Afin
düzeltme perspektifin *foreshortening* bileşenini tam kapatmaz — kadranın uzak
yarısı hâlâ biraz sıkışık kalır — ama ölçümde kalan hata küçük (bkz. 11.08
raporu). Tam homografi için kadranın üzerinde bilinen dört nokta gerekir;
gerçek göstergede bu, çizgi konumlarından çıkarılabilir ve ileri bir iş olarak
duruyor.

**Zarar vermeme:** düzeltme kabul kapılarından geçemezse `None` döner ve çağıran
ham kareyle devam eder. Yanlış bir düzleştirme, düzleştirmemekten daha kötüdür.

Açı konvansiyonu: bu dosya açı ÜRETMEZ. Yalnızca görüntüyü ve merkez/yarıçapı
dönüştürür; açı ölçümü düzleştirilmiş karede `read/needle.py` tarafından
yapılır. Böylece konvansiyon tek yerde kalır.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- Kenar seçimi (refine.py ile aynı mantık, oradaki gerekçeler geçerli) ---
CALISMA_PX = 320
ESIK_K = 1.5
HALKA_MIN = 0.80        # elips olduğu için halka refine.py'dakinden GENİŞ
HALKA_MAX = 1.30
COS_ESIK = 0.90         # elipste gradyan radyalden biraz sapar, tolerans gevşek
MIN_DESTEK = 60         # elips 5 parametreli; daireden çok nokta ister

# --- Kabul kapıları ---
# Eksen oranı 1'e çok yakınsa zaten dik bakılıyor demektir; düzeltmek gürültü
# ekler. 0,45'ten küçükse (≈63° eğim) kadranın uzak yarısı okunamaz durumdadır.
MIN_EKSEN_ORANI = 0.45
DUZELTME_ESIGI = 0.97   # bu oranın üstünde düzeltme yapılmaz — gereksiz
# Uydurulan elipsin merkezi kaba merkezden bu kadar uzaksa başka şeye oturmuş.
MAX_KAYMA_ORANI = 0.25
# Elips kenarlarının uyum kalitesi: noktaların elipse ortalama uzaklığı / yarıçap.
MAX_ARTIK_ORANI = 0.08


@dataclass(frozen=True)
class Duzlestirme:
    """Düzleştirilmiş kare ve yeni kadran geometrisi."""

    image: np.ndarray
    center_px: tuple[int, int]
    radius_px: float
    axis_ratio: float        # kısa/uzun eksen — 1,0 = dik bakış
    tilt_deg: float          # eksen oranından çıkarılan eğim tahmini
    residual_ratio: float    # elips uyum kalitesi
    matrix: np.ndarray       # uygulanan 2×3 afin dönüşüm


def _kenar_noktalari(image, center, radius):
    """Kadran kenarı olmaya aday piksel bulutu (radyal gradyanlı, halkada)."""
    h, w = image.shape[:2]
    yari = radius * 1.45
    x0, y0 = max(0, int(center[0] - yari)), max(0, int(center[1] - yari))
    x1, y1 = min(w, int(center[0] + yari) + 1), min(h, int(center[1] + yari) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None

    roi = image[y0:y1, x0:x1]
    gri = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    olcek = min(1.0, CALISMA_PX / max(gri.shape))
    if olcek < 1.0:
        gri = cv2.resize(gri, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)

    merkez_s = np.array([(center[0] - x0) * olcek, (center[1] - y0) * olcek])
    r_s = radius * olcek
    if r_s < 10:
        return None, None

    gri = cv2.GaussianBlur(gri, (3, 3), 0)
    gx = cv2.Scharr(gri, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gri, cv2.CV_32F, 0, 1)
    buyukluk = cv2.magnitude(gx, gy)

    ys, xs = np.nonzero(buyukluk > buyukluk.mean() + ESIK_K * buyukluk.std())
    if xs.size < MIN_DESTEK:
        return None, None

    dx, dy = xs - merkez_s[0], ys - merkez_s[1]
    uzaklik = np.hypot(dx, dy)
    halkada = (uzaklik > HALKA_MIN * r_s) & (uzaklik < HALKA_MAX * r_s) & (uzaklik > 1e-6)
    if np.count_nonzero(halkada) < MIN_DESTEK:
        return None, None

    xs, ys, dx, dy, uzaklik = (a[halkada] for a in (xs, ys, dx, dy, uzaklik))
    gvx, gvy = gx[ys, xs], gy[ys, xs]
    gnorm = np.hypot(gvx, gvy)
    gnorm[gnorm == 0] = 1.0
    radyal = np.abs((gvx * dx + gvy * dy) / (gnorm * uzaklik)) > COS_ESIK
    if np.count_nonzero(radyal) < MIN_DESTEK:
        return None, None

    # Tam çözünürlüğe geri taşı: elips parametreleri orijinal ölçekte olsun.
    noktalar = np.column_stack((xs[radyal] / olcek + x0, ys[radyal] / olcek + y0))
    return noktalar.astype(np.float32), r_s / olcek


def _elips_artigi(noktalar, elips) -> float:
    """Noktaların elipse ortalama bağıl uzaklığı — uyum kalitesi ölçüsü."""
    (ex, ey), (d1, d2), aci = elips
    a, b = d1 / 2.0, d2 / 2.0
    if a <= 0 or b <= 0:
        return float("inf")
    t = np.radians(aci)
    dx, dy = noktalar[:, 0] - ex, noktalar[:, 1] - ey
    # Elips kendi eksenlerine döndürülüp birim çembere normalize ediliyor;
    # noktanın yarıçapı 1'den ne kadar saparsa uyum o kadar kötü.
    u = (dx * np.cos(t) + dy * np.sin(t)) / a
    v = (-dx * np.sin(t) + dy * np.cos(t)) / b
    return float(np.mean(np.abs(np.hypot(u, v) - 1.0)))


def duzlestir(image: np.ndarray, center: tuple[int, int], radius: float,
              *, zorla: bool = False) -> Duzlestirme | None:
    """Kadranı elipsten daireye götürerek düzleştirir.

    `zorla=True` eksen oranı eşiğini yok sayar (ölçüm/ablasyon için).
    Güvenilir bir elips bulunamazsa None döner.
    """
    if radius <= 0:
        return None

    noktalar, _ = _kenar_noktalari(image, center, radius)
    if noktalar is None or len(noktalar) < 5:
        return None

    try:
        elips = cv2.fitEllipse(noktalar)
    except cv2.error:
        return None

    (ex, ey), (d1, d2), aci_deg = elips
    kisa, uzun = min(d1, d2), max(d1, d2)
    if uzun <= 0:
        return None
    oran = kisa / uzun

    if not (MIN_EKSEN_ORANI <= oran <= 1.0):
        return None
    if np.hypot(ex - center[0], ey - center[1]) > MAX_KAYMA_ORANI * radius:
        return None

    artik = _elips_artigi(noktalar, elips)
    if artik > MAX_ARTIK_ORANI:
        return None

    # Zaten dik bakılıyorsa dokunma: her warp bir kez daha enterpolasyon demek
    # ve ibrenin ince kenarını gereksiz yere yumuşatır.
    if oran > DUZELTME_ESIGI and not zorla:
        return None

    # `fitEllipse` açıyı UZUN değil ilk eksene göre verir; kısa ekseni bulmak
    # için hangisinin küçük olduğuna bakmak gerekiyor.
    kisa_eksen_aci = aci_deg if d1 <= d2 else aci_deg + 90.0

    # Düzeltme: kısa ekseni 1/oran kadar gererek elipsi daireye götür.
    # Sırayla: merkezi orijine al → kısa ekseni x'e döndür → x'i ger → geri
    # döndür → merkeze taşı. Tek 2×3 matriste birleştiriliyor.
    t = np.radians(kisa_eksen_aci)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    S = np.array([[1.0 / oran, 0.0], [0.0, 1.0]])
    A = R.T @ S @ R

    merkez = np.array([ex, ey])
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = A
    M[:, 2] = merkez - A @ merkez      # merkez sabit kalsın

    duz = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Düzleştirmeden sonra kadran daire; yarıçapı uzun eksenin yarısı.
    yeni_yaricap = uzun / 2.0
    # Kadran yüzü bezelin içinde: refine.py'daki oranla tutarlı kalınıyor.
    egim = float(np.degrees(np.arccos(np.clip(oran, 0.0, 1.0))))

    return Duzlestirme(image=duz,
                       center_px=(int(round(ex)), int(round(ey))),
                       radius_px=float(yeni_yaricap),
                       axis_ratio=float(oran),
                       tilt_deg=egim,
                       residual_ratio=float(artik),
                       matrix=M)
