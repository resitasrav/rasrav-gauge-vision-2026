"""İbrenin GÖRÜNTÜDEKİ açısını ölçer — analog okuma zincirinin ilk adımı (İP6).

    from gauge_vision.read.needle import read_needle_angle

    okuma = read_needle_angle(img, center=(256, 256), radius=200)
    okuma.angle_img_deg   # 90.2  → İP7 bunu değere çevirecek
    okuma.confidence      # 0.94  → İP15 eşiğin altını `unreadable` basacak

**Hangi açı ölçülüyor:** `angle_img_deg` — kadranın kendi çerçevesindeki açı
(`angle_deg`) değil. Kamera yatıksa ikisi `roll_deg` kadar farklıdır. Etiketteki
karşılığı da `angle_img_deg`; karşılaştırma o alanla yapılır. İkisinin
karıştırılması yatık her karede okumayı sessizce kaydırır (bkz. 30.07 raporu).

**Merkez ve yarıçap dışarıdan verilir.** Bu iş paketinde ölçülen şey açı
yöntemidir; kadranın karede nerede olduğunu İP5 (YOLO tespiti) bulacaktır.
Ölçüm sırasında merkez/yarıçap ground truth'tan gelir ki hata yalnızca açı
yöntemine ait olsun. Merkez hatasının açıya etkisi ayrıca ölçülür
(`scripts/olc_ip6.py --merkez-hatasi`).

**İki yöntem, tek arayüz (K3 kararı):**

| method     | Nasıl çalışır | Güçlü yanı |
|------------|---------------|------------|
| `"polar"`  | Merkezden ışın gönderip koyu pikselin kesintisiz uzandığı yönü arar | Kadran çizgileri ve sayılar ışına takılmaz — ibre merkezden başlayan tek uzun koyu şerittir |
| `"hough"`  | Canny + olasılıksal Hough ile çizgi parçaları, merkezden geçenler süzülür | Literatürdeki klasik referans; kıyas için gerekli |

Ground truth bedava olduğu için iki yöntem sentetik veride yan yana ölçülür;
kazanan İP7'ye girdi olur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

# --- Tarama halkası (hepsi kadran yarıçapına oran) -----------------------------
# İbre merkezden `NEEDLE_LEN_RATIO`=0.78'e uzanır, kuyruğu 0.16'da biter, göbek
# 0.07'dir. Taramaya 0.22'den başlamak kuyruğu ve göbeği dışarıda bırakır: ibrenin
# 180° tersi yön kendiliğinden elenir, ayrıca "hangi uç?" sorusu hiç doğmaz.
SCAN_R_MIN_RATIO = 0.22
# 0.72'de bitmek ana çizgileri (0.86'dan başlar) tarama dışında tutar.
SCAN_R_MAX_RATIO = 0.72
SCAN_SAMPLES = 64          # ışın başına örnek — 0.5·r'de ~1,5 piksel adım
SCAN_STEP_DEG = 0.5        # açı çözünürlüğü; plato ortalaması bunun altına iner

# Platoyu oluşturan açılar: en uzun kesintisiz şeridin bu oranı kadarını yakalayanlar.
PLATEAU_RATIO = 0.90

# --- DENENDİ ve ELENDİ: ibre ucu / kuyruk ayrımı dış uzanımla (27.08) ----------
# `SCAN_R_MIN_RATIO` yorumundaki "kuyruk 0,16R'de biter, 180° tersi kendiliğinden
# elenir" ifadesi **kendi üretecimizin** geometrisidir, evrensel değildir.
# Bağımsız sette (`Synanthropic/reading-analog-gauge`, gerçek endüstriyel zemin)
# kadranların belirgin bir karşı-ağırlık kuyruğu var. Ölçülen: 400 karede merkez
# ETİKETTEN gelirken %1,0, merkez `refine_dial` ile kestirilirken **%8,1 kare
# 180° ters** okunuyor.
#
# İki deneme yapıldı, ikisi de ölçümle elendi — aynı yolu üçüncü kez deneme:
#
# 1. "Şerit 0,72R'yi aşıyor mu" (mutlak sonda) → İP8 ekran düzeneğinde okumanın
#    TAMAMINI reddetti (kapsama 10/10 → 0/10). Orada kutu kadranın çevresinde
#    pay bıraktığı için yarıçap fazla kestiriliyor ve gerçek ibre 0,72R'ye hiç
#    ulaşmıyor. Kapı ibrenin kısalığını değil YARIÇAP KESTİRİMİNİN hatasını
#    ölçüyordu.
# 2. "İki yönün uzanımı birbirine göre" (göreli, ölçekten bağımsız) → ayırt
#    etmedi: 749 karede fark medyanı doğru okumalarda 0,52, ters okumalarda
#    0,48; dağılımlar örtüşüyor. Eşik taraması 0,05'te 73 tersin 0'ını yakalıyor.
#
# **Kök sebep ibre mantığında değil MERKEZ DOĞRULUĞUNDA.** İP6'nın kendi
# duyarlılık tablosu bunu zaten söylüyor: merkez kadran ÇAPININ %2'si kadar
# kayınca ortalama hata 8,1°, **maksimum 178,8°** oluyor. `refine_dial` bu sette
# ortalama %1,8 (yarıçapın) sapıyor ama p95'i %4,0 — yani dağılımın kuyruğu tam
# o kritik bölgeye giriyor ve orada okuma sessizce ters dönüyor. Uzanım ölçütü
# de merkezden ölçüldüğü için aynı hatadan etkileniyor; hastalığın kendisiyle
# teşhis konmuyor.
#
# **Refine'in kendi kanıt sayıları da bu hatayı ÖNGÖRMÜYOR** (1007 kare,
# 27.08 · elenen ters / feda edilen doğru):
#   artık > 0,030   → %6 / %4       ·  yayılma > 0,030 → %12 / %16
#   destek/r < 2,0  → %27 / %6      ·  destek/r < 4,0  → %38 / %14
# `destek/r` en iyi ayıran, o bile terslerin dörtte birini yakalayıp doğruların
# %6'sını feda ediyor. Bu oranla bir kapı koymak sorunu çözmez, sadece kapsamı
# düşürüp bakılacak bir eşik daha ekler — bu yüzden KONULMADI.
#
# Sıradaki denenecek yol ZAMANSAL tutarlılıktır: zincir videoda çalışıyor ve
# 180°'lik bir sıçrama ardışık karelerde fiziksel olarak imkânsızdır.
# `gauge_vision/temporal.py` bu iş için zaten var; tek karede ayrılamayan şey
# kare dizisinde ayrılabilir.
# Plato bu orandan geniş bir açı yayına yayılıyorsa görüntü koyu/bozuk demektir.
SPREAD_MAX = 0.15          # tüm dairenin oranı olarak (0.15 → 54°)

# --- Hough ---------------------------------------------------------------------
HOUGH_MASK_R_RATIO = 0.82  # çizgiler ve bezel Canny'ye hiç girmesin
HOUGH_MIN_LEN_RATIO = 0.30
HOUGH_MAX_GAP_RATIO = 0.05
HOUGH_THRESHOLD_RATIO = 0.25
HOUGH_HUB_TOL_RATIO = 0.40  # parçanın yakın ucu merkeze bu kadar yakın olmalı
HOUGH_RADIAL_TOL_DEG = 12.0  # parça yönü ile uç noktanın açısı bu kadar tutmalı
CANNY_LO, CANNY_HI = 50, 150

METHODS = ("polar", "hough")


@dataclass(frozen=True)
class NeedleReading:
    """Tek bir karede ölçülen ibre açısı.

    `confidence` 0-1 arasıdır ve İP15'in `unreadable` eşiğine girdi olacaktır.
    Sayı uydurmamak için: yöntem hiçbir aday bulamazsa `read_needle_angle`
    None döner — düşük güvenli bir tahmin bile üretmez.
    """

    angle_img_deg: float                 # 0° = saat 3 yönü, CCW pozitif
    confidence: float
    method: str
    tip_px: tuple[int, int]              # görselleştirme/hata ayıklama için


def angle_difference_deg(measured: float, truth: float) -> float:
    """İki açının farkı, (-180, 180] aralığına sarılmış hâlde.

    Çıplak çıkarma 359° ile 1° arasını 358° gösterir; kadran başında ölçülen
    hata bu yüzden devasa görünür. Hata tablosu bu fonksiyonla üretilir.
    """
    return (measured - truth + 180.0) % 360.0 - 180.0


def read_needle_angle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    *,
    method: str = "polar",
) -> NeedleReading | None:
    """`center`/`radius` ile verilen kadranda ibrenin görüntüdeki açısını ölçer.

    Aday bulunamazsa None döner (yanlış okumaktansa okumamak — 3. kural).
    """
    if method not in METHODS:
        raise ValueError(f"bilinmeyen yöntem '{method}' — {METHODS}")
    if radius <= 0:
        raise ValueError(f"yarıçap pozitif olmalı, {radius} verildi")

    # Önce ROI, sonra iş: maliyet KADRANIN boyutuna bağlı olmalı, karenin
    # boyutuna değil. Kırpma olmadan `_polarity_mask` bütün kareyi Otsu'dan
    # geçirip iki kez eşikliyor ve `cvtColor` da tüm kareye iniyordu — ölçülen
    # (27.08, tek çekirdek, aynı kadran):
    #     360 px kesit → 3,26 ms · 1080p tam kare → 18,13 ms · 1440p → 30,88 ms
    # Gömülü hedefte (RK3588) bu fark doğrudan devriye bütçesinden yeniyor.
    # `refine_dial` bu kırpmayı zaten yapıyordu; okuyucu yapmıyordu.
    #
    # Sonuç DEĞİŞMEZ: kırpım kadranın tamamını içeriyor, açı merkeze göreli
    # ölçülüyor ve `tip_px` çıkışta tam kare koordinatına geri taşınıyor.
    # İP6'nın 0,123° sayısı bu değişiklikten sonra birebir aynı kaldı.
    kirpim, ofset = _roi_kirp(image, center, float(radius))
    gray = kirpim if kirpim.ndim == 2 else cv2.cvtColor(kirpim, cv2.COLOR_BGR2GRAY)
    yerel_merkez = (center[0] - ofset[0], center[1] - ofset[1])

    okuma = (_read_polar if method == "polar" else _read_hough)(
        gray, yerel_merkez, float(radius))
    if okuma is None or ofset == (0, 0):
        return okuma
    return replace(okuma, tip_px=(okuma.tip_px[0] + ofset[0],
                                  okuma.tip_px[1] + ofset[1]))


# Kırpım yarıçapın bu katına kadar alınır. 1,0'dan büyük olmalı: `_polarity_mask`
# kadran yüzünü 0,95R'ye kadar örnekliyor ve Otsu eşiği o bölgeden geliyor —
# daha dar bir kırpım eşiği değiştirir, yani SONUCU değiştirir.
ROI_ORANI = 1.15


def _roi_kirp(image: np.ndarray, center: tuple[int, int],
              radius: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Kadranın çevresini kırpar. `(kırpım, (dx, dy))` döner.

    Kırpım kare sınırlarına taşarsa daraltılır; merkez yine kırpımın içinde
    kalır çünkü kaydırma da birlikte taşınıyor. Kırpmaya gerek yoksa (kadran
    zaten kareyi dolduruyorsa) görüntü olduğu gibi ve ofset (0, 0) döner —
    gereksiz bir kopya çıkarmamak için.
    """
    h, w = image.shape[:2]
    yari = int(radius * ROI_ORANI) + 1
    x0 = max(0, center[0] - yari)
    y0 = max(0, center[1] - yari)
    x1 = min(w, center[0] + yari)
    y1 = min(h, center[1] + yari)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return image, (0, 0)
    if x0 == 0 and y0 == 0 and x1 == w and y1 == h:
        return image, (0, 0)
    return image[y0:y1, x0:x1], (x0, y0)


# ------------------------------------------------------------ kutupsal tarama --

def _read_polar(gray: np.ndarray, center: tuple[int, int], radius: float) -> NeedleReading | None:
    """Merkezden çıkan ışınlar boyunca kesintisiz koyu VEYA açık şeridi arar.

    Kadran çizgileri ve sayı etiketleri de zeminden farklı renktedir; ayırt
    edici özellik onların merkezden BAŞLAMAMASIDIR. Bu yüzden ölçüt "kaç farklı
    piksel var" değil, "merkezin hemen dışından itibaren kesintisiz ne kadar
    uzanıyor" olarak seçildi. Sayı etiketi ışının ortasında bir kopukluğun
    ardında kalır ve şeridi uzatmaz.

    İbre zeminden KOYU olabilir (beyaz kadranda siyah/kırmızı ibre — envanterin
    varsaydığı durum) ya da AÇIK olabilir (koyu kadranda beyaz ibre — örn.
    araç göstergeleri). Hangisi olduğu envanterde beyan edilmiyor, biri
    varsayılırsa öteki türdeki hiçbir gösterge okunamaz. Bu yüzden iki kutbu da
    dene, ibre imzasına (dar plato = yüksek güven) hangisi daha çok uyuyorsa
    onu kullan — "cevap makul mü" değil "kanıt hangisinde güçlü" sorusu.
    """
    en_iyi: NeedleReading | None = None
    for koyu in (True, False):
        maske = _polarity_mask(gray, center, radius, koyu)
        if maske is None:
            continue
        aday = _polar_from_mask(maske, center, radius)
        if aday is not None and (en_iyi is None or aday.confidence > en_iyi.confidence):
            en_iyi = aday
    return en_iyi


def _polar_from_mask(maske: np.ndarray, center: tuple[int, int], radius: float) -> NeedleReading | None:
    runs = _radial_runs(maske, center, radius)
    if runs.max() <= 0:
        return None

    esik = runs.max() * PLATEAU_RATIO
    plato = runs >= esik

    yaylar = _contiguous_arcs(plato)
    if not yaylar:
        return None

    # En uzun yay ibre; birden fazla yay varsa görüntüde ibreye benzeyen başka
    # bir şey daha var demektir — açı yine üretilir ama güven bölünür.
    en_uzun = max(yaylar, key=len)
    aci = _circular_mean_deg([i * SCAN_STEP_DEG for i in en_uzun],
                             [runs[i] for i in en_uzun])

    doluluk = runs.max() / (SCAN_SAMPLES - 1)
    yayilma = sum(len(y) for y in yaylar) / len(runs)
    guven = doluluk * max(0.0, 1.0 - yayilma / SPREAD_MAX) / len(yaylar)

    return NeedleReading(
        angle_img_deg=aci,
        confidence=float(np.clip(guven, 0.0, 1.0)),
        method="polar",
        tip_px=_polar_px(center, radius * SCAN_R_MAX_RATIO, aci),
    )


def _polarity_mask(gray: np.ndarray, center: tuple[int, int], radius: float,
                    koyu: bool) -> np.ndarray | None:
    """Kadran yüzünü Otsu ile ikiye ayırır: ibre adayı ve zemin.

    Eşik sabit değil çünkü ibre siyah da olabilir kırmızı da (envanterdeki
    `needle_color`); kırmızının grisi ~91, siyahınki ~20, beyaz yüz 255.
    Otsu'yu bütün kareye değil YALNIZCA kadran yüzüne uygulamak şart: kare
    zemini (gri 140-225) ve bezel (70) eşiği kadranın dışına kaçırır.

    `koyu=True` ibrenin zeminden koyu olduğunu varsayar (`gray < esik`),
    `koyu=False` açık olduğunu (`gray > esik`) — `_read_polar` ikisini de
    dener.
    """
    h, w = gray.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    yuz = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= (radius * 0.95) ** 2
    pikseller = gray[yuz]
    if pikseller.size < 16:
        return None   # kadran karenin dışında kalmış — okuma denenmez

    esik, _ = cv2.threshold(pikseller.reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray < esik if koyu else gray > esik


def _radial_runs(dark: np.ndarray, center: tuple[int, int], radius: float) -> np.ndarray:
    """Her açı için: tarama halkasının başından itibaren kesintisiz koyu örnek sayısı."""
    acilar = np.arange(0.0, 360.0, SCAN_STEP_DEG)
    yaricaplar = np.linspace(radius * SCAN_R_MIN_RATIO, radius * SCAN_R_MAX_RATIO,
                             SCAN_SAMPLES)

    rad = np.radians(acilar)[:, None]
    xs = np.rint(center[0] + yaricaplar[None, :] * np.cos(rad)).astype(int)
    ys = np.rint(center[1] - yaricaplar[None, :] * np.sin(rad)).astype(int)

    h, w = dark.shape
    icerde = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    koyu = np.zeros_like(icerde)
    koyu[icerde] = dark[ys[icerde], xs[icerde]]

    # İlk koyu-olmayan örneğin indisi = kesintisiz şeridin uzunluğu.
    acik = ~koyu
    return np.where(acik.any(axis=1), acik.argmax(axis=1), SCAN_SAMPLES - 1)


def _contiguous_arcs(mask: np.ndarray) -> list[list[int]]:
    """Dairesel maskedeki bitişik indis öbekleri (0 ile son indis komşudur)."""
    (indisler,) = np.nonzero(mask)
    if indisler.size == 0:
        return []

    yaylar: list[list[int]] = [[int(indisler[0])]]
    for i in indisler[1:]:
        if i == yaylar[-1][-1] + 1:
            yaylar[-1].append(int(i))
        else:
            yaylar.append([int(i)])

    # 0° ile 360°'nin iki yana düşen parçaları tek ibredir, birleştirilir.
    if len(yaylar) > 1 and yaylar[0][0] == 0 and yaylar[-1][-1] == len(mask) - 1:
        yaylar[0] = yaylar[-1] + yaylar[0]
        yaylar.pop()
    return yaylar


def _circular_mean_deg(acilar: list[float], agirliklar: list[float]) -> float:
    """Açıların ağırlıklı dairesel ortalaması.

    Düz aritmetik ortalama 359° ile 1°'nin ortasını 180° bulur; ibre kadranın
    başındayken okuma tam ters yöne düşerdi.
    """
    rad = np.radians(acilar)
    w = np.asarray(agirliklar, dtype=float)
    return float(np.degrees(np.arctan2((w * np.sin(rad)).sum(), (w * np.cos(rad)).sum())))


# ----------------------------------------------------------------------- Hough --

def _read_hough(gray: np.ndarray, center: tuple[int, int], radius: float) -> NeedleReading | None:
    """Canny + olasılıksal Hough; merkezden çıkan en uzun radyal parçayı seçer.

    Ana çizgiler de radyaldir ve uzatıldıklarında merkezden geçerler — "çizgi
    merkezden geçiyor mu" süzgeci onları elemez. Ayırt edici olan parçanın YAKIN
    ucunun merkeze uzaklığıdır: ibrenin yakın ucu göbekte, çizginin yakın ucu
    kadranın kenarındadır. Yine de çizgiler daha maskeleme aşamasında dışarıda
    bırakılıyor; süzgeç ikinci savunma hattı.
    """
    maske = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.circle(maske, center, int(radius * HOUGH_MASK_R_RATIO), 255, -1)
    yuz = cv2.bitwise_and(gray, gray, mask=maske)

    kenarlar = cv2.Canny(yuz, CANNY_LO, CANNY_HI)
    parcalar = cv2.HoughLinesP(
        kenarlar, rho=1, theta=np.pi / 360,
        threshold=max(10, int(radius * HOUGH_THRESHOLD_RATIO)),
        minLineLength=radius * HOUGH_MIN_LEN_RATIO,
        maxLineGap=radius * HOUGH_MAX_GAP_RATIO,
    )
    if parcalar is None:
        return None

    acilar: list[float] = []
    agirliklar: list[float] = []
    # OpenCV sürümüne göre (N,1,4) veya (1,N,4) dönüyor; şekli sabitliyoruz.
    for x1, y1, x2, y2 in parcalar.reshape(-1, 4):
        p_yakin, p_uzak = sorted(((x1, y1), (x2, y2)), key=lambda p: _uzaklik(p, center))
        if _uzaklik(p_yakin, center) > radius * HOUGH_HUB_TOL_RATIO:
            continue   # yakın ucu göbekte değil → kadran çizgisi veya yazı

        uc_acisi = _aci_deg(center, p_uzak)
        parca_acisi = _aci_deg(p_yakin, p_uzak)
        if abs(angle_difference_deg(parca_acisi, uc_acisi)) > HOUGH_RADIAL_TOL_DEG:
            continue   # merkezden dışa doğru uzanmıyor → ibre değil

        acilar.append(uc_acisi)
        agirliklar.append(_uzaklik(p_yakin, p_uzak))

    if not acilar:
        return None

    aci = _circular_mean_deg(acilar, agirliklar)

    # Güven: kabul edilen parçaların ne kadar hemfikir olduğu (dairesel bileşke
    # uzunluğu). Kalın ibre iki kenarını da verir; ikisi aynı yönü gösterdiğinde
    # bileşke 1'e yaklaşır, dağılırlarsa düşer.
    rad = np.radians(acilar)
    w = np.asarray(agirliklar, dtype=float)
    bileske = math.hypot((w * np.cos(rad)).sum(), (w * np.sin(rad)).sum()) / w.sum()

    return NeedleReading(
        angle_img_deg=aci,
        confidence=float(np.clip(bileske, 0.0, 1.0)),
        method="hough",
        tip_px=_polar_px(center, radius * SCAN_R_MAX_RATIO, aci),
    )


# ------------------------------------------------------------------ yardımcılar --

def _uzaklik(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _aci_deg(kaynak: tuple[float, float], hedef: tuple[float, float]) -> float:
    """`kaynak`tan `hedef`e bakan açı. y ekseni aşağı arttığı için dy eksili."""
    return math.degrees(math.atan2(-(hedef[1] - kaynak[1]), hedef[0] - kaynak[0]))


def _polar_px(center: tuple[int, int], radius: float, angle_deg: float) -> tuple[int, int]:
    rad = math.radians(angle_deg)
    return (round(center[0] + radius * math.cos(rad)),
            round(center[1] - radius * math.sin(rad)))
