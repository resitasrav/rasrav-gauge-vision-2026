"""Kameranın yatıklığını kadranın KENDİ çizgilerinden kestirir (İP8 / K2).

    from gauge_vision.read.roll import estimate_roll

    kestirim = estimate_roll(kare, merkez, yaricap, gauge)
    kestirim.roll_deg     # +4.2  → İP7'ye verilir, okuma bu kadar geri döndürülür

**Neden gerekli:** 06.08 bütçesine göre uçtan uca hatanın %90'ı düzeltilmeyen
yatıklıktan geliyor (1,710 puanın 1,890'ı). Kamera 4° yatıksa ibre açısı da 4°
kayar; 270°'lik kadranda bu doğrudan %1,5 tam skala hatadır. Tespit merkezini
düzeltmek (`detect/refine.py`) merkez kalemini 0,681'den 0,051'e indirdi ama
uçtan uca sayıyı oynatmadı — çünkü darboğaz burasıydı.

**Yöntem: beklenen çizgi deseniyle dairesel çapraz korelasyon.**

Kadranın çizgilerinin NEREDE olması gerektiğini biliyoruz: `gauges.yaml`'daki
çizgi sayısı ve ölçek kuralı her çizginin kadran çerçevesindeki açısını verir.
Görüntüde ölçtüğümüz desen ise bunun `roll` kadar dönmüş hâlidir. İki deseni
dairesel kaydırarak en iyi örtüşmeyi arıyoruz; kayma miktarı yatıklıktır.

Denenmeyen basit alternatif ve neden: "ölü bölgeyi (çizgisiz yay) bul, ortasını
al". Tek bir özelliğe dayanır — ölü bölgenin bir kenarı parlama veya örtme
yüzünden kaybolursa kestirim yarım kadran kayar. Korelasyon BÜTÜN çizgileri
birden kullanır, birkaçının kaybolması sonucu bozmaz.

**Karekök ölçekli kadran (FI-310) burada AVANTAJ:** çizgileri eşit aralıklı
değildir, desen kendini tekrar etmez, korelasyon tepesi tektir. Eşit aralıklı
kadranda ise desen ~çizgi aralığı kadar periyodiktir; bu belirsizliği iki şey
kırar: ölü bölge (deseni periyodik olmaktan çıkarır) ve `MAX_ROLL_DEG` sınırı.

Açı konvansiyonu: 0° = saat 3 yönü, CCW pozitif (bkz. CLAUDE.md §3).
`roll_deg` sentetik üreteçteki `DialLook.roll_deg` ile aynı işarettedir:
`angle_img = angle_dial + roll`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gauge_vision.config import Gauge

# --- Çizgi halkası (kadran yarıçapına oran) ---
# Ana çizgiler 1,00R'den 0,86R'ye, ara çizgiler 0,93R'ye uzanır (synth/dial.py).
# İbre 0,78R'de biter, sayı etiketleri 0,70R'dedir: halkayı 0,84'ten başlatmak
# ikisini de dışarıda bırakır. İbrenin desene karışması yatıklığı ibrenin
# durduğu yöne doğru çeker — okumanın kendisini bozacak türden bir hata.
TICK_R_MIN_RATIO = 0.84
TICK_R_MAX_RATIO = 1.00
TICK_SAMPLES = 12          # halka boyunca yarıçap örneği
STEP_DEG = 1.0             # açı çözünürlüğü; alt-derece tepe parabolle bulunur

# Pan-tilt platformu kadranı kabaca dik görecek şekilde duruyor; ±25° dışındaki
# bir "yatıklık" kestirimi düzeltme değil, yanlış çizgiye kilitlenmedir.
MAX_ROLL_DEG = 25.0

# Ara çizgiler ana çizgilerin yarısı kadar uzun; halkanın da yaklaşık yarısını
# kaplarlar. Beklenen desende ağırlıkları bu yüzden yarım.
MINOR_WEIGHT = 0.5
# Beklenen desendeki her çizgi bu genişlikte bir tümsek olarak çiziliyor.
# Sıfır genişlikte darbe kullanılırsa korelasyon tepesi bir derecelik kaymada
# çöker; tümsek onu sürekli hale getirir.
TICK_SIGMA_DEG = 1.5

MIN_KONTRAST = 0.02        # halkada çizgi hiç görünmüyorsa kestirim üretilmez

# --- İki kapı: biri GÖRELİ, biri MUTLAK ---
#
# Tepe / ikinci en iyi tepe oranı: aynı desene iki farklı kaymada eşit iyi
# oturuluyorsa (eşit aralıklı kadranda periyodiklik) hangisinin doğru olduğu
# bilinemez. Ölçülen dağılımlar (06.08):
#   gerçek kadran  min 1,18 · medyan 1,68 · max 6,94
#   gürültü        min 1,01 · medyan 1,21 · max 6,16
# **Bu kapı gürültüyü AYIRT ETMİYOR** — dağılımlar iç içe. Yine de duruyor,
# çünkü koruduğu şey farklı bir arıza: iki kaymanın yarışması. Gürültüyü
# aşağıdaki uyum kapısı eliyor.
MIN_TEPE_ORANI = 1.08

# Tepedeki normalize korelasyon — desen GERÇEKTEN tutuyor mu. Ölçülen dağılımlar:
#   gerçek kadran     min 0,629 · medyan 0,724 · max 0,804
#   rastgele gürültü  min 0,012 · medyan 0,099 · max 0,176
#   YANLIŞ gösterge   min 0,149 · medyan 0,196 · max 0,219
# 0,40 üç kümenin de dışında, gerçek kadranın en kötüsünün epey altında.
#
# Üçüncü satır beklenmeyen bir kazanç: göstergeye YANLIŞ kimlik verilirse
# (U11 — waypoint sözlüğü tanımsız, kimlik şu an elle geliyor) desen tutmaz ve
# yatıklık kestirimi susar. Yanlış kimliğin sessizce yanlış değer üretmesine
# karşı elimizdeki tek otomatik işaret budur.
MIN_UYUM = 0.40


@dataclass(frozen=True)
class RollEstimate:
    """Kestirilen kamera yatıklığı.

    `confidence` korelasyon tepesinin ne kadar ayrık olduğundan gelir; İP15'in
    eşiğine girmek üzere 0-1 aralığındadır. Belirsizse `estimate_roll` None
    döner — yanlış yatıklık, yatıklığı hiç düzeltmemekten daha kötüdür çünkü
    hatayı azaltmak yerine rastgele bir yöne taşır.
    """

    roll_deg: float
    confidence: float
    peak_ratio: float      # en iyi tepe / ikinci en iyi tepe (GÖRELİ ayrıklık)
    match: float           # tepedeki normalize korelasyon (MUTLAK uyum, -1..1)
    contrast: float        # ölçülen çizgi deseninin genliği


def expected_tick_profile(gauge: Gauge, step_deg: float = STEP_DEG) -> np.ndarray:
    """Kadranın çizgilerinin BEKLENEN açısal deseni (kadran çerçevesinde).

    Envanterden türetilir, görüntüye bakmaz. Aynı gösterge için sabit olduğundan
    çağıran tarafından bir kez hesaplanıp saklanabilir.
    """
    n = int(round(360.0 / step_deg))
    profil = np.zeros(n)
    majors, minors = gauge.tick_values()

    aci_derece = np.arange(n) * step_deg
    for degerler, agirlik in ((majors, 1.0), (minors, MINOR_WEIGHT)):
        for v in degerler:
            aci = gauge.scale.angle_for_value(v) % 360.0
            # Dairesel mesafe: 359° ile 1° arası 2°'dir, 358° değil.
            d = (aci_derece - aci + 180.0) % 360.0 - 180.0
            profil += agirlik * np.exp(-0.5 * (d / TICK_SIGMA_DEG) ** 2)
    return profil


# Beklenen desen göstergeye özgü ve sabittir; saha döngüsünde her karede yeniden
# üretmenin anlamı yok. Anahtar deseni BELİRLEYEN alanlardan kuruluyor: envanter
# değişip yeniden yüklenirse anahtar da değişir, bayat desen kullanılmaz.
_PROFIL_ONBELLEK: dict[tuple, np.ndarray] = {}


def cached_tick_profile(gauge: Gauge) -> np.ndarray:
    """`expected_tick_profile`'ın önbellekli hâli."""
    s = gauge.scale
    anahtar = (gauge.id, s.min, s.max, s.angle_min, s.angle_max, s.direction, s.linear,
               gauge.synthetic.get("tick_major"), gauge.synthetic.get("tick_minor"))
    if anahtar not in _PROFIL_ONBELLEK:
        _PROFIL_ONBELLEK[anahtar] = expected_tick_profile(gauge)
    return _PROFIL_ONBELLEK[anahtar]


def measured_tick_profile(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    step_deg: float = STEP_DEG,
) -> np.ndarray | None:
    """Çizgi halkasındaki koyuluğun açıya göre deseni (görüntü çerçevesinde).

    Kadranın kendi zemininden eşiklenir, sabit eşikle değil: çizginin ve yüzün
    tonu göstergeden göstergeye ve ışığa göre değişir.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    acilar = np.arange(0.0, 360.0, step_deg)
    yaricaplar = np.linspace(radius * TICK_R_MIN_RATIO, radius * TICK_R_MAX_RATIO,
                             TICK_SAMPLES)
    rad = np.radians(acilar)[:, None]
    xs = np.rint(center[0] + yaricaplar[None, :] * np.cos(rad)).astype(int)
    ys = np.rint(center[1] - yaricaplar[None, :] * np.sin(rad)).astype(int)

    icerde = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if icerde.mean() < 0.5:
        return None      # kadranın yarısından fazlası karenin dışında

    ornek = np.full(xs.shape, np.nan)
    ornek[icerde] = gray[ys[icerde], xs[icerde]]

    # Koyuluk = zemine göre ne kadar karanlık. Halkanın kendi üst yüzdeliği
    # zemin sayılıyor: çizgiler halkanın küçük bir kısmını kaplar, kalanı yüzdür.
    gecerli = ornek[~np.isnan(ornek)]
    if gecerli.size < 32:
        return None
    zemin = np.percentile(gecerli, 80)
    koyuluk = np.clip(zemin - ornek, 0, None)
    return np.nanmean(koyuluk, axis=1)


def _dairesel_korelasyon(olculen: np.ndarray, beklenen: np.ndarray) -> np.ndarray:
    """Her kayma için örtüşme skoru. İndis k → `roll = k * step_deg`.

    Ölçülen desen beklenenin `roll` kadar dönmüş hâlidir; beklenen deseni k kadar
    ileri kaydırıp çarpıyoruz, en iyi örtüşme k = roll'da olur. FFT ile çünkü
    360 kaymanın hepsini tek geçişte verir (gömülü hedefte de milisaniye altı).
    """
    o = olculen - olculen.mean()
    b = beklenen - beklenen.mean()
    return np.fft.irfft(np.fft.rfft(o) * np.conj(np.fft.rfft(b)), n=o.size)


def _tepe_ince_ayar(skor: np.ndarray, k: int, step_deg: float) -> float:
    """Tepe indisini komşularına parabol uydurup alt-derece çözünürlüğe taşır.

    Adım 1° iken kestirimi 1°'ye yuvarlamak, düzeltmenin kendisi kadar hata
    bırakırdı (1° ≈ %0,37 tam skala).
    """
    n = skor.size
    y0, y1, y2 = skor[(k - 1) % n], skor[k], skor[(k + 1) % n]
    payda = y0 - 2 * y1 + y2
    kayma = 0.0 if abs(payda) < 1e-12 else 0.5 * (y0 - y2) / payda
    return (k + np.clip(kayma, -1.0, 1.0)) * step_deg


def estimate_roll(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    gauge: Gauge,
    *,
    max_roll_deg: float = MAX_ROLL_DEG,
    beklenen: np.ndarray | None = None,
) -> RollEstimate | None:
    """Kadranın çizgilerinden kamera yatıklığını kestirir.

    Kestirim belirsizse None döner; çağıran yatıklığı 0 kabul eder. `beklenen`
    önceden hesaplanmış desen verilirse yeniden üretilmez (aynı gösterge için
    sabittir, saha döngüsünde her karede hesaplamaya gerek yok).
    """
    if gauge.scale is None:
        return None
    if radius <= 0:
        return None

    olculen = measured_tick_profile(image, center, radius)
    if olculen is None:
        return None

    kontrast = float(olculen.max() - olculen.min())
    # Genlik yoksa halkada çizgi görünmüyor demektir (kadran çok küçük, aşırı
    # bulanık ya da yarıçap yanlış). Gürültüye desen uydurmanın anlamı yok.
    if kontrast < MIN_KONTRAST * 255:
        return None

    if beklenen is None:
        beklenen = cached_tick_profile(gauge)
    skor = _dairesel_korelasyon(olculen, beklenen)

    n = skor.size
    sinir = int(round(max_roll_deg / STEP_DEG))
    # Aranan aralık: 0'ın iki yanı (negatif yatıklık dizinin sonundan sarar).
    adaylar = np.concatenate([np.arange(0, sinir + 1), np.arange(n - sinir, n)])
    en_iyi = int(adaylar[np.argmax(skor[adaylar])])

    # İkinci tepe: en iyinin komşuluğu dışındaki en yüksek skor. Aynı tümseğin
    # yamacı "rakip" sayılırsa oran hep 1'e yakın çıkar ve kapı hiç çalışmaz.
    komsuluk = max(2, int(round(2 * TICK_SIGMA_DEG / STEP_DEG)))
    uzak = adaylar[np.minimum(np.abs(adaylar - en_iyi), n - np.abs(adaylar - en_iyi)) > komsuluk]
    ikinci = float(skor[uzak].max()) if uzak.size else 0.0

    tepe = float(skor[en_iyi])
    if tepe <= 0:
        return None
    oran = tepe / max(ikinci, 1e-9) if ikinci > 0 else float("inf")

    # Normalize korelasyon: tepenin MUTLAK uyumu. Tepe oranı yalnızca "en iyi
    # kayma ikinciden ne kadar iyi" der ve gürültüde de tesadüfen yüksek çıkar
    # (06.08: 10 gürültü karesinin 8'i oran kapısını geçiyordu). Bu sayı ise
    # ölçülen desenin beklenene gerçekten benzeyip benzemediğini söyler.
    norm = np.linalg.norm(olculen - olculen.mean()) * np.linalg.norm(beklenen - beklenen.mean())
    uyum = tepe / norm if norm > 0 else 0.0

    if uyum < MIN_UYUM or oran < MIN_TEPE_ORANI:
        return None      # desen tutmuyor ya da belirsiz — düzeltme yapılmaz

    roll = _tepe_ince_ayar(skor, en_iyi, STEP_DEG)
    roll = (roll + 180.0) % 360.0 - 180.0
    if abs(roll) > max_roll_deg:
        return None

    # Güven iki kapının ikisine birden bakar: desen tutuyor mu VE tepe ayrık mı.
    # Tek başına hiçbiri yetmiyor; çarpım en zayıf halkayı öne çıkarır.
    g_uyum = np.clip((uyum - MIN_UYUM) / (1.0 - MIN_UYUM), 0.0, 1.0)
    g_oran = 1.0 if not np.isfinite(oran) else np.clip(1.0 - MIN_TEPE_ORANI / oran, 0.0, 1.0)
    return RollEstimate(roll_deg=float(roll), confidence=float(g_uyum * g_oran),
                        peak_ratio=float(oran), match=float(uyum), contrast=kontrast)
