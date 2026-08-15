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

Açı konvansiyonu: 0° = saat 3 yönü, CCW pozitif (bkz. `configs/gauges.yaml` başlığı).
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

# --- İki kapı: biri MUTLAK, biri AYRIKLIK ---
#
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

# ⚠ 13.08'in bulgusu: uyum kapısı TEK BAŞINA "kanıt var mı" sorusunu sormuyor.
# Tanımadığımız bir kadran stili (araç hız göstergesi) çizgi halkası taşıdığı
# için beklenen desenle 0,40'ın ÜSTÜNDE korelasyon üretebiliyor — ama desen ona
# BENZEDİĞİ için değil, "çember üstünde çizgiler" olduğu için: eşit aralıklı
# yabancı desen birçok kaymada benzer skor verir ve en iyisi rastgele bir
# kaymadır (ölçülen vaka: gerçekte ~0° yatık panelde 21,3° sahte yatıklık).
# Kanıt, desenin YALNIZCA tek bir kaymada oturmasıdır. Bunu ayrıklık ölçer:
#
#   ayriklik = (tepe - tepe komşuluğu DIŞINDAKİ küresel en iyi skor) / norm
#
# İkinci tepe araması eskiden ±MAX_ROLL_DEG penceresinin İÇİNDEydi; yabancı
# desen pencerenin dışında daha da iyi oturuyorsa görünmüyordu bile. Artık tüm
# çember taranıyor — desen başka bir kaymada eşit oturuyorsa fark sıfıra düşer
# ve kestirim susar. Eşik `scripts/olc_roll_kaniti.py` ile ölçülen dağılımlardan
# (13.08 · 146 kare · outputs/metrics/roll_kaniti.json):
#   gerçek kadran (doğru kimlik, ±15° yatık)  min 0,112 · medyan 0,147
#   rastgele gürültü                          max 0,036
#   yanlış gösterge kimliği                   max 0,024
#   yabancı stil (saat düzeni + araç paneli)  max 0,012
# 0,10 aradaki boşlukta ve bilinçli olarak gerçek kümenin dibine yakın: sahte
# kestirim (yanlış yöne düzeltme) kestirimsizlikten (roll=0) daha tehlikeli,
# şüphe aleyhte kullanılıyor. Aynı koşuda eski kapının yabancı stilde 26 karenin
# 6'sında sahte kestirim ürettiği, yenisinin 0 ürettiği ölçüldü; doğru kümede
# yatıklık hatası değişmedi (ortanca 0,023° · max 0,075°).
MIN_AYRIKLIK = 0.10

# Eski tepe/ikinci ORAN kapısı kaldırıldı (13.08): ölçülen dağılımları zaten iç
# içeydi (gerçek min 1,18 · gürültü medyan 1,21) ve koruduğu "iki kaymanın
# yarışması" arızasını ayrıklık kapısı mutlak ölçekte, tüm çemberde ölçüyor.
# İki kapı aynı soruyu sorunca zayıf olanı yanlış bir güven kaynağı oluyordu.


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
    match: float           # tepedeki normalize korelasyon (MUTLAK uyum, -1..1)
    separation: float      # tepe - komşuluk dışı küresel en iyi (normalize fark)
    contrast: float


@dataclass(frozen=True)
class RollEvidence:
    """Kapılar uygulanmadan ÖNCEKİ iç sayılar.

    İki tüketicisi var: `estimate_roll` (kapıları bunun üstüne uygular) ve
    teşhis/ölçüm scriptleri (`tani_ip8.py`, `olc_roll_kaniti.py` — kapıya
    takılan karelerin NEDEN takıldığını görmek ve eşikleri dağılımdan seçmek
    için). Teşhis tarafı eskiden bu hesabın bir kopyasını taşıyordu; kopya,
    buradaki bir değişiklikte sessizce bayatlayacaktı.
    """

    roll_deg: float        # pencere içi en iyi kayma (ince ayarlı, ±180)
    match: float           # tepedeki normalize korelasyon
    separation: float      # (tepe - komşuluk dışı KÜRESEL en iyi) / norm
    contrast: float
    global_best_deg: float # tüm çemberdeki en iyi kaymanın yeri (teşhis için)


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


def roll_evidence(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    gauge: Gauge,
    *,
    max_roll_deg: float = MAX_ROLL_DEG,
    beklenen: np.ndarray | None = None,
) -> RollEvidence | None:
    """Yatıklık kestiriminin KANIT sayılarını üretir; kapı uygulamaz.

    None yalnızca sayı üretilemediğinde döner (profil yok, kontrast yok);
    "kanıt zayıf" burada elenmez — o karar `estimate_roll`'un eşiklerine ait.
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
    # Kestirim alanı: 0'ın iki yanı (negatif yatıklık dizinin sonundan sarar).
    # Platform kadranı kabaca dik gördüğü için gerçek yatıklık bu pencerededir;
    # pencere dışı da kanıt aramasına dahildir (aşağıda).
    adaylar = np.concatenate([np.arange(0, sinir + 1), np.arange(n - sinir, n)])
    en_iyi = int(adaylar[np.argmax(skor[adaylar])])
    tepe = float(skor[en_iyi])

    # İkinci tepe: en iyinin komşuluğu dışındaki, TÜM ÇEMBERDEKİ en yüksek
    # skor. Komşuluk dışlanıyor çünkü aynı tümseğin yamacı "rakip" değildir.
    # Arama pencereyle SINIRLANMIYOR: yabancı bir desen 40°'de daha iyi
    # oturuyorsa bu, penceredeki tepenin kanıt olmadığının ta kendisidir —
    # pencere içinde aransaydı görünmezdi (13.08'in sahte yatıklık vakası).
    komsuluk = max(2, int(round(2 * TICK_SIGMA_DEG / STEP_DEG)))
    mesafe = np.arange(n)
    mesafe = np.minimum(np.abs(mesafe - en_iyi), n - np.abs(mesafe - en_iyi))
    uzak = mesafe > komsuluk
    ikinci = float(skor[uzak].max()) if uzak.any() else 0.0
    kuresel = int(np.argmax(skor))

    norm = (np.linalg.norm(olculen - olculen.mean())
            * np.linalg.norm(beklenen - beklenen.mean()))
    if norm <= 0:
        return None
    uyum = tepe / norm
    ayriklik = (tepe - ikinci) / norm

    roll = _tepe_ince_ayar(skor, en_iyi, STEP_DEG)
    roll = (roll + 180.0) % 360.0 - 180.0
    kuresel_deg = (kuresel * STEP_DEG + 180.0) % 360.0 - 180.0
    return RollEvidence(roll_deg=float(roll), match=float(uyum),
                        separation=float(ayriklik), contrast=kontrast,
                        global_best_deg=float(kuresel_deg))


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

    İki kapı, iki ayrı soru:
      uyum     ≥ MIN_UYUM      desen bu kaymada GERÇEKTEN oturuyor mu
      ayrıklık ≥ MIN_AYRIKLIK  YALNIZCA bu kaymada mı oturuyor
    İkincisi olmadan ilki "cevap makul mü"den öteye geçmiyordu: yabancı bir
    kadran stili uyumu geçip rastgele bir kaymaya kilitlenebiliyordu (13.08).
    """
    kanit = roll_evidence(image, center, radius, gauge,
                          max_roll_deg=max_roll_deg, beklenen=beklenen)
    if kanit is None:
        return None
    if kanit.match < MIN_UYUM or kanit.separation < MIN_AYRIKLIK:
        return None      # desen tutmuyor ya da tek kaymaya özgü değil
    if abs(kanit.roll_deg) > max_roll_deg:
        return None

    # Güven iki kapının ikisine birden bakar: desen tutuyor mu VE tepe ayrık mı.
    # Tek başına hiçbiri yetmiyor; çarpım en zayıf halkayı öne çıkarır.
    g_uyum = np.clip((kanit.match - MIN_UYUM) / (1.0 - MIN_UYUM), 0.0, 1.0)
    g_ayrik = np.clip((kanit.separation - MIN_AYRIKLIK) / (0.5 - MIN_AYRIKLIK),
                      0.0, 1.0)
    return RollEstimate(roll_deg=kanit.roll_deg,
                        confidence=float(g_uyum * g_ayrik),
                        match=kanit.match, separation=kanit.separation,
                        contrast=kanit.contrast)
