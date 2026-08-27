"""Tespit kutusunun verdiği kaba merkezi kadran çemberinden rafine eder.

    from gauge_vision.detect.refine import refine_dial
    daire = refine_dial(kare, merkez, yaricap)      # None = rafine güvenilmez

06.08 bütçesine göre zincir hatasının %35'i (0,681 puan) tespit kutusunun merkez
sapmasından geliyordu. Kutu kadranı iyi çevreliyor ama merkezi kadran çapının
~%1,3'ü kadar kayıyor; İP6'nın kutupsal taraması merkeze duyarlı olduğu için bu
doğrudan okuma hatasına dönüşüyor.

**Yaklaşım — neden Hough akümülatörü değil:**

Genel çember tespitinde (görüntüde çember nerede?) `HoughCircles` üç boyutlu
(x, y, r) akümülatör doldurur; maliyeti kenar pikseli × yarıçap adedi kadardır.
Bizim problemimiz genel değil: YOLO merkezi zaten ±%2 biliyor, yarıçapı da
yaklaşık biliyor. Elimizde güçlü bir ön bilgi varken arama yapmak israftır.

Onun yerine geometriden yararlanıyoruz: **eş merkezli bir çemberin her kenar
pikselinde gradyan merkezden geçen bir doğru tanımlar.** Bu doğruların en küçük
kareler kesişimi merkezi verir ve kapalı formda çözülür (2×2 denklem). Akümülatör
yok, arama yok, tek geçiş.

Eş merkezliliğe dikkat: kadran yüzü (1,0R) ve bezel dışı (1,07R) iki ayrı
çemberdir. Yarıçap uyduran bir çözüm (Kåsa vb.) bu karışımda ikisinin arasına
düşer; gradyan doğruları ise İKİSİ İÇİN DE merkezden geçtiği için karışım
merkezi bozmaz. Yöntemin bu problemde seçilme sebebi budur.

**Ana çizgiler neden bu yöntemi bozmuyor:** ana çizgiler 0,86R–1,0R arasında,
yani taradığımız halkanın içinde. Ama çizgi radyal uzanır → kenarının gradyanı
radyale DİKTİR. Çember kenarının gradyanı ise radyale PARALELDİR. Tek bir nokta
çarpımı eşiği (`COS_ESIK`) çizgileri, rakamları ve ibreyi eler.

Fikir kaynağı: Reşit'in `Cascaded-Soft-Hough` çalışmasındaki gradyan yönü
oylaması, piramitle küçültme ve medyan mesafeyle yarıçap doğrulama. Üçü de
alındı, akümülatörlü çok ölçekli arama alınmadı — ROI'yi YOLO veriyor.

Açı konvansiyonu bu dosyayı İLGİLENDİRMEZ: burada yapılan iş saf 2B geometridir,
görüntü koordinatlarında (y aşağı artar) doğrudan çalışır. Açıya çevirme yok,
dolayısıyla işaret hatası riski de yok.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- ROI ve çalışma çözünürlüğü ---
# Halka yarıçapın 1,18 katına kadar taranıyor, kutu merkezi de kaymış olabilir;
# 1,30 ikisine birden pay bırakır.
ROI_ORANI = 1.30

# ROI bu kenara indirilir. Sabit olması bilinçli: gömülü hedefte maliyet kamera
# çözünürlüğünden BAĞIMSIZ kalsın — 4K karede de 640×480'de de aynı işi yapar.
#
# Değer ÖLÇÜLEREK seçildi (06.08, sarsıntı %4): 128 px → %0,58 · 192 → %0,39 ·
# 256 → %0,15 · 320 → %0,046 · 384 → %0,022; süre 1,03 → 1,74 ms. 320'den sonra
# kazanç durdu, maliyet artmaya devam etti.
#
# **Ölçüm bir varsayımı çürüttü:** küçültmenin çizgi/rakam kenarlarını eritip
# işi KOLAYLAŞTIRACAĞI beklenmişti; tam tersi çıktı, çözünürlük düştükçe hata
# arttı. Sebebi şu: çizgileri zaten gradyan yönü filtresi eliyor, küçültmenin
# eleyecek bir şeyi kalmıyor, sadece kenar konumu bulanıyor. Yani piramidin
# bu problemdeki değeri gürültü bastırmak DEĞİL, maliyeti tavanlamaktır.
CALISMA_PX = 320

# --- Kenar seçimi ---
# Sabit eşik yerine görüntünün kendi istatistiği: parlama/karanlık altında sabit
# eşik ya her şeyi ya hiçbir şeyi geçirir (İP14'ün konusu, şimdiden hazırlıklı).
ESIK_K = 1.5
HALKA_MIN = 0.85      # kadran yüzü kenarı 1,00R · bezel dışı 1,07R
HALKA_MAX = 1.18
COS_ESIK = 0.94       # gradyan ile radyal yön arası ~±20°

# --- Kabul kapıları ---
# Rafine "iyileştirme" iddiasındadır; iddiayı ispatlayamazsa kutu tahmini kalır.
# Kötüleştirmemek, iyileştirmekten önce gelir.
MIN_DESTEK = 40           # bu kadar kenar pikseli bulunamazsa fit anlamsız
# Yarıçapın %20'sinden (= kadran çapının %10'u) fazla kayma → başka bir yapıya
# oturmuş demektir. Ölçümle seçildi: 0,12'de %6 sapmalı karelerin 37'si geri
# çevriliyordu — kapı düzeltilebilecek kareleri eliyordu. 0,20'de %8 sapma bile
# kurtarılıyor, 0,25'e çıkarmak ek kazanç vermedi. Medyan kapısı arkada duruyor.
MAX_KAYMA_ORANI = 0.20
MEDYAN_MIN = 0.85         # fitten sonra kenarların medyan mesafesi bu aralıkta
MEDYAN_MAX = 1.25         # değilse bulunan merkez çemberin merkezi değil
# Kanıt kalitesi kapıları. Eşikler ÖLÇÜLEN dağılımlardan seçildi (06.08):
#
#                        artık (medyan/p95/max)   yayılma (medyan/p95/max)
#   gerçek kadran        0,0142 / 0,0159 / 0,0171   0,0142 / 0,0217 / 0,0252
#   boş arka plan        0,1342 / 0,1922 / 0,2065   0,0703 / 0,1095 / 0,1237
#   rastgele gürültü     0,1754 / 0,1886 / 0,1947   0,0802 / 0,0881 / 0,0910
#
# 0,05 iki kümenin ortasında: gerçek kadranın en kötüsünün ~2-3 katı, sahtenin
# en iyisinin ~1,5-3 katı altında. Düz kenar (duvar/boru) zaten daha önceki
# kapılara takılıyor, buraya hiç gelmiyor.
#
# ⚠ Bu sayılar SENTETİK görüntüden. Gerçek fotoğrafta cam yansıması ve bulanıklık
# artığı yükseltecektir; eşik İP8'de gerçek karelerle yeniden ölçülmelidir.
#
# --- ÖLÇÜLDÜ (27.08): tahmin doğru çıktı, ama GEVŞETME ÇÖZÜM DEĞİL --------------
# 14 gerçek videoda rafine **%0,6 oranında kabul ediliyor** — yani sahada fiilen
# kapalı. Reddin nedeni ölçüldü (373 gerçek kadran kutusu):
#     artık büyük   %74,3   ← gerçek kadranda medyan artık 0,1195 (eşik 0,05)
#     kayma büyük   %19,8   ← medyan 0,111 ama p95 0,383 (eşik 0,20)
#     radyal az      %4,0
#     KABUL          %0,8
# Yukarıdaki uyarı aynen gerçekleşmiş: gerçek fotoğrafta artık 2,4 katına çıkıyor.
#
# Eşiği gevşetmek DENENDİ ve ELENDİ. Ground truth yok, ama 180° sıçrama fiziksel
# olarak imkânsız olduğu için hata sinyali ground truth gerektirmiyor. Dört
# ayar, dört gerçek kadran videosu, 466 okunan kare:
#
#     ayar                 rafine   toplam flip   oynama (4.mp4)
#     kapalı                  %0       10/466        3,24°
#     mevcut eşik           %0-1       10/466        3,25°
#     gevşek 0,15/0,30     %31-37      10/466        4,50°
#     gevşek 0,25/0,40     %34-44      10/466        5,00°
#
# Rafine üç kat daha sık ateşliyor ve **flip sayısı hiç değişmiyor**, açı
# oynaması ise artıyor. Yani gevşetmenin kazandırdığı bir şey yok, kaybettirdiği
# var. Kapılar reddederken haklı: gerçek kadranda artığı yükselten şey (cam
# yansıması, bulanıklık) merkezi de o kadar güvenilmez yapıyor.
#
# Sonuç: eşikler DEĞİŞTİRİLMEDİ. Rafinenin sahada kapalı olması bir kusur değil,
# dürüst bir ret — kutu merkezi zaten yeterince iyi. Kalan %2-10'luk flip'in yolu
# buradan geçmiyor; ZAMANSAL tutarlılık (bkz. read/needle.py sonundaki not).
MAX_ARTIK_ORANI = 0.05
MAX_YAYILMA_ORANI = 0.05

# Aykırı kenarları eleyen tek tur: artığı medyanın bu katından büyük olan atılır.
IRLS_KATSAYI = 2.5


@dataclass(frozen=True)
class DialCircle:
    """Rafine edilmiş kadran dairesi.

    `radius_px` bilinçli olarak KUTUDAN gelen değerdir, fitten değil: kutupsal
    tarama bandı [0,22R–0,72R] yarıçaptaki ±%7 hataya dayanıklıdır (ibre 0,78R,
    çizgiler 0,86R), merkeze ise duyarlıdır. Ölçülmemiş bir iyileştirme uğruna
    yarıçapı da oynatmak, hatayı tek değişkende izlemeyi imkânsız kılardı.
    """

    center_px: tuple[int, int]
    radius_px: float
    support: int           # fite giren kenar pikseli
    shift_px: float        # kaba merkeze göre kayma (tam çözünürlükte)
    residual_ratio: float  # doğruların çözüme medyan dik uzaklığı / yarıçap
    spread_ratio: float    # kenarların merkeze uzaklığındaki medyan sapma / yarıçap


def _dogru_kesisimi(nokta: np.ndarray, yon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """N doğrunun en küçük kareler kesişimi. (çözüm, dik artıklar) döner.

    Doğru i: `nokta[i]` noktasından geçer, `yon[i]` birim yönünde uzanır.
    Bir x noktasının doğruya dik uzaklığının karesi (x-p)ᵀ(I-ddᵀ)(x-p) olduğundan
    toplamı en küçükleyen x, (ΣPᵢ)x = ΣPᵢpᵢ denkleminin çözümüdür — 2×2.
    """
    dx, dy = yon[:, 0], yon[:, 1]
    # P = I - ddᵀ (dik izdüşüm), bileşenleri toplanarak 2×2 sistem kuruluyor.
    p11 = 1.0 - dx * dx
    p12 = -dx * dy
    p22 = 1.0 - dy * dy

    A = np.array([[p11.sum(), p12.sum()],
                  [p12.sum(), p22.sum()]])
    b = np.array([(p11 * nokta[:, 0] + p12 * nokta[:, 1]).sum(),
                  (p12 * nokta[:, 0] + p22 * nokta[:, 1]).sum()])

    # Tüm gradyanlar aynı yöne bakıyorsa (düz kenar, çember değil) A tekil olur.
    if abs(np.linalg.det(A)) < 1e-9:
        raise np.linalg.LinAlgError("doğrular paralel — çember kenarı yok")

    cozum = np.linalg.solve(A, b)
    fark = cozum - nokta
    # Dik artık: farkın doğru yönüne dik bileşeninin boyu.
    artik = np.abs(fark[:, 0] * dy - fark[:, 1] * dx)
    return cozum, artik


def refine_dial(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    *,
    calisma_px: int = CALISMA_PX,
) -> DialCircle | None:
    """Kaba (merkez, yarıçap) tahminini kadran çemberinden rafine eder.

    Rafine güvenilmezse `None` döner — çağıran kaba tahminde kalır. Hata
    yükseltmez: saha döngüsünde tek kare için tur düşürülmez.
    """
    if radius <= 0:
        return None

    h, w = image.shape[:2]
    yari = radius * ROI_ORANI
    x0 = max(0, int(center[0] - yari))
    y0 = max(0, int(center[1] - yari))
    x1 = min(w, int(center[0] + yari) + 1)
    y1 = min(h, int(center[1] + yari) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    roi = image[y0:y1, x0:x1]
    gri = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    # Sadece küçültülür, asla büyütülmez: 60 px'lik bir kadranı 192'ye şişirmek
    # bilgi eklemez, sadece maliyet ekler.
    olcek = min(1.0, calisma_px / max(gri.shape))
    if olcek < 1.0:
        gri = cv2.resize(gri, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)

    merkez_s = np.array([(center[0] - x0) * olcek, (center[1] - y0) * olcek])
    r_s = radius * olcek
    if r_s < 6:
        return None

    # Hafif bulanıklaştırma: gürültü pikselleri gradyan yönünü rastgele çevirir,
    # yön filtresi de tam olarak yöne güvendiği için önce yumuşatılır.
    gri = cv2.GaussianBlur(gri, (3, 3), 0)
    gx = cv2.Scharr(gri, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gri, cv2.CV_32F, 0, 1)
    buyukluk = cv2.magnitude(gx, gy)

    esik = float(buyukluk.mean() + ESIK_K * buyukluk.std())
    ys, xs = np.nonzero(buyukluk > esik)
    if xs.size < MIN_DESTEK:
        return None

    # Halka filtresi: kadranın içi (ibre, rakamlar, göbek) ve dışı (arka plan)
    # elenir. Kalan sadece çember kenarı ve ana çizgilerin dış uçlarıdır.
    dx_m = xs - merkez_s[0]
    dy_m = ys - merkez_s[1]
    uzaklik = np.hypot(dx_m, dy_m)
    halkada = (uzaklik > HALKA_MIN * r_s) & (uzaklik < HALKA_MAX * r_s)
    if halkada.sum() < MIN_DESTEK:
        return None

    xs, ys, uzaklik = xs[halkada], ys[halkada], uzaklik[halkada]
    dx_m, dy_m = dx_m[halkada], dy_m[halkada]
    gvx, gvy = gx[ys, xs], gy[ys, xs]

    gnorm = np.hypot(gvx, gvy)
    gnorm[gnorm == 0] = 1.0
    # Gradyan yönü radyal yönle ne kadar paralel? Çember kenarında |cos|≈1,
    # ana çizgide ≈0. İşaret önemsiz: içeri de dışarı da bakabilir.
    hizalanma = np.abs((gvx * dx_m + gvy * dy_m) / (gnorm * uzaklik))
    radyal = hizalanma > COS_ESIK
    if radyal.sum() < MIN_DESTEK:
        return None

    nokta = np.column_stack((xs[radyal], ys[radyal])).astype(np.float64)
    yon = np.column_stack((gvx[radyal] / gnorm[radyal], gvy[radyal] / gnorm[radyal]))

    try:
        cozum, artik = _dogru_kesisimi(nokta, yon)
        # Tek tur aykırı eleme: yön filtresinden kaçan çizgi ucu / cam yansıması
        # olursa merkezi çeker. İkinci tura gerek yok, ölçüm farkı göstermedi.
        kalan = artik <= IRLS_KATSAYI * max(np.median(artik), 1e-6)
        if kalan.sum() >= MIN_DESTEK:
            nokta, yon = nokta[kalan], yon[kalan]
            cozum, artik = _dogru_kesisimi(nokta, yon)
    except np.linalg.LinAlgError:
        return None

    kayma_s = float(np.hypot(*(cozum - merkez_s)))
    if kayma_s > MAX_KAYMA_ORANI * r_s:
        return None

    uzaklik_son = np.hypot(nokta[:, 0] - cozum[0], nokta[:, 1] - cozum[1])
    medyan = float(np.median(uzaklik_son))
    if not (MEDYAN_MIN * r_s <= medyan <= MEDYAN_MAX * r_s):
        return None

    # --- Kanıt kalitesi: BURADA GERÇEKTEN ÇEMBER VAR MI? ---
    # Yukarıdaki iki kapı sadece "cevap makul mü" diye sorar ve rastgele gürültüde
    # de geçer (06.08'de ölçüldü: 50/50 gürültü kabul ediliyordu). Aşağıdaki iki
    # sayı kanıtın kendisine bakar:
    #   artik_orani  — gradyan doğruları TEK bir noktada kesişiyor mu? Çemberde
    #                  hepsi merkezden geçer, artık ~0; gürültüde yön rastgeledir.
    #   yayilma      — kenarlar merkezden EŞİT uzaklıkta mı? Çemberde dar bir
    #                  bantta toplanır; gürültüde halkaya yayılır.
    artik_orani = float(np.median(artik)) / r_s
    yayilma = float(np.median(np.abs(uzaklik_son - medyan))) / r_s
    if artik_orani > MAX_ARTIK_ORANI or yayilma > MAX_YAYILMA_ORANI:
        return None

    return DialCircle(
        center_px=(round(cozum[0] / olcek + x0), round(cozum[1] / olcek + y0)),
        radius_px=radius,
        support=int(nokta.shape[0]),
        shift_px=kayma_s / olcek,
        residual_ratio=artik_orani,
        spread_ratio=yayilma,
    )
