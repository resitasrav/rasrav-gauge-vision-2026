"""7-segment / dijital panel okur — İP11.

    from gauge_vision.read.digital import read_digital

    okuma = read_digital(kare, gauge)
    okuma.value    # 123.4
    okuma.status   # "ok" | "unreadable" | "out_of_range" | "alarm"

**Neden genel amaçlı OCR değil de segment geometrisi.**

İP4'ün taraması (K5) genel OCR'ın 7-segment ekranlarda zorlandığını gösteriyordu;
`letsgodigital` gibi ayrı eğitilmiş modeller tam bu yüzden var. Sebep şu:
7-segment rakamlar **bağlantısız** çizgilerden oluşur. Genel OCR modelleri
bitişik gövdeli fontlarla eğitilmiştir ve "7" ile "1"i, "8" ile "0"ı ayıran
şey onlarda gövde şeklidir. Segment ekranda ise ayırt edici bilgi tamamen
**hangi çubuğun yandığındadır**.

O bilgi doğrudan okunabiliyorken araya bir öğrenilmiş model koymak, çözülmüş
bir problemi belirsizleştirir. Burada yapılan: haneyi bul, yedi segment
bölgesinin her birinin parlaklığını ölç, yanık/sönük diye ikile, elde edilen
7 bitlik deseni rakama çevir.

**Bunun sınırı da açıktır ve gizlenmiyor:** yöntem panelin dik ve düz
göründüğünü varsayar. Eğik bakışta hane kutuları kayar. Gerçek panel
fotoğraflarında `letsgodigital` ile kıyaslanmalıdır — kıyas yapılmadan
"daha iyi" denmez (İP6'da kutupsal tarama Hough'u ölçümle geçmişti, burada da
aynı disiplin geçerli).

**Belirsiz desen uydurulmaz.** Yedi bitlik desen tabloda yoksa hane `None`
döner ve okuma `unreadable` olur — 3. kural burada da geçerli.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gauge_vision.config import Gauge
from gauge_vision.read.calibrate import (
    DURUM_ALARM,
    DURUM_KADRAN_DISI,
    DURUM_OK,
    DURUM_OKUNAMADI,
    GaugeReading,
)

# --- Hane bulma ---
# Panelin karanlık zemininde yanık segmentler parlak; eşik Otsu ile bulunur.
MIN_HANE_EN_ORANI = 0.02      # görüntü enine oran — gürültü lekelerini eler
MIN_HANE_BOY_ORANI = 0.20     # rakamlar panelin belirgin bir kısmını kaplar
MAX_HANE_EN_ORANI = 0.40

# --- Segment örnekleme ---
# Her segmentin merkezinden küçük bir pencere örneklenir. Pencere segment
# KALINLIĞINDAN dar olmalı (kalınlık ≈ hane eninin %16'sı): taşarsa zemin
# ortalamaya karışır, yanık segment sönüğe yaklaşır ve güven düşer. 0,22 ile
# ölçülen güvenler 0,34-0,72 arasındaydı ve DP-401'in 0,80 eşiğini geçemiyordu —
# rakamlar DOĞRU çözülüyor ama okuma reddediliyordu.
ORNEK_PENCERE_ORANI = 0.13    # hane enine oran
# Yanık/sönük ayrımı: hane içindeki en parlak ve en sönük segmentin ortası.
# Sabit eşik kullanılamaz — panel parlaklığı ışığa ve pozlamaya göre değişir.
AYRIM_PAYI = 0.35             # ayrım noktasının bu kadar uzağındaki değerler net

# --- Aydınlatma bantları (27.08) ---
# `_panel_seviyeleri` panelin TAMAMI için tek bir yanık/sönük referansı üretir ve
# bu, aydınlatmanın panel boyunca sabit olmasını gerektirir. Gerçek fotoğrafta
# gerektirmiyor: ekranda yansıma gradyanı var ve sol üçte birin zemin medyanı
# sağın **1,53-1,71 katı** çıkıyor (21.08 ölçümü). Tek referansla bir uçta yanık
# segment eşiğin altında, öbür uçta zemin üstünde kalıyor.
#
# Referans YİNE PANELDEN geliyor — hane içinden ölçmek "8" ve boş haneyi
# çözemez, gerekçesi `_haneyi_coz` docstring'inde — ama sabit değil, hanenin x
# konumundaki aydınlatmaya göre ölçekleniyor.
#
# **Ölçek ZEMİNDEN kestiriliyor, rakamdan değil:** rakam içeriği bantla değişir
# (bir bantta "8", öbüründe "1") ama zemin yalnız aydınlatmayla değişir.
#
# Model ÇARPIMSAL; toplamsal da denendi ve daha zayıf çıktı (İP8'in beş gerçek
# fotoğrafında çarpımsal 1/5, toplamsal 0/5). Yansıma toplamsal olsa da ekranın
# kendi parlaklığı çarpımsal ölçekleniyor ve baskın olan o.
BANT_SAYISI = 8
# Bozuk bir bant referansı savurmasın: kazanç bu aralığa kırpılır.
BANT_KAZANC_ALT, BANT_KAZANC_UST = 0.4, 2.5

# Segment düzeni `synth/digital.py` ile aynı: a b c d e f g
SEGMENT_KONUMLARI: dict[str, tuple[float, float]] = {
    "a": (0.50, 0.06),   # üst
    "b": (0.90, 0.28),   # sağ üst
    "c": (0.90, 0.74),   # sağ alt
    "d": (0.50, 0.95),   # alt
    "e": (0.10, 0.74),   # sol alt
    "f": (0.10, 0.28),   # sol üst
    "g": (0.50, 0.50),   # orta
}

DESEN_RAKAM: dict[frozenset, str] = {
    frozenset("abcdef"): "0", frozenset("bc"): "1", frozenset("abdeg"): "2",
    frozenset("abcdg"): "3", frozenset("bcfg"): "4", frozenset("acdfg"): "5",
    frozenset("acdefg"): "6", frozenset("abc"): "7", frozenset("abcdefg"): "8",
    frozenset("abcdfg"): "9", frozenset("g"): "-", frozenset(): " ",
}


@dataclass(frozen=True)
class DigitalReading:
    """Panel okuması + izlenebilirlik için ara sonuçlar."""

    text: str | None                    # "123.4" — çözülemeyen hane varsa None
    value: float | None
    confidence: float
    digit_confidences: list[float]
    digit_boxes: list[tuple[int, int, int, int]]
    reason: str = ""


def _segment_maskesi(gri: np.ndarray) -> np.ndarray:
    """Segment pikselleri (YANIK **ve** SÖNÜK) — panel zemininden ayrılmış hâlde.

    **Tek eşik burada çalışmaz ve ilk sürüm tam bu yüzden çuvalladı.** Panelde
    üç parlaklık seviyesi var: zemin (~28), sönük segment (~53), yanık segment
    (~166). Tek Otsu eşiği en büyük boşluğa oturur, yani sönük ile yanık
    arasına — geriye yalnızca YANIK pikseller kalır.

    Sonuç: "1" rakamının bağlantılı bileşeni yalnızca sağdaki iki çubuktur ve
    kutusu 20 piksel genişliğinde çıkar. Hane kutusu yanlış olunca segment
    örnekleme noktaları da yanlış yere düşer ve hiçbir rakam çözülemez.

    Çözüm iki kademeli eşikleme: önce yanık/geri kalan ayrılır, sonra geri
    kalanın içinde zemin/sönük ayrılır. İkinci eşik hane KUTUSUNU verir.
    """
    otsu1, _ = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alt_kume = gri[gri <= otsu1]
    if alt_kume.size < 32:
        return (gri > otsu1).astype(np.uint8) * 255

    otsu2, _ = cv2.threshold(alt_kume.reshape(-1, 1), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maske = (gri > otsu2).astype(np.uint8) * 255

    # Kenara değen bileşenleri at — panel ÇERÇEVESİ maskeye sızıyor.
    #
    # Parlaklık sıralaması sezgiye aykırı: zemin ~28, SÖNÜK segment ~53,
    # çerçeve ~70, yanık segment ~166. Yani çerçeve sönük segmentten DAHA
    # PARLAK. Sönük segmentleri yakalayan her eşik çerçeveyi de yakalar ve
    # hane kutuları tüm karenin boyuna yayılır (ölçülen: 4 hane yerine
    # 0-127-255-383 diye kareyi dörde bölen kutular, hiçbiri okunmuyor).
    #
    # Ayırt edici özellik konum: çerçeve görüntü kenarına değer, rakamlar
    # değmez. Eşikle uğraşmak yerine bu geometrik gerçek kullanılıyor.
    n, etiket, ist, _ = cv2.connectedComponentsWithStats(maske, 8)
    h, w = maske.shape
    temiz = np.zeros_like(maske)
    for i in range(1, n):
        x, y, cw, ch, _ = ist[i]
        if x <= 0 or y <= 0 or x + cw >= w or y + ch >= h:
            continue
        temiz[etiket == i] = 255
    # Hiçbir şey kalmadıysa (panel kareyi tamamen dolduruyor) ham maskeye dön.
    return temiz if np.count_nonzero(temiz) > 50 else maske


def _egim_kestir(maske: np.ndarray, kutular: list) -> float:
    """Rakamların sağa yatıklığı. Hane kutularının MEDYANI alınır.

    Gerçek 7-segment panellerin çoğu yatıktır. Yatıklık hesaba katılmazsa üst
    segmentler (a, b, f) sistematik olarak sola kayık örneklenir; "7" ile "1"
    gibi yalnızca üst segmentlerle ayrılan rakamlar karışır.

    **Panel genelinden kestirmek çalışmaz** — ilk sürüm bunu deniyordu ve 0,10
    yerine 0,002 buluyordu. Sebep: haneler yatayda yayılı olduğu için üst ve alt
    şeritlerin x ortalaması yatıklığı değil yerleşimi ölçüyor, fark yıkanıyor.
    Kestirim hane hane yapılmalı.

    Hane içinde yatıklık, x ile y arasındaki kovaryanstan çıkar: rakam sağa
    yatıksa y küçüldükçe (yukarı çıkıldıkça) x büyür, yani kovaryans negatiftir.
    Medyan alınıyor çünkü "1" gibi tek çubuklu rakamlarda kestirim dejenere olur.
    """
    egimler = []
    for (x1, y1, x2, y2) in kutular:
        alt = maske[y1:y2, x1:x2]
        ys, xs = np.nonzero(alt)
        if ys.size < 60:
            continue
        var_y = float(np.var(ys))
        if var_y < 4.0:
            continue
        kov = float(np.mean((xs - xs.mean()) * (ys - ys.mean())))
        egimler.append(-kov / var_y)

    if not egimler:
        return 0.0
    # Aşırı değerleri kırp: sağlıklı bir panelde yatıklık ±%35'i geçmez.
    return float(np.clip(np.median(egimler), -0.35, 0.35))


def _haneleri_bul(gri: np.ndarray, beklenen: int) -> list[tuple[int, int, int, int]]:
    """Hane kutuları, soldan sağa sıralı.

    Ondalık nokta da bir bileşendir ama boyu rakamların çok altındadır;
    `MIN_HANE_BOY_ORANI` onu eler — nokta hane sayılırsa tüm dizge kayar.
    """
    h, w = gri.shape[:2]
    maske = _segment_maskesi(gri)

    # Ondalık noktayı KAPAMA ÖNCESİ ele. Nokta küçük bir lekedir; kapama onu
    # komşu haneye yapıştırır ve o hanenin kutusu sola/sağa genişler. Kutu
    # kayınca segment örnekleme noktaları da kayar — ölçümde son hanenin
    # güveni sistematik olarak düşük çıkıyor, bazen hiç çözülemiyordu.
    # Ölçüt YÜKSEKLİK DEĞİL, iki boyutun BÜYÜĞÜ olmalı. Yükseklik kullanmak
    # yatay segmentleri (a, d, g) de siler: onlar hane eni kadar geniş ama
    # yalnızca segment kalınlığı kadar yüksektir. İlk denemede tam bu oldu ve
    # hane kutuları 71 px yerine 19 px çıktı.
    en_kucuk = 0.5 * MIN_HANE_BOY_ORANI * h
    n0, etiket0, ist0, _ = cv2.connectedComponentsWithStats(maske, 8)
    temiz = np.zeros_like(maske)
    for i in range(1, n0):
        if max(ist0[i, cv2.CC_STAT_WIDTH], ist0[i, cv2.CC_STAT_HEIGHT]) >= en_kucuk:
            temiz[etiket0 == i] = 255
    maske = temiz if np.count_nonzero(temiz) > 50 else maske

    # Bir hanenin segmentleri BAĞLANTISIZDIR; dikey kapama üst ve alt çubukları
    # tek bileşende toplar.
    cekirdek = cv2.getStructuringElement(cv2.MORPH_RECT,
                                         (max(3, w // 80), max(3, h // 10)))
    kapali = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, cekirdek)

    n, _, istatistik, _ = cv2.connectedComponentsWithStats(kapali, 8)
    adaylar = []
    for i in range(1, n):
        x, y, cw, ch, _ = istatistik[i]
        if ch < MIN_HANE_BOY_ORANI * h:
            continue
        if not (MIN_HANE_EN_ORANI * w <= cw <= MAX_HANE_EN_ORANI * w):
            continue
        adaylar.append((int(x), int(y), int(x + cw), int(y + ch)))

    adaylar.sort(key=lambda k: k[0])
    if len(adaylar) > beklenen:
        # Panel çerçevesinin kenarı da bileşen üretebilir; en uzunları tut.
        adaylar = sorted(sorted(adaylar, key=lambda k: k[3] - k[1], reverse=True)
                         [:beklenen], key=lambda k: k[0])

    if len(adaylar) == beklenen:
        return adaylar

    # --- Yedek yol: tüm bölgeyi eşit parçaya böl ---
    #
    # ⚠ Bu yol İSABETSİZDİR ve bilinen bir sınırdır. Denenip GERİ ALINAN
    # alternatif: bulunan hanelerin adımından ızgara türetip eksikleri sola
    # uzatmak. Varsayımı "eksik haneler baştadır" idi ve iki bitişik hane tek
    # bileşende birleştiğinde çöküyor — ölçümde dizge doğruluğu %93,3'ten
    # %45'e düştü. Kör bölme daha kötü değil, sadece daha öngörülebilir.
    #
    # Yedek yolun devreye girdiği başlıca durum EKSİ İŞARETİDİR: yalnızca orta
    # çubuk yanar, yüksekliği segment kalınlığı kadardır ve hane filtresine
    # takılır. Sonuç: negatif değerlerde güven ~0,75'e düşer ve DP-401'in 0,80
    # eşiğini geçemez — okuma DOĞRU çözülür ama reddedilir. Bu, yanlış okumaktan
    # iyidir (3. kural) ama kapsama kaybıdır.
    #
    # Kalıcı çözüm hane kutularını görüntüden değil TESPİTTEN almaktır: İP5
    # panelin kutusunu veriyor, hane sayısı envanterde yazılı, dolayısıyla
    # ızgara doğrudan kurulabilir. Zincire bağlandığında (İP13) yapılacak.
    ys, xs = np.nonzero(maske)
    if xs.size < 50:
        return adaylar
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    if (y2 - y1) < MIN_HANE_BOY_ORANI * h:
        return adaylar
    genislik = (x2 - x1) / beklenen
    return [(int(x1 + i * genislik), y1, int(x1 + (i + 1) * genislik), y2)
            for i in range(beklenen)]


def _segment_parlakliklari(gri: np.ndarray, kutu, egim: float = 0.0) -> dict[str, float]:
    """Her segmentin merkezindeki ortalama parlaklık.

    `egim` rakamın yatıklığı: örnekleme noktası, hanenin ALTINDAN yukarı
    çıktıkça sağa kaydırılır. Sıfır verilirse yatık panellerde üst segmentler
    kaçırılır.

    Segment merkezinden ORTALAMA alınıyor, maksimum değil: maksimum, komşu
    segmentin kenarından sızan tek bir parlak pikselle yanılır ve sönük bir
    segmenti yanık gösterir — "0"ı "8" yapan hata tam olarak budur.
    """
    x1, y1, x2, y2 = kutu
    w, h = x2 - x1, y2 - y1

    # Kutu, yatık rakamın TÜM yüksekliğini kapsadığı için rakamın kendi
    # genişliğinden `egim × yükseklik` kadar geniştir: altta sola, üstte sağa
    # taşar. Örnekleme oranları rakamın genişliğine göre olmalı.
    #
    # Bu ayrım atlandığında konumlar kutu genişliğine yayılıyor ve sağdaki
    # segmentler (b, c) rakamın dışına düşüyordu: `c` yanıkken 78 okunuyor,
    # sönük sanılıyordu. "3" ile "7", "4" ile "1" tam bu segmentle ayrılır.
    rakam_w = max(4.0, w - abs(egim) * h)
    sol_pay = abs(egim) * h if egim < 0 else 0.0
    pencere = max(2, int(min(rakam_w, h) * ORNEK_PENCERE_ORANI))

    cikti = {}
    for ad, (ox, oy) in SEGMENT_KONUMLARI.items():
        cy = y1 + oy * h
        cx = x1 + sol_pay + ox * rakam_w + egim * (y2 - cy)
        px1, py1 = int(cx - pencere / 2), int(cy - pencere / 2)
        px2, py2 = px1 + pencere, py1 + pencere
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(gri.shape[1], px2), min(gri.shape[0], py2)
        bolge = gri[py1:py2, px1:px2]
        cikti[ad] = float(bolge.mean()) if bolge.size else 0.0
    return cikti


def _panel_seviyeleri(gri: np.ndarray) -> tuple[float, float]:
    """Panelin YANIK ve SÖNÜK referans parlaklıkları. `(sonuk, yanik)` döner.

    Hane içi eşikleme tek başına yetmez: tüm segmentleri aynı durumda olan bir
    hane ("8" ya da boş) kendi içinde ayrım üretmez. Ayrımı ancak panelin
    genelinden gelen bir referans yapabilir.
    """
    otsu, _ = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yanik_px = gri[gri > otsu]
    kalan = gri[gri <= otsu]
    if yanik_px.size < 16 or kalan.size < 16:
        return float(gri.min()), float(gri.max())

    # Sönük seviye: kalanın üst dilimi (zemin değil, sönük segment).
    sonuk = float(np.percentile(kalan, 92))
    return sonuk, float(np.median(yanik_px))


def _aydinlatma_bantlari(gri: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Panel boyunca aydınlatma profili: bant başına zemin medyanı + genel medyan.

    Zemin = Otsu eşiğinin ALTINDA kalan pikseller (yanmayan yüzey). Bant başına
    medyan alınıyor; tek bozuk bant referansı savurmasın diye üçlü kayan
    medyanla yumuşatılıyor.
    """
    otsu, _ = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    zemin = gri <= otsu
    if int(zemin.sum()) < 64:
        return None, 0.0

    genel = float(np.median(gri[zemin]))
    if genel <= 1e-6:
        return None, 0.0

    w = gri.shape[1]
    kenar = np.linspace(0, w, BANT_SAYISI + 1).astype(int)
    bantlar = np.full(BANT_SAYISI, genel, dtype=float)
    for i in range(BANT_SAYISI):
        dilim = gri[:, kenar[i]:kenar[i + 1]]
        maske = zemin[:, kenar[i]:kenar[i + 1]]
        px = dilim[maske]
        if px.size >= 16:
            bantlar[i] = float(np.median(px))

    yumusak = bantlar.copy()
    for i in range(BANT_SAYISI):
        a, b = max(0, i - 1), min(BANT_SAYISI, i + 2)
        yumusak[i] = float(np.median(bantlar[a:b]))
    return yumusak, genel


def _yerel_seviyeler(bantlar: np.ndarray | None, genel: float, x_merkez: float,
                     genislik: int, sonuk: float, yanik: float) -> tuple[float, float]:
    """Panel referansını hanenin bulunduğu bandın aydınlatmasına ölçekler."""
    if bantlar is None or genel <= 1e-6 or genislik <= 0:
        return sonuk, yanik
    i = min(BANT_SAYISI - 1, max(0, int(x_merkez / genislik * BANT_SAYISI)))
    kazanc = float(np.clip(bantlar[i] / genel, BANT_KAZANC_ALT, BANT_KAZANC_UST))
    return sonuk * kazanc, yanik * kazanc


def _haneyi_coz(parlakliklar: dict[str, float], sonuk_ref: float,
                yanik_ref: float) -> tuple[str | None, float]:
    """Segment parlaklıklarını rakama çevirir. `(karakter, güven)` döner.

    **Eşik PANEL genelinden gelir, hane içinden değil.** İlk sürüm hane içi
    en düşük/en yüksek arasını ikiye bölüyordu; bu, hanenin kendi içinde ayrım
    üretmediği iki gerçek durumda çöker:

        "8"        yedi segmenti de yanık — hane tekdüze parlak
        boş hane   hiçbiri yanık değil    — hane tekdüze sönük

    Ölçümde her başarısızlık bu iki durumdan biriydi: 40 panelin 15'i
    okunamıyordu ve hepsinde bir "8" vardı. Panel referansına (sönük/yanık
    ortası) göre eşiklemek ikisini de doğal olarak çözer ve ışık değişimine
    de dayanıklıdır, çünkü referans her karede yeniden ölçülür.
    """
    panel_araligi = max(1.0, yanik_ref - sonuk_ref)
    esik = (sonuk_ref + yanik_ref) / 2.0

    yanik = {ad for ad, v in parlakliklar.items() if v > esik}
    karakter = DESEN_RAKAM.get(frozenset(yanik))
    if karakter is None:
        return None, 0.0

    # Her segmentin eşiğe uzaklığı, panel aralığına oranla — en kararsız segment
    # güveni belirler. "8" ile "0"ı ayıran tek bir segmenttir; o segment
    # kararsızsa sayı sessizce yanlış olur.
    marjlar = [abs(v - esik) / (panel_araligi / 2.0) for v in parlakliklar.values()]
    guven = float(np.clip(min(marjlar) / AYRIM_PAYI, 0.0, 1.0))
    return karakter, guven


def _noktayi_yerlestir(haneler: list[str], decimals: int) -> str:
    """Ondalık noktayı envanterdeki basamak sayısına göre koyar.

    Noktanın kendisi görüntüden okunmuyor: nokta küçük ve tek pikselli bir
    lekedir, düşük çözünürlükte kaybolur. Envanter panelin kaç ondalık
    gösterdiğini zaten söylüyor (2. kural — gösterge bilgisi YAML'dan gelir).
    """
    metin = "".join(haneler).strip()
    isaret = ""
    if metin.startswith("-"):
        isaret, metin = "-", metin[1:]
    if decimals <= 0 or len(metin) <= decimals:
        return isaret + metin
    return f"{isaret}{metin[:-decimals]}.{metin[-decimals:]}"


def read_digital(image: np.ndarray, gauge: Gauge) -> GaugeReading:
    """Dijital paneli okur ve `inspect/reading` gövdesi üretir.

    Hata yükseltmez; sorunlar `status` ile bildirilir.
    """
    if gauge.type != "digital":
        raise ValueError(f"{gauge.id}: read_digital sadece dijital panelde çalışır "
                         f"(tip: {gauge.type})")

    d = gauge.digits or {}
    count = int(d.get("count", 4))
    decimals = int(d.get("decimals", 1))
    esik = float((gauge.raw.get("reading") or {}).get("conf_threshold",
                                                      gauge.conf_threshold))

    gri = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kutular = _haneleri_bul(gri, count)
    egim = _egim_kestir(_segment_maskesi(gri), kutular)
    sonuk_ref, yanik_ref = _panel_seviyeleri(gri)

    def bos(status: str, conf: float = 0.0) -> GaugeReading:
        return GaugeReading(gauge_id=gauge.id, type=gauge.type, value=None,
                            unit=gauge.unit, conf=conf, status=status,
                            raw_angle=0.0, dial_angle=None)

    if not kutular:
        return bos(DURUM_OKUNAMADI)

    bantlar, zemin_genel = _aydinlatma_bantlari(gri)
    genislik = gri.shape[1]

    haneler, guvenler = [], []
    for kutu in kutular:
        s, y = _yerel_seviyeler(bantlar, zemin_genel, (kutu[0] + kutu[2]) / 2.0,
                                genislik, sonuk_ref, yanik_ref)
        ch, g = _haneyi_coz(_segment_parlakliklari(gri, kutu, egim), s, y)
        if ch is None:
            # Tek bir çözülemeyen hane tüm sayıyı geçersiz kılar: "12?4" diye
            # bir okuma yayınlanamaz, kısmi sayı yanlış sayıdan farksızdır.
            return bos(DURUM_OKUNAMADI)
        haneler.append(ch)
        guvenler.append(g)

    # Zincir güveni en zayıf haneden yüksek olamaz.
    guven = float(min(guvenler)) if guvenler else 0.0
    metin = _noktayi_yerlestir(haneler, decimals)

    if metin.startswith("-") and not gauge.allow_minus:
        # Panel eksi gösteremiyorsa, yanan o tek orta çubuk eksi işareti DEĞİLDİR
        # — kirli bir segment, bir yansıma ya da yarım yanmış bir hanedir. İşareti
        # atıp pozitif sayıyı yayınlamak, bilinen bir okuma hatasının üstüne
        # makul görünen bir değer koymaktır (3. kural).
        return bos(DURUM_OKUNAMADI, guven)

    try:
        deger = float(metin)
    except ValueError:
        return bos(DURUM_OKUNAMADI, guven)

    if guven < esik:
        return bos(DURUM_OKUNAMADI, guven)

    aralik = gauge.raw.get("range") or {}
    if aralik:
        alt, ust = float(aralik.get("min", -1e18)), float(aralik.get("max", 1e18))
        if not alt <= deger <= ust:
            # OCR sağlık kontrolü: panelin fiziksel aralığı dışındaki bir sayı
            # okuma hatasıdır, gerçek bir ölçüm değil.
            return GaugeReading(gauge_id=gauge.id, type=gauge.type, value=None,
                                unit=gauge.unit, conf=guven,
                                status=DURUM_KADRAN_DISI, raw_angle=0.0,
                                dial_angle=None)

    alarm = gauge.alarm or {}
    durum = DURUM_OK
    for anahtar, asiliyor in (("crit_above", lambda x: deger > x),
                              ("crit_below", lambda x: deger < x),
                              ("warn_above", lambda x: deger > x),
                              ("warn_below", lambda x: deger < x)):
        v = alarm.get(anahtar)
        if v is not None and asiliyor(float(v)):
            durum = DURUM_ALARM
            break

    return GaugeReading(gauge_id=gauge.id, type=gauge.type,
                        value=round(deger, decimals), unit=gauge.unit,
                        conf=guven, status=durum, raw_angle=0.0, dial_angle=None)
