"""Uçtan uca okuma zinciri: görüntü → tespit → kırp → açı → değer.

    from gauge_vision.pipeline import read_frame
    sonuc = read_frame(kare, model, gauge)
    sonuc.reading.value      # 7.9

İP5, İP6 ve İP7'yi birbirine bağlayan tek yer burasıdır. Ölçüm scripti
(`olc_zincir.py`) ile canlı demo (`canli_oku.py`) aynı kodu çalıştırsın diye
`src/` altında duruyor: demoda gördüğün sayı ile raporda yazan sayı aynı
hattan çıkmazsa ikisi de güvenilmez olur.

**Ölçüm scriptlerinden farkı:** `olc_ip6.py` ve `olc_ip7.py` kadranın merkezini
ve yarıçapını **etiketten** alır — orada ölçülen şey okuma yöntemidir. Burada
ikisi de **tespitten** gelir. Aradaki fark zincirin gerçek hatasıdır ve 05.08
ölçümüne göre 13 kattır (%0,129 → %1,72).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gauge_vision.config import Gauge
from gauge_vision.detect.perspective import Duzlestirme, duzlestir
from gauge_vision.detect.refine import refine_dial
from gauge_vision.read.calibrate import GaugeReading, read_value
from gauge_vision.read.needle import NeedleReading, read_needle_angle
from gauge_vision.read.roll import RollEstimate, estimate_roll

# Kadran yüzünün yarıçapı tespit kutusundan türetilir. Kutu bezeli de içerdiğinden
# ham yarının tamamı alınmaz: sentetik üreteçte dış yarıçap = kadran yarıçapı × 1,07
# (dial.BEZEL_WIDTH_RATIO). Fazla büyük yarıçap tarama halkasını kadranın dışına
# taşırır ve ana çizgiler ibre sanılabilir.
KUTU_YARICAP_ORANI = 1 / 1.07

# Kutu kare değilse (açılı bakış, kısmi örtme) kısa kenar esas alınır; uzun kenara
# göre alınan yarıçap kadranın dışını tarar.
MIN_YARICAP_PX = 12

# Tespit sınıfı → envanterdeki gösterge tipi. Dört sınıflı model (İP5
# genişletmesi) bu adları kullanır; tek sınıflı eski ağırlıklarda yalnız
# `gauge` vardır ve eşleme yine doğru çalışır.
#
# ⚠ `keypad` BİLEREK YOK. 27.08'de eklenen buton paneli tipinin okuyucusu
# hazır ama TESPİT sınıfı yok: yeni bir sınıf eklemek modeli yeniden eğitmek
# demektir ve o eğitim, sahadan gerçek pano fotoğrafı gelmeden yapılırsa yine
# kendi çizdiğimiz panoyu öğrenir. Bu yüzden buton paneli şimdilik kırpılmış
# görüntüde okunuyor (`canli_oku.py --tespitsiz`) — dijital, lamba ve vana da
# İP11/İP12'de tam olarak böyle doğrulanmıştı, tespit sonra geldi.
SINIF_TIP: dict[str, str] = {
    "gauge": "analog", "digital": "digital", "lamp": "lamp", "valve": "valve",
}


def _tipe_uyan_kutular(sonuc, gauge: Gauge):
    """`gauge.type` ile uyuşan kutuların indisleri, güvene göre azalan.

    **Neden sınıf filtresi gerekli.** Zincir eskiden karedeki EN GÜVENLİ kutuyu
    alıp beyan edilen göstergeymiş gibi okuyordu. Dört sınıflı model gelince bu
    sessiz bir hata kaynağı oldu: bir ikaz lambası, bir dijital panelden yüksek
    güvenle çıkabilir ve zincir lambanın kırpımını 7-segment çözücüye verir.
    Ölçülen örnek (14.08, üretilmiş panel videosu): aynı karede `digital`,
    `lamp` ve yanlış pozitif `valve` kutuları birlikte bulunuyor.

    Sınıf filtresi **kimlik doğrulaması değildir** ve öyle sunulmamalı: tipin
    doğru olması, kutunun beyan edilen GÖSTERGE olduğunu göstermez. Bir
    termometre de `gauge` sınıfındadır. Kimlik, robotun durağından beyanla
    gelir (U11) — 14.08'de yatıklık kanıtının kimlik ayrımı yapıp yapamadığı
    ölçüldü ve YAPAMADIĞI görüldü (doğru kimlik medyan ayrıklık 0,011; yanlış
    kimlik -0,103; dağılımlar örtüşüyor).
    """
    adlar = getattr(sonuc, "names", None) or {}
    kutular = sonuc.boxes
    sira = sorted(range(len(kutular)),
                  key=lambda i: float(kutular.conf[i]), reverse=True)
    if len(adlar) <= 1:
        return sira
    uyan = [i for i in sira
            if SINIF_TIP.get(adlar.get(int(kutular.cls[i]), ""), "") == gauge.type]
    return uyan


@dataclass(frozen=True)
class FrameResult:
    """Tek karenin zincir çıktısı. `reading` None ise okuma üretilememiştir."""

    box_xyxy: tuple[float, float, float, float] | None
    detect_conf: float
    center_px: tuple[int, int] | None
    radius_px: float
    needle: NeedleReading | None
    reading: GaugeReading | None
    reason: str = ""
    # Merkez kadran çemberinden rafine edilebildi mi? Rafine kapılardan geçemezse
    # kutu merkezinde kalınır ve bu False olur — ölçümde ikisi ayrılabilsin.
    center_refined: bool = False
    # Uygulanan yatıklık ve nereden geldiği. `roll` None ise kestirim yapılmamış
    # ya da başarısız olmuştur; `roll_deg` o zaman dışarıdan verilen değerdir.
    roll_deg: float = 0.0
    roll: RollEstimate | None = None
    # Perspektif düzeltmesi uygulanabildiyse dolu. `axis_ratio` 1'e yakınsa
    # kadrana zaten dik bakılıyor demektir.
    perspective: Duzlestirme | None = None

    @property
    def ok(self) -> bool:
        return self.reading is not None and self.reading.value is not None


def _bos(sebep: str, kutu=None, guven: float = 0.0) -> FrameResult:
    return FrameResult(box_xyxy=kutu, detect_conf=guven, center_px=None,
                       radius_px=0.0, needle=None, reading=None, reason=sebep)


# Tespit kutusundan gösterge yüzünü kırparken bırakılan pay. Kutu kadranı sıkı
# sarar; dijital panelde ve lambada kenar bilgisi (çerçeve, pano zemini) okuma
# için GEREKLİDİR — `read_state` yanık/sönük ayrımını çevre parlaklığına göre
# yapıyor, `read_digital` çerçeveyi eleyebilmek için onu görmek zorunda.
KIRPIM_PAYI = 0.12


def _kirp(image: np.ndarray, kutu, pay: float = KIRPIM_PAYI) -> np.ndarray:
    """Kutuyu payla birlikte kırpar. Kare dışına taşarsa sınırlara kırpılır."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = kutu
    dx, dy = (x2 - x1) * pay, (y2 - y1) * pay
    return image[max(0, int(y1 - dy)):min(h, int(y2 + dy)),
                 max(0, int(x1 - dx)):min(w, int(x2 + dx))]


@dataclass(frozen=True)
class Tespit:
    """Karedeki tek bir nesne — okunmadan önce, yalnız TİP düzeyinde."""

    box_xyxy: tuple[float, float, float, float]
    conf: float
    sinif: str          # modelin sınıf adı: gauge | digital | lamp | valve
    tip: str            # envanter tipi karşılığı: analog | digital | lamp | valve


def detect_objects(image: np.ndarray, model, *, conf: float = 0.25) -> list[Tespit]:
    """Karedeki bütün göstergeleri TİPİYLE döndürür — hiçbirini okumadan.

    `read_gauge` bir göstergeyi okur ve okumak için o göstergenin envanterdeki
    kalibrasyonuna ihtiyaç duyar. Ama karede envanterde olmayan göstergeler de
    bulunur ve onları görmezden gelmek yanıltıcıdır: 14.08 demosunda karede iki
    kadran varken ekranda tek kutu görünüyordu ve karşılaştırma haksız çıkıyordu.

    Bu yol tam olarak "sistemin dürüstçe bilebildiği kadarını" verir: karede ne
    var ve **ne tipte**. Hangi GÖSTERGE olduğunu söylemez — bunu görüntüden
    çıkarmak ölçümle denendi ve olmadığı görüldü (bkz. `_tipe_uyan_kutular`).
    """
    sonuc = model.predict(image, conf=conf, verbose=False)[0]
    adlar = getattr(sonuc, "names", None) or {}
    kutular = sonuc.boxes
    cikti = []
    for i in range(len(kutular)):
        ad = adlar.get(int(kutular.cls[i]), "gauge") if adlar else "gauge"
        cikti.append(Tespit(
            box_xyxy=tuple(float(v) for v in kutular.xyxy[i].tolist()),
            conf=float(kutular.conf[i]), sinif=ad,
            tip=SINIF_TIP.get(ad, "bilinmiyor")))
    cikti.sort(key=lambda t: t.conf, reverse=True)
    return cikti


@dataclass(frozen=True)
class AnalogKutuOkuma:
    """Kimliği bilinmeyen TEK analog kutunun geometrik okuması.

    Değer ve birim bilinçli olarak YOK: kalibrasyon (min/max, süpürme, birim)
    göstergenin kimliğine aittir ve kimlik görüntüden çıkarılamıyor (ölçüldü,
    bkz. `_tipe_uyan_kutular`). Kimliksiz kutuya envanterden bir kalibrasyon
    uygulamak "termometreyi bar okumak" sınıfı sessiz hata üretir — 26.08'de
    dört videoda fiilen gözlendi (devir saati "0,8 bar ok", termometre
    "2,2 bar ok"). Burada yalnızca görüntüden ölçülebilen söylenir: kadran
    çemberi ve ibre açısı.
    """

    box_xyxy: tuple[float, float, float, float]
    conf: float
    center_px: tuple[int, int] | None
    radius_px: float
    center_refined: bool
    needle: NeedleReading | None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.needle is not None


def read_all_analog(
    image: np.ndarray,
    model,
    *,
    conf: float = 0.25,
    tespitler: list["Tespit"] | None = None,
) -> list[AnalogKutuOkuma]:
    """Karedeki BÜTÜN analog kutulara daire rafinesi + ibre açısı uygular.

    `read_frame` tek (beyan edilen) göstergeyi değere çevirir; bu fonksiyon
    onun tamamlayıcısıdır: karede birden çok analog gösterge varken diğerleri
    görmezden gelinmesin diye HER analog kutu tek tek okunur. Çıktı açı
    düzeyindedir — değere çevirme, kimliği beyan edilen kutu için
    `read_frame`'de kalır.

    `tespitler` verilirse tespit tekrarlanmaz (demo aynı karede
    `detect_objects` zaten çağırıyor; modeli iki kez koşturmak kare hızını
    yarılar).
    """
    if tespitler is None:
        tespitler = detect_objects(image, model, conf=conf)

    cikti: list[AnalogKutuOkuma] = []
    for t in tespitler:
        if t.tip != "analog":
            continue
        merkez, yaricap = dial_from_box(t.box_xyxy)
        if yaricap < MIN_YARICAP_PX:
            cikti.append(AnalogKutuOkuma(t.box_xyxy, t.conf, None, 0.0, False,
                                         None, f"kadran çok küçük ({yaricap:.0f} px)"))
            continue
        daire = refine_dial(image, merkez, yaricap)
        rafine = daire is not None
        if rafine:
            merkez, yaricap = daire.center_px, daire.radius_px
        aci = read_needle_angle(image, merkez, yaricap, method="polar")
        cikti.append(AnalogKutuOkuma(
            t.box_xyxy, t.conf, merkez, yaricap, rafine, aci,
            "" if aci is not None else "ibre bulunamadı"))
    return cikti


def read_gauge(
    image: np.ndarray,
    model,
    gauge: Gauge,
    *,
    detect_conf: float = 0.25,
    esik: float | None = None,
    **analog_kw,
) -> FrameResult:
    """**Tipten bağımsız giriş noktası** — İP13'ün zincir birleştirmesi.

    Gösterge tipine göre doğru okuyucuya dallanır:

        analog   → read_frame (tespit → perspektif → merkez → yatıklık → açı → değer)
        digital  → read_digital (7-segment)
        lamp     → read_state (HSV)
        valve    → read_state (kol açısı)

    Dört tip de aynı `FrameResult`'ı döndürür ve `reading` alanı aynı
    `GaugeReading` gövdesidir; dolayısıyla yayın katmanı (İP10) tipi hiç
    bilmeden çalışır. Tip bilgisi **envanterden** gelir, görüntüden çıkarılmaz —
    hangi durakta hangi gösterge olduğunu robotun turu söyler (U11).

    Analog dışındaki tipler tespit kutusunu KIRPMA için kullanır; kadran
    geometrisi (merkez, yarıçap, ibre) onlarda anlamlı değildir.
    """
    if gauge.type == "analog":
        return read_frame(image, model, gauge, detect_conf=detect_conf,
                          esik=esik, **analog_kw)

    sonuc = model.predict(image, conf=detect_conf, verbose=False)[0]
    if len(sonuc.boxes) == 0:
        return _bos("gösterge bulunamadı")

    uyan = _tipe_uyan_kutular(sonuc, gauge)
    if not uyan:
        return _bos(f"karede {gauge.type} tipinde gösterge yok")

    en_iyi = uyan[0]
    kutu = tuple(float(v) for v in sonuc.boxes.xyxy[en_iyi].tolist())
    tespit_guveni = float(sonuc.boxes.conf[en_iyi])
    kesit = _kirp(image, kutu)
    if kesit.size == 0:
        return _bos("kırpım boş", kutu, tespit_guveni)

    if gauge.type == "digital":
        from gauge_vision.read.digital import read_digital
        okuma = read_digital(kesit, gauge)
    elif gauge.type in ("lamp", "valve"):
        from gauge_vision.read.state import read_state
        okuma = read_state(kesit, gauge)
    elif gauge.type == "keypad":
        from gauge_vision.read.keypad import read_keypad
        okuma = read_keypad(kesit, gauge)
    else:
        return _bos(f"desteklenmeyen tip: {gauge.type}", kutu, tespit_guveni)

    # Güven tespitle çarpılıyor — analog dalıyla aynı ilke: zincirin güveni en
    # zayıf halkasından yüksek olamaz.
    from dataclasses import replace as _replace
    okuma = _replace(okuma, conf=okuma.conf * tespit_guveni)

    return FrameResult(box_xyxy=kutu, detect_conf=tespit_guveni,
                       center_px=None, radius_px=0.0, needle=None,
                       reading=okuma)


def dial_from_box(box_xyxy) -> tuple[tuple[int, int], float]:
    """Tespit kutusundan kadran merkezi ve yarıçapı."""
    x1, y1, x2, y2 = box_xyxy
    merkez = (round((x1 + x2) / 2), round((y1 + y2) / 2))
    yaricap = min(x2 - x1, y2 - y1) / 2 * KUTU_YARICAP_ORANI
    return merkez, yaricap


def read_frame(
    image: np.ndarray,
    model,
    gauge: Gauge,
    *,
    detect_conf: float = 0.25,
    method: str = "polar",
    roll_deg: float | None = None,
    refine: bool = True,
    perspektif: bool = False,
    esik: float | None = None,
) -> FrameResult:
    """Karede göstergeyi bulur, ibresini ölçer, değere çevirir.

    Hata yükseltmez; her başarısızlık `reason` ile bildirilir. Zincir saha
    döngüsünde çalışacaktır, tek bir okunamayan kare turu düşürmemelidir.

    `roll_deg=None` (varsayılan) kamera yatıklığını kadranın çizgilerinden
    KESTİRİR (`read/roll.py`). Kestirim güvenilmezse 0 kabul edilir — yanlış bir
    yatıklık, düzeltmemekten daha kötüdür. Sayı verilirse kestirim yapılmaz;
    ölçümde ablasyon (0 = düzeltme yok, etiket = ideal düzeltme) böyle kurulur.

    `refine` kutu merkezini kadran çemberinden düzeltir (bkz. `detect/refine.py`).
    Ablasyon anahtarı olarak kapatılabilir — kazancı ölçülebilsin diye parametre.
    """
    sonuc = model.predict(image, conf=detect_conf, verbose=False)[0]
    if len(sonuc.boxes) == 0:
        return _bos("gösterge bulunamadı")

    # En güvenli UYUMLU kutu: karede birden çok gösterge olabilir, zincir tek
    # gösterge okur. Sınıfı beyan edilen tiple uyuşmayan kutular elenir; hangi
    # göstergenin okunacağı saha döngüsünde robotun durağıyla belirlenir (U11).
    uyan = _tipe_uyan_kutular(sonuc, gauge)
    if not uyan:
        return _bos(f"karede {gauge.type} tipinde gösterge yok")

    en_iyi = uyan[0]
    kutu = tuple(float(v) for v in sonuc.boxes.xyxy[en_iyi].tolist())
    tespit_guveni = float(sonuc.boxes.conf[en_iyi])

    merkez, yaricap = dial_from_box(kutu)
    if yaricap < MIN_YARICAP_PX:
        return _bos(f"kadran çok küçük ({yaricap:.0f} px)", kutu, tespit_guveni)

    # Perspektif düzeltmesi EN ÖNDE: sonraki her adım (merkez rafinesi, yatıklık,
    # ibre) düzleştirilmiş karede çalışmalı. Sonraya bırakılırsa her biri eğik
    # geometride ölçüm yapar ve düzeltmenin anlamı kalmaz.
    duzlestirme = None
    if perspektif:
        duzlestirme = duzlestir(image, merkez, yaricap)
        if duzlestirme is not None:
            image = duzlestirme.image
            merkez, yaricap = duzlestirme.center_px, duzlestirme.radius_px * KUTU_YARICAP_ORANI

    # Merkez rafinesi ibre ölçümünden ÖNCE: kutupsal tarama merkeze duyarlıdır,
    # düzeltmeyi sonradan uygulamak açıyı geriye dönük kurtarmaz.
    rafine_edildi = False
    if refine:
        daire = refine_dial(image, merkez, yaricap)
        if daire is not None:
            merkez, yaricap = daire.center_px, daire.radius_px
            rafine_edildi = True

    # Yatıklık ibreden BAĞIMSIZ ölçülür: kaynağı kadranın çizgileridir, ibre
    # tarama halkasının dışında kalır. Bu yüzden sırası önemli değil, ama
    # merkezden sonra gelmeli — çizgi halkası da merkeze göre taranıyor.
    yatiklik = None
    if roll_deg is None:
        yatiklik = estimate_roll(image, merkez, yaricap, gauge)
        roll_deg = yatiklik.roll_deg if yatiklik else 0.0

    aci = read_needle_angle(image, merkez, yaricap, method=method)
    if aci is None:
        return _bos("ibre bulunamadı", kutu, tespit_guveni)

    # Güvenler çarpılıyor: zincirin güveni en zayıf halkasından yüksek olamaz.
    # İP15'in `unreadable` eşiği bu birleşik sayıya uygulanacaktır.
    okuma = read_value(gauge, aci.angle_img_deg, roll_deg=roll_deg,
                       confidence=aci.confidence * tespit_guveni, esik=esik)

    return FrameResult(box_xyxy=kutu, detect_conf=tespit_guveni, center_px=merkez,
                       radius_px=yaricap, needle=aci, reading=okuma,
                       center_refined=rafine_edildi,
                       roll_deg=roll_deg, roll=yatiklik,
                       perspective=duzlestirme)
