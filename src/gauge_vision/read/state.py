"""İkaz lambası ve vana pozisyonu okur — İP12.

    from gauge_vision.read.state import read_state

    okuma = read_state(kare, gauge)      # gauge.type: "lamp" | "valve"
    okuma.value    # "green" | "red" | "off"  ·  "open" | "closed"

**Durum isimleri envanterden gelir, koda gömülmez** (2. kural). `gauges.yaml`
LM-501 için `off/green/red`, VL-601 için `open/closed` tanımlar; yeni bir renk
ya da pozisyon eklemek YAML'a satır eklemektir.

**Lamba — neden HSV ve neden doygunluk şart.**

Renk eşiği RGB'de yapılmaz: parlaklık değiştiğinde üç kanal birden kayar ve
"kırmızı" tanımı ışıkla birlikte kayar. HSV'de renk (H) parlaklıktan (V)
ayrışır. Ama H tek başına yetmez — SÖNÜK bir lambanın da bir H değeri vardır ve
gürültüden gelir. Ayırt edici olan doygunluktur (S): yanan lamba doygun,
sönük lamba gri. Bu yüzden karar sırası: önce "yanıyor mu" (V ve S), sonra
"hangi renk" (H).

**Vana — neden açı ve neden ±20°.**

Kol boru hattına paralelse açık, dikse kapalı. Ölçülen şey kolun açısıdır;
envanterdeki not "±20° içindeyse o duruma sayılır" der. Arada kalan açı
`unreadable` olur — yarı açık bir vana gerçek bir durumdur ve "açık" diye
yayınlanması tehlikelidir.

Kolun açısı, en büyük koyu bileşenin ikinci moment eksenlerinden (PCA)
çıkarılır. Hough çizgisi değil: kol kalın bir dikdörtgendir, Hough onun İKİ
kenarını iki ayrı çizgi olarak bulur ve hangisinin kol olduğu belirsiz kalır.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from gauge_vision.config import Gauge
from gauge_vision.read.calibrate import (
    DURUM_ALARM,
    DURUM_OK,
    DURUM_OKUNAMADI,
    GaugeReading,
)

# --- Lamba ---
# Yanık sayılmak için gereken doygunluk (0-255). Doygunluk ışık kazancına
# BAĞIMSIZDIR (S = (max-min)/max), bu yüzden mutlak eşik kullanılabilir.
LAMBA_MIN_DOYGUNLUK = 70

# Parlaklık için MUTLAK eşik KULLANILAMAZ — ilk sürümün en tehlikeli hatası
# buydu. `V > 90` kuralı ×0,15 kazançta yanan lambayı 35'e düşürüp "sönük"
# yapıyordu: ölçümde 180 karenin 60'ı SESSİZCE yanlış sınıflandı. Yanlış durum
# yayınlamak, okuyamadığını söylemekten çok daha tehlikelidir (3. kural).
#
# Ayırt edici özellik ışık seviyesi değil, KONTRAST: yanan lamba pano
# zemininden parlaktır, sönük lamba (koyu mercek) zeminden karanlıktır. Bu oran
# kazançla ölçeklenmez, dolayısıyla her ışıkta geçerlidir.
LAMBA_PARLAKLIK_ORANI = 1.25   # mercek / çevre parlaklık oranı
# Renk aralıkları (OpenCV H: 0-179). Kırmızı sarmalı için iki dilim.
RENK_ARALIKLARI: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 10), (170, 179)],
    "green": [(45, 85)],
    "yellow": [(20, 35)],
    "blue": [(95, 130)],
}
# Lamba bölgesi: kadran gibi bir yarıçap yok, karenin ortasındaki daire
# örnekleniyor. Gerçek zincirde bu bölgeyi İP5'in kutusu verir.
LAMBA_BOLGE_ORANI = 0.45

# --- Vana ---
VANA_TOLERANS_DEG = 20.0     # envanterdeki nota karşılık gelir
# Kolun bileşeni bu orandan küçükse gürültüdür.
VANA_MIN_ALAN_ORANI = 0.005
# Kol uzun ve ince olmalı: iki eksenin oranı bunun altındaysa şekil kol değil.
VANA_MIN_UZAMA = 2.0


def _lamba_bolgesi(image: np.ndarray) -> np.ndarray:
    """Karenin ortasındaki dairesel bölgenin maskesi."""
    h, w = image.shape[:2]
    maske = np.zeros((h, w), np.uint8)
    cv2.circle(maske, (w // 2, h // 2), int(min(h, w) * LAMBA_BOLGE_ORANI), 255, -1)
    return maske


def _cevre_bolgesi(image: np.ndarray) -> np.ndarray:
    """Merceğin DIŞINDA kalan halka — parlaklık referansı.

    Yanan lamba bu referanstan parlak, sönük lamba karanlıktır ve bu oran ışık
    kazancıyla ölçeklenmez. Referans olmadan mutlak eşik kullanmak zorunda
    kalınır; o da düşük ışıkta yanan lambayı "sönük" gösterir.
    """
    h, w = image.shape[:2]
    dis = np.zeros((h, w), np.uint8)
    cv2.circle(dis, (w // 2, h // 2), int(min(h, w) * 0.48), 255, -1)
    cv2.circle(dis, (w // 2, h // 2), int(min(h, w) * LAMBA_BOLGE_ORANI), 0, -1)
    return dis


def _lamba_durumu(image: np.ndarray, izinli: list[str]) -> tuple[str | None, float]:
    """Lambanın durumu ve güveni. `izinli` envanterdeki durum adlarıdır."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    maske = _lamba_bolgesi(image)
    cevre = _cevre_bolgesi(image)
    h_k, s_k, v_k = cv2.split(hsv)

    toplam = int(np.count_nonzero(maske))
    if toplam == 0 or int(np.count_nonzero(cevre)) == 0:
        return None, 0.0

    # Parlaklık eşiği ÇEVREDEN türetiliyor, sabit değil — bkz. modül başındaki
    # `LAMBA_PARLAKLIK_ORANI` notu.
    cevre_v = float(np.median(v_k[cevre > 0]))
    v_esik = max(12.0, cevre_v * LAMBA_PARLAKLIK_ORANI)

    # Parlak piksellerin oranı: lamba küçük olabilir, ortalama V yanıltır.
    parlak = (v_k > v_esik) & (s_k > LAMBA_MIN_DOYGUNLUK) & (maske > 0)

    oran = float(np.count_nonzero(parlak)) / toplam
    if oran < 0.04:
        # Yanan hiçbir bölge yok → sönük. Güven, ne kadar net söndüğünden.
        if "off" in izinli:
            return "off", float(np.clip(1.0 - oran / 0.04, 0.0, 1.0))
        return None, 0.0

    # Yanıyor: baskın rengi bul. Yalnızca envanterde tanımlı renkler aranır.
    h_parlak = h_k[parlak]
    skorlar: dict[str, float] = {}
    for ad in izinli:
        dilimler = RENK_ARALIKLARI.get(ad)
        if not dilimler:
            continue
        sayim = sum(int(np.count_nonzero((h_parlak >= a) & (h_parlak <= b)))
                    for a, b in dilimler)
        skorlar[ad] = sayim / h_parlak.size

    if not skorlar:
        return None, 0.0

    sirali = sorted(skorlar.items(), key=lambda kv: kv[1], reverse=True)
    en_iyi, en_iyi_skor = sirali[0]
    ikinci_skor = sirali[1][1] if len(sirali) > 1 else 0.0

    if en_iyi_skor < 0.35:
        # Hiçbir tanımlı renge yeterince benzemiyor — uydurma.
        return None, float(en_iyi_skor)

    # Güven: baskın renk ikinciden ne kadar ayrık. İki renk yarışıyorsa
    # (turuncu bir lamba kırmızı ile sarı arasında kalır) güven düşmeli.
    guven = float(np.clip(en_iyi_skor - ikinci_skor, 0.0, 1.0))
    return en_iyi, guven


def _kol_acisi(image: np.ndarray) -> tuple[float, float] | None:
    """Vana kolunun açısı ve uzama oranı. `(açı_deg, uzama)` döner.

    Açı konvansiyonu projedeki gibi: 0° = saat 3 yönü, CCW pozitif. Kol iki
    yönlü olduğu için açı 180° modunda anlamlıdır (yatay kol 0° ya da 180°).
    """
    gri = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ikili = cv2.threshold(gri, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n, etiket, ist, _ = cv2.connectedComponentsWithStats(ikili, 8)
    if n < 2:
        return None

    h, w = gri.shape[:2]
    en_buyuk, en_buyuk_alan = -1, 0
    for i in range(1, n):
        alan = int(ist[i, cv2.CC_STAT_AREA])
        if alan > en_buyuk_alan and alan > VANA_MIN_ALAN_ORANI * h * w:
            en_buyuk, en_buyuk_alan = i, alan
    if en_buyuk < 0:
        return None

    ys, xs = np.nonzero(etiket == en_buyuk)
    if xs.size < 20:
        return None

    # PCA: kovaryans matrisinin baskın özvektörü kolun uzun eksenidir.
    # Hough kullanılmıyor çünkü kalın bir kolun İKİ kenarını iki ayrı çizgi
    # olarak bulur ve hangisinin eksen olduğu belirsiz kalır.
    noktalar = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    noktalar -= noktalar.mean(axis=0)
    kov = np.cov(noktalar, rowvar=False)
    ozdeger, ozvektor = np.linalg.eigh(kov)
    if ozdeger[1] <= 1e-9:
        return None

    uzama = float(math.sqrt(max(ozdeger[1], 1e-9) / max(ozdeger[0], 1e-9)))
    vx, vy = ozvektor[:, 1]
    # y ekseni AŞAĞI arttığı için işaret çevriliyor (CLAUDE.md §3).
    aci = math.degrees(math.atan2(-vy, vx)) % 180.0
    return aci, uzama


def _vana_durumu(image: np.ndarray, izinli: list[str]) -> tuple[str | None, float]:
    """Vananın durumu ve güveni.

    Referans: kol YATAY ise (0° / 180°) boru hattına paralel → `open`,
    DİK ise (90°) → `closed`. Gerçek montajda bu eşleşme değişebilir; kalıcı
    çözüm envantere `open_angle` alanı eklemektir (bkz. docs/SORULAR.md).
    """
    sonuc = _kol_acisi(image)
    if sonuc is None:
        return None, 0.0
    aci, uzama = sonuc

    if uzama < VANA_MIN_UZAMA:
        # Şekil uzun ve ince değil — kol değil, gürültü ya da başka bir nesne.
        return None, 0.0

    # 180° modunda yataya ve dikeye uzaklık.
    yatay_fark = min(aci, 180.0 - aci)
    dikey_fark = abs(aci - 90.0)

    adaylar = []
    if "open" in izinli:
        adaylar.append(("open", yatay_fark))
    if "closed" in izinli:
        adaylar.append(("closed", dikey_fark))
    if not adaylar:
        return None, 0.0

    adaylar.sort(key=lambda kv: kv[1])
    en_iyi, en_iyi_fark = adaylar[0]
    ikinci_fark = adaylar[1][1] if len(adaylar) > 1 else 90.0

    # Kapı envanterden: tolerans dışındaki açı hiçbir duruma sayılmaz. Yarı
    # açık bir vana GERÇEK bir durumdur ve "açık" diye yayınlanması tehlikelidir.
    if en_iyi_fark > VANA_TOLERANS_DEG:
        return None, 0.0

    # Güven, doğru duruma ne kadar YAKIN olduğundan değil, DİĞER durumdan ne
    # kadar AYRIK olduğundan gelir.
    #
    # İlk sürüm `1 - fark/tolerans` kullanıyordu; bu, tolerans sınırında güveni
    # sıfıra indiriyor ve 0,70 eşiğiyle birlikte fiilî toleransı ±6°'ye
    # düşürüyordu. Envanter ±20° diyor, ölçüm ±6° yapıyordu — sessiz bir
    # uyuşmazlık. Ölçülen sonuç: temiz koşulda vana doğruluğu %63,3, 22 kare
    # boşuna reddedilmiş.
    #
    # İki durum 90° ayrık olduğuna göre, 20° sapmış bir kol hâlâ diğerinden
    # 70° uzaktır ve karışma riski yoktur. Güven bunu yansıtmalı.
    guven = float(np.clip((ikinci_fark - en_iyi_fark) / 45.0, 0.0, 1.0))
    return en_iyi, guven


def read_state(image: np.ndarray, gauge: Gauge) -> GaugeReading:
    """Lamba ya da vana durumunu okur; `inspect/reading` gövdesi üretir.

    `value` alanı burada SAYI DEĞİL DİZGEDİR (durum adı). Şema bunu kabul
    ediyor; tüketen taraf (Özgür'ün tur raporu) `type` alanına bakarak ayırır.
    """
    if gauge.type not in ("lamp", "valve"):
        raise ValueError(f"{gauge.id}: read_state sadece lamba/vana okur "
                         f"(tip: {gauge.type})")

    izinli = gauge.state_names
    if not izinli:
        raise ValueError(f"{gauge.id}: envanterde `states` tanımlı değil")

    if gauge.type == "lamp":
        durum, guven = _lamba_durumu(image, izinli)
    else:
        durum, guven = _vana_durumu(image, izinli)

    if durum is None or guven < gauge.conf_threshold:
        return GaugeReading(gauge_id=gauge.id, type=gauge.type, value=None,
                            unit=gauge.unit, conf=guven, status=DURUM_OKUNAMADI,
                            raw_angle=0.0, dial_angle=None)

    # Alarm durumu envanterden: `states` içinde `alarm: true` işaretli olan.
    alarmli = {s["name"] for s in gauge.states if s.get("alarm")}
    status = DURUM_ALARM if durum in alarmli else DURUM_OK

    return GaugeReading(gauge_id=gauge.id, type=gauge.type, value=durum,
                        unit=gauge.unit, conf=guven, status=status,
                        raw_angle=0.0, dial_angle=None)
