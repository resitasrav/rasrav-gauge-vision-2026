"""İbre açısı ölçümü doğru sayıyı üretiyor mu (İP6).

Testler sentetik kadranın ground truth'una karşı yazıldı: değer biz verdiğimiz
için doğru açı bilinmektedir. Ölçülen ile beklenen arasındaki eşikler bilinçli
olarak **dar** tutuldu — yöntem bozulduğunda test bunu ölçümden önce yakalasın.

Dikkat edilen nokta: karşılaştırma daima `angle_img_deg` iledir. `angle_deg` ile
karşılaştıran bir test, yatık kamerada yanlış olan bir yöntemi doğru sanardı.
"""

import numpy as np
import pytest

from gauge_vision.config import load_gauges
from gauge_vision.read.needle import (
    METHODS,
    NeedleReading,
    angle_difference_deg,
    read_needle_angle,
)
from gauge_vision.synth.dial import DialLook, render_analog

GAUGES = load_gauges()
ANALOG_IDS = [gid for gid, g in GAUGES.items() if g.type == "analog"]

# Temiz sentetik kadranda kabul edilen üst sınır. Ölçülen ortalamalar bunun çok
# altında (polar ~0,1° · Hough ~0,4°); eşik gürültü payıyla birlikte konuldu.
TOLERANS_DEG = 2.0


def _oku(gauge, value, yontem, look=None):
    img, truth = render_analog(gauge, value, look=look)
    okuma = read_needle_angle(img, truth.center_px, truth.radius_px, method=yontem)
    assert okuma is not None, f"{gauge.id} @ {value}: {yontem} okuma üretemedi"
    return okuma, truth


# ------------------------------------------------------------------- temel --

def test_aci_farki_daireyi_sariyor():
    """359° ile 1° arası 2°'dir, 358° değil.

    Sarma yapılmazsa kadranın başındaki okumalar hata tablosunda devasa
    görünür ve ortalama hata gerçekte olduğundan kötü çıkar.
    """
    assert angle_difference_deg(1.0, 359.0) == pytest.approx(2.0)
    assert angle_difference_deg(359.0, 1.0) == pytest.approx(-2.0)
    assert angle_difference_deg(90.0, 90.0) == pytest.approx(0.0)
    assert abs(angle_difference_deg(180.0, 0.0)) == pytest.approx(180.0)


@pytest.mark.parametrize("yontem", METHODS)
@pytest.mark.parametrize("gid", ANALOG_IDS)
def test_kadran_ortasinda_aci_dogru(yontem, gid):
    g = GAUGES[gid]
    okuma, truth = _oku(g, (g.scale.min + g.scale.max) / 2, yontem)
    hata = abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_img_deg))
    assert hata < TOLERANS_DEG, f"{gid} · {yontem}: {hata:.2f}° sapma"


@pytest.mark.parametrize("yontem", METHODS)
def test_kadran_boyunca_hic_180_atlamasi_yok(yontem):
    """İbre ile kuyruğu karıştırılırsa hata ~180° çıkar; her değerde kontrol edilir.

    Kuyruk ayrı bir "yakın" tehlike: yönü tam ters olduğu için üretilen değer
    kadranın öteki ucuna düşer ve sayı yine makul görünür.
    """
    g = GAUGES["PT-101"]
    for value in np.linspace(g.scale.min, g.scale.max, 12):
        okuma, truth = _oku(g, float(value), yontem)
        hata = abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_img_deg))
        assert hata < TOLERANS_DEG, f"{yontem} @ {value:.1f}: {hata:.1f}° — kuyruk mu okundu?"


@pytest.mark.parametrize("yontem", METHODS)
def test_kadran_ucunda_sarma_bozulmuyor(yontem):
    """FI-310 max'ta ibre 225°'de; dairesel ortalama burada patlamamalı."""
    g = GAUGES["FI-310"]
    okuma, truth = _oku(g, g.scale.max, yontem)
    assert abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_img_deg)) < TOLERANS_DEG


# ------------------------------------------------------- yatık kamera (kritik) --

@pytest.mark.parametrize("yontem", METHODS)
def test_yatik_kamerada_goruntu_acisi_okunuyor(yontem):
    """Yatıklıkta ölçülen `angle_img_deg`'e yakın, `angle_deg`'e uzak olmalı.

    Bu testin ikinci iddiası birincisinden önemli: yöntem kadran çerçevesindeki
    açıyı üretiyor olsaydı ilk iddia da geçerdi (roll küçükken fark azdır) ve
    hata yalnızca çok yatık karelerde ortaya çıkardı.
    """
    g = GAUGES["PT-101"]
    roll = 20.0
    okuma, truth = _oku(g, 5.0, yontem, look=DialLook(roll_deg=roll))

    goruntu_hatasi = abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_img_deg))
    kadran_hatasi = abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_deg))

    assert goruntu_hatasi < TOLERANS_DEG
    assert kadran_hatasi > roll / 2, "yöntem görüntü açısını değil kadran açısını ölçüyor"


# ------------------------------------------------------------------ varyasyon --

@pytest.mark.parametrize("yontem", METHODS)
@pytest.mark.parametrize("radius_ratio", [0.28, 0.42])
def test_kadran_boyu_degisince_calisiyor(yontem, radius_ratio):
    """Veri setindeki yarıçap aralığının iki ucu — sabit piksel eşiği kalmasın."""
    g = GAUGES["TI-205"]
    okuma, truth = _oku(g, 75.0, yontem, look=DialLook(radius_ratio=radius_ratio))
    assert abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_img_deg)) < TOLERANS_DEG


@pytest.mark.parametrize("yontem", METHODS)
def test_merkez_kaymasi_sorun_degil(yontem):
    g = GAUGES["PT-101"]
    okuma, truth = _oku(g, 3.0, yontem, look=DialLook(center_offset_px=(20, -15)))
    assert abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_img_deg)) < TOLERANS_DEG


@pytest.mark.parametrize("yontem", METHODS)
def test_ince_ve_kalin_ibre(yontem):
    g = GAUGES["PT-101"]
    for w in (0.75, 1.35):
        okuma, truth = _oku(g, 7.0, yontem, look=DialLook(needle_w_scale=w))
        assert abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_img_deg)) < TOLERANS_DEG


# ------------------------------------------------- okunamayan giriş / sözleşme --

def test_bos_goruntude_deger_uydurulmuyor():
    """İbre yoksa ya None ya da düşük güven — sayı uydurulmaz (3. kural).

    Düz gri karede "en koyu yön" diye bir şey yoktur; yöntem yine de bir açı
    üretiyorsa güveni yerlerde olmalı ki İP15 onu `unreadable` basabilsin.
    """
    bos = np.full((300, 300, 3), 200, dtype=np.uint8)
    for yontem in METHODS:
        okuma = read_needle_angle(bos, (150, 150), 120, method=yontem)
        assert okuma is None or okuma.confidence < 0.5, f"{yontem}: boş karede güven yüksek"


def test_temiz_kadranda_guven_yuksek():
    """Eşik ayarlanabilsin diye alt sınır: temiz kadran belirgin biçimde güvenli."""
    g = GAUGES["PT-101"]
    for yontem in METHODS:
        okuma, _ = _oku(g, 5.0, yontem)
        assert okuma.confidence > 0.5, f"{yontem}: temiz kadranda güven {okuma.confidence:.2f}"


def test_bilinmeyen_yontem_reddediliyor():
    img, truth = render_analog(GAUGES["PT-101"], 5.0)
    with pytest.raises(ValueError, match="bilinmeyen yöntem"):
        read_needle_angle(img, truth.center_px, truth.radius_px, method="sihir")


def test_gecersiz_yaricap_reddediliyor():
    img, truth = render_analog(GAUGES["PT-101"], 5.0)
    with pytest.raises(ValueError, match="yarıçap pozitif"):
        read_needle_angle(img, truth.center_px, 0)


def test_gri_goruntu_de_kabul_ediliyor():
    """Zincirin ilerisinde kırpım gri gelebilir; iki kanal sayısı da çalışmalı."""
    import cv2

    img, truth = render_analog(GAUGES["PT-101"], 5.0)
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    okuma = read_needle_angle(gri, truth.center_px, truth.radius_px)
    assert isinstance(okuma, NeedleReading)
    assert abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_img_deg)) < TOLERANS_DEG
