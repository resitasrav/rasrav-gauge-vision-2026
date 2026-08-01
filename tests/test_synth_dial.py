"""Sentetik kadran çizici doğru geometriyi üretiyor mu (İP3).

Buradaki testlerin çoğu döndürülen sayıya değil **piksele** bakar: ground truth
"ibre 90°'de" diyorsa görüntüde de gerçekten orada olmalı. Etiket ile görüntü
birbirinden kayarsa İP6 kusursuz çalışsa bile hatalı ölçülür — ve bu, sessizce
yanlış sonuç üreten tam olarak o hata sınıfıdır.
"""

import math

import numpy as np
import pytest

from gauge_vision.config import load_gauges
from gauge_vision.synth.dial import CANVAS_PX, NEEDLE_LEN_RATIO, DialLook, render_analog

GAUGES = load_gauges()
ANALOG_IDS = [gid for gid, g in GAUGES.items() if g.type == "analog"]


def _aci_olc(center, nokta) -> float:
    """Piksel çiftinden açı — OpenCV'de y aşağı arttığı için sinüs terimi eksili."""
    return math.degrees(math.atan2(-(nokta[1] - center[1]), nokta[0] - center[0]))


def _aci_farki(a: float, b: float) -> float:
    """İki açı arasındaki en kısa fark (derece, işaretsiz)."""
    return abs((a - b + 180) % 360 - 180)


def _parlaklik(img, truth, aci: float) -> float:
    """İbrenin bulunması gereken yarıçapta, verilen açıdaki piksel parlaklığı."""
    r = truth.radius_px * NEEDLE_LEN_RATIO * 0.6
    x = round(truth.center_px[0] + r * math.cos(math.radians(aci)))
    y = round(truth.center_px[1] - r * math.sin(math.radians(aci)))
    return float(img[y, x].mean())


@pytest.mark.parametrize("gid", ANALOG_IDS)
def test_gorsel_boyutu_ve_tipi(gid):
    img, _ = render_analog(GAUGES[gid], GAUGES[gid].scale.min)
    assert img.shape == (CANVAS_PX, CANVAS_PX, 3)
    assert img.dtype == np.uint8


def test_ground_truth_acisi_olcekle_ayni():
    """truth.angle_deg elle değil, Scale üzerinden gelmeli — tek kaynak kuralı."""
    g = GAUGES["PT-101"]
    _, truth = render_analog(g, 5.0)
    assert truth.angle_deg == pytest.approx(g.scale.angle_for_value(5.0))
    assert truth.angle_deg == pytest.approx(90.0)


@pytest.mark.parametrize("gid", ANALOG_IDS)
def test_ibre_ucu_beyan_edilen_acida(gid):
    """Uç pikselinden geri hesaplanan açı, etiketteki açıyla aynı çıkmalı.

    y ekseninin işareti burada yakalanır: eksi unutulursa açı aynada
    yansımış gibi çıkar ve tüm sentetik etiketler sessizce bozulur.
    """
    g = GAUGES[gid]
    deger = (g.scale.min + g.scale.max) / 2
    _, truth = render_analog(g, deger)

    olculen = _aci_olc(truth.center_px, truth.tip_px)
    assert _aci_farki(olculen, truth.angle_img_deg) < 1.0, \
        f"{gid}: uç {olculen:.1f}°, etiket {truth.angle_img_deg:.1f}°"


@pytest.mark.parametrize("gid", ANALOG_IDS)
def test_ibre_gercekten_cizilmis(gid):
    """Etiketin gösterdiği yönde koyu piksel, 90° yanında açık piksel olmalı.

    Metadata doğru olup çizim yanlış olabilirdi; bu test ikisini birbirine bağlar.
    """
    g = GAUGES[gid]
    img, truth = render_analog(g, (g.scale.min + g.scale.max) / 2)

    assert _parlaklik(img, truth, truth.angle_img_deg) < 100, f"{gid}: ibre yönünde koyu piksel yok"
    assert _parlaklik(img, truth, truth.angle_img_deg + 90) > 200, f"{gid}: kadran beyaz değil"


def test_yatik_kamerada_iki_aci_ayriliyor():
    """Kamera yatıksa kadran çerçevesindeki açı sabit kalır, görüntüdeki kayar.

    Bu ayrım İP6/İP7 arasındaki sınır: İP6 görüntüden `angle_img_deg`'i ölçer,
    İP7 değere çevirmek için `angle_deg`'e ihtiyaç duyar. İkisi karıştırılırsa
    yatık her karede okuma sessizce kayar.
    """
    g = GAUGES["PT-101"]
    roll = 20.0
    img, truth = render_analog(g, 5.0, look=DialLook(roll_deg=roll))

    assert truth.angle_deg == pytest.approx(90.0), "kadran çerçevesi yatıklıktan etkilenmez"
    assert truth.angle_img_deg == pytest.approx(90.0 + roll)

    olculen = _aci_olc(truth.center_px, truth.tip_px)
    assert _aci_farki(olculen, truth.angle_img_deg) < 1.0

    # Görüntü gerçekten döndü mü: ibre yeni açıda koyu, eski açıda beyaz olmalı.
    assert _parlaklik(img, truth, truth.angle_img_deg) < 100
    assert _parlaklik(img, truth, truth.angle_deg) > 200


def test_varyasyon_goruntuyu_degistiriyor():
    """DialLook alanları çiziciye gerçekten geçiyor mu — sessizce yok sayılmasın."""
    g = GAUGES["PT-101"]
    temel, t_temel = render_analog(g, 5.0)
    farkli, t_farkli = render_analog(g, 5.0, look=DialLook(radius_ratio=0.30,
                                                          center_offset_px=(15, -10),
                                                          background_bgr=(150, 150, 150)))
    assert t_farkli.radius_px < t_temel.radius_px
    assert t_farkli.center_px != t_temel.center_px
    assert not np.array_equal(temel, farkli)


@pytest.mark.parametrize("gid", ANALOG_IDS)
def test_kadran_kutusu_goruntunun_icinde(gid):
    """bbox İP5'in (YOLO) etiketi olacak — görüntü dışına taşarsa işe yaramaz."""
    _, truth = render_analog(GAUGES[gid], GAUGES[gid].scale.max)
    x1, y1, x2, y2 = truth.bbox_xyxy
    assert 0 <= x1 < x2 <= CANVAS_PX
    assert 0 <= y1 < y2 <= CANVAS_PX


def test_analog_olmayan_gosterge_reddediliyor():
    with pytest.raises(ValueError, match="sadece analog"):
        render_analog(GAUGES["LM-501"], 1.0)


def test_kadran_disi_deger_cizilmiyor():
    with pytest.raises(ValueError, match="kadran aralığı dışında"):
        render_analog(GAUGES["PT-101"], 12.0)


def test_farkli_degerler_farkli_goruntu_uretiyor():
    """Aynı görüntüyü 100 kez üretip 'veri seti' sanmayalım."""
    a, _ = render_analog(GAUGES["PT-101"], 2.0)
    b, _ = render_analog(GAUGES["PT-101"], 8.0)
    assert not np.array_equal(a, b)
