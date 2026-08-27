"""Pano tipi metre: kare çerçeve, yay skala, kenardan dönen ibre (İP18).

27.08'de 3.mp4'te (Pioneer amplifikatörün iki dikdörtgen VU metresi) ölçülen
boşluk: 330 karenin hiçbirinde tespit yok, çünkü model bu geometriyi hiç
görmemişti. Okuma tarafında da üç varsayım birden kırılıyor — bu dosya o üçünü
tek tek sınıyor.

Yuvarlak kadran için doğru olan ve burada YANLIŞ olan varsayımlar:
    1. ibre kutunun ortasından döner        → pivot kenara yakın
    2. skala tam çemberdir                  → ~120°'lik yay
    3. yarıçap kutunun yarısıdır            → beyan edilen orandan gelir
"""

from __future__ import annotations

import numpy as np
import pytest

from gauge_vision.config import ConfigError, load_gauges
from gauge_vision.pipeline import dial_from_box
from gauge_vision.read.needle import angle_difference_deg, read_needle_angle
from gauge_vision.synth.panel import render_panel_meter

GAUGES = load_gauges()
PANEL = GAUGES["EM-501"]
YUVARLAK = GAUGES["PT-101"]


def _oku(gauge, deger: float, *, pencere: bool = True, envanter_pivot: bool = True):
    img, truth = render_panel_meter(gauge, deger)
    merkez, yaricap = dial_from_box(truth.bbox_xyxy, gauge if envanter_pivot else None)
    a0, a1 = gauge.scale.ccw_araligi
    okuma = read_needle_angle(img, merkez, yaricap, method="polar",
                              aci_penceresi=(a0, a1) if pencere else None)
    return okuma, truth


# --- envanter beyanı -----------------------------------------------------------

def test_yuvarlak_kadran_varsayilan_davranisi_KORUYOR():
    """`face` beyan etmeyen gösterge bugünkü davranışını aynen sürdürmeli.

    İP6'nın 0,123°'si ve İP7'nin %0,129'u penceresiz, kutu-merkezli koşuya ait.
    Yeni geometri desteği o sayıları geçersiz kılarsa kazanç değil kayıptır.
    """
    assert YUVARLAK.face_shape == "round"
    assert YUVARLAK.pivot_ratio == (0.5, 0.5)
    assert YUVARLAK.sweep_radius_ratio is None

    kutu = (100.0, 100.0, 300.0, 300.0)
    merkez, _ = dial_from_box(kutu, YUVARLAK)
    merkez_gaugesiz, _ = dial_from_box(kutu, None)
    assert merkez == merkez_gaugesiz == (200, 200)


def test_pivot_kutunun_ortasinda_DEGIL():
    assert PANEL.face_shape == "panel"
    kutu = (0.0, 0.0, 200.0, 100.0)
    merkez, _ = dial_from_box(kutu, PANEL)
    assert merkez == (100, 86), merkez          # pivot [0.5, 0.86]
    assert merkez != (100, 50)                  # kutu merkezi olsaydı burası


def test_pivot_ve_yaricap_KUTUDAN_birebir_geri_geliyor():
    """Üreteç ile okuyucunun geometri anlayışı aynı olmalı.

    Ayrışırlarsa ölçüm sessizce kendi hatasını ölçer: sentetik veride her şey
    tutar, sahada hiçbir şey tutmaz.
    """
    _, truth = render_panel_meter(PANEL, 0.5)
    merkez, yaricap = dial_from_box(truth.bbox_xyxy, PANEL)
    assert merkez == truth.pivot_px
    assert abs(yaricap - truth.sweep_radius_px) < 1.0


def test_ccw_araligi_cw_kadranda_TERSINE_ceviriyor():
    """`cw` kadranda açı azalır; yay CCW yönünde max'tan min'e uzanır.

    Düz çıkarma EM-501'de 120° yerine 240°'lik yayı verir ve pencerenin
    eleyeceği çerçeve tam o fazladan 120°'nin içinde kalır — 27.08'de ölçüldü,
    pencere açıkken bile hata 107,6°'de sabit kaldı.
    """
    a0, a1 = PANEL.scale.ccw_araligi
    assert (a0, a1) == (30.0, 150.0)
    assert (a1 - a0) % 360.0 == pytest.approx(PANEL.scale.sweep_deg)


# --- okuma ---------------------------------------------------------------------

@pytest.mark.parametrize("deger", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_yay_penceresiyle_aci_dogru_okunuyor(deger):
    okuma, truth = _oku(PANEL, deger)
    assert okuma is not None
    hata = abs(angle_difference_deg(okuma.angle_img_deg, truth.angle_img_deg))
    assert hata < 2.0, f"aci hatasi {hata:.2f}° (deger {deger})"


def test_PENCERESIZ_tarama_cerceveyi_ibre_saniyor():
    """Pencereyi kaldırmak hatayı geri getirmeli — kapının neden var olduğunu kilitler.

    Ölçülen (300 kare): pencere yokken ortalama açı hatası 107,6°, okumaların
    45'i 180° ters. Sebep: pivot alt kenara yakın olduğu için aşağı bakan
    ışınlar siyah ÇERÇEVEYE çarpıyor ve orada da kesintisiz koyu şerit var.
    """
    hatalar = []
    for deger in (0.1, 0.5, 0.9):
        okuma, truth = _oku(PANEL, deger, pencere=False)
        assert okuma is not None
        hatalar.append(abs(angle_difference_deg(okuma.angle_img_deg,
                                                truth.angle_img_deg)))
    assert min(hatalar) > 20.0, f"pencere olmadan da dogru okundu: {hatalar}"


def test_KUTU_MERKEZI_pivot_okumayi_kaydiriyor():
    """Yanlış pivot okumayı kırmaz, SESSİZCE kaydırır — en sinsi hata türü."""
    hatalar = []
    for deger in (0.1, 0.5, 0.9):
        okuma, truth = _oku(PANEL, deger, envanter_pivot=False)
        if okuma is None:
            continue
        hatalar.append(abs(angle_difference_deg(okuma.angle_img_deg,
                                                truth.angle_img_deg)))
    assert hatalar and max(hatalar) > 10.0, f"kutu merkeziyle de dogru cikti: {hatalar}"


# --- üreteç sınırları ----------------------------------------------------------

def test_yuvarlak_kadrani_panel_urecteciyle_cizmek_HATA():
    with pytest.raises(ValueError, match="panel"):
        render_panel_meter(YUVARLAK, 5.0)


def test_pivot_araligi_dogrulaniyor(tmp_path):
    """Aralık dışı pivot yazım hatasıdır ve sessiz kaydırma üretir."""
    from gauge_vision.config import _dogrula_face
    with pytest.raises(ConfigError, match="pivot"):
        _dogrula_face({"shape": "panel", "pivot": [0.5, 1.4]}, "X-1", "test")


def test_bilinmeyen_face_sekli_HATA():
    from gauge_vision.config import _dogrula_face
    with pytest.raises(ConfigError, match="face.shape"):
        _dogrula_face({"shape": "oval"}, "X-1", "test")


def test_ibre_gercekten_kutunun_icinde():
    """Çizim kutudan taşarsa etiket yanlış olur ve tespit eğitimi bozulur."""
    for deger in (0.0, 0.5, 1.0):
        _, truth = render_panel_meter(PANEL, deger)
        x1, y1, x2, y2 = truth.bbox_xyxy
        tx, ty = truth.tip_px
        assert x1 <= tx <= x2 and y1 <= ty <= y2, (deger, truth.tip_px, truth.bbox_xyxy)


def test_renderin_urettigi_aci_envanterle_tutuyor():
    """Ground truth üretecin kendi kabulünden değil, envanterin ölçeğinden gelmeli."""
    for deger in (0.0, 0.33, 1.0):
        _, truth = render_panel_meter(PANEL, deger)
        assert truth.angle_deg == pytest.approx(PANEL.scale.angle_for_value(deger))


def test_bos_face_beyani_yuvarlak_sayiliyor():
    from gauge_vision.config import _dogrula_face
    _dogrula_face({}, "X-1", "test")            # hata yükseltmemeli
    assert YUVARLAK.face == {}


def test_panel_metre_goruntusu_uretiliyor():
    img, truth = render_panel_meter(PANEL, 0.5)
    assert img.ndim == 3 and img.shape[2] == 3
    assert truth.sweep_radius_px > 0
    assert np.asarray(img).dtype == np.uint8
