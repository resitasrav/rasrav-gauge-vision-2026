"""Zor koşul üreteci (`synth/degrade.py`) — İP14'ün zemini.

Buradaki testlerin çoğu **ground truth'un bozulmayla birlikte doğru taşındığını**
sınar. Görüntüyü bozmak kolaydır; etiketi bozulmadan taşımak değildir. Etiket
yanlış taşınırsa İP14'ün tüm hata tablosu sessizce yanlış çıkar ve bu, sayılar
"makul" göründüğü için fark edilmez.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gauge_vision.config import load_gauges
from gauge_vision.synth.degrade import (
    EKSENLER,
    Bozulma,
    bozulmalar_uygula,
    perspektif_matrisi,
)
from gauge_vision.synth.dial import render_analog


@pytest.fixture(scope="module")
def kadran():
    return render_analog(load_gauges()["PT-101"], value=5.0)


def test_bozulma_yoksa_goruntu_degismiyor(kadran):
    """Kimlik durumu: `Bozulma()` hiçbir şeyi oynatmamalı.

    Ölçüm tablosunun ilk satırı (bozulmasız) referanstır; oradaki sayı zincirin
    temiz hâlinden farklıysa tüm tablo kayar.
    """
    img, truth = kadran
    yeni, t2 = bozulmalar_uygula(img, truth, Bozulma())
    assert np.array_equal(yeni, img)
    assert t2 == truth


def test_perspektif_degeri_degistirmiyor(kadran):
    """Kamera açısı değişince gösterge farklı bir değer göstermez.

    `value` ve `angle_deg` (kadran çerçevesi) sabit kalmalı; okunması gereken
    sayı odur. Yalnızca görüntüdeki geometri değişir.
    """
    img, truth = kadran
    _, t2 = bozulmalar_uygula(img, truth, Bozulma(egiklik_deg=30))
    assert t2.value == truth.value
    assert t2.angle_deg == truth.angle_deg


def test_perspektif_goruntu_acisini_degistiriyor(kadran):
    """`angle_img_deg` artık `angle_deg + roll` DEĞİLDİR.

    Bu ayrım İP14'ün varlık sebebi: naif okuyucu görüntüdeki açıyı ölçer, ama
    eğik bakışta o açı kadranın kendi açısı değildir. Fark ölçülmezse
    perspektifin maliyeti görünmez.
    """
    img, truth = kadran
    # İbre 90°'de (tam yukarı) simetri yüzünden kaymaz; 30° eğim + yan eksen seç.
    img2, _ = bozulmalar_uygula(img, truth, Bozulma())
    _, t2 = bozulmalar_uygula(img, truth, Bozulma(egiklik_deg=40, egiklik_yon_deg=55))
    assert abs(t2.angle_img_deg - truth.angle_img_deg) > 0.5


def test_perspektif_merkezi_tasiyor(kadran):
    """Merkez ve ibre ucu homografiyle birlikte gitmeli."""
    img, truth = kadran
    _, t2 = bozulmalar_uygula(img, truth, Bozulma(egiklik_deg=35, egiklik_yon_deg=20))
    assert t2.center_px != truth.center_px or t2.tip_px != truth.tip_px
    # Kadran karenin içinde kalmalı — testin kendisi anlamsızlaşmasın.
    assert 0 < t2.center_px[0] < img.shape[1]
    assert 0 < t2.center_px[1] < img.shape[0]


def test_perspektif_matrisi_sifir_egimde_birim(kadran):
    img, _ = kadran
    M = perspektif_matrisi(img.shape, 0.0)
    assert np.allclose(M / M[2, 2], np.eye(3), atol=1e-6)


@pytest.mark.parametrize("egim", [15, 30, 45])
def test_daire_elipse_donuyor(kadran, egim):
    """Eğik bakışta kadranın kutusu bir yönde kısalmalı — foreshortening.

    Afin bir kaydırma (shear) bunu üretmez; testin ayırt ettiği şey budur.
    """
    img, truth = kadran
    _, t2 = bozulmalar_uygula(img, truth, Bozulma(egiklik_deg=egim, egiklik_yon_deg=0))
    x1, y1, x2, y2 = t2.bbox_xyxy
    en, boy = x2 - x1, y2 - y1
    # yon=0 → yatay eksen etrafında dönme → dikey kısalma
    assert boy < en * math.cos(math.radians(egim)) * 1.25


def test_parlama_parlaklik_ekliyor(kadran):
    """Yansıma toplamalı: sahnenin üstüne ışık ekler, kontrastı ölçeklemez."""
    img, truth = kadran
    parlak, _ = bozulmalar_uygula(img, truth, Bozulma(parlama=0.8))
    assert parlak.mean() > img.mean()
    assert parlak.max() >= img.max()


def test_dusuk_isik_karartiyor(kadran):
    img, truth = kadran
    karanlik, _ = bozulmalar_uygula(img, truth, Bozulma(isik_kazanci=0.3))
    assert karanlik.mean() < img.mean() * 0.6


def test_bulaniklik_kenarlari_yumusatiyor(kadran):
    """Bulanıklık gradyan enerjisini düşürmeli — 90° civarında da.

    Önceki üreteçte `tan(90°)` patlıyor, çekirdekte tek piksel kalıyor ve
    görüntü hiç bulanmıyordu; yani "dik yönde bulanıklık" testi aslında
    bulanıklıksız görüntüyü ölçüyordu.
    """
    import cv2
    img, truth = kadran
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keskin = cv2.Laplacian(gri, cv2.CV_64F).var()

    for aci in (0, 45, 90, 135):
        bulanik, _ = bozulmalar_uygula(img, truth,
                                       Bozulma(bulaniklik_px=21, bulaniklik_aci=aci))
        b_gri = cv2.cvtColor(bulanik, cv2.COLOR_BGR2GRAY)
        assert cv2.Laplacian(b_gri, cv2.CV_64F).var() < keskin * 0.6, f"açı {aci}"


def test_jpeg_boyut_ve_tip_koruyor(kadran):
    img, truth = kadran
    sikisik, _ = bozulmalar_uygula(img, truth, Bozulma(jpeg_kalite=20))
    assert sikisik.shape == img.shape and sikisik.dtype == img.dtype


def test_ayni_tohum_ayni_sonuc(kadran):
    """Rastgele bileşenler (parlama yeri, gürültü) tohumlu olmalı."""
    img, truth = kadran
    b = Bozulma(parlama=0.6, isik_kazanci=0.4)
    a1, _ = bozulmalar_uygula(img, truth, b, np.random.default_rng(7))
    a2, _ = bozulmalar_uygula(img, truth, b, np.random.default_rng(7))
    assert np.array_equal(a1, a2)


def test_eksenler_ilk_seviyesi_bozulmasiz():
    """Her eksenin ilk seviyesi referans olmalı — tablo oradan okunuyor."""
    for eksen, seviyeler in EKSENLER.items():
        _, ilk = seviyeler[0]
        assert not ilk.etkin, f"{eksen} ekseninin ilk seviyesi bozulmasız değil"
