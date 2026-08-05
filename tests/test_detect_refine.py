"""Kadran merkezi rafinesi (`detect/refine.py`).

Testlerin ayrıldığı iki soru:
  1. Kaydırılmış merkezi gerçekten geri buluyor mu? (fayda)
  2. Bulamadığında bunu itiraf ediyor mu? (zarar vermeme)

İkincisi daha önemli: rafine başarısız olduğunda `None` dönmezse kutu tahminini
sessizce bozar ve bu, projenin "yanlış okumaktansa okumama" kuralını deler.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gauge_vision.config import load_gauges
from gauge_vision.detect.refine import MAX_KAYMA_ORANI, refine_dial
from gauge_vision.synth.dial import DialLook, render_analog


@pytest.fixture(scope="module")
def kadran():
    """Tek bir sentetik kadran: (görüntü, merkez, yarıçap)."""
    gauge = load_gauges()["PT-101"]
    img, truth = render_analog(gauge, value=5.0)
    return img, truth.center_px, float(truth.radius_px)


def _kaydir(merkez, mesafe, aci_deg):
    a = math.radians(aci_deg)
    return (round(merkez[0] + mesafe * math.cos(a)),
            round(merkez[1] + mesafe * math.sin(a)))


def _sapma(bulunan, gercek) -> float:
    return math.hypot(bulunan[0] - gercek[0], bulunan[1] - gercek[1])


# ------------------------------------------------------------------ fayda --

@pytest.mark.parametrize("aci_deg", [0, 45, 120, 200, 300])
def test_kaydirilmis_merkez_geri_bulunuyor(kadran, aci_deg):
    """Her yönden kaydırılan merkez kadran çemberinden geri getirilir.

    Yön parametrik: tek yönde çalışan bir düzeltme, işaret hatasını gizleyebilir.
    """
    img, gercek, r = kadran
    kaba = _kaydir(gercek, 0.08 * 2 * r, aci_deg)     # kadran çapının %8'i

    daire = refine_dial(img, kaba, r)
    assert daire is not None
    assert _sapma(daire.center_px, gercek) < 0.2 * _sapma(kaba, gercek)


def test_dogru_merkez_bozulmuyor(kadran):
    """Merkez zaten doğruysa rafine onu kaydırmamalı."""
    img, gercek, r = kadran
    daire = refine_dial(img, gercek, r)
    assert daire is not None
    assert _sapma(daire.center_px, gercek) < 0.01 * 2 * r


def test_calisma_cozunurlugu_dogrulugu_belirler(kadran):
    """Düşük çalışma çözünürlüğü hatayı ARTIRIR — küçültme gürültü bastırmaz.

    Bu test bir varsayımı kilitliyor: piramidin buradaki işi maliyeti tavanlamak,
    kenar gürültüsünü elemek değil. Tersi olsaydı 96 px daha iyi çıkardı.
    """
    img, gercek, r = kadran
    kaba = _kaydir(gercek, 0.04 * 2 * r, 30)

    kaba_cozunurluk = refine_dial(img, kaba, r, calisma_px=96)
    ince_cozunurluk = refine_dial(img, kaba, r, calisma_px=320)
    assert kaba_cozunurluk is not None and ince_cozunurluk is not None
    assert _sapma(ince_cozunurluk.center_px, gercek) < _sapma(kaba_cozunurluk.center_px, gercek)


def test_ana_cizgiler_merkezi_cekmiyor(kadran):
    """Ana çizgiler halkanın içinde ama radyal; gradyan filtresi onları elemeli.

    Elenmeseydi merkez, çizgilerin yoğun olduğu tarafa (kadranın süpürme yayına)
    doğru sistematik kayardı — 270°'lik kadranda alt taraf çizgisizdir.
    """
    img, gercek, r = kadran
    daire = refine_dial(img, gercek, r)
    assert daire is not None
    # Kayma yönünden bağımsız olarak büyüklüğü çok küçük olmalı.
    assert daire.shift_px < 0.01 * 2 * r


# --------------------------------------------------------- zarar vermeme --

def test_rastgele_gurultu_reddediliyor():
    """Çember olmayan yerde `None` dönmeli — kanıt kalitesi kapısı budur.

    İlk sürüm bu testi geçemiyordu: kayma ve medyan kapıları gürültüde de
    sağlanıyordu, çünkü ikisi de "cevap makul mü" diye sorar, "kanıt var mı"
    diye değil (06.08).
    """
    rng = np.random.default_rng(0)
    for _ in range(10):
        gurultu = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        assert refine_dial(gurultu, (256, 256), 200.0) is None


def test_duz_zemin_reddediliyor():
    """Tek renk zeminde kenar yok; fit kurulamaz."""
    duz = np.full((256, 256, 3), 180, dtype=np.uint8)
    assert refine_dial(duz, (128, 128), 90.0) is None


def test_cok_kucuk_kadran_reddediliyor(kadran):
    """Yarıçap çalışma çözünürlüğünde birkaç piksele düşerse fit anlamsızdır."""
    img, gercek, _ = kadran
    assert refine_dial(img, gercek, 3.0) is None
    assert refine_dial(img, gercek, 0.0) is None


def test_kadranin_disina_bakinca_reddediliyor(kadran):
    """Kadran ROI'nin dışında kalırsa rafine kabul edilmemeli."""
    img, _, r = kadran
    kose = (int(r * 0.5), int(r * 0.5))
    assert refine_dial(img, kose, r) is None


def test_kayma_kapisi_asilmiyor(kadran):
    """Kabul edilen hiçbir rafine `MAX_KAYMA_ORANI`'nı geçmemeli."""
    img, gercek, r = kadran
    for aci in range(0, 360, 40):
        kaba = _kaydir(gercek, 0.09 * 2 * r, aci)
        daire = refine_dial(img, kaba, r)
        if daire is not None:
            assert daire.shift_px <= MAX_KAYMA_ORANI * r + 1.0


def test_kare_kenarindaki_kadran_cokmuyor():
    """ROI görüntü sınırına taşarsa kırpılır; hata yükseltilmez."""
    gauge = load_gauges()["PT-101"]
    img, truth = render_analog(
        gauge, value=5.0, look=DialLook(radius_ratio=0.30, center_offset_px=(150, 150)))
    sonuc = refine_dial(img, truth.center_px, float(truth.radius_px))
    assert sonuc is None or _sapma(sonuc.center_px, truth.center_px) < 0.05 * 2 * truth.radius_px


def test_yaricap_kutudan_geliyor(kadran):
    """Rafine yarıçapa dokunmaz — tek değişken merkezdir.

    Yarıçap da oynasaydı zincir hatasındaki değişimin hangisinden geldiği
    ayrılamazdı; ölçüm scripti tek kalemi izleyebilsin diye sabit tutuluyor.
    """
    img, gercek, r = kadran
    daire = refine_dial(img, gercek, r)
    assert daire is not None
    assert daire.radius_px == r
