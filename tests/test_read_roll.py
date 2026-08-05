"""Kamera yatıklığının kadran çizgilerinden kestirimi (`read/roll.py`).

Bu modülün yanlış çalışması ÖZELLİKLE tehlikeli: yatıklık okumaya doğrudan
eklenir, dolayısıyla hatalı bir kestirim düzeltmemekten daha kötüdür — hatayı
azaltmak yerine başka bir yöne taşır. Testlerin çoğu bu yüzden "bulamayınca
None döndürüyor mu" sorusuna ayrılmıştır.
"""

from __future__ import annotations

import numpy as np
import pytest

from gauge_vision.config import load_gauges
from gauge_vision.read.roll import (
    MAX_ROLL_DEG,
    cached_tick_profile,
    estimate_roll,
    expected_tick_profile,
    measured_tick_profile,
)
from gauge_vision.synth.dial import DialLook, render_analog

ANALOG = ["PT-101", "TI-205", "FI-310"]


@pytest.fixture(scope="module")
def gauges():
    return load_gauges()


def _kare(gauge, deger, roll=0.0):
    img, truth = render_analog(gauge, value=deger, look=DialLook(roll_deg=roll))
    return img, truth


# ------------------------------------------------------------- beklenen desen --

@pytest.mark.parametrize("gid", ANALOG)
def test_beklenen_desen_cizgilerin_acisinda_tepe_yapiyor(gauges, gid):
    """Desen envanterden türetiliyor; tepeler ana çizgilerin açısında olmalı."""
    g = gauges[gid]
    profil = expected_tick_profile(g)
    majors, _ = g.tick_values()

    for v in majors:
        aci = int(round(g.scale.angle_for_value(v) % 360.0))
        # Tepe noktası kendi komşuluğunda en yüksek olmalı (±2°).
        pencere = [profil[(aci + d) % 360] for d in range(-2, 3)]
        assert profil[aci] == pytest.approx(max(pencere), rel=0.05)


def test_olu_bolge_bos(gauges):
    """270°'lik kadranın çizgisiz yayında desen sıfıra yakın olmalı.

    Bu yay olmasaydı eşit aralıklı desen periyodik olurdu ve korelasyon hangi
    çizgiye kilitlendiğini ayırt edemezdi.
    """
    g = gauges["PT-101"]
    profil = expected_tick_profile(g)
    # PT-101: 225° → -45°, ölü bölgenin ortası 270°.
    assert profil[270] < 0.01 * profil.max()


def test_onbellek_ayni_diziyi_veriyor(gauges):
    g = gauges["PT-101"]
    assert cached_tick_profile(g) is cached_tick_profile(g)
    assert np.allclose(cached_tick_profile(g), expected_tick_profile(g))


# ----------------------------------------------------------------- doğruluk --

@pytest.mark.parametrize("gid", ANALOG)
@pytest.mark.parametrize("roll", [-8.0, -3.5, 0.0, 3.5, 8.0])
def test_yatiklik_geri_bulunuyor(gauges, gid, roll):
    """İşaret ve büyüklük birlikte sınanıyor.

    İşaret ters olsaydı hata 2·roll çıkardı ve `abs()` ile bakan bir test bunu
    yakalayamazdı; bu yüzden fark doğrudan ölçülüyor.
    """
    g = gauges[gid]
    img, _ = _kare(g, deger=(g.scale.min + g.scale.max) / 2, roll=roll)

    est = estimate_roll(img, (256, 256), 205.0, g)
    assert est is not None
    assert est.roll_deg == pytest.approx(roll, abs=0.5)


@pytest.mark.parametrize("deger_orani", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ibre_kestirimi_etkilemiyor(gauges, deger_orani):
    """İbre nereyi gösterirse göstersin yatıklık aynı çıkmalı.

    Tarama halkası 0,84R'den başlıyor, ibre 0,78R'de bitiyor. Halka aşağı
    kaydırılsaydı kestirim ibrenin durduğu yöne çekilirdi ve hata OKUMAYLA
    İLİŞKİLİ olurdu — yani sistematik, fark edilmesi zor bir kayma.
    """
    g = gauges["PT-101"]
    deger = g.scale.min + deger_orani * (g.scale.max - g.scale.min)
    img, _ = _kare(g, deger=deger, roll=5.0)

    est = estimate_roll(img, (256, 256), 205.0, g)
    assert est is not None
    assert est.roll_deg == pytest.approx(5.0, abs=0.5)


def test_karekok_kadran_daha_ayrik_tepe_veriyor(gauges):
    """FI-310'un çizgileri eşit aralıklı değil; desen kendini tekrar etmez.

    Beklenen: karekök kadranda tepe oranı doğrusal kadrandan yüksek. Bu bir
    doğruluk testi değil, tasarım gerekçesinin ölçümle tutup tutmadığının
    kontrolü — tutmasaydı belirsizlik kapısının eşiği yeniden düşünülmeliydi.
    """
    kk = gauges["FI-310"]
    dg = gauges["PT-101"]
    assert not kk.scale.linear and dg.scale.linear

    img_kk, _ = _kare(kk, deger=(kk.scale.min + kk.scale.max) / 2, roll=4.0)
    img_dg, _ = _kare(dg, deger=(dg.scale.min + dg.scale.max) / 2, roll=4.0)

    e_kk = estimate_roll(img_kk, (256, 256), 205.0, kk)
    e_dg = estimate_roll(img_dg, (256, 256), 205.0, dg)
    assert e_kk is not None and e_dg is not None
    assert e_kk.peak_ratio > e_dg.peak_ratio


# ------------------------------------------------------- bulamayınca susmak --

def test_gurultude_kestirim_uretilmiyor(gauges):
    """Rastgele gürültüye desen uydurulmamalı."""
    g = gauges["PT-101"]
    rng = np.random.default_rng(0)
    uretilen = sum(
        estimate_roll(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8),
                      (256, 256), 205.0, g) is not None
        for _ in range(10)
    )
    assert uretilen == 0


def test_yanlis_gosterge_kimligi_yakalaniyor(gauges):
    """Kadrana BAŞKA bir göstergenin deseni sorulursa kestirim üretilmemeli.

    Beklenmeyen ama değerli bir yan fayda: gösterge kimliği şu an elle/waypoint
    ile geliyor ve waypoint sözlüğü henüz tanımsız (U11). Yanlış kimlik normalde
    SESSİZCE yanlış değer üretir; desen uyuşmazlığı bunun tek otomatik işareti.
    """
    img, _ = _kare(gauges["PT-101"], deger=5.0, roll=3.0)
    assert estimate_roll(img, (256, 256), 205.0, gauges["FI-310"]) is None


def test_duz_zeminde_kestirim_uretilmiyor(gauges):
    """Kontrast yoksa çizgi de yok."""
    duz = np.full((512, 512, 3), 200, dtype=np.uint8)
    assert estimate_roll(duz, (256, 256), 205.0, gauges["PT-101"]) is None


def test_sinirin_disindaki_yatiklik_reddediliyor(gauges):
    """Aranan aralığın dışındaki dönme düzeltilmeye çalışılmamalı.

    Pan-tilt platformu kadranı kabaca dik görür; 60°'lik bir "yatıklık" ya
    yanlış çizgiye kilitlenmedir ya da kadran zaten okunamayacak durumdadır.
    """
    g = gauges["PT-101"]
    img, _ = _kare(g, deger=5.0, roll=60.0)
    est = estimate_roll(img, (256, 256), 205.0, g)
    assert est is None or abs(est.roll_deg) <= MAX_ROLL_DEG


def test_kadran_disina_bakinca_kestirim_yok(gauges):
    """Yarıçap veya merkez saçmaysa profil çıkarılamaz."""
    g = gauges["PT-101"]
    img, _ = _kare(g, deger=5.0)
    assert estimate_roll(img, (256, 256), 0.0, g) is None
    assert measured_tick_profile(img, (-4000, -4000), 205.0) is None


def test_analog_olmayan_gosterge_reddediliyor(gauges):
    """Dijital panelin çizgi deseni yoktur; sessizce sayı üretilmemeli."""
    img = np.full((512, 512, 3), 200, dtype=np.uint8)
    dijital = [g for g in gauges.values() if g.type != "analog"]
    assert dijital, "envanterde analog olmayan gösterge kalmamış"
    assert estimate_roll(img, (256, 256), 205.0, dijital[0]) is None


def test_guven_araligi(gauges):
    """`confidence` 0-1 arası olmalı — İP15'in eşiğine girecek."""
    g = gauges["PT-101"]
    img, _ = _kare(g, deger=5.0, roll=3.0)
    est = estimate_roll(img, (256, 256), 205.0, g)
    assert est is not None and 0.0 <= est.confidence <= 1.0
