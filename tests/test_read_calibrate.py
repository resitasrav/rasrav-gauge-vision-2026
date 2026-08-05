"""Açı → değer dönüşümü ve okuma durumu doğru mu (İP7).

Testlerin belkemiği **gidiş-dönüş** iddiasıdır: `angle_for_value` ile çizilen
ibrenin açısı `value_for_angle`'a verildiğinde başlangıçtaki değer geri gelmelidir.
Bu, iki formülün birbirinin tersi olduğunu tek bir cümlede sınar — ve karekök
ölçekli kadranda üssün ters uygulanmasını da kapsar.

Beklenen değerlerin bir bölümü kâğıt üzerinde hesaplanmıştır; bir formülün kendi
formülüyle doğrulanması bilgi üretmez.
"""

import pytest

from gauge_vision.config import load_gauges
from gauge_vision.read.calibrate import (
    DURUM_ALARM,
    DURUM_KADRAN_DISI,
    DURUM_OK,
    DURUM_OKUNAMADI,
    read_value,
)

GAUGES = load_gauges()
ANALOG_IDS = [gid for gid, g in GAUGES.items() if g.type == "analog"]


# ------------------------------------------------------------- gidiş-dönüş --

@pytest.mark.parametrize("gid", ANALOG_IDS)
def test_gidis_donus_deger_korunuyor(gid):
    """value → açı → value: kadranın her yerinde başlangıç değeri geri gelmeli."""
    g = GAUGES[gid]
    aralik = g.scale.max - g.scale.min
    for i in range(21):
        deger = g.scale.min + aralik * i / 20
        aci = g.scale.angle_for_value(deger)
        geri = g.scale.value_for_angle(aci)
        assert geri == pytest.approx(deger, abs=aralik * 1e-6), \
            f"{gid} @ {deger:.3f}: açı {aci:.2f}° → {geri:.3f}"


def test_karekok_kadranda_ters_us_uygulaniyor():
    """FI-310'da doğrusal ters çevirme ıskalar; üs gerçekten tersleniyor mu.

    Kadranın yarısındaki açı (135° süpürme ilerlemiş) doğrusal kabulde 50 m³/h
    verir; karekök ölçekte doğrusu √0,5 × 100 = 70,7 m³/h'dir. Aradaki 20,7
    birimlik fark, `linear: false` alanının koda ulaşmadığında oluşacak sessiz
    hatanın büyüklüğüdür.
    """
    g = GAUGES["FI-310"]
    yarim_aci = g.scale.angle_min + g.scale.sweep_deg / 2   # ccw kadran

    assert g.scale.fraction_for_angle(yarim_aci) == pytest.approx(0.5)
    assert g.scale.value_for_angle(yarim_aci) == pytest.approx(70.71, abs=0.01)
    assert g.scale.value_for_angle(yarim_aci) != pytest.approx(50.0, abs=1.0)


def test_capalar_kagit_uzerinde_hesaplanan_degerlerde():
    """29.07'de elle hesaplanan çapa açıları ters yönde de tutmalı."""
    assert GAUGES["PT-101"].scale.value_for_angle(90.0) == pytest.approx(5.0)
    assert GAUGES["TI-205"].scale.value_for_angle(90.0) == pytest.approx(75.0)
    assert GAUGES["FI-310"].scale.value_for_angle(22.5) == pytest.approx(50.0, abs=0.01)


@pytest.mark.parametrize("gid", ANALOG_IDS)
def test_kadran_uclari(gid):
    g = GAUGES[gid]
    assert g.scale.value_for_angle(g.scale.angle_min) == pytest.approx(g.scale.min)
    assert g.scale.value_for_angle(g.scale.angle_max) == pytest.approx(g.scale.max)


# ------------------------------------------------- kadran dışı ve sarma tuzağı --

def test_kadranin_biraz_gerisi_devasa_deger_uretmiyor():
    """min'in 1° gerisindeki ibre, mod 360 yüzünden kadranın ötesi sanılmamalı.

    Sarma yapılmasaydı oran 1,33 çıkar, tolerans testini de geçemeyip
    `out_of_range` olurdu — ama yanlış SEBEPLE: ibre kadranın gerisinde diye
    değil, ötesinde diye. Uçtaki okumaların davranışı buna bağlı.
    """
    g = GAUGES["PT-101"]                      # cw: 225° → -45°
    oran = g.scale.fraction_for_angle(226.0)  # min'in 1° gerisi
    assert -0.02 < oran < 0.0, f"oran {oran:.3f} — sarma düzeltmesi çalışmıyor"


def test_kadran_disi_aci_hata_yukseltiyor():
    """PT-101 kadranı 225° → -45°; -80° ölü bölgede, max ucunun 35° ötesindedir."""
    g = GAUGES["PT-101"]
    with pytest.raises(ValueError, match="kadranın dışında"):
        g.scale.value_for_angle(-80.0)


def test_dayanaga_yaslanan_ibre_uca_kirpiliyor():
    """Süpürmenin ucundan 2° taşan ibre `ok` sayılıp uç değere kırpılmalı."""
    g = GAUGES["PT-101"]
    okuma = read_value(g, angle_img_deg=227.0)   # min'in 2° gerisi
    assert okuma.status == DURUM_OK
    assert okuma.value == pytest.approx(g.scale.min)


def test_kadranin_cok_disi_deger_uretmiyor():
    g = GAUGES["PT-101"]
    okuma = read_value(g, angle_img_deg=-120.0)
    assert okuma.status == DURUM_KADRAN_DISI
    assert okuma.value is None, "kadran dışında değer uydurulmamalı"


# ------------------------------------------------------- okuma katmanı kararları --

def test_dusuk_guvende_deger_yayinlanmiyor():
    """Eşiğin altında `unreadable` ve value None — İP15'in çekirdek davranışı."""
    g = GAUGES["PT-101"]
    okuma = read_value(g, angle_img_deg=90.0, confidence=g.conf_threshold - 0.01)
    assert okuma.status == DURUM_OKUNAMADI
    assert okuma.value is None
    assert okuma.raw_angle == pytest.approx(90.0), "ham açı yine de taşınmalı"


def test_esigin_tam_ustunde_okuma_yapiliyor():
    g = GAUGES["PT-101"]
    okuma = read_value(g, angle_img_deg=90.0, confidence=g.conf_threshold)
    assert okuma.status != DURUM_OKUNAMADI
    assert okuma.value == pytest.approx(5.0)


def test_yatiklik_duzeltiliyor():
    """Kamera yatıkken ham açı kayar, değer kaymamalı.

    Bu testin olmaması hâlinde yatık her karede okuma sessizce kayar; İP6'nın
    0,12°'lik hassasiyeti yatıklık düzeltilmediğinde hiçbir şey ifade etmez.
    """
    g = GAUGES["PT-101"]
    roll = 15.0
    okuma = read_value(g, angle_img_deg=90.0 + roll, roll_deg=roll)
    assert okuma.value == pytest.approx(5.0)
    assert okuma.raw_angle == pytest.approx(105.0)
    assert okuma.dial_angle == pytest.approx(90.0)

    kayitsiz = read_value(g, angle_img_deg=90.0 + roll)   # düzeltme uygulanmazsa
    assert kayitsiz.value != pytest.approx(5.0, abs=0.2)


def test_alarm_esikleri_envanterden_geliyor():
    """PT-101: warn_above 8,0 · TI-205: crit_above 120."""
    pt = GAUGES["PT-101"]
    assert read_value(pt, pt.scale.angle_for_value(5.0)).status == DURUM_OK
    assert read_value(pt, pt.scale.angle_for_value(8.5)).status == DURUM_ALARM

    ti = GAUGES["TI-205"]
    assert read_value(ti, ti.scale.angle_for_value(100.0)).status == DURUM_OK
    assert read_value(ti, ti.scale.angle_for_value(130.0)).status == DURUM_ALARM


def test_ondalik_envanterdeki_kadar():
    g = GAUGES["PT-101"]                       # decimals: 1
    okuma = read_value(g, g.scale.angle_for_value(4.7777))
    assert okuma.value == pytest.approx(4.8)


# ------------------------------------------------------------------ mesaj gövdesi --

def test_mesaj_sozlesmedeki_alanlari_tasiyor():
    """`inspect/reading` alan adları sözleşmeden; burada ikinci isim türetilmez."""
    g = GAUGES["PT-101"]
    mesaj = read_value(g, 90.0, confidence=0.88).as_message()
    assert set(mesaj) == {"gauge_id", "type", "value", "unit", "conf", "status", "raw_angle"}
    assert mesaj["gauge_id"] == "PT-101"
    assert mesaj["type"] == "analog"
    assert mesaj["unit"] == "bar"
    assert mesaj["value"] == pytest.approx(5.0)


def test_okunamayan_mesajda_deger_null():
    g = GAUGES["PT-101"]
    mesaj = read_value(g, 90.0, confidence=0.1).as_message()
    assert mesaj["value"] is None
    assert mesaj["status"] == DURUM_OKUNAMADI


def test_analog_olmayan_gosterge_reddediliyor():
    with pytest.raises(ValueError, match="sadece analog"):
        read_value(GAUGES["LM-501"], 90.0)
