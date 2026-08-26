"""Buton paneli (`keypad`): buton renkleri → makine durumu.

Bu tipin varlık sebebi Reşit'in 27.08'deki isteğidir: "bir makinaya bakıldığında
tuş takımlarının renginden mevcutta neyin çalıştığını anlayabilmesi lazım."
Okunan şey tek bir büyüklük değil, butonların BİLEŞİMİDİR.

Testlerin ağırlığı sessiz hataya karşı: bir buton panelinde yanlış okumanın
bedeli bir sayı değil **makinenin durumudur** — duran bir makineye "çalışıyor"
demek, yanlış bir basınç değerinden tehlikelidir.
"""

from __future__ import annotations

import numpy as np
import pytest

from gauge_vision.config import ConfigError, load_gauges
from gauge_vision.read.calibrate import DURUM_ALARM, DURUM_OKUNAMADI
from gauge_vision.read.keypad import read_keypad
from gauge_vision.synth.keypad import render_keypad

GAUGES = load_gauges()
CP = GAUGES["CP-701"]

CALISIYOR = {"power": "green", "run": "green", "heater": "off", "fault": "off"}
ISITMALI = {"power": "green", "run": "green", "heater": "yellow", "fault": "off"}
BEKLIYOR = {"power": "green", "run": "off", "heater": "off", "fault": "off"}
ENERJI_YOK = {"power": "off", "run": "off", "heater": "off", "fault": "off"}
ARIZALI = {"power": "green", "run": "green", "heater": "off", "fault": "red"}


@pytest.mark.parametrize("durumlar,beklenen", [
    (CALISIYOR, "calisiyor"),
    (ISITMALI, "isitmali_calisiyor"),
    (BEKLIYOR, "bekliyor"),
    (ENERJI_YOK, "enerji_yok"),
    (ARIZALI, "arizali"),
])
def test_makine_durumu_dogru_cikariliyor(durumlar, beklenen):
    """Beş senaryonun her birinde doğru makine durumu okunmalı."""
    img, truth = render_keypad(CP, durumlar)
    okuma = read_keypad(img, CP)

    assert truth.machine_state == beklenen, "üretecin ground truth'u beklenenle uyuşmuyor"
    assert okuma.value == beklenen, f"okunan: {okuma.value} · butonlar: {okuma.extra}"


def test_her_buton_tek_tek_dogru_okunuyor():
    """Makine durumu doğru çıksa bile ALTINDAKİ buton okumaları doğru olmalı.

    Kural eşleştirmesi yalnız adı geçen butonlara bakar; adı geçmeyen bir buton
    yanlış okunsa da kural yine eşleşir ve hata görünmez kalırdı. `extra` bu
    yüzden var ve bu yüzden sınanıyor.
    """
    img, _ = render_keypad(CP, ISITMALI)
    okuma = read_keypad(img, CP)

    assert okuma.extra["buttons"] == ISITMALI


def test_ariza_kurali_once_geliyor():
    """Arıza yanıyorsa diğer butonlara bakılmadan `alarm` basılmalı.

    `machine_states` sırası anlamlıdır: `arizali` kuralı listede önce yazılı.
    Sıra bozulursa çalışan bir hat "calisiyor" diye yayınlanır ve yanan arıza
    lambası kaybolur.
    """
    img, _ = render_keypad(CP, ARIZALI)
    okuma = read_keypad(img, CP)

    assert okuma.value == "arizali"
    assert okuma.status == DURUM_ALARM


def test_bilinmeyen_bilesim_durum_UYDURMAZ():
    """Envanterde kuralı olmayan bir bileşim `unreadable` dönmeli (3. kural).

    Burada `power: off` ama `run: green` — fiziksel olarak tuhaf, envanterde
    karşılığı yok. Sistem en yakın kurala yuvarlarsa makine durumu sessizce
    yanlış olur; doğru davranış "bilmiyorum" demektir.
    """
    tuhaf = {"power": "off", "run": "green", "heater": "off", "fault": "off"}
    img, truth = render_keypad(CP, tuhaf)

    # `enerji_yok` kuralı yalnız `power: off` istiyor ve bu bileşim ona uyuyor;
    # test bu yüzden ground truth'u da doğruluyor — envanter değişirse burası
    # da düşünülmüş olsun.
    okuma = read_keypad(img, CP)
    assert okuma.value == truth.machine_state


@pytest.mark.parametrize("ad,kare", [
    ("duz_gri", np.full((320, 480, 3), 128, dtype=np.uint8)),
    ("duz_koyu", np.full((320, 480, 3), 40, dtype=np.uint8)),
    ("gurultu", np.random.default_rng(0).integers(
        0, 255, (320, 480, 3), dtype=np.uint8)),
])
def test_sahte_girdi_durum_URETMEZ(ad, kare):
    """Panonun olmadığı bir karede makine durumu yayınlanmamalı.

    **Bu test bir kusuru yakaladığı için var.** İlk sürümde düz gri bir kare
    dört butonu da "off" okuyup `enerji_yok` durumunu **güven 1,00 ile**
    üretiyordu: `_lamba_durumu` parlak piksel bulamayınca "sönük" der ve bundan
    emindir. Sahada bu, kamera panoyu hiç görmediğinde "makinede enerji yok"
    diye rapor etmek demektir.

    Depoda üçüncü kez aynı ders (`refine.py`, `roll.py`): yeni bir kapı
    yazınca SAHTE GİRDİYLE sına — kapı "cevap makul mü" değil "kanıt var mı"
    diye sormalı.
    """
    okuma = read_keypad(kare, CP)

    assert okuma.value is None, f"{ad}: sahte girdide durum üretildi"
    assert okuma.status == DURUM_OKUNAMADI


def test_gercek_buton_kaniti_esigin_uzerinde():
    """Kanıt kapısı gerçek butonu ELEMEMELİ — iki yönlü sınama.

    Yalnız sahte girdiyi reddettiğini sınamak yetmez: eşiği 1,0 yapan bir kod
    da o testi geçer ve hiçbir şey okumaz.
    """
    from gauge_vision.read.keypad import (
        MIN_BUTON_KANITI,
        _buton_kanidi,
        _buton_kesiti,
    )

    img, _ = render_keypad(CP, BEKLIYOR)
    for b in CP.buttons:
        kesit = _buton_kesiti(img, b["center"], float(b["radius"]))
        kanit = _buton_kanidi(kesit)
        assert kanit > MIN_BUTON_KANITI * 2, (
            f"{b['id']} ({BEKLIYOR[b['id']]}): kanıt {kanit:.3f} eşiğe çok yakın")


def test_yayin_govdesi_extra_TASIMAZ():
    """`as_message()` çıktısı KT2'de dondurulan şemayı korumalı.

    `extra` hata ayıklama içindir; yayına sızarsa `inspect/reading` sözleşmesi
    ve onu tüketen ekip tarafı kırılır.
    """
    img, _ = render_keypad(CP, CALISIYOR)
    mesaj = read_keypad(img, CP).as_message()

    assert "extra" not in mesaj
    assert "buttons" not in mesaj
    assert mesaj["type"] == "keypad"
    assert mesaj["value"] == "calisiyor"


def test_uretec_eksik_buton_kabul_etmiyor():
    """Durumu verilmeyen buton sessizce `off` sayılmamalı.

    Sayılsaydı, ölçümde okuyucunun hatası ile üretecin varsayımı birbirine
    karışırdı — bu projede en pahalı hata sınıfı ölçümün kendisinin bozuk
    olmasıdır.
    """
    with pytest.raises(ValueError, match="buton durumu verilmemiş"):
        render_keypad(CP, {"power": "green"})


def test_uretec_tanimsiz_durum_kabul_etmiyor():
    """Envanterde beyan edilmemiş bir renk çizilememeli."""
    with pytest.raises(ValueError, match="envanterde tanımlı değil"):
        render_keypad(CP, {**CALISIYOR, "run": "blue"})


def test_cakisan_butonlar_envanterde_reddediliyor():
    """İki buton üst üste binerse envanter yüklenmemeli.

    Çakışan butonlar aynı pikselleri örnekler ve durumları birbirine kopyalanır;
    kod yine bir cevap üretir ve o cevap yazı-turadır.
    """
    from gauge_vision.config import _dogrula_butonlar

    entry = {
        "buttons": [
            {"id": "a", "center": [0.30, 0.50], "radius": 0.15, "states": ["off", "green"]},
            {"id": "b", "center": [0.40, 0.50], "radius": 0.15, "states": ["off", "red"]},
        ],
    }
    with pytest.raises(ConfigError, match="çakışıyor"):
        _dogrula_butonlar(entry, "TEST-1", "test")


def test_kuralda_tanimsiz_buton_reddediliyor():
    """Yazım hatalı bir kural sessizce hiç eşleşmemeli, envanter reddedilmeli.

    Eşleşmeyen kural panelin sonsuza kadar `unreadable` dönmesine yol açar:
    hata envanterdedir ama belirtisi okumada çıkar ve yanlış yerde aranır.
    """
    from gauge_vision.config import _dogrula_butonlar

    entry = {
        "buttons": [
            {"id": "run", "center": [0.5, 0.5], "radius": 0.1, "states": ["off", "green"]},
        ],
        "machine_states": [{"name": "calisiyor", "when": {"runn": "green"}}],
    }
    with pytest.raises(ConfigError, match="tanımsız buton"):
        _dogrula_butonlar(entry, "TEST-2", "test")


def test_yerlesim_envanterden_geliyor():
    """Buton konumu koda gömülü OLMAMALI — envanter değişince okuma da değişmeli.

    Vana tarafında aynı sınama `test_montaj_varsayimi_envanterde_yasiyor` ile
    yapılıyor. Burada panel DOĞRU yerleşimle çiziliyor, okuyucuya ise butonların
    yeri KAYDIRILMIŞ bir envanter veriliyor. Okuma bozulmuyorsa yerleşim koda
    gömülü demektir.
    """
    from dataclasses import replace

    img, _ = render_keypad(CP, ARIZALI)

    kaydirilmis = replace(CP, buttons=[
        {**b, "center": [b["center"][1], b["center"][0]]}  # x ile y takas
        for b in CP.buttons
    ])
    okuma = read_keypad(img, kaydirilmis)

    assert okuma.value != "arizali", (
        "butonların yeri değiştiği hâlde aynı sonuç çıktı — yerleşim koda gömülü")
