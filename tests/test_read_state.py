"""Lamba ve vana durumu (`read/state.py`) — İP12.

En kritik test `test_dusuk_isikta_yanlis_sinif_uretmiyor`. Lamba okumasında
yanlış durum yayınlamak, okuyamadığını söylemekten çok daha tehlikelidir:
kırmızı (arıza) lambayı "off" okumak, arızayı görünmez yapar.
"""

from __future__ import annotations

import numpy as np
import pytest

from gauge_vision.config import load_gauges
from gauge_vision.read.state import VARSAYILAN_KOL_ACILARI, read_state
from gauge_vision.synth.degrade import Bozulma, bozulmalar_uygula
from gauge_vision.synth.dial import DialTruth
from gauge_vision.synth.state import render_lamp, render_valve


@pytest.fixture(scope="module")
def lamba():
    return load_gauges()["LM-501"]


@pytest.fixture(scope="module")
def vana():
    return load_gauges()["VL-601"]


def _boz(img, bozulma, rng=None):
    h, w = img.shape[:2]
    sahte = DialTruth(gauge_id="x", value=0.0, angle_deg=0.0, roll_deg=0.0,
                      angle_img_deg=0.0, center_px=(w // 2, h // 2), tip_px=(0, 0),
                      radius_px=min(h, w) // 3, bbox_xyxy=(0, 0, w, h))
    return bozulmalar_uygula(img, sahte, bozulma, rng or np.random.default_rng(0))[0]


# -------------------------------------------------------------------- lamba --

@pytest.mark.parametrize("durum", ["off", "green", "red"])
def test_lamba_durumu_geri_okunuyor(lamba, durum):
    img, _ = render_lamp(lamba, durum)
    okuma = read_state(img, lamba)
    assert okuma.value == durum


def test_sonuk_renkli_mercek_off_okunuyor(lamba):
    """Sönmüş kırmızı lamba siyah değil KOYU KIRMIZIDIR ve `off`tur.

    Renk bilgisine bakan bir okuyucu onu "red" sanar; ayırt edici olan
    parlaklıktır. Gerçek panolarda sönük mercek her zaman görünür.
    """
    for renk in ("red", "green"):
        img, _ = render_lamp(lamba, "off", renk=renk)
        assert read_state(img, lamba).value == "off"


def test_alarm_durumu_envanterden(lamba):
    """`states` içinde `alarm: true` işaretli durum `status: alarm` üretmeli."""
    img, _ = render_lamp(lamba, "red")
    okuma = read_state(img, lamba)
    assert okuma.status == "alarm"

    img, _ = render_lamp(lamba, "green")
    assert read_state(img, lamba).status == "ok"


def test_dusuk_isikta_yanlis_sinif_uretmiyor(lamba):
    """⚠ Bu testin koruduğu şey projenin 3. kuralıdır.

    İlk sürüm mutlak parlaklık eşiği (`V > 90`) kullanıyordu; ×0,15 kazançta
    yanan lamba 35'e düşüp "off" okunuyordu — ölçümde 180 karenin 60'ı SESSİZCE
    yanlış sınıflandı. Eşik artık çevre parlaklığına göre, yani ışık kazancından
    bağımsız.

    Test doğru okumayı ZORUNLU KILMIYOR: karanlıkta okuyamamak kabul edilebilir.
    Zorunlu olan, YANLIŞ bir durum yayınlamamaktır.
    """
    for durum in ("off", "green", "red"):
        for kazanc in (0.4, 0.25, 0.15):
            img, _ = render_lamp(lamba, durum)
            karanlik = _boz(img, Bozulma(isik_kazanci=kazanc))
            okuma = read_state(karanlik, lamba)
            assert okuma.value in (durum, None), \
                f"{durum} → {okuma.value} (kazanç {kazanc}) — sessiz yanlış sınıf"


def test_lamba_gurultude_susuyor(lamba):
    rng = np.random.default_rng(0)
    uretilen = sum(
        read_state(rng.integers(0, 255, (320, 320, 3), dtype=np.uint8),
                   lamba).value not in (None, "off")
        for _ in range(10)
    )
    assert uretilen == 0


# --------------------------------------------------------------------- vana --

@pytest.mark.parametrize("durum", ["open", "closed"])
def test_vana_durumu_geri_okunuyor(vana, durum):
    img, _ = render_valve(vana, durum)
    okuma = read_state(img, vana)
    assert okuma.value == durum


@pytest.mark.parametrize("sapma", [-18, -10, 0, 10, 18])
def test_tolerans_icindeki_sapma_kabul_ediliyor(vana, sapma):
    """Envanter "±20° içindeyse o duruma sayılır" diyor — ölçüm de öyle demeli.

    İlk sürümde güven `1 - fark/tolerans` idi; sınırda sıfıra iniyor ve 0,70
    eşiğiyle birlikte fiilî toleransı ±6°'ye düşürüyordu. Envanterle kod
    arasındaki bu sessiz uyuşmazlık ancak ölçümle görüldü (doğruluk %63,3).
    """
    img, _ = render_valve(vana, "open", sapma_deg=float(sapma))
    assert read_state(img, vana).value == "open"


@pytest.mark.parametrize("sapma", [30, 40, 45])
def test_arada_kalan_kol_okunmuyor(vana, sapma):
    """Yarı açık vana GERÇEK bir durumdur; "açık" diye yayınlanamaz."""
    img, _ = render_valve(vana, "open", sapma_deg=float(sapma))
    assert read_state(img, vana).value is None


def test_tolerans_siniri_envanterle_tutarli(vana):
    """Kabul edilen en büyük sapma envanterdeki `tolerance_deg` olmalı.

    Tolerans artık koddan değil YAML'dan geliyor; test de sabit bir sayıya
    değil envanterin kendi beyanına bakıyor. Beyan değişirse test onunla
    birlikte kayar — ayrışma imkânsız hâle gelir.
    """
    kabul = [s for s in range(0, 46)
             if read_state(render_valve(vana, "open", sapma_deg=float(s))[0],
                           vana).value is not None]
    assert max(kabul) <= vana.tolerance_deg + 1
    assert max(kabul) >= vana.tolerance_deg - 3


def test_montaj_varsayimi_envanterde_yasiyor(vana):
    """`lever_angle` takas edilince okuma da takas olmalı — kod değişmeden.

    S2'nin asıl riski buydu: montajda kol ters takılıysa vana kapalıyken
    "açık" yayınlanır ve hiçbir test bunu yakalayamaz, çünkü kod ile sentetik
    üreteç aynı varsayımı paylaşır. Bu test o paylaşımı KIRIYOR: görüntüyü
    üreteç kendi varsayımıyla çizerken okuyucuya ters envanter veriliyor.
    Cevap ters dönmüyorsa varsayım hâlâ koda gömülü demektir.
    """
    from dataclasses import replace

    ters = replace(vana, states=[
        {"name": "open", "lever_angle": 90},
        {"name": "closed", "lever_angle": 0},
    ])
    img, _ = render_valve(vana, "open")           # üreteç: kol yatay çiziliyor
    assert read_state(img, vana).value == "open"
    assert read_state(img, ters).value == "closed"


def test_beyansiz_envanterde_geri_dusus_calisiyor(vana):
    """Hiçbir durum açı beyan etmezse belgelenmiş varsayıma düşülür."""
    from dataclasses import replace

    beyansiz = replace(vana, states=[{"name": "open"}, {"name": "closed"}])
    assert beyansiz.state_angles == {}
    img, _ = render_valve(vana, "closed")
    assert read_state(img, beyansiz).value == "closed"
    assert VARSAYILAN_KOL_ACILARI["closed"] == 90.0


def test_vana_bos_karede_susuyor(vana):
    duz = np.full((320, 320, 3), 120, dtype=np.uint8)
    assert read_state(duz, vana).value is None


# ------------------------------------------------------------------- ortak --

def test_yanlis_tip_reddediliyor(lamba):
    img, _ = render_lamp(lamba, "green")
    with pytest.raises(ValueError):
        read_state(img, load_gauges()["PT-101"])


def test_mesaj_govdesi(lamba):
    img, _ = render_lamp(lamba, "green")
    m = read_state(img, lamba).as_message()
    assert m["type"] == "lamp" and m["value"] == "green"
    for alan in ("gauge_id", "type", "value", "conf", "status"):
        assert alan in m
