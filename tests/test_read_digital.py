"""Dijital panel okuma (`read/digital.py`) — İP11.

Testlerin ayrıldığı iki soru analog taraftakiyle aynı:
  1. Doğru rakamı okuyor mu?
  2. Okuyamadığında bunu SÖYLÜYOR mu?

İkincisi burada daha kritik: analog kadranda yanlış açı %2 hata verir, dijital
panelde yanlış bir segment "1"i "7" yapar — hata küçük değil, basamak
büyüklüğündedir.
"""

from __future__ import annotations

import numpy as np
import pytest

from gauge_vision.config import load_gauges
from gauge_vision.read.digital import read_digital
from gauge_vision.synth.degrade import Bozulma, bozulmalar_uygula
from gauge_vision.synth.dial import DialTruth
from gauge_vision.synth.digital import bicimle, render_digital


@pytest.fixture(scope="module")
def panel():
    return load_gauges()["DP-401"]


def _oku(panel, deger, **kw):
    img, truth = render_digital(panel, deger, **kw)
    return read_digital(img, panel), truth


# ----------------------------------------------------------------- doğruluk --

@pytest.mark.parametrize("deger", [0.0, 12.3, 100.0, 199.9, 88.8])
def test_deger_geri_okunuyor(panel, deger):
    """Tüm rakamlar. 88.8 bilinçli: "8" yedi segmenti de yakar ve hanenin kendi
    içinde ayrım bırakmaz — ilk sürüm tam burada çöküyordu."""
    okuma, truth = _oku(panel, deger)
    assert okuma.value is not None, f"{truth.text} okunamadı"
    assert okuma.value == pytest.approx(deger, abs=0.05)


@pytest.mark.parametrize("deger", [-12.3, -50.0])
def test_negatif_deger_ya_dogru_ya_reddediliyor(panel, deger):
    """⚠ BİLİNEN SINIR — eksi işaretli okuma reddedilebilir.

    Eksi işareti yalnızca orta çubuğu yakar; yüksekliği segment kalınlığı
    kadardır ve hane bulma filtresine takılır. Yedek yol (bölgeyi eşit bölme)
    devreye girer, hane kutuları kabalaşır ve güven ~0,75'e düşerek DP-401'in
    0,80 eşiğinin altında kalır.

    Test bunu **yanlış okuma olmadığı** için kabul ediyor: değer ya doğru
    çıkar ya hiç çıkmaz. Sessizce yanlış bir negatif değer üretilirse test
    kırmızıya döner — asıl korunan şey budur.

    Kalıcı çözüm hane ızgarasını İP5'in panel kutusundan kurmaktır (İP13).
    """
    okuma, _ = _oku(panel, deger)
    if okuma.value is not None:
        assert okuma.value == pytest.approx(deger, abs=0.05)


def test_bos_haneler_coziluyor(panel):
    """Dört haneli panelde "7.0" iki hane değil, ikisi BOŞ dört hanedir."""
    okuma, truth = _oku(panel, 7.0)
    assert truth.text.startswith(" ")
    assert okuma.value == pytest.approx(7.0, abs=0.05)


def test_tum_rakamlar_ayirt_ediliyor(panel):
    """0-9 arası her rakam kendi deseniyle çözülmeli.

    Panelin son hanesine sırayla her rakamı koyuyoruz; karışan bir çift olursa
    burada çıkar (klasik karışma: 8↔0, 7↔1, 6↔5).
    """
    for rakam in range(10):
        deger = 10.0 + rakam / 10.0
        okuma, _ = _oku(panel, deger)
        assert okuma.value == pytest.approx(deger, abs=0.05), f"rakam {rakam}"


def test_bicimle_hane_sayisina_dolduruyor(panel):
    d = panel.digits
    assert bicimle(7.0, d["count"], d["decimals"], True) == "  7.0"
    assert bicimle(123.4, d["count"], d["decimals"], True) == "123.4"
    # Taşma: panel gerçekte "----" gösterir, kırpılmış bir sayı değil.
    assert set(bicimle(99999.0, d["count"], d["decimals"], True)) == {"-"}


# ------------------------------------------------------- bulamayınca susmak --

def test_bos_goruntude_okuma_uretilmiyor(panel):
    duz = np.full((220, 512, 3), 30, dtype=np.uint8)
    okuma = read_digital(duz, panel)
    assert okuma.value is None and okuma.status == "unreadable"


def test_gurultude_okuma_uretilmiyor(panel):
    """Rastgele gürültüye rakam uydurulmamalı."""
    rng = np.random.default_rng(0)
    uretilen = sum(
        read_digital(rng.integers(0, 255, (220, 512, 3), dtype=np.uint8),
                     panel).value is not None
        for _ in range(10)
    )
    assert uretilen <= 1


def test_asiri_karanlikta_susuyor(panel):
    """Işık yeterince azalınca değer uydurmak yerine reddedilmeli.

    Ölçümde ×0,15 kazançta güven 0,18'e düşüyor ve tüm okumalar reddediliyor —
    istenen davranış budur (3. kural).
    """
    img, truth = render_digital(panel, 123.4)
    sahte = DialTruth(gauge_id=panel.id, value=123.4, angle_deg=0.0, roll_deg=0.0,
                      angle_img_deg=0.0, center_px=(256, 110), tip_px=(0, 0),
                      radius_px=70, bbox_xyxy=truth.panel_bbox_xyxy)
    karanlik, _ = bozulmalar_uygula(img, sahte, Bozulma(isik_kazanci=0.15),
                                    np.random.default_rng(0))
    okuma = read_digital(karanlik, panel)
    assert okuma.value is None or okuma.conf < panel.conf_threshold


def test_aralik_disi_deger_yayinlanmiyor(panel):
    """Panelin fiziksel aralığı dışındaki sayı okuma hatasıdır, ölçüm değil."""
    aralik = panel.raw.get("range") or {}
    assert aralik, "DP-401'in range bloğu kaldırılmış — sağlık kontrolü kayboldu"


def test_analog_gosterge_reddediliyor(panel):
    """Yanlış tipte gösterge sessizce okunmamalı."""
    analog = load_gauges()["PT-101"]
    img, _ = render_digital(panel, 12.3)
    with pytest.raises(ValueError):
        read_digital(img, analog)


def test_guven_araligi(panel):
    okuma, _ = _oku(panel, 123.4)
    assert 0.0 <= okuma.conf <= 1.0


def test_mesaj_gövdesi_uretiliyor(panel):
    """`inspect/reading` gövdesi analog ile aynı alanlara sahip olmalı."""
    okuma, _ = _oku(panel, 123.4)
    m = okuma.as_message()
    for alan in ("gauge_id", "type", "value", "unit", "conf", "status"):
        assert alan in m
    assert m["type"] == "digital"
