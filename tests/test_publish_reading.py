"""`inspect/reading` yayını ve şema doğrulaması (`publish/reading.py`) — İP10.

Testlerin çoğu **şemaya uymayan mesajın yayınlanmadığını** sınar. Bozuk bir
mesajı brokera bırakmak, onu tüketen tarafta (Özgür'ün tur raporu) sessiz bir
hataya dönüşür; hatayı kaynağında durdurmak entegrasyonda gün kazandırır.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gauge_vision.config import load_gauges
from gauge_vision.publish.reading import (
    GECERLI_DURUMLAR,
    KONU,
    SEMA_SURUMU,
    ReadingPublisher,
    SemaHatasi,
    mesaj_dogrula,
    mesaj_kur,
)
from gauge_vision.read.calibrate import read_value
from gauge_vision.read.state import read_state
from gauge_vision.synth.state import render_lamp


@pytest.fixture(scope="module")
def gauges():
    return load_gauges()


@pytest.fixture
def okuma(gauges):
    """Gerçek bir okuma — elle kurulmuş sözlük değil.

    Şema testinin gerçek üretim yolundan geçmesi önemli: elle yazılmış bir
    sözlük, `as_message()` değiştiğinde sessizce eskir ve testi işe yaramaz
    hale getirir.
    """
    return read_value(gauges["PT-101"], angle_img_deg=90.0, confidence=0.95)


# --------------------------------------------------------------- mesaj kurma --

def test_mesaj_tam_alanlara_sahip(okuma):
    m = mesaj_kur(okuma, img_ref="frames/1.jpg", source="test")
    for alan in ("ts", "schema", "source", "img_ref", "gauge_id", "type",
                 "value", "unit", "conf", "status"):
        assert alan in m
    assert m["schema"] == SEMA_SURUMU
    mesaj_dogrula(m)


def test_zaman_damgasi_utc(okuma):
    """Yerel saat karşılaştırmayı sessizce bozar; ISO-8601 + UTC zorunlu."""
    m = mesaj_kur(okuma)
    assert m["ts"].endswith("+00:00") or m["ts"].endswith("Z")


def test_govde_okuma_katmanindan_geliyor(okuma):
    """Alan adları yayın katmanında ikinci kez tanımlanmamalı."""
    m = mesaj_kur(okuma)
    govde = okuma.as_message()
    for anahtar, deger in govde.items():
        assert m[anahtar] == deger


# ------------------------------------------------------------- doğrulamalar --

def test_eksik_alan_reddediliyor(okuma):
    m = mesaj_kur(okuma)
    for alan in ("ts", "schema", "gauge_id", "status", "conf"):
        bozuk = {k: v for k, v in m.items() if k != alan}
        with pytest.raises(SemaHatasi):
            mesaj_dogrula(bozuk)


def test_taninmayan_durum_reddediliyor(okuma):
    m = mesaj_kur(okuma)
    m["status"] = "belki"
    with pytest.raises(SemaHatasi):
        mesaj_dogrula(m)


def test_gecerli_durumlar_sozlesmedeki_dortlu():
    assert GECERLI_DURUMLAR == {"ok", "unreadable", "out_of_range", "alarm"}


@pytest.mark.parametrize("conf", [-0.1, 1.5, "yüksek", None])
def test_aralik_disi_guven_reddediliyor(okuma, conf):
    m = mesaj_kur(okuma)
    m["conf"] = conf
    with pytest.raises(SemaHatasi):
        mesaj_dogrula(m)


def test_okunamadi_ile_deger_birlikte_olamaz(okuma):
    """⚠ 3. kuralın şema seviyesindeki karşılığı.

    `status: unreadable` ile birlikte bir değer yayınlamak, tüketen tarafın o
    değeri kullanmasına yol açar — "okuyamadım" bilgisi böylece kaybolur.
    """
    m = mesaj_kur(okuma)
    m["status"] = "unreadable"
    with pytest.raises(SemaHatasi):
        mesaj_dogrula(m)


def test_ok_ile_bos_deger_birlikte_olamaz(okuma):
    m = mesaj_kur(okuma)
    m["value"] = None
    with pytest.raises(SemaHatasi):
        mesaj_dogrula(m)


def test_dusuk_guvenli_okuma_gecerli_mesaj_uretiyor(gauges):
    """Eşiğin altındaki okuma şemaya UYGUN olmalı: value null + unreadable."""
    dusuk = read_value(gauges["PT-101"], angle_img_deg=90.0, confidence=0.10)
    m = mesaj_kur(dusuk)
    mesaj_dogrula(m)
    assert m["value"] is None and m["status"] == "unreadable"


def test_lamba_durumu_dizge_olarak_gecerli(gauges):
    """Lamba/vanada `value` sayı değil durum adıdır; şema bunu kabul etmeli."""
    img, _ = render_lamp(gauges["LM-501"], "green")
    m = mesaj_kur(read_state(img, gauges["LM-501"]))
    mesaj_dogrula(m)
    assert m["value"] == "green" and m["type"] == "lamp"


def test_sema_surumu_uyusmazligi_reddediliyor(okuma):
    m = mesaj_kur(okuma)
    m["schema"] = SEMA_SURUMU + 1
    with pytest.raises(SemaHatasi):
        mesaj_dogrula(m)


# ------------------------------------------------------------------- yayıncı --

def test_dosya_moduna_dusuyor(okuma, tmp_path: Path):
    """Broker yoksa yayın ölçülemez hale gelmemeli."""
    y = ReadingPublisher(dosya_dizini=str(tmp_path), zorla_dosya=True)
    assert y.baglan() == "dosya"
    y.yayinla(okuma, img_ref="a.jpg")
    y.yayinla(okuma)
    y.kapat()

    satirlar = [json.loads(s) for p in tmp_path.glob("*.jsonl")
                for s in p.read_text(encoding="utf-8").splitlines()]
    assert len(satirlar) == 2
    for m in satirlar:
        mesaj_dogrula(m)


def test_bozuk_mesaj_yayinlanmiyor_ve_hata_yutulmuyor(gauges, tmp_path: Path):
    """Doğrulama başarısızsa mesaj gitmemeli VE çağıran haberdar olmalı.

    Sessizce atmak, yayının çalıştığı yanılsamasını yaratır.
    """
    from dataclasses import replace

    okuma = read_value(gauges["PT-101"], angle_img_deg=90.0, confidence=0.95)
    bozuk = replace(okuma, status="unreadable")   # değer dolu ama okunamadı

    y = ReadingPublisher(dosya_dizini=str(tmp_path), zorla_dosya=True)
    y.baglan()
    with pytest.raises(SemaHatasi):
        y.yayinla(bozuk)
    y.kapat()

    assert y.gonderilen == 0 and y.reddedilen == 1
    assert not any(p.read_text(encoding="utf-8").strip()
                   for p in tmp_path.glob("*.jsonl"))


def test_baglanmadan_yayin_hata_veriyor(okuma):
    y = ReadingPublisher(zorla_dosya=True)
    with pytest.raises(RuntimeError):
        y.yayinla(okuma)


def test_konu_sozlesmedeki_ad():
    assert KONU == "inspect/reading"


def test_context_manager(okuma, tmp_path: Path):
    with ReadingPublisher(dosya_dizini=str(tmp_path), zorla_dosya=True) as y:
        y.yayinla(okuma)
        assert y.mod == "dosya"
    assert y.mod == "kapalı"
