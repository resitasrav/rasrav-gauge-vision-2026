"""configs/gauges.yaml gerçekten yüklenebiliyor mu ve doğrulama tutuyor mu.

Bu testler envanterin sağlığını korur: birisi (ben dahil) YAML'a bozuk bir
satır eklerse İP6/İP7 saatler sonra tuhaf sayılarla değil, burada patlar.
"""

import pytest
import yaml

from gauge_vision.config import ConfigError, load_gauges


def test_envanter_yukleniyor():
    gauges = load_gauges()
    assert gauges, "envanter boş dönmemeli"
    assert "PT-101" in gauges, "referans gösterge PT-101 envanterde olmalı"


def test_dort_tip_de_temsil_ediliyor():
    """Şema H4'te (dijital/lamba/vana) yeniden tasarlanmasın diye hepsi başından var."""
    tipler = {g.type for g in load_gauges().values()}
    assert tipler == {"analog", "digital", "lamp", "valve"}


def test_supurme_acisi_makul():
    """Açı konvansiyonunun sayısal karşılığı: her analog kadran 0-350° arası süpürür."""
    for g in load_gauges().values():
        if g.type == "analog":
            assert 0 < g.scale.sweep_deg <= 350, f"{g.id}: süpürme {g.scale.sweep_deg}°"


def test_pt101_referans_degerleri():
    """PT-101 klasik 270° kadran — kalibrasyon buna göre oturtulacak (İP7)."""
    s = load_gauges()["PT-101"].scale
    assert (s.min, s.max) == (0.0, 10.0)
    assert s.direction == "cw"
    assert s.sweep_deg == pytest.approx(270.0)


def test_ccw_supurme_dogru_hesaplaniyor():
    """FI-310 ters yönde dönüyor; formülün yönü ayırt ettiğini gösterir."""
    s = load_gauges()["FI-310"].scale
    assert s.direction == "ccw"
    assert s.sweep_deg == pytest.approx(270.0)
    assert s.linear is False, "karekök ölçekli debimetre — İP7'de ayrı ele alınacak"


def test_conf_esikleri_gecerli():
    for g in load_gauges().values():
        assert 0.0 < g.conf_threshold <= 1.0


def test_bozuk_yon_beyanla_yakalaniyor(tmp_path):
    """Projedeki en sinsi hata: 'direction' yanlış yazılırsa geometri bunu göremez.

    225 → -45 arası ccw okunursa süpürme 90° çıkar; bu da "makul" bir sayı olduğu için
    kod çalışmaya devam eder ve TÜM okumalar sessizce yanlış olur. Tek savunma,
    envanterdeki `sweep_deg` beyanı ile çapraz kontrol.
    """
    bozuk = {
        "version": 1,
        "gauges": [{
            "id": "X-1", "name": "test", "type": "analog", "unit": "bar",
            "scale": {"min": 0, "max": 10, "angle_min": 225, "angle_max": -45,
                      "direction": "ccw",    # doğrusu cw idi
                      "sweep_deg": 270},     # beyan: 270 · hesaplanan: 90 → hata
        }],
    }
    p = tmp_path / "bozuk.yaml"
    p.write_text(yaml.safe_dump(bozuk), encoding="utf-8")

    with pytest.raises(ConfigError, match="beyan edilen süpürme"):
        load_gauges(p)


def test_ayni_aci_hata_veriyor(tmp_path):
    """angle_min == angle_max → süpürme 0°, kalibrasyon imkânsız."""
    bozuk = {
        "version": 1,
        "gauges": [{
            "id": "X-2", "name": "test", "type": "analog", "unit": "bar",
            "scale": {"min": 0, "max": 10, "angle_min": 90, "angle_max": 90,
                      "direction": "cw"},
        }],
    }
    p = tmp_path / "sifir.yaml"
    p.write_text(yaml.safe_dump(bozuk), encoding="utf-8")

    with pytest.raises(ConfigError, match="süpürme açısı"):
        load_gauges(p)


def test_beyan_edilen_supurme_gerceklesiyor():
    """Envanterdeki her analog göstergede sweep_deg beyanı var mı ve tutuyor mu."""
    for g in load_gauges().values():
        if g.type == "analog":
            assert g.scale.sweep_declared is not None, f"{g.id}: sweep_deg beyanı eksik"
            assert g.scale.sweep_deg == pytest.approx(g.scale.sweep_declared, abs=0.5)


def test_ayni_id_iki_kez_hata_veriyor(tmp_path):
    ikiz = {
        "version": 1,
        "gauges": [
            {"id": "L-1", "name": "a", "type": "lamp",
             "states": [{"name": "off"}, {"name": "green"}]},
            {"id": "L-1", "name": "b", "type": "lamp",
             "states": [{"name": "off"}, {"name": "red"}]},
        ],
    }
    p = tmp_path / "ikiz.yaml"
    p.write_text(yaml.safe_dump(ikiz), encoding="utf-8")

    with pytest.raises(ConfigError, match="iki kez"):
        load_gauges(p)
