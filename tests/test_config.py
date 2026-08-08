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


# --------------------------------------------------------------------------
# değer → açı  (İP3 ibreyi buna göre çizecek, İP7 bunun tersini alacak)
# --------------------------------------------------------------------------

def test_uc_degerler_beyan_edilen_acilara_oturuyor():
    """min → angle_min, max → angle_max. Formül YAML'ın iki ucunu da tutturmalı."""
    for g in load_gauges().values():
        if g.type != "analog":
            continue
        s = g.scale
        assert s.angle_for_value(s.min) == pytest.approx(s.angle_min), f"{g.id} alt uç"
        assert s.angle_for_value(s.max) == pytest.approx(s.angle_max), f"{g.id} üst uç"


def test_pt101_orta_deger_saat_12de():
    """0-10 bar'lık 270° kadranın ortası (5 bar) tam yukarıyı, yani 90°'yi göstermeli.

    Elle hesabı: 225° - (0.5 × 270°) = 90°. Çapa değeri bilerek elle seçildi —
    formülü kendi formülüyle doğrulamak bir şey kanıtlamaz.
    """
    s = load_gauges()["PT-101"].scale
    assert s.angle_for_value(5.0) == pytest.approx(90.0)
    assert s.angle_for_value(2.5) == pytest.approx(157.5)   # 225 - 67.5


def test_ti205_dar_kadranda_da_dogru():
    """240°'lik kadran: 75 °C ortada → 210° - 120° = 90°."""
    s = load_gauges()["TI-205"].scale
    assert s.angle_for_value(75) == pytest.approx(90.0)


def test_fi310_karekok_olcegi_dogrusaldan_ayriliyor():
    """Karekök ölçekli debimetrede ibre yarıda değil, süpürmenin 1/4'ünde olur.

    50 m³/h → oran (0.5)² = 0.25 → -45° + 67.5° = 22.5°.
    Doğrusal sansaydı 90° çıkardı; aradaki 67.5°'lik fark İP7'de doğrusal
    formülün bu göstergede neden ıskalayacağını sayısal olarak gösteriyor.
    """
    s = load_gauges()["FI-310"].scale
    assert s.angle_for_value(50) == pytest.approx(22.5)
    assert s.angle_for_value(50) != pytest.approx(90.0)


def test_ccw_kadranda_aci_artiyor():
    """FI-310 saat yönünün tersine dönüyor → değer arttıkça açı da artmalı."""
    s = load_gauges()["FI-310"].scale
    assert s.angle_for_value(80) > s.angle_for_value(20)


def test_kadran_disi_deger_reddediliyor():
    """11 bar'lık bir ibre 10 bar'lık kadrana çizilemez — sessizce kırpmak yerine patla."""
    s = load_gauges()["PT-101"].scale
    with pytest.raises(ValueError, match="kadran aralığı dışında"):
        s.angle_for_value(11.0)
    with pytest.raises(ValueError, match="kadran aralığı dışında"):
        s.angle_for_value(-0.1)


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


# ------------------------------------------------- vana: kol açısı beyanları --

def _vana_dosyasi(tmp_path, states, reading=None):
    doc = {"version": 1, "gauges": [
        {"id": "V-1", "name": "test vana", "type": "valve", "states": states,
         **({"reading": reading} if reading else {})}]}
    p = tmp_path / "v.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return p


def test_kol_acilari_envanterden_okunuyor():
    """VL-601'in beyanı `Gauge.state_angles` üzerinden görünmeli."""
    v = load_gauges()["VL-601"]
    assert v.state_angles == {"open": 0.0, "closed": 90.0}
    assert v.tolerance_deg == 20.0


def test_yarim_beyan_reddediliyor(tmp_path):
    """Bir durum açı beyan edip diğeri etmezse envanter yarı yarıya karışır."""
    p = _vana_dosyasi(tmp_path, [{"name": "open", "lever_angle": 0},
                                 {"name": "closed"}])
    with pytest.raises(ConfigError, match="ya hepsi ya hiçbiri"):
        load_gauges(p)


def test_ayirt_edilemeyen_acilar_reddediliyor(tmp_path):
    """0° ve 10° ±20° toleransla ayrılamaz — kod yine cevap üretirdi, o cevap
    yazı-tura olurdu. Çelişki okumada değil envanterde."""
    p = _vana_dosyasi(tmp_path, [{"name": "open", "lever_angle": 0},
                                 {"name": "closed", "lever_angle": 10}])
    with pytest.raises(ConfigError, match="ayırt edilemez"):
        load_gauges(p)


def test_kol_acisi_180_modunda_sarmaliyor(tmp_path):
    """Kol iki uçludur: 180° ile 0° aynı fiziksel duruştur."""
    p = _vana_dosyasi(tmp_path, [{"name": "open", "lever_angle": 180},
                                 {"name": "closed", "lever_angle": 90}])
    assert load_gauges(p)["V-1"].state_angles["open"] == 0.0


def test_gecersiz_tolerans_reddediliyor(tmp_path):
    p = _vana_dosyasi(tmp_path, [{"name": "open", "lever_angle": 0},
                                 {"name": "closed", "lever_angle": 90}],
                      reading={"tolerance_deg": 0})
    with pytest.raises(ConfigError, match="tolerance_deg"):
        load_gauges(p)


def test_allow_minus_varsayilani_acik():
    """Envanter susarsa negatif değere izin verilir; DP-401 açıkça true diyor."""
    assert load_gauges()["DP-401"].allow_minus is True
