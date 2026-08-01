"""Sentetik veri seti üreteci doğru ve TEKRAR ÜRETİLEBİLİR mi (İP3).

Buradaki en önemli test tekrar üretilebilirlik: aynı tohum aynı veri setini
vermezse İP6'nın "ortalama açı hatası 2.1°" ölçümü bir hafta sonra
doğrulanamaz — ve doğrulanamayan sayı ölçüm değildir.
"""

import json

import pytest

from gauge_vision.config import load_gauges
from gauge_vision.synth.generate import (
    LABELS_FILE,
    META_FILE,
    generate_dataset,
    load_labels,
)

SAYI = 12          # testler hızlı kalsın; davranış sayıdan bağımsız
GAUGES = load_gauges()


@pytest.fixture(scope="module")
def veri_seti(tmp_path_factory):
    yol = tmp_path_factory.mktemp("sentetik") / "v_test"
    ozet = generate_dataset(yol, count=SAYI, seed=0)
    return ozet


def test_istenen_sayida_goruntu_ve_etiket(veri_seti):
    kayitlar = load_labels(veri_seti.out_dir)
    assert len(kayitlar) == SAYI
    for k in kayitlar:
        assert (veri_seti.out_dir / k["file"]).exists(), f"eksik dosya: {k['file']}"


def test_ayni_tohum_ayni_veri_setini_veriyor(tmp_path):
    """Ölçümün tekrarlanabilirliği buna bağlı."""
    a = generate_dataset(tmp_path / "a", count=SAYI, seed=42)
    b = generate_dataset(tmp_path / "b", count=SAYI, seed=42)
    assert a.labels_path.read_text(encoding="utf-8") == b.labels_path.read_text(encoding="utf-8")


def test_farkli_tohum_farkli_veri_setini_veriyor(tmp_path):
    a = generate_dataset(tmp_path / "a", count=SAYI, seed=1)
    b = generate_dataset(tmp_path / "b", count=SAYI, seed=2)
    assert a.labels_path.read_text(encoding="utf-8") != b.labels_path.read_text(encoding="utf-8")


def test_etiketteki_aci_olcekle_tutarli(veri_seti):
    """Etiketin değeri ile açısı aynı kadrandan gelmiş olmalı."""
    for k in load_labels(veri_seti.out_dir):
        scale = GAUGES[k["gauge_id"]].scale
        assert k["angle_deg"] == pytest.approx(scale.angle_for_value(k["value"]))
        assert k["angle_img_deg"] == pytest.approx(k["angle_deg"] + k["roll_deg"])


def test_degerler_kadran_icinde(veri_seti):
    for k in load_labels(veri_seti.out_dir):
        scale = GAUGES[k["gauge_id"]].scale
        assert scale.min <= k["value"] <= scale.max


def test_degerler_kadrana_yayilmis(tmp_path):
    """Katmanlı örnekleme: değerler kadranın bir köşesine kümelenmemeli.

    Düz rastgele çekimde bu test şansa bağlı geçerdi; katmanlı örnekleme
    her dilimden bir örnek aldığı için garanti.
    """
    ozet = generate_dataset(tmp_path / "yayilim", count=30, seed=3, gauge_ids=["PT-101"])
    scale = GAUGES["PT-101"].scale
    oranlar = [(k["value"] - scale.min) / (scale.max - scale.min)
               for k in load_labels(ozet.out_dir)]
    assert min(oranlar) < 0.15, "kadranın alt ucundan örnek yok"
    assert max(oranlar) > 0.85, "kadranın üst ucundan örnek yok"


def test_gostergeler_dengeli_paylasilmis(veri_seti):
    sayilar = list(veri_seti.per_gauge.values())
    assert sum(sayilar) == SAYI
    assert max(sayilar) - min(sayilar) <= 1


def test_kadran_kutusu_goruntu_icinde(veri_seti):
    """Merkez kayması kadranı görüntü dışına taşırsa İP5'e yanlış kutu öğretiriz."""
    boyut = json.loads((veri_seti.out_dir / META_FILE).read_text(encoding="utf-8"))["image_size"]
    for k in load_labels(veri_seti.out_dir):
        x1, y1, x2, y2 = k["bbox_xyxy"]
        assert 0 <= x1 < x2 <= boyut, f"{k['file']}: kutu x ekseninde taşmış"
        assert 0 <= y1 < y2 <= boyut, f"{k['file']}: kutu y ekseninde taşmış"


def test_meta_uretimi_kayit_altina_aliyor(veri_seti):
    """Veri seti tohumsuz/tarihsiz durursa hangi ölçümün hangi veriye ait olduğu kaybolur."""
    meta = json.loads((veri_seti.out_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["seed"] == 0
    assert meta["count"] == SAYI
    assert "variation" in meta and "created" in meta


def test_etiket_dosyasi_satir_satir_json(veri_seti):
    satirlar = (veri_seti.out_dir / LABELS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(satirlar) == SAYI
    for s in satirlar:
        json.loads(s)          # her satır tek başına geçerli JSON


def test_bilinmeyen_gosterge_reddediliyor(tmp_path):
    with pytest.raises(ValueError, match="analog gösterge bulunamadı"):
        generate_dataset(tmp_path / "x", count=SAYI, gauge_ids=["YOK-1"])


def test_analog_olmayan_gosterge_reddediliyor(tmp_path):
    with pytest.raises(ValueError, match="analog gösterge bulunamadı"):
        generate_dataset(tmp_path / "x", count=SAYI, gauge_ids=["LM-501"])


def test_gosterge_sayisindan_az_kare_reddediliyor(tmp_path):
    with pytest.raises(ValueError, match="küçük olamaz"):
        generate_dataset(tmp_path / "x", count=2)
