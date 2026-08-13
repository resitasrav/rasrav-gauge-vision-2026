"""YOLO etiket dönüşümü doğru mu (İP5).

Bu dosyanın varlık sebebi tek bir hata sınıfıdır: **kutu dönüşümü sessizce
yanlış olabilir.** YOLO normalize edilmiş MERKEZ + boyut ister, COCO ise sol-üst
köşe + boyut verir; ikisi karıştırıldığında eğitim çalışır, kayıp düşer, model
bir şeyler öğrenir — sadece kutular yarım kadran kaymış olur. Sonuç: yüksek
görünen bir mAP ve İP6'ya sistematik olarak kaymış merkezler.

Testler bu yüzden gidiş-dönüş üzerinden yazıldı: satırdan geri çözülen kutu
başlangıçtakiyle aynı olmalı.
"""

import pytest

from gauge_vision.detect.dataset import (
    GAUGE_SINIF_ID,
    IMAGES_DIR,
    LABELS_DIR,
    SINIFLAR,
    SINIFLAR_COK,
    sentetik_disa_aktar,
    veri_yaml_yaz,
    yolo_satiri,
)
from gauge_vision.synth.generate import generate_dataset


def _coz(satir: str, genislik: int, yukseklik: int):
    """YOLO satırını xyxy piksele geri çevirir — testin karşılaştırma tarafı."""
    sinif, cx, cy, w, h = satir.split()
    cx, cy, w, h = float(cx) * genislik, float(cy) * yukseklik, float(w) * genislik, float(h) * yukseklik
    return int(sinif), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ------------------------------------------------------------ kutu dönüşümü --

def test_gidis_donus_kutu_korunuyor():
    kutu = (100.0, 60.0, 300.0, 260.0)
    sinif, geri = _coz(yolo_satiri(kutu, 512, 512), 512, 512)
    assert sinif == GAUGE_SINIF_ID
    assert geri == pytest.approx(kutu, abs=0.05)


def test_merkez_gercekten_merkez():
    """Sol-üst köşe ile merkez karıştırılırsa bu test kırmızı olur."""
    _, cx, cy, w, h = yolo_satiri((0.0, 0.0, 100.0, 200.0), 1000, 1000).split()
    assert float(cx) == pytest.approx(0.05)    # 50/1000, köşe olsaydı 0.0
    assert float(cy) == pytest.approx(0.10)    # 100/1000
    assert float(w) == pytest.approx(0.10)
    assert float(h) == pytest.approx(0.20)


def test_kare_disina_tasan_kutu_kirpiliyor():
    """Taşan kutu kırpılmalı; kırpılmazsa merkez sistematik olarak dışa kayar."""
    _, geri = _coz(yolo_satiri((-40.0, -30.0, 200.0, 200.0), 512, 512), 512, 512)
    assert geri[0] == pytest.approx(0.0, abs=0.05)
    assert geri[1] == pytest.approx(0.0, abs=0.05)
    assert geri[2] == pytest.approx(200.0, abs=0.05)


def test_dikdortgen_olmayan_kutu_reddediliyor():
    with pytest.raises(ValueError, match="geçersiz kutu"):
        yolo_satiri((200.0, 100.0, 200.0, 300.0), 512, 512)   # genişlik sıfır

    with pytest.raises(ValueError, match="geçersiz kutu"):
        yolo_satiri((600.0, 100.0, 700.0, 300.0), 512, 512)   # tamamen kare dışında


def test_normalize_degerler_birimin_icinde():
    """YOLO 0-1 dışındaki değeri kabul etmez; sessizce yazmayalım."""
    _, cx, cy, w, h = yolo_satiri((10.0, 10.0, 500.0, 500.0), 512, 512).split()
    assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, w, h))


# ----------------------------------------------------------- sentetik aktarım --

def test_sentetik_aktarim_esli_dosya_uretiyor(tmp_path):
    """Her görüntünün etiketi olmalı; biri eksikse eğitim sessizce etiketsiz kalır."""
    veri = tmp_path / "sentetik"
    generate_dataset(veri, count=6, seed=3)

    hedef = tmp_path / "yolo"
    n = sentetik_disa_aktar(veri, hedef)

    goruntuler = sorted(p.stem for p in (hedef / IMAGES_DIR).glob("*.png"))
    etiketler = sorted(p.stem for p in (hedef / LABELS_DIR).glob("*.txt"))
    assert n == 6
    assert goruntuler == etiketler
    assert all(ad.startswith("syn_") for ad in goruntuler), "kaynak öneki yok"


def test_sentetik_etiket_kadranin_uzerine_dusuyor(tmp_path):
    """Etiketteki kutu gerçekten kadranı çevreliyor mu — merkez ibre merkezinde mi.

    Dosya eşleşmesi doğru olup kutunun yanlış görüntüye yazılması mümkündür;
    bu test etiketi ground truth'un merkeziyle karşılaştırarak onu yakalar.
    """
    import json

    veri = tmp_path / "sentetik"
    generate_dataset(veri, count=4, seed=1)
    hedef = tmp_path / "yolo"
    sentetik_disa_aktar(veri, hedef)

    boyut = json.loads((veri / "meta.json").read_text(encoding="utf-8"))["image_size"]
    with (veri / "labels.jsonl").open(encoding="utf-8") as f:
        for satir in f:
            k = json.loads(satir)
            ad = f"syn_{k['file'].split('/')[-1].removesuffix('.png')}"
            _, kutu = _coz((hedef / LABELS_DIR / f"{ad}.txt").read_text(encoding="utf-8").strip(),
                           boyut, boyut)
            merkez = ((kutu[0] + kutu[2]) / 2, (kutu[1] + kutu[3]) / 2)
            assert merkez == pytest.approx(tuple(k["center_px"]), abs=1.5), \
                f"{ad}: kutu merkezi kadran merkezinde değil"


# ------------------------------------------------------------------ veri yaml --

def test_veri_yaml_mutlak_yol_yaziyor(tmp_path):
    """Göreli yol, eğitim başka dizinden koşturulunca boş kümeye işaret eder."""
    yol = veri_yaml_yaz(tmp_path / "gauge.yaml",
                        train=tmp_path / "t", val=tmp_path / "v", test=tmp_path / "s")
    icerik = yol.read_text(encoding="utf-8")
    assert f"nc: {len(SINIFLAR)}" in icerik
    assert "gauge" in icerik
    for anahtar in ("train:", "val:", "test:"):
        deger = next(s.split(": ", 1)[1] for s in icerik.splitlines() if s.startswith(anahtar))
        assert deger.startswith(("/", tmp_path.drive.lower(), tmp_path.drive)), \
            f"{anahtar} göreli yazılmış: {deger}"


# ------------------------------------------------------- dört sınıflı küme (13.08) --

def test_tek_sinif_listesi_degismedi():
    """İP5'in üç yapılandırması tek sınıflıdır ve öyle kalmalıdır.

    `SINIFLAR`'a sınıf eklemek `veri_yaml_yaz`'ın yazdığı `nc`'yi büyütür; İP5'in
    yapılandırmaları yeniden üretildiğinde tespit başlığı değişir ve 05.08'de
    ölçülen mAP50 0,967 bugünkü koşuyla **karşılaştırılamaz** hâle gelir. Dört
    tipli küme ayrı bir sabit kullanır. Bu test o ayrımı koruyor.
    """
    assert SINIFLAR == ("gauge",)


def test_cok_sinifta_gauge_sifirda_kaliyor():
    """Geriye dönük uyum: mevcut tek sınıflı etiketler yeniden yazılmadan
    dört tipli kümeye girebilmeli. `gauge` yer değiştirirse hepsi bozulur."""
    assert SINIFLAR_COK[0] == "gauge"
    assert SINIFLAR_COK.index("gauge") == GAUGE_SINIF_ID
    assert set(SINIFLAR).issubset(SINIFLAR_COK)


def test_veri_yaml_sinif_listesini_dinliyor(tmp_path):
    """`siniflar` geçilmezse tek sınıf, geçilirse verilen liste yazılmalı."""
    tek = veri_yaml_yaz(tmp_path / "tek.yaml", train=tmp_path / "t", val=tmp_path / "v")
    assert "nc: 1" in tek.read_text(encoding="utf-8")

    dort = veri_yaml_yaz(tmp_path / "dort.yaml", train=tmp_path / "t", val=tmp_path / "v",
                         siniflar=SINIFLAR_COK)
    icerik = dort.read_text(encoding="utf-8")
    assert "nc: 4" in icerik
    for ad in SINIFLAR_COK:
        assert ad in icerik


def test_yolo_satiri_sinif_kimligini_yaziyor():
    """Sınıf kimliği satırın ilk alanıdır; yanlış yazılırsa lamba vana olur."""
    for kimlik, ad in enumerate(SINIFLAR_COK):
        satir = yolo_satiri((10, 20, 110, 120), 512, 512, sinif=kimlik)
        assert satir.split()[0] == str(kimlik), f"{ad} kimliği satıra geçmedi"
