"""İP8 ekran düzeneğinin kendi kendini sınaması (`ekran_kadran` + `olc_ip8`).

Bu projede en pahalı hata sınıfı ölçümün kendisinin bozuk olmasıydı (veri
sızıntısı, dairesel kalibrasyon, bozuk bulanıklık üreteci…). İP8'in düzeneği de
bir ölçüm aracıdır ve fotoğraflar çekilmeden ÖNCE sahte girdiyle sınanmalıdır:
fotoğraf yerine ekran karesinin KENDİSİ ölçümden geçirilir. Gerçek optik yol
yokken zincir ground truth'u bire bir bulmalı — bulamıyorsa fotoğraftan önce
düzenek bozuktur ve çıkacak tablo hiçbir soruyu cevaplamaz.

Tespit burada YOLO değil "ideal kutu"dur (içerik bölgesini bire bir veren
stub): sınanan şey tespit değil, kare üretimi + okuma + tipe göre
değerlendirme hattıdır. YOLO'nun ekran fotoğrafındaki başarısı İP8 ölçümünün
kendisinde raporlanır, birim testte değil.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ekran_kadran  # noqa: E402
import olc_ip8  # noqa: E402

from gauge_vision.config import load_gauges  # noqa: E402

BOYUT = 512
ADET_ANALOG = 5
ADET_DIJITAL = 4


@pytest.fixture(scope="module")
def gauges():
    return load_gauges()


@pytest.fixture(scope="module")
def kareler_ve_kayitlar(gauges):
    cizelge = ekran_kadran.plan_kur(gauges, "hepsi", ADET_ANALOG, ADET_DIJITAL)
    kareler, kayitlar = [], []
    for i, (gauge, spec) in enumerate(cizelge, start=1):
        kare, kayit = ekran_kadran.kare_uret(gauge, spec, i, len(cizelge), BOYUT)
        kareler.append(kare)
        kayitlar.append(kayit)
    return kareler, kayitlar


class _IdealKutu:
    """Manifestteki `icerik_bbox`'ı bire bir veren tespit stub'ı.

    Kutuyu kare düzeninden yeniden HESAPLAMAZ — üretici ne yazdıysa onu verir.
    İkinci kez hesaplasaydı, kare düzeni değişince (örn. tampon bandı eklendi)
    test sessizce yanlış bölgeyi okur ve yine geçerdi ya da yanlış sebeple
    kalırdı.
    """

    def __init__(self, bbox):
        self._bbox = [float(v) for v in bbox]

    class _Kutular:
        def __init__(self, bbox):
            self.xyxy = np.array([bbox])
            self.conf = np.array([1.0])

        def __len__(self):
            return 1

    class _Sonuc:
        def __init__(self, bbox):
            self.boxes = _IdealKutu._Kutular(bbox)

    def predict(self, image, **_):
        return [_IdealKutu._Sonuc(self._bbox)]


# ------------------------------------------------------------------ plan ----

def test_plan_dort_tipi_kapsiyor(gauges, kareler_ve_kayitlar):
    _, kayitlar = kareler_ve_kayitlar
    tipler = {k["type"] for k in kayitlar}
    assert tipler == {"analog", "digital", "lamp", "valve"}


def test_sira_numaralari_bosluksuz(kareler_ve_kayitlar):
    """Eşleştirme sıraya dayanıyor; planda delik olursa çekim talimatı şaşar."""
    _, kayitlar = kareler_ve_kayitlar
    assert [k["sira"] for k in kayitlar] == list(range(1, len(kayitlar) + 1))


def test_analog_uclari_tariyor(gauges, kareler_ve_kayitlar):
    """Min ve max MUTLAKA planda: işaret/yön hatası en çok uçlarda görünür."""
    _, kayitlar = kareler_ve_kayitlar
    for gid in {k["gauge_id"] for k in kayitlar if k["type"] == "analog"}:
        degerler = [k["value"] for k in kayitlar if k["gauge_id"] == gid]
        s = gauges[gid].scale
        assert min(degerler) == s.min and max(degerler) == s.max


def test_dijital_negatif_ucu_tariyor(gauges, kareler_ve_kayitlar):
    """Negatif uç, eksi işaretinin çözülmesini tek başına test eden karedir."""
    _, kayitlar = kareler_ve_kayitlar
    degerler = [k["value"] for k in kayitlar if k["type"] == "digital"]
    assert degerler and min(degerler) < 0


def test_vana_ara_konum_unreadable_bekliyor(kareler_ve_kayitlar):
    """Ara konumda doğru cevap okumamaktır (6. kural) — plan bunu beyan etmeli."""
    _, kayitlar = kareler_ve_kayitlar
    ara = [k for k in kayitlar if k["type"] == "valve"
           and k["beklenen"] == "unreadable"]
    assert len(ara) == 1 and ara[0]["sapma_deg"] == ekran_kadran.VANA_ARA_KONUM_DEG


# ------------------------------------------- uçtan uca (fotoğrafsız) ölçüm ----

@pytest.fixture(scope="module")
def satirlar(tmp_path_factory, gauges, kareler_ve_kayitlar):
    kareler, kayitlar = kareler_ve_kayitlar
    klasor = tmp_path_factory.mktemp("ip8_ekran")
    yollar = []
    for kare, kayit in zip(kareler, kayitlar):
        yol = klasor / f"IMG-{kayit['sira']}.png"
        # imwrite dönüşü kontrol ediliyor: sessiz 0 baytlık dosya daha önce
        # bir ölçümü çöpe çevirdi (inline python + kaçış karakteri kazası).
        assert cv2.imwrite(str(yol), kare)
        yollar.append(yol)
    bulunan, uyarilar = olc_ip8.fotograflari_bul(klasor)
    assert not uyarilar and bulunan == yollar
    sonuc = []
    for yol, kayit in zip(bulunan, kayitlar):
        sonuc += olc_ip8.olc([yol], [kayit], gauges,
                             _IdealKutu(kayit["icerik_bbox"]), 0.25)
    return sonuc


def test_analog_temiz_karede_okunuyor(satirlar):
    analog = [s for s in satirlar if s["type"] == "analog"]
    assert all("hata_yuzde" in s for s in analog), \
        [s.get("sebep") for s in analog if "hata_yuzde" not in s]
    hatalar = [s["hata_yuzde"] for s in analog]
    # Optik yol yokken zincir sentetik tabanında kalmalı. Sınır cömert (%2):
    # burada hassasiyet değil düzeneğin sağlığı sınanıyor; hassasiyet ölçümün
    # kendisinde raporlanır.
    assert max(hatalar) < 2.0, hatalar


def test_dijital_temiz_karede_dizge_dogru(satirlar):
    dijital = [s for s in satirlar if s["type"] == "digital"]
    assert dijital and all(s.get("dogru") for s in dijital), \
        [(s.get("okunan"), s.get("sebep")) for s in dijital]


def test_lamba_temiz_karede_dogru(satirlar):
    lamba = [s for s in satirlar if s["type"] == "lamp"]
    assert lamba and all(s.get("dogru") for s in lamba), \
        [(s["beklenen"], s.get("okunan")) for s in lamba]


def test_vana_temiz_karede_dogru_ara_konum_unreadable(satirlar):
    vana = [s for s in satirlar if s["type"] == "valve"]
    assert vana and all(s.get("dogru") for s in vana), \
        [(s["beklenen"], s.get("okunan"), s.get("durum")) for s in vana]
    ara = next(s for s in vana if s["beklenen"] == "unreadable")
    assert ara["durum"] == "unreadable" and "okunan" not in ara


def test_ozet_tip_basina_referans_tasiyor(satirlar):
    """Rapora giden sayı sentetik referansıyla yan yana durmalı."""
    tipler = olc_ip8.ozetle(satirlar)
    assert set(tipler) == {"analog", "digital", "lamp", "valve"}
    assert tipler["analog"]["kapsama"] == 1.0
    for tip in ("digital", "lamp", "valve"):
        assert tipler[tip]["dogruluk"] == 1.0, (tip, tipler[tip])
        assert "referans" in tipler[tip]
