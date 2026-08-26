"""Karede birden çok analog gösterge varken zincir ne yapıyor (`read_all_analog`).

26.08'de dört videoda ölçülen iki kusurun testi burada:

1. **Karede birden çok analog vardı, yalnız biri okunuyordu.** `read_gauge` tek
   gösterge okur ve bu doğrudur (kalibrasyon göstergeye özeldir), ama diğerleri
   hiç dokunulmadan bırakılıyordu. `read_all_analog` hepsine bakar.
2. **Kimliği bilinmeyen kutuya değer/birim üretiliyordu** — termometre "2,2 bar",
   devir saati "0,8 bar". `AnalogKutuOkuma`'da `value`/`unit` alanı YOKTUR; bu
   testler o alanların geri gelmediğini de sınar.
"""

from __future__ import annotations

import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.pipeline import AnalogKutuOkuma, Tespit, read_all_analog
from gauge_vision.synth.dial import render_analog

GAUGES = load_gauges()
KADRAN_PX = 220


def _kadran(gauge_id: str, deger: float) -> np.ndarray:
    """Tek kadranı kendi karesinde çizer (sentetik üreteç, ground truth bedava)."""
    goruntu, _ = render_analog(GAUGES[gauge_id], deger, size=KADRAN_PX)
    return goruntu


def _sahne(kadranlar: list[np.ndarray]) -> tuple[np.ndarray, list[Tespit]]:
    """Kadranları yan yana bir zemine yerleştirir ve kutularını döndürür.

    Tespit taklit ediliyor: sınanan şey YOLO değil, "kaç kutu verilirse o kadar
    okuma üretiliyor mu" davranışıdır. Gerçek tespitle sınamak testi modelin
    ağırlıklarına bağlar ve ağırlık değişince test sebepsiz kırılır.
    """
    pay = 30
    h = KADRAN_PX + 2 * pay
    w = len(kadranlar) * KADRAN_PX + (len(kadranlar) + 1) * pay
    sahne = np.full((h, w, 3), 200, dtype=np.uint8)
    tespitler = []
    for i, k in enumerate(kadranlar):
        x = pay + i * (KADRAN_PX + pay)
        sahne[pay:pay + KADRAN_PX, x:x + KADRAN_PX] = k
        tespitler.append(Tespit(
            box_xyxy=(float(x), float(pay), float(x + KADRAN_PX), float(pay + KADRAN_PX)),
            conf=0.9, sinif="gauge", tip="analog"))
    return sahne, tespitler


def test_karedeki_her_analog_okunuyor():
    """Üç kadran varsa üç okuma çıkar — biri değil.

    26.08'in birinci kusuru: `termometre.mp4`'ün ilk karesinde dört termometre
    vardı, zincir tek değer üretiyordu.
    """
    sahne, tespitler = _sahne([_kadran("PT-101", 3.0),
                               _kadran("PT-101", 6.0),
                               _kadran("PT-101", 9.0)])

    okumalar = read_all_analog(sahne, model=None, tespitler=tespitler)

    assert len(okumalar) == 3
    assert all(o.ok for o in okumalar), [o.reason for o in okumalar]


def test_farkli_degerler_farkli_acilar_uretiyor():
    """Her kutu KENDİ ibresini okumalı — hepsine aynı cevap dönmemeli.

    Döngü yanlış kurulursa (hep aynı kırpım okunursa) sayı üretilir ve test
    'üç okuma var' diye geçerdi; ayırt edici olan açıların farklı olmasıdır.
    """
    sahne, tespitler = _sahne([_kadran("PT-101", 1.0), _kadran("PT-101", 9.0)])

    okumalar = read_all_analog(sahne, model=None, tespitler=tespitler)

    aci_1, aci_2 = (o.needle.angle_img_deg for o in okumalar)
    assert abs(aci_1 - aci_2) > 45, (aci_1, aci_2)


def test_analog_olmayan_kutular_atlanir():
    """Dijital/lamba/vana kutusuna kadran geometrisi uygulanmaz.

    `read_all_analog` yalnız analogla ilgilenir; diğer tipler için merkez ve
    ibre kavramı anlamsızdır ve zorlanırsa uydurma açı üretir.
    """
    sahne, tespitler = _sahne([_kadran("PT-101", 5.0)])
    tespitler.append(Tespit(box_xyxy=(0.0, 0.0, 50.0, 50.0), conf=0.95,
                            sinif="lamp", tip="lamp"))

    okumalar = read_all_analog(sahne, model=None, tespitler=tespitler)

    assert len(okumalar) == 1


def test_okuma_deger_ve_birim_TASIMAZ():
    """Kimliksiz kutuya değer/birim üretilmez (3. kural).

    Bu testin varlık sebebi 26.08'de ölçülen sessiz hatadır: 0-120 °C'lik bir
    termometre "2,2 bar · status ok · güven 0,724" diye yayınlandı. Değer ve
    birim göstergenin KİMLİĞİNE aittir; kimlik görüntüden çıkarılamıyor.
    Alanlar geri eklenirse test kırılır ve sebebini burada okur.
    """
    alanlar = AnalogKutuOkuma.__dataclass_fields__
    assert "value" not in alanlar
    assert "unit" not in alanlar
    assert "reading" not in alanlar


def test_cok_kucuk_kutu_sayi_uretmez():
    """Uzaktaki minik kadran okunamaz diye işaretlenir, uydurulmaz."""
    sahne = np.full((60, 60, 3), 200, dtype=np.uint8)
    tespitler = [Tespit(box_xyxy=(10.0, 10.0, 26.0, 26.0), conf=0.8,
                        sinif="gauge", tip="analog")]

    okumalar = read_all_analog(sahne, model=None, tespitler=tespitler)

    assert len(okumalar) == 1
    assert not okumalar[0].ok
    assert "küçük" in okumalar[0].reason
