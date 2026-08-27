"""Beş sınıflı tespit kümesi: dört tip + `keypad`, artı ZOR NEGATİFLER (İP17).

    python scripts/hazirla_karistiricilar.py      # once karistiricilar
    python scripts/hazirla_ip17_keypad.py
    python scripts/hazirla_ip17_keypad.py --tip-basina 600 --karistirici-olasilik 0.5

`hazirla_ip5_cok_sinif.py`'nin üstüne iki şey ekliyor:

**1. `keypad` sınıfı.** `read_keypad` pano kırpımını alıp envanterdeki oranlarla
buton kutularını çıkarıyor; eksik olan panoyu BULACAK bir tespit sınıfıydı.
Sentetik üreteç (`synth/keypad.py`) panoyu ground truth olarak veriyor, yani
etiket bedava. YALNIZ pano etiketleniyor, içindeki butonlar değil — gerekçe
`_keypad_bindir` içinde. Ayrı bir `button` sınıfı da yok (bkz. `SINIFLAR_KEYPAD`).

**2. Gerçek videodan kırpımlar.** 27.08'de ölçülen kusur: kadranın olmadığı
karelerde 383 "gauge" kutusu üretildi. `hazirla_karistiricilar.py` o kutuları
gerçek videolardan kırpıp **etiketine göre kovalara** ayırır ve burada eğitim
karelerine yapıştırılırlar:

    negatif/   teker, vantilatör kanadı, makine gövdesi → ETİKETSİZ
    lamp/      ikaz lambası camı                        → `lamp` etiketli
    keypad/    butonlu kontrol panosu                   → `keypad` etiketli

Etiketsiz yapıştırmanın anlamı YOLO'da nettir: kutu yoksa o bölge arka plandır
ve orada tespit üreten model ceza alır. Yani model "bu teker gösterge değil"i
doğrudan öğrenir — eşiği yükseltmek gibi dolaylı bir bastırma değil.

**Ayrım ÖLÇÜMLE öğrenildi.** İlk sürümde her kırpım negatifti. Gerçek videoda
sonuç: kadran olmayan videolarda `gauge` kutusu 64→4 (istenen), ama 10.mp4'te
`lamp` kutusu 49→1 (gerileme). Sebep: o kırpım gerçek bir ikaz lambasının
camıydı; "gösterge değil" diye öğretilince model onu LAMBA olarak da göremez
oldu. Bir kırpım gerçekten bir sınıfın örneğiyse o sınıfla etiketlenir —
"bu bir lamba" demek "bu gösterge değil" demekten daha çok şey öğretir.

**Neden tüm kareyi negatif olarak koymuyoruz:** boş etiketli bir kare "burada
hiç gösterge yok" der; buton panosu videolarında bu YANLIŞ ve onları arka plan
diye öğretmek yeni bir hata açar. Kırpım yaklaşımı yalnız NESNEYİ alır,
bağlamını değil.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

from gauge_vision.config import load_gauges                          # noqa: E402
from gauge_vision.detect.dataset import (IMAGES_DIR, LABELS_DIR,     # noqa: E402
                                         SINIFLAR_KEYPAD, veri_yaml_yaz,
                                         yolo_satiri)
from gauge_vision.synth.keypad import render_keypad                  # noqa: E402
import hazirla_ip5_cok_sinif as C                                    # noqa: E402

HEDEF_KOK = KOK / "data" / "detect" / "keypad5"
TABAN_KOK = KOK / "data" / "detect" / "cok_sinif"
KARISTIRICI_KOK = KOK / "data" / "detect" / "karistiricilar"
METRIK = KOK / "outputs" / "metrics" / "ip17_keypad_veri.json"
KARE = C.KARE
VAL_ORANI = 0.2

KEYPAD_SINIF = SINIFLAR_KEYPAD.index("keypad")


def _kopyala(kaynak_gor: Path, hedef_gor: Path, hedef_et: Path) -> int:
    """Var olan kümeyi olduğu gibi taşır (0-3 sınıf kimlikleri değişmiyor)."""
    n = 0
    for g in sorted(kaynak_gor.glob("*.*")):
        etiket = kaynak_gor.parent / LABELS_DIR / f"{g.stem}.txt"
        if not etiket.exists():
            continue
        shutil.copy2(g, hedef_gor / g.name)
        shutil.copy2(etiket, hedef_et / etiket.name)
        n += 1
    return n


def _uret_analog(gauge, sinif: int, rng: random.Random):
    """Analog kadranı sahneye girecek nesne olarak çizer.

    `hazirla_ip5_cok_sinif._uret_nesne` analogu üretmiyor; orada analog kareler
    mevcut tek sınıflı kümeden KOPYALANIYOR. Burada ayrıca çiziliyor çünkü zor
    negatifin değeri BAĞLAMDA: teker ile kadranın AYNI karede bulunması gerek,
    yoksa model ikisini hiç yan yana görmeden ayırmayı öğrenmek zorunda kalır.
    """
    from gauge_vision.synth.dial import render_analog
    olcek = gauge.scale
    img, truth = render_analog(gauge, rng.uniform(olcek.min, olcek.max))
    return C.Nesne(img, truth.bbox_xyxy, sinif, "daire")


def _keypad_bindir(zemin: np.ndarray, gauge, rng: random.Random,
                   dolu: list) -> list[str]:
    """Sentetik buton panosunu sahneye yapıştırır; pano + buton etiketleri döner.

    Buton kutuları panonun kendi koordinatlarında geliyor; pano kırpılıp
    ölçeklendiği için aynı dönüşüm butonlara da uygulanmalı. Bu hesabı
    `_bindir`e gömmek yerine burada açıkça yapıyoruz — `_bindir` tek kutu
    döndürüyor, buradaki nesne İÇ İÇE kutular taşıyor.
    """
    bilesim = {b["id"]: rng.choice(list(b.get("states") or ["off"]))
               for b in gauge.buttons}
    img, truth = render_keypad(gauge, bilesim,
                               etiket_goster=rng.random() < 0.7)

    px1, py1, px2, py2 = (int(v) for v in truth.bbox_xyxy)
    kirpim = img[max(0, py1):py2, max(0, px1):px2]
    if kirpim.size == 0:
        return []

    hedef = rng.randint(220, KARE) if rng.random() < 0.4 else rng.randint(90, 260)
    oran = hedef / max(kirpim.shape[:2])
    yeni = (max(16, int(kirpim.shape[1] * oran)), max(16, int(kirpim.shape[0] * oran)))
    kirpim = cv2.resize(kirpim, yeni, interpolation=cv2.INTER_AREA)

    for _ in range(30):
        ox = rng.randint(0, KARE - yeni[0])
        oy = rng.randint(0, KARE - yeni[1])
        kutu = (ox, oy, ox + yeni[0], oy + yeni[1])
        if not all(kutu[0] >= d[2] or kutu[2] <= d[0] or kutu[1] >= d[3] or kutu[3] <= d[1]
                   for d in dolu):
            continue
        zemin[oy:oy + yeni[1], ox:ox + yeni[0]] = kirpim
        dolu.append(kutu)

        # YALNIZ pano etiketleniyor, içindeki butonlar DEĞİL. İki sebep:
        #
        # 1. `synth/keypad.py` butonları KARE çiziyor, gerçek buton yuvarlak.
        #    Onları `lamp` etiketlemek modele "lamba karedir" dedirtir ve
        #    `render_lamp`'ın yuvarlak lambalarıyla doğrudan çelişir — tek
        #    sınıfa iki uyumsuz şekil öğretmek ikisini birden bozar.
        # 2. Zincirin buton tespitine İHTİYACI YOK: `read_keypad` pano
        #    kırpımını alıp buton kutularını envanterdeki oranlardan çıkarıyor.
        #    Tespitten istenen tek şey panoyu bulmak.
        return [yolo_satiri(kutu, KARE, KARE, sinif=KEYPAD_SINIF)]
    return []


def _karistirici_bindir(zemin: np.ndarray, yol: Path, sinif: int | None,
                        rng: random.Random, dolu: list) -> str | None:
    """Gerçek videodan gelen kırpımı sahneye yapıştırır.

    `sinif` None ise ETİKET ÜRETİLMEZ (zor negatif); değilse o sınıfla
    etiketlenir. Ayrım ölçümle öğrenildi: her şeyi negatif yapmak 10.mp4'te
    `lamp` tespitini 49'dan 1'e düşürdü, çünkü oradaki kırpım gerçek bir ikaz
    lambasıydı ve "gösterge değil" diye öğretilince lamba olarak da görülemedi.
    """
    kirpim = cv2.imread(str(yol))
    if kirpim is None or kirpim.size == 0:
        return None
    hedef = rng.randint(70, 300)
    oran = hedef / max(kirpim.shape[:2])
    yeni = (max(12, int(kirpim.shape[1] * oran)), max(12, int(kirpim.shape[0] * oran)))
    kirpim = cv2.resize(kirpim, yeni, interpolation=cv2.INTER_AREA)

    # Kenar yumuşatma ŞART: sert dikdörtgen yama yapıştırılırsa model
    # "yamanın kenarı" ile "gösterge değil"i ilişkilendirebilir ve sahada
    # kenarsız gerçek tekere bakınca öğrendiği ipucu yok olur. Negatifin
    # değeri NESNEDE, yapıştırma izinde değil.
    maske = np.ones((yeni[1], yeni[0]), np.float32)
    pay = max(2, int(min(yeni) * 0.10))
    maske[:pay, :] *= np.linspace(0, 1, pay)[:, None]
    maske[-pay:, :] *= np.linspace(1, 0, pay)[:, None]
    maske[:, :pay] *= np.linspace(0, 1, pay)[None, :]
    maske[:, -pay:] *= np.linspace(1, 0, pay)[None, :]
    maske = maske[..., None]

    for _ in range(20):
        ox = rng.randint(0, KARE - yeni[0])
        oy = rng.randint(0, KARE - yeni[1])
        kutu = (ox, oy, ox + yeni[0], oy + yeni[1])
        if all(kutu[0] >= d[2] or kutu[2] <= d[0] or kutu[1] >= d[3] or kutu[3] <= d[1]
               for d in dolu):
            pencere = zemin[oy:oy + yeni[1], ox:ox + yeni[0]].astype(np.float32)
            harman = kirpim.astype(np.float32) * maske + pencere * (1.0 - maske)
            zemin[oy:oy + yeni[1], ox:ox + yeni[0]] = np.clip(harman, 0, 255).astype(np.uint8)
            dolu.append(kutu)
            if sinif is None:
                return None
            # Etiket kutusu yumuşatma payı KADAR daraltılıyor: kenardaki alfa
            # geçişi nesnenin kendisi değil, yapıştırma izi.
            ic = (kutu[0] + pay, kutu[1] + pay, kutu[2] - pay, kutu[3] - pay)
            return yolo_satiri(ic, KARE, KARE, sinif=sinif)
    return None


def uret_kare(gauges, keypadler, karistiricilar, rng: random.Random,
              karistirici_olasilik: float) -> tuple[np.ndarray, list[str]]:
    zemin = C._zemin(rng)
    dolu: list = []
    satirlar: list[str] = []

    # Gerçek kırpımlar ÖNCE: sonra konursa gösterge kutularının üstüne oturmaya
    # çalışıp yer bulamıyor ve karelerin çoğunda hiç görünmüyorlar.
    if karistiricilar and rng.random() < karistirici_olasilik:
        for _ in range(rng.randint(1, 2)):
            yol, sinif = rng.choice(karistiricilar)
            satir = _karistirici_bindir(zemin, yol, sinif, rng, dolu)
            if satir:
                satirlar.append(satir)

    if keypadler and rng.random() < 0.35:
        satirlar += _keypad_bindir(zemin, rng.choice(keypadler), rng, dolu)

    for _ in range(rng.randint(1, 3)):
        gauge = rng.choice(gauges)
        sinif = SINIFLAR_KEYPAD.index(C._SINIF_ADI[gauge.type])
        nesne = _uret_analog(gauge, sinif, rng) if gauge.type == "analog" \
            else C._uret_nesne(gauge, sinif, rng)
        if nesne is None:
            continue
        kutu = C._bindir(zemin, nesne, rng, dolu)
        if kutu is not None:
            satirlar.append(yolo_satiri(kutu, KARE, KARE, sinif=sinif))

    if rng.random() < 0.5:
        kalite = rng.randint(45, 92)
        _, tampon = cv2.imencode(".jpg", zemin, [cv2.IMWRITE_JPEG_QUALITY, kalite])
        zemin = cv2.imdecode(tampon, cv2.IMREAD_COLOR)
    return zemin, satirlar


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hedef", type=Path, default=HEDEF_KOK)
    p.add_argument("--taban", type=Path, default=TABAN_KOK,
                   help="mevcut dort sinifli kume - oldugu gibi tabana girer")
    p.add_argument("--karistirici", type=Path, default=KARISTIRICI_KOK)
    p.add_argument("--tip-basina", type=int, default=700, help="uretilecek kare")
    p.add_argument("--karistirici-olasilik", type=float, default=0.45)
    p.add_argument("--tohum", type=int, default=17)
    a = p.parse_args(argv)

    gauges = [g for g in load_gauges().values()
              if g.type in ("analog", "digital", "lamp", "valve")]
    keypadler = [g for g in load_gauges().values() if g.type == "keypad"]
    # Kırpımlar alt klasöre göre etiketleniyor: `negatif/` etiketsiz girer,
    # diğerleri klasör adındaki sınıfla. Ayrımın gerekçesi
    # `hazirla_karistiricilar.py` içindeki KAYNAKLAR yorumunda.
    karistiricilar: list[tuple[Path, int | None]] = []
    kirpim_sayim: dict[str, int] = {}
    if a.karistirici.exists():
        for kova in sorted(p for p in a.karistirici.iterdir() if p.is_dir()):
            if kova.name == "negatif":
                sinif = None
            elif kova.name in SINIFLAR_KEYPAD:
                sinif = SINIFLAR_KEYPAD.index(kova.name)
            else:
                print(f"UYARI: bilinmeyen kirpim klasoru atlandi: {kova.name}")
                continue
            yollar = sorted(kova.glob("*.png"))
            karistiricilar += [(y, sinif) for y in yollar]
            kirpim_sayim[kova.name] = len(yollar)
    if not karistiricilar:
        print(f"UYARI: karistirici yok ({a.karistirici}) — once "
              f"scripts/hazirla_karistiricilar.py")
    print(f"envanter: {len(gauges)} gosterge, {len(keypadler)} keypad")
    print(f"gercek kirpim: {kirpim_sayim}")

    if a.hedef.exists():
        shutil.rmtree(a.hedef)
    for bolum in ("train", "val"):
        (a.hedef / bolum / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        (a.hedef / bolum / LABELS_DIR).mkdir(parents=True, exist_ok=True)

    # Mevcut dört sınıflı küme olduğu gibi tabana giriyor: `keypad` SONA (4)
    # eklendiği için 0-3 etiketleri hiç dokunulmadan geçerli. İP5'in ölçülmüş
    # tabanını yeniden üretmek, karşılaştırmayı da kaybettirirdi.
    tasinan = 0
    for bolum in ("train", "val"):
        kaynak = a.taban / bolum / IMAGES_DIR
        if not kaynak.exists():
            continue
        tasinan += _kopyala(kaynak, a.hedef / bolum / IMAGES_DIR,
                            a.hedef / bolum / LABELS_DIR)
    print(f"taban kumeden tasinan: {tasinan} kare ({a.taban.name})")

    rng = random.Random(a.tohum)
    n_val = int(a.tip_basina * VAL_ORANI)
    sayim = {ad: 0 for ad in SINIFLAR_KEYPAD}
    negatif_kare = 0
    for i in range(a.tip_basina):
        img, satirlar = uret_kare(gauges, keypadler, karistiricilar, rng,
                                  a.karistirici_olasilik)
        bolum = "val" if i < n_val else "train"
        ad = f"ip17_{i:05d}"
        cv2.imwrite(str(a.hedef / bolum / IMAGES_DIR / f"{ad}.png"), img)
        (a.hedef / bolum / LABELS_DIR / f"{ad}.txt").write_text(
            "\n".join(satirlar) + ("\n" if satirlar else ""), encoding="utf-8")
        if not satirlar:
            negatif_kare += 1
        for s in satirlar:
            sayim[SINIFLAR_KEYPAD[int(s.split()[0])]] += 1

    yol = veri_yaml_yaz(a.hedef / "gauge5.yaml",
                        train=a.hedef / "train" / IMAGES_DIR,
                        val=a.hedef / "val" / IMAGES_DIR,
                        siniflar=SINIFLAR_KEYPAD)

    ozet = {"kare": a.tip_basina, "val": n_val, "sinif_basina_ornek": sayim,
            "tamamen_negatif_kare": negatif_kare,
            "gercek_kirpim": kirpim_sayim,
            "karistirici_olasilik": a.karistirici_olasilik,
            "siniflar": list(SINIFLAR_KEYPAD)}
    METRIK.parent.mkdir(parents=True, exist_ok=True)
    METRIK.write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ozet, ensure_ascii=False, indent=2))
    print(f"\nveri: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
