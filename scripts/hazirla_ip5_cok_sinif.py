"""Dört gösterge tipini kapsayan YOLO eğitim kümesini kurar (İP5 genişletmesi).

    python scripts/hazirla_ip5_cok_sinif.py
    python scripts/hazirla_ip5_cok_sinif.py --tip-basina 400 --tohum 7

**Neden bu script var.** 17.08 ölçümünde (İP13) zincirin dört gösterge tipini de
okuduğu ama **tespitin** yalnızca analog kadranı bulduğu görüldü: dijital panel,
ikaz lambası ve vana görüntülerinde YOLO "gösterge bulunamadı" dönüyordu. Okuma
katmanı 480 karede sıfır sessiz hata veriyordu; eksik olan tespitti. Bu, o günün
raporunda üç gün üst üste "yarın" olarak yazılan işin kendisidir.

**Sınıf kimlikleri geriye dönük uyumludur.** `gauge` 0'da kalır; böylece
`data/detect/karisik` altındaki mevcut etiketler ve İP5'in ölçülmüş ağırlıkları
geçersizleşmez, yeni küme eskisinin üstüne **eklenir**.

**Neden nesneler zemine yerleştiriliyor.** `render_digital`/`render_lamp`/
`render_valve` nesneyi kareyi dolduracak biçimde çizer — okuma ölçümü için doğru
olan budur, çünkü orada kırpım zaten tespitten gelir. Ama bu kareler doğrudan
tespit eğitimine verilirse model "nesne = tüm görüntü" öğrenir ve sahnede
gösterge aramayı hiç öğrenmez. Bu yüzden her nesne rastgele ölçek ve konumda,
pano benzeri bir zemine bindirilir; karede birden fazla nesne olabilir.

**Doğrulama/test bölümü bilinçli olarak ikiye ayrılmıştır.** Analog tarafta
05.08'in **aynı gerçek test kümesi** kullanılır, yoksa çıkan mAP o günün
sayısıyla karşılaştırılamaz. Yeni tipler kendi tutulmuş bölümlerini alır.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.detect.dataset import (IMAGES_DIR, LABELS_DIR, SINIFLAR_COK,
                                         veri_yaml_yaz, yolo_satiri)
from gauge_vision.synth.digital import render_digital
from gauge_vision.synth.state import render_lamp, render_valve

HEDEF_KOK = Path("data/detect/cok_sinif")
KAYNAK_KARISIK = Path("data/detect/karisik/train")
GERCEK_KOK = Path("data/detect/_gercek")
ENVANTER = Path("configs/gauges.yaml")
METRIK_YOLU = Path("outputs/metrics/ip5_cok_sinif_veri.json")

KARE = 512                      # eğitim karesinin kenarı
VAL_ORANI = 0.2                 # yeni tiplerin tutulan bölümü


@dataclass(frozen=True)
class Nesne:
    """Zemine bindirilecek tek gösterge."""

    goruntu: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    sinif: int
    maske: str = "dikdortgen"   # "dikdortgen" | "daire" | "koyu"


def _maske_uret(kirpim: np.ndarray, tur: str) -> np.ndarray:
    """Nesnenin gövdesini kendi çizim zemininden ayıran alfa maskesi.

    Üreteçler nesneyi gri bir kareye çizer; bu kare olduğu gibi yapıştırılırsa
    zeminde **dikdörtgen bir süreksizlik** oluşur ve model göstergeyi değil
    yamanın kenarını öğrenir. Maske tipi göstergenin fiziksel biçiminden gelir,
    genel bir eşikten değil:

    - `daire` — ikaz lambası yuvarlaktır, muhafazası da öyle.
    - `koyu`  — vananın gövdesi koldur; kolun çevresindeki gri, çizim zeminidir
                ve sahada onun yerinde boru/duvar vardır.
    - `dikdortgen` — dijital panel gerçekten dikdörtgen bir gövdedir; çerçevesi
                sahnede de keskin kenarla durur, maskelemek gerçeği bozar.
    """
    h, w = kirpim.shape[:2]
    if tur == "daire":
        m = np.zeros((h, w), np.float32)
        cv2.circle(m, (w // 2, h // 2), int(min(h, w) * 0.5) - 1, 1.0, -1, cv2.LINE_AA)
        return m
    if tur == "koyu":
        gri = cv2.cvtColor(kirpim, cv2.COLOR_BGR2GRAY)
        # Çizim zemini kırpımın köşelerinde durur; gövde ondan koyudur.
        zemin_tonu = float(np.median([gri[0, 0], gri[0, -1], gri[-1, 0], gri[-1, -1]]))
        m = (gri < zemin_tonu - 25).astype(np.float32)
        return cv2.GaussianBlur(m, (0, 0), 1.0)
    return np.ones((h, w), np.float32)


def _zemin(rng: random.Random) -> np.ndarray:
    """Pano benzeri zemin: düz ton + eğim + doku + birkaç kenar çizgisi.

    Zeminin gerçekçi olması değil, **tekdüze olmaması** gerekiyor. Tek renk bir
    arka planda model nesneyi kenar yoğunluğundan ayırt etmeyi öğrenir ve
    sahadaki dolu panoda çalışmaz.
    """
    ton = rng.randint(55, 190)
    img = np.full((KARE, KARE, 3), ton, np.uint8)

    # Yumuşak aydınlatma eğimi — sanayi panosunda ışık tek yönden gelir.
    egim = np.linspace(-rng.uniform(10, 45), rng.uniform(10, 45), KARE, dtype=np.float32)
    img = np.clip(img.astype(np.float32) + (egim[None, :, None] if rng.random() < 0.5
                                            else egim[:, None, None]), 0, 255).astype(np.uint8)

    # Doku: hafif gürültü, sonra bulanıklık — keskin gürültü YOLO'ya kenar gibi görünür.
    gurultu = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 6, img.shape)
    img = np.clip(img.astype(np.float32) + gurultu, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.8))

    # Pano bölmeleri / boru izleri: nesneyle karışabilecek düz kenarlar.
    for _ in range(rng.randint(0, 4)):
        p1 = (rng.randint(0, KARE), rng.randint(0, KARE))
        p2 = (rng.randint(0, KARE), rng.randint(0, KARE))
        renk = tuple(int(max(0, min(255, ton + rng.randint(-45, 45)))) for _ in range(3))
        cv2.line(img, p1, p2, renk, rng.randint(2, 9), cv2.LINE_AA)
    return img


def _uret_nesne(gauge, sinif: int, rng: random.Random) -> Nesne | None:
    """Envanterdeki göstergeden tek bir çizim üretir."""
    if gauge.type == "digital":
        d = gauge.digits or {}
        alt, ust = 0.0, 10.0 ** int(d.get("count", 4) - int(d.get("decimals", 1)) - 1)
        img, truth = render_digital(gauge, rng.uniform(alt, ust))
        return Nesne(img, truth.panel_bbox_xyxy, sinif, "dikdortgen")
    if gauge.type == "lamp":
        img, truth = render_lamp(gauge, rng.choice(gauge.state_names))
        return Nesne(img, truth.bbox_xyxy, sinif, "daire")
    if gauge.type == "valve":
        img, truth = render_valve(gauge, rng.choice(gauge.state_names),
                                  sapma_deg=rng.uniform(-8, 8))
        return Nesne(img, truth.bbox_xyxy, sinif, "koyu")
    return None


def _bindir(zemin: np.ndarray, nesne: Nesne, rng: random.Random,
            dolu: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    """Nesneyi kutusuna kırpıp rastgele ölçek/konumda zemine yapıştırır.

    Kırpımdan sonra ölçekleniyor: ham kare nesnenin çevresinde geniş boşluk
    taşıyor, o boşluk yapıştırılırsa zemini dikdörtgen bir yamayla örter ve
    model **yamanın kenarını** öğrenir.
    """
    x1, y1, x2, y2 = (int(v) for v in nesne.bbox_xyxy)
    kirpim = nesne.goruntu[max(0, y1):y2, max(0, x1):x2]
    if kirpim.size == 0:
        return None

    # Ölçek iki kipli: devriye karesinde gösterge uzaktan küçük görünür, ama
    # zincir okumaya geçerken kadranı **kırpıp yakınlaştırır** ve tespit o
    # kırpımda yeniden koşabilir. İlk sürümde yalnız küçük kip vardı; İP13'ün
    # kareyi dolduran görüntülerinde tespit %10-65'te kaldı çünkü o ölçek
    # eğitim dağılımının dışındaydı. Yakın çekim kipi bu yüzden eklendi.
    if rng.random() < 0.35:
        hedef = rng.randint(300, KARE)                 # yakın çekim / kırpım
    else:
        hedef = rng.randint(64, 230)                   # devriye mesafesi
    oran = hedef / max(kirpim.shape[:2])
    yeni = (max(8, int(kirpim.shape[1] * oran)), max(8, int(kirpim.shape[0] * oran)))
    kirpim = cv2.resize(kirpim, yeni, interpolation=cv2.INTER_AREA)
    maske = _maske_uret(kirpim, nesne.maske)[..., None]

    for _ in range(30):                                # çakışmayan yer ara
        ox = rng.randint(0, KARE - yeni[0])
        oy = rng.randint(0, KARE - yeni[1])
        kutu = (ox, oy, ox + yeni[0], oy + yeni[1])
        if all(kutu[0] >= d[2] or kutu[2] <= d[0] or kutu[1] >= d[3] or kutu[3] <= d[1]
               for d in dolu):
            pencere = zemin[oy:oy + yeni[1], ox:ox + yeni[0]].astype(np.float32)
            harman = kirpim.astype(np.float32) * maske + pencere * (1.0 - maske)
            zemin[oy:oy + yeni[1], ox:ox + yeni[0]] = np.clip(harman, 0, 255).astype(np.uint8)
            dolu.append(kutu)
            return kutu
    return None


def uret_kare(gauges: list, rng: random.Random) -> tuple[np.ndarray, list[str]]:
    """Bir eğitim karesi: zemin + 1-3 gösterge + YOLO etiket satırları."""
    zemin = _zemin(rng)
    dolu: list[tuple[int, int, int, int]] = []
    satirlar: list[str] = []

    for _ in range(rng.randint(1, 3)):
        gauge = rng.choice(gauges)
        sinif = SINIFLAR_COK.index(_SINIF_ADI[gauge.type])
        nesne = _uret_nesne(gauge, sinif, rng)
        if nesne is None:
            continue
        kutu = _bindir(zemin, nesne, rng, dolu)
        if kutu is not None:
            satirlar.append(yolo_satiri(kutu, KARE, KARE, sinif=sinif))

    # Yayın hattı etkisi en sonda: JPEG artefaktı yapıştırmadan önce uygulanırsa
    # yamanın içinde kalır, gerçekte ise tüm kareye birden uygulanır.
    if rng.random() < 0.5:
        kalite = rng.randint(45, 92)
        _, tampon = cv2.imencode(".jpg", zemin, [cv2.IMWRITE_JPEG_QUALITY, kalite])
        zemin = cv2.imdecode(tampon, cv2.IMREAD_COLOR)
    return zemin, satirlar


_SINIF_ADI = {"analog": "gauge", "digital": "digital", "lamp": "lamp", "valve": "valve"}


def _kopyala(kaynak_gor: Path, hedef_gor: Path, hedef_et: Path) -> int:
    """Var olan tek sınıflı kümeyi olduğu gibi taşır (gauge zaten 0'dır)."""
    n = 0
    for g in sorted(kaynak_gor.glob("*.*")):
        etiket = kaynak_gor.parent / LABELS_DIR / f"{g.stem}.txt"
        if not etiket.exists():
            continue
        shutil.copy2(g, hedef_gor / g.name)
        shutil.copy2(etiket, hedef_et / etiket.name)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Dört tipli tespit kümesi (İP5 genişletmesi)")
    p.add_argument("--tip-basina", type=int, default=300,
                   help="dijital/lamba/vana için üretilecek kare sayısı (toplam)")
    p.add_argument("--tohum", type=int, default=0)
    args = p.parse_args(argv)

    if not (KAYNAK_KARISIK / IMAGES_DIR).exists():
        print(f"karisik kümesi yok ({KAYNAK_KARISIK}) — önce scripts/hazirla_ip5_veri.py")
        return 1

    envanter = load_gauges(ENVANTER)
    hepsi = list(envanter.values()) if isinstance(envanter, dict) else list(envanter)
    yeni_tipler = [g for g in hepsi if g.type in ("digital", "lamp", "valve")]
    if not yeni_tipler:
        print("envanterde dijital/lamba/vana yok")
        return 1

    rng = random.Random(args.tohum)
    if HEDEF_KOK.exists():
        shutil.rmtree(HEDEF_KOK)
    for bolum in ("train", "val"):
        (HEDEF_KOK / bolum / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        (HEDEF_KOK / bolum / LABELS_DIR).mkdir(parents=True, exist_ok=True)

    # 1) Mevcut analog kümesi olduğu gibi girer — İP5'in ölçülmüş tabanı korunur.
    analog_n = _kopyala(KAYNAK_KARISIK / IMAGES_DIR,
                        HEDEF_KOK / "train" / IMAGES_DIR,
                        HEDEF_KOK / "train" / LABELS_DIR)

    # 2) Yeni tipler üretilir ve train/val'e bölünür.
    n_val = int(args.tip_basina * VAL_ORANI)
    sayim: dict[str, int] = {ad: 0 for ad in SINIFLAR_COK}
    for i in range(args.tip_basina):
        bolum = "val" if i < n_val else "train"
        img, satirlar = uret_kare(yeni_tipler, rng)
        if not satirlar:
            continue
        ad = f"cs_{i:05d}"
        cv2.imwrite(str(HEDEF_KOK / bolum / IMAGES_DIR / f"{ad}.png"), img)
        (HEDEF_KOK / bolum / LABELS_DIR / f"{ad}.txt").write_text(
            "\n".join(satirlar) + "\n", encoding="utf-8")
        for s in satirlar:
            sayim[SINIFLAR_COK[int(s.split()[0])]] += 1
    sayim["gauge"] += analog_n  # yaklaşık: analog kümede kare başına ~1 kadran

    # 3) Veri tanımı. Doğrulama iki kaynaktan: gerçek analog val + yeni tip val.
    yaml_yolu = veri_yaml_yaz(
        HEDEF_KOK / "gauge4.yaml",
        train=HEDEF_KOK / "train" / IMAGES_DIR,
        val=HEDEF_KOK / "val" / IMAGES_DIR,
        test=GERCEK_KOK / "test" / IMAGES_DIR,
        siniflar=SINIFLAR_COK)

    train_n = len(list((HEDEF_KOK / 'train' / IMAGES_DIR).glob('*.*')))
    val_n = len(list((HEDEF_KOK / 'val' / IMAGES_DIR).glob('*.*')))

    ozet = {
        "is_paketi": "IP5-cok-sinif",
        "siniflar": list(SINIFLAR_COK),
        "tohum": args.tohum,
        "train_kare": train_n,
        "val_kare": val_n,
        "devralinan_analog_kare": analog_n,
        "uretilen_yeni_tip_kare": train_n + val_n - analog_n,
        "kutu_sayisi_sinif_basina": sayim,
        "veri_yaml": str(yaml_yolu),
        "test_kumesi": "05.08 ile AYNI gerçek test bölümü (yalnız gauge sınıfı)",
    }
    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    METRIK_YOLU.write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"train {train_n} kare · val {val_n} kare")
    print(f"  devralınan analog: {analog_n} · üretilen yeni tip: {train_n + val_n - analog_n}")
    print("  kutu sayısı:", ", ".join(f"{k} {v}" for k, v in sayim.items()))
    print(f"veri tanımı: {yaml_yolu}")
    print(f"özet: {METRIK_YOLU}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
