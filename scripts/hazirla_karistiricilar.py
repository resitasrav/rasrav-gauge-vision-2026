"""Modeli fiilen yanıltan nesneleri gerçek videolardan kırpıp toplar (İP17).

    python scripts/hazirla_karistiricilar.py
    python scripts/hazirla_karistiricilar.py --kirpim-basina 40 --conf 0.25

27.08'de 14 gerçek video işlendi ve karede TEK BİR kadran olmadığı hâlde 383
"gauge" kutusu üretildiği ölçüldü. Gözle doğrulananlar: forkliftin ön tekerleği,
elektrikli vantilatör, beyaz ikaz lambasının düz camı, panoya basılı direnç
sembolü.

Bu script tam olarak o kutuları kırpar. Ürettiği şey **zor negatif**tir: rastgele
arka plan değil, modelin *bu ağırlıkla* yanıldığı nesnelerin ta kendisi. Sonra
`hazirla_ip17_buton.py` bunları eğitim karelerine ETİKETSİZ yapıştırır ve model
"bunlar gösterge değil" demeyi öğrenir.

**Neden tüm kareyi negatif olarak almıyoruz:** bir kareyi boş etiketle koymak
"bu karede hiç gösterge yok" der. Fabrika geniş çekiminde bu doğru, ama buton
panosu videolarında değil — orada gerçek lamba ve butonlar var ve onları
arka plan diye öğretmek yeni bir hata sınıfı açardı. Kırpım yaklaşımı bu tuzağı
tamamen atlıyor: sadece yanlış nesne alınıyor, bağlamı değil.

Kaynak videolar bilinçle seçili — hepsinde okunabilir kadran YOK:
    1, 2, genis, genis2   fabrika/depo geniş çekimi (teker, lamba, makine)
    10, 11                buton ve kontrol panosu (buton, kol, basılı sembol)
    7                     vantilatör (dönen kanat — her karede başka açı)
7.mp4'te gerçek bir yuvarlak saat de var; kutu bazlı kırpım aldığımız için o
kutular ayıklanıyor (bkz. `--en-az-yaricap` ve elle dışlama listesi).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

VARSAYILAN_AGIRLIK = KOK / "runs/detect/models/ip5/cok_sinif/weights/best.pt"
VIDEO_KOK = KOK.parent / "demo" / "girdi" / "video"
HEDEF = KOK / "data" / "detect" / "karistiricilar"
METRIK = KOK / "outputs" / "metrics" / "karistiricilar.json"

# (video, hedef_etiket, aciklama)
#
# `hedef_etiket` kırpımın eğitimde NE OLARAK kullanılacağını söyler ve bu ayrım
# ÖLÇÜMLE öğrenildi. İlk sürümde hepsi negatifti; sonuç (27.08, gerçek video):
#     kadran olmayan videolarda gauge kutusu   64 -> 4    (istenen)
#     10.mp4'te lamp kutusu                    49 -> 1    (GERİLEME)
# Sebep açık: 10.mp4'ün kırpımı gerçek bir ikaz lambasının camı. "Gösterge
# değil" diye öğretince model onu LAMBA olarak da göremez oldu.
#
# Doğrusu: bir kırpım gerçekten bir sınıfın örneğiyse o sınıfla etiketlenir.
# Negatif yalnız hiçbir sınıfa girmeyen nesneler içindir (teker, vantilatör).
# Bu ayrıca daha bilgilendirici: "bu bir lamba" demek, "bu gösterge değil"
# demekten daha çok şey öğretir.
KAYNAKLAR = [
    ("1", "negatif", "fabrika genis cekim - makine govdesi, uzak isik"),
    ("2", "negatif", "depo - forklift tekeri (bijon deseni kadran taklidi)"),
    ("genis", "negatif", "tekstil fabrikasi genis"),
    ("genis2", "negatif", "fabrika genis"),
    ("7", "negatif", "vantilator kanadi - donen kanat her karede baska aci"),
    ("10", "lamp", "buton panosu - isikli buton = ikaz lambasi"),
    ("11", "keypad", "kontrol panosu - butonlu pano"),
]

# 7.mp4'te gerçek bir yuvarlak saat var ve o kırpılmamalı. Saat karenin sağ-alt
# bölgesinde duruyor, vantilatör sağ-üstte; kaba bir bölge dışlaması yeterli
# çünkü amaç mükemmel ayıklama değil, temiz negatif toplamak.
DISLA = {"7": lambda x1, y1, x2, y2, w, h: (y1 + y2) / 2 > h * 0.45}

MIN_KENAR_PX = 48          # bundan küçük kırpım eğitimde bilgi taşımıyor


def _model(yol: Path):
    from ultralytics import YOLO
    if not yol.exists():
        raise SystemExit(f"agirlik yok: {yol}")
    return YOLO(str(yol))


def video_kirp(ad: str, model, hedef: Path, conf: float,
               kirpim_basina: int, ornek_kare: int) -> int:
    yol = VIDEO_KOK / f"{ad}.mp4"
    if not yol.exists():
        print(f"  {ad}: video yok, atlandi")
        return 0
    cap = cv2.VideoCapture(str(yol))
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    from gauge_vision.pipeline import detect_objects

    n = 0
    for f in np.linspace(0, max(toplam - 1, 0), ornek_kare).astype(int):
        if n >= kirpim_basina:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, kare = cap.read()
        if not ok:
            continue
        h, w = kare.shape[:2]
        for d in detect_objects(kare, model, conf=conf):
            if d.tip != "analog" or n >= kirpim_basina:
                continue
            x1, y1, x2, y2 = (int(v) for v in d.box_xyxy)
            if min(x2 - x1, y2 - y1) < MIN_KENAR_PX:
                continue
            if ad in DISLA and DISLA[ad](x1, y1, x2, y2, w, h):
                continue
            kirpim = kare[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if kirpim.size == 0:
                continue
            cv2.imwrite(str(hedef / f"{ad}_{int(f):06d}_{n:03d}.png"), kirpim)
            n += 1
    cap.release()
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agirlik", type=Path, default=VARSAYILAN_AGIRLIK)
    p.add_argument("--hedef", type=Path, default=HEDEF)
    p.add_argument("--conf", type=float, default=0.25,
                   help="dusuk tutulur: amac modelin YANILDIGI her kutuyu toplamak")
    p.add_argument("--kirpim-basina", type=int, default=60, help="video basina kirpim")
    p.add_argument("--ornek-kare", type=int, default=120, help="video basina taranan kare")
    a = p.parse_args(argv)

    a.hedef.mkdir(parents=True, exist_ok=True)
    for eski in a.hedef.rglob("*.png"):
        eski.unlink()

    model = _model(a.agirlik)
    sayim: dict[str, dict[str, int]] = {}
    print(f"karistirici kirpimlari -> {a.hedef}")
    for ad, etiket, aciklama in KAYNAKLAR:
        kova = a.hedef / etiket
        kova.mkdir(parents=True, exist_ok=True)
        n = video_kirp(ad, model, kova, a.conf, a.kirpim_basina, a.ornek_kare)
        sayim.setdefault(etiket, {})[ad] = n
        print(f"  {ad:8s} -> {etiket:8s} {n:4d} kirpim   ({aciklama})")

    toplam = sum(n for kova in sayim.values() for n in kova.values())
    METRIK.parent.mkdir(parents=True, exist_ok=True)
    METRIK.write_text(json.dumps(
        {"toplam": toplam, "etiket_basina": sayim, "conf": a.conf,
         "agirlik": str(a.agirlik.relative_to(KOK))}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\ntoplam {toplam} kirpim · {METRIK.relative_to(KOK)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
