r"""Yatıklık kapısının kanıt istatistiklerini dört kümede ölçer (13.08 bulgusu).

    python scripts\olc_roll_kaniti.py
    python scripts\olc_roll_kaniti.py --araba ..\demo\girdi\araba.mp4

**Neden var.** 13.08'de `estimate_roll` tanımadığı bir kadran stilinde (araç
hız göstergesi, gerçekte ~0° yatık) 21,3° sahte yatıklık üretti. Uyum kapısı
"desen buraya oturuyor mu" diye soruyordu; yabancı ama çizgili bir kadran bunu
rastgele bir kaymada geçebiliyor. Yeni kapı (ayrıklık) "desen YALNIZCA buraya
mı oturuyor" diye soruyor. Bu script iki istatistiğin (uyum, ayrıklık)
dağılımını dört kümede ölçer ki eşik tahminle değil ölçümle konsun — refine.py
ve roll.py'ın ilk kapılarındaki hatanın (07.08) tekrarı olmasın.

Kümeler:
    dogru    sentetik kadran, DOĞRU kimlik, bilinen yatıklık   → kapı GEÇMELİ
    gurultu  rastgele gürültü                                  → SUSMALI
    yanlis   sentetik kadran, YANLIŞ gösterge kimliği          → SUSMALI
    yabanci  tam çember eşit aralıklı çizgi (saat/araç düzeni) → SUSMALI
             + (varsa) araç paneli videosundan gerçek kareler  → SUSMALI

`dogru` kümesinde yatıklık hatası da ölçülür: kapıyı sıkılaştırırken doğruluk
bozulmadığı buradan görülür.
"""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import gauge_vision.read.roll as RL
from gauge_vision.config import load_gauges
from gauge_vision.detect.refine import refine_dial
from gauge_vision.pipeline import dial_from_box
from gauge_vision.synth.dial import DialLook, render_analog

METRIK_YOLU = Path("outputs/metrics/roll_kaniti.json")
ANALOG = ["PT-101", "TI-205", "FI-310"]
KARE_BASINA = 20        # küme başına gösterge/çift başına kare
ROLL_ARALIK = 15.0      # dogru kümesindeki gerçek yatıklık aralığı (±)
MERKEZ, YARICAP = (256, 256), 205.0


def _kanit(img, gauge, merkez=MERKEZ, yaricap=YARICAP) -> dict | None:
    k = RL.roll_evidence(img, merkez, yaricap, gauge)
    if k is None:
        return None
    return {"uyum": k.match, "ayriklik": k.separation, "roll_deg": k.roll_deg}


def dogru_kumesi(gauges, rng) -> list[dict]:
    satirlar = []
    for gid in ANALOG:
        g = gauges[gid]
        for _ in range(KARE_BASINA):
            deger = g.scale.min + rng.random() * (g.scale.max - g.scale.min)
            roll = rng.uniform(-ROLL_ARALIK, ROLL_ARALIK)
            img, _ = render_analog(g, deger, look=DialLook(roll_deg=roll))
            k = _kanit(img, g)
            if k:
                k.update(gauge=gid, gercek_roll=round(roll, 2),
                         roll_hata=round(abs(k["roll_deg"] - roll), 3))
                satirlar.append(k)
    return satirlar


def gurultu_kumesi(gauges, rng, adet=30) -> list[dict]:
    g = gauges["PT-101"]
    satirlar = []
    for _ in range(adet):
        img = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        k = _kanit(img, g)
        if k:
            satirlar.append(k)
    return satirlar


def yanlis_kumesi(gauges, rng) -> list[dict]:
    # Çiftler bilinçli: PT↔FI süpürme yönü de farklı, TI→PT çizgi sayısı farklı.
    ciftler = [("PT-101", "FI-310"), ("FI-310", "PT-101"), ("TI-205", "PT-101")]
    satirlar = []
    for cizilen, sorulan in ciftler:
        g_ciz, g_sor = gauges[cizilen], gauges[sorulan]
        for _ in range(10):
            deger = g_ciz.scale.min + rng.random() * (g_ciz.scale.max - g_ciz.scale.min)
            img, _ = render_analog(g_ciz, deger, look=DialLook())
            k = _kanit(img, g_sor)
            if k:
                k.update(cizilen=cizilen, sorulan=sorulan)
                satirlar.append(k)
    return satirlar


def yabanci_cizim(rng) -> np.ndarray:
    """Tam çember, eşit aralıklı çizgi halkası — saat/araç göstergesi düzeni.

    13.08 vakasının damıtılmış hâli: çizgiler var, ölü bölge yok, desen
    periyodik. Rastgele bir faz kayması ekleniyor ki korelasyonun kilitleneceği
    "doğru" kayma diye bir şey hiç olmasın.
    """
    img = np.full((512, 512, 3), 235, np.uint8)
    faz = rng.uniform(0, 6.0)
    for i in range(60):
        aci = np.radians(i * 6.0 + faz)
        r1 = YARICAP * (0.86 if i % 5 == 0 else 0.93)
        p1 = (int(256 + r1 * np.cos(aci)), int(256 - r1 * np.sin(aci)))
        p2 = (int(256 + YARICAP * np.cos(aci)), int(256 - YARICAP * np.sin(aci)))
        cv2.line(img, p1, p2, (30, 30, 30), 3, cv2.LINE_AA)
    return img


def yabanci_kumesi(gauges, rng, araba: Path | None, agirlik: Path | None) -> list[dict]:
    g = gauges["PT-101"]
    satirlar = []
    for _ in range(15):
        k = _kanit(yabanci_cizim(rng), g)
        if k:
            k["kaynak"] = "sentetik_saat"
            satirlar.append(k)

    # Gerçek yabancı stil: araç paneli videosu (13.08'in vakası). Video ve
    # ağırlık depo dışında; yoksa sentetik saat düzeni tek başına kalır.
    if araba and araba.exists() and agirlik and agirlik.exists():
        from ultralytics import YOLO
        model = YOLO(str(agirlik))
        cap = cv2.VideoCapture(str(araba))
        toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(0, max(toplam - 1, 0), 12).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, kare = cap.read()
            if not ok:
                continue
            sonuc = model.predict(kare, conf=0.25, verbose=False)[0]
            if len(sonuc.boxes) == 0:
                continue
            i = int(sonuc.boxes.conf.argmax())
            kutu = tuple(float(v) for v in sonuc.boxes.xyxy[i].tolist())
            merkez, yaricap = dial_from_box(kutu)
            daire = refine_dial(kare, merkez, yaricap)
            if daire is not None:
                merkez, yaricap = daire.center_px, daire.radius_px
            k = _kanit(kare, g, merkez, yaricap)
            if k:
                k["kaynak"] = "araba_mp4"
                satirlar.append(k)
        cap.release()
    return satirlar


def _dagilim(satirlar: list[dict], alan: str) -> dict | None:
    v = [s[alan] for s in satirlar if alan in s]
    if not v:
        return None
    return {"n": len(v), "min": round(min(v), 4),
            "ortanca": round(float(np.median(v)), 4), "max": round(max(v), 4)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="yatıklık kapısı kanıt dağılımları")
    ap.add_argument("--araba", type=Path, default=Path("../demo/girdi/araba.mp4"))
    ap.add_argument("--agirlik", type=Path, default=Path("../demo/best.pt"),
                    help="araba videosunda kadranı bulacak YOLO (kaggle_v1)")
    ap.add_argument("--cikti", type=Path, default=METRIK_YOLU)
    args = ap.parse_args()

    gauges = load_gauges()
    rng = np.random.default_rng(0)

    kumeler = {
        "dogru": dogru_kumesi(gauges, rng),
        "gurultu": gurultu_kumesi(gauges, rng),
        "yanlis": yanlis_kumesi(gauges, rng),
        "yabanci": yabanci_kumesi(gauges, rng, args.araba, args.agirlik),
    }

    print("| kume | n | uyum min/ortanca/max | ayriklik min/ortanca/max |")
    print("|---|---|---|---|")
    ozet = {"esikler": {"MIN_UYUM": RL.MIN_UYUM, "MIN_AYRIKLIK": RL.MIN_AYRIKLIK}}
    for ad, satirlar in kumeler.items():
        u, a = _dagilim(satirlar, "uyum"), _dagilim(satirlar, "ayriklik")
        ozet[ad] = {"uyum": u, "ayriklik": a, "satirlar": satirlar}
        if u:
            print(f"| {ad} | {u['n']} | {u['min']} / {u['ortanca']} / {u['max']} "
                  f"| {a['min']} / {a['ortanca']} / {a['max']} |")
        else:
            print(f"| {ad} | 0 | profil uretilemedi | — |")

    dogru = kumeler["dogru"]
    if dogru:
        h = _dagilim(dogru, "roll_hata")
        ozet["dogru_roll_hata"] = h
        print(f"\ndogru kumesinde yatıklık hatası: ortanca {h['ortanca']}° "
              f"· max {h['max']}°")

    # Eski kapının (yalnız uyum ≥ MIN_UYUM) aleyhte kümelerde kaç sahte
    # kestirim üreteceği — hatanın büyüklüğünü rapora taşımak için.
    for ad in ("gurultu", "yanlis", "yabanci"):
        eski = sum(1 for s in kumeler[ad] if s["uyum"] >= RL.MIN_UYUM)
        yeni = sum(1 for s in kumeler[ad] if s["uyum"] >= RL.MIN_UYUM
                   and s["ayriklik"] >= RL.MIN_AYRIKLIK)
        n = len(kumeler[ad])
        ozet.setdefault("sahte_kestirim", {})[ad] = {
            "n": n, "eski_kapi": eski, "yeni_kapi": yeni}
        print(f"{ad}: {n} karede sahte kestirim — eski kapı {eski}, "
              f"yeni kapı {yeni}")

    args.cikti.parent.mkdir(parents=True, exist_ok=True)
    args.cikti.write_text(json.dumps(ozet, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nOlcum: {args.cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
