"""Okuma doğruluğunu BAĞIMSIZ bir sette ölçer (gerçek endüstriyel zemin).

    python scripts/olc_gercek_zemin.py
    python scripts/olc_gercek_zemin.py --ornek 1000 --kaydet-figur

**Neden bu ölçüm var.** Bugüne kadarki bütün okuma sayıları (İP6 0,123° ·
İP7 %0,129 · İP8 %0,473) **bizim ürettiğimiz** kadranlarda alındı — sentetik
üreteçte de, ekran çekiminde de kareyi biz çizdik. "Yöntem çalışıyor" iddiası
başkasının verisinde sınanmadan tamamlanmaz.

Kaynak `Synanthropic/reading-analog-gauge` (Hugging Face, herkese açık,
34.370 görüntü). Kadran render, **zemin gerçek endüstriyel fotoğraf**; ibre
ucu, kadran merkezi ve skala uçları etiketli. Etiketin görüntüyle tuttuğu
27.08'de sınandı: 150 karede okuyucumuz etiketle 149 kez 15° içinde uyuştu,
yani etiket ibre UCUdur ve set okuma ölçümüne uygundur.

**İki mod ayrı ayrı ölçülür, çünkü iki farklı şeyi söylerler:**

1. `--mod etiket` — merkez ETİKETTEN alınır. Ölçülen şey **yalnız ibre
   okuma yöntemidir** (İP6'nın kıyaslanabilir karşılığı).
2. `--mod rafine` — merkez `detect/refine.py` ile GÖRÜNTÜDEN bulunur.
   Aradaki fark merkez kestiriminin maliyetidir; 05.08'de bu fark sentetikte
   13 kat ölçülmüştü ve zincirin en duyarlı büyüklüğü budur.

**Açı hatası yüzdeye çevrilir.** Skala uçları etiketli olduğu için her
kadranın kendi süpürmesi biliniyor; hata `açı_hatası / süpürme × 100` ile
tam skala yüzdesine dönüyor ve **hedef %5 ile doğrudan kıyaslanabilir** hâle
geliyor. Süpürme kadran başına değiştiği için sabit bir bölen kullanılmaz.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.detect.refine import refine_dial
from gauge_vision.read.needle import read_needle_angle

KAYNAK_ZIP = Path("data/raw/hf_analog_gauge/keypoint.zip")
METRIK_YOLU = Path("outputs/metrics/gercek_zemin_okuma.json")
FIGUR_YOLU = Path("outputs/figures/gercek_zemin_okuma.png")

# Tarama halkası kadranın kenarına değil biraz içine oturmalı: bezel ve
# rakamlar halkaya girerse ibre imzası bozulur (aynı oran `pipeline`'da da var).
HALKA_ORANI = 0.92


def _noktalar(stem: str) -> list[tuple[int, int]] | None:
    """Dosya adından dört anahtar nokta: start · center · end · tip."""
    pts: list[tuple[int, int]] = []
    for parca in stem.split("_"):
        if "-" not in parca:
            continue
        a, _, b = parca.partition("-")
        if not (a.isdigit() and b.isdigit()):
            continue
        pts.append((int(a), int(b)))
        if len(pts) == 4:
            return pts
    return None


def _aci(merkez, uc) -> float:
    """Görüntü koordinatından açı — y ekseni aşağı arttığı için eksi işareti."""
    return math.degrees(math.atan2(-(uc[1] - merkez[1]), uc[0] - merkez[0]))


def _sapma(a: float, b: float) -> float:
    """İki açı arasındaki en kısa mutlak fark (0-180)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _supurme(aci_start: float, aci_end: float) -> float:
    """Skalanın süpürme açısı — saat yönünde start'tan end'e.

    Tipik kadran saat yönünde (CCW konvansiyonunda AZALAN açıyla) ilerler;
    `(start - end) mod 360` bu yayı verir. Sonuç 0'a çok yakınsa etiket
    bozuktur ve kare elenir — sıfıra bölüp sonsuz hata üretmek, ölçümü tek
    bir kötü etiketle çöpe atardı.
    """
    return (aci_start - aci_end) % 360


def olc(zip_yolu: Path, ornek: int, mod: str, tohum: int) -> dict:
    if not zip_yolu.exists():
        raise SystemExit(
            f"kaynak yok: {zip_yolu}\n"
            "indir: curl -L -o data/raw/hf_analog_gauge/keypoint.zip "
            "https://huggingface.co/datasets/Synanthropic/reading-analog-gauge/"
            "resolve/main/keypoint.zip")

    z = zipfile.ZipFile(zip_yolu)
    adlar = [a for a in z.namelist() if a.lower().endswith((".jpg", ".png"))]
    rng = np.random.default_rng(tohum)
    sec = [adlar[i] for i in rng.choice(len(adlar), min(ornek, len(adlar)),
                                        replace=False)]

    aci_hatalari: list[float] = []
    yuzde_hatalari: list[float] = []
    merkez_sapmalari: list[float] = []
    okunamayan = 0
    elenen = 0

    for i, ad in enumerate(sec):
        pts = _noktalar(Path(ad).stem)
        if pts is None:
            elenen += 1
            continue
        start, merkez_gt, end, tip = pts
        img = cv2.imdecode(np.frombuffer(z.read(ad), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            elenen += 1
            continue

        yaricap = max(math.dist(merkez_gt, start), math.dist(merkez_gt, end))
        supurme = _supurme(_aci(merkez_gt, start), _aci(merkez_gt, end))
        if supurme < 30:
            elenen += 1
            continue

        merkez = tuple(merkez_gt)
        r = yaricap
        if mod == "rafine":
            daire = refine_dial(img, merkez, yaricap)
            if daire is None:
                okunamayan += 1
                continue
            merkez, r = daire.center_px, daire.radius_px
            merkez_sapmalari.append(math.dist(merkez, merkez_gt) / yaricap * 100)

        olcum = read_needle_angle(img, merkez, r * HALKA_ORANI, method="polar")
        if olcum is None:
            okunamayan += 1
            continue

        hata = _sapma(_aci(merkez_gt, tip), olcum.angle_img_deg)
        aci_hatalari.append(hata)
        yuzde_hatalari.append(hata / supurme * 100)

        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(sec)}")

    if not aci_hatalari:
        raise SystemExit("hiç okuma üretilemedi — kaynak ya da mod hatalı")

    a = np.array(aci_hatalari)
    y = np.array(yuzde_hatalari)
    denenen = len(sec) - elenen
    return {
        "mod": mod,
        "ornek": len(sec),
        "elenen_etiket": elenen,
        "okunan": len(a),
        "okunamayan": okunamayan,
        "kapsama_yuzde": round(100 * len(a) / max(denenen, 1), 2),
        "aci_hatasi_deg": {
            "ortalama": round(float(a.mean()), 4),
            "medyan": round(float(np.median(a)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "maks": round(float(a.max()), 4)},
        "tam_skala_hatasi_yuzde": {
            "ortalama": round(float(y.mean()), 4),
            "medyan": round(float(np.median(y)), 4),
            "p95": round(float(np.percentile(y, 95)), 4),
            "maks": round(float(y.max()), 4),
            "hedefin_altinda_oran": round(float((y < 5).mean() * 100), 2)},
        "merkez_sapmasi_yuzde": (
            {"ortalama": round(float(np.mean(merkez_sapmalari)), 4),
             "p95": round(float(np.percentile(merkez_sapmalari, 95)), 4)}
            if merkez_sapmalari else None),
        "_ham_yuzde": y.tolist(),
    }


def _figur(sonuclar: list[dict], yol: Path) -> None:
    """Hata dağılımı — rapora giren figür. Kod `scripts/` altında ki üretilebilsin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eksen = plt.subplots(1, len(sonuclar), figsize=(6 * len(sonuclar), 4),
                              squeeze=False)
    for i, s in enumerate(sonuclar):
        ax = eksen[0][i]
        y = np.array(s["_ham_yuzde"])
        ax.hist(np.clip(y, 0, 20), bins=60, color="#3b7dd8", edgecolor="none")
        ax.axvline(5, color="#d94f4f", linestyle="--", linewidth=2,
                   label="hedef %5")
        ax.axvline(y.mean(), color="#2b8a3e", linewidth=2,
                   label=f"ortalama %{y.mean():.2f}")
        ax.set_title(f"mod: {s['mod']} · n={s['okunan']} · "
                     f"%5 altı: {s['tam_skala_hatasi_yuzde']['hedefin_altinda_oran']:.1f}%")
        ax.set_xlabel("tam skala hatası (%)")
        ax.set_ylabel("kare sayısı")
        ax.legend()
    fig.suptitle("Bağımsız sette okuma hatası — gerçek endüstriyel zemin")
    fig.tight_layout()
    yol.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(yol, dpi=130)
    print(f"figür: {yol}")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Bağımsız sette okuma ölçümü")
    p.add_argument("--zip", default=str(KAYNAK_ZIP))
    p.add_argument("--ornek", type=int, default=500)
    p.add_argument("--tohum", type=int, default=0)
    p.add_argument("--mod", choices=("etiket", "rafine", "ikisi"), default="ikisi")
    p.add_argument("--kaydet-figur", action="store_true")
    args = p.parse_args(argv)

    modlar = ["etiket", "rafine"] if args.mod == "ikisi" else [args.mod]
    sonuclar = []
    for mod in modlar:
        print(f"\nmod: {mod} · örnek {args.ornek}")
        sonuclar.append(olc(Path(args.zip), args.ornek, mod, args.tohum))

    print(f"\n{'mod':10s} {'okunan':>7s} {'kapsama':>8s} {'açı ort':>9s} "
          f"{'%TS ort':>9s} {'%TS p95':>9s} {'%5 altı':>8s}")
    for s in sonuclar:
        print(f"{s['mod']:10s} {s['okunan']:>7d} {s['kapsama_yuzde']:>7.1f}% "
              f"{s['aci_hatasi_deg']['ortalama']:>8.3f}° "
              f"{s['tam_skala_hatasi_yuzde']['ortalama']:>8.3f}% "
              f"{s['tam_skala_hatasi_yuzde']['p95']:>8.3f}% "
              f"{s['tam_skala_hatasi_yuzde']['hedefin_altinda_oran']:>7.1f}%")

    if args.kaydet_figur:
        _figur(sonuclar, FIGUR_YOLU)

    kayit = {"tarih": date.today().isoformat(),
             "kaynak": "Synanthropic/reading-analog-gauge (HuggingFace)",
             "not": "kadran render, zemin gerçek endüstriyel fotoğraf; "
                    "etiket ibre ucu (27.08'de 150 karede doğrulandı)",
             "sonuclar": [{k: v for k, v in s.items() if k != "_ham_yuzde"}
                          for s in sonuclar]}
    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    METRIK_YOLU.write_text(json.dumps(kayit, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nyazıldı: {METRIK_YOLU}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
