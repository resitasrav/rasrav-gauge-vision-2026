"""İP5'in eğitim kümelerini kurar: sentetik · gerçek · karışık.

    python scripts/hazirla_ip5_veri.py
    python scripts/hazirla_ip5_veri.py --sentetik-orani 0.25

Üç yapılandırma kurulur ve üçü de **aynı** doğrulama/test kümesini kullanır
(gerçek fotoğrafların val ve test bölümleri). Değişen tek şey eğitim verisidir;
böylece K1 kararı ("sentetik tek başına yetersiz") kendi verimizde sınanabilir:

    sentetik   yalnızca kendi ürettiğimiz 100 kadran
    gercek     yalnızca gerçek endüstriyel fotoğraflar
    karisik    gerçek + sentetik oranı `--sentetik-orani` olacak kadar sentetik

Gerçek veri: Roboflow-100 `gauge-u2lwv` setinin Hugging Face aynası
(`Francesco/gauge-u2lwv`, CC, 235 gerçek fotoğraf, COCO kutu).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import date
from pathlib import Path

from gauge_vision.detect.dataset import (
    IMAGES_DIR,
    LABELS_DIR,
    hf_parquet_disa_aktar,
    sentetik_disa_aktar,
    veri_yaml_yaz,
)

HAM_GERCEK = "data/raw/hf_gauge_u2lwv"
SENTETIK = "data/synthetic/v0"
CIKTI = "data/detect"
OZET_YOLU = "outputs/metrics/ip5_veri_ozeti.json"

PARQUET = {
    "train": "train-00000-of-00001-70b9c9c9d9c94518.parquet",
    "val": "validation-00000-of-00001-4804bbd661df3206.parquet",
    "test": "test-00000-of-00001-3aa894bcc80dc8b3.parquet",
}


def _tasi(kaynak: Path, hedef: Path, adlar: list[str]) -> None:
    """Seçilen görüntü/etiket çiftlerini kopyalar. İkisi birlikte taşınır."""
    (hedef / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    (hedef / LABELS_DIR).mkdir(parents=True, exist_ok=True)
    for ad in adlar:
        shutil.copyfile(kaynak / IMAGES_DIR / f"{ad}.png", hedef / IMAGES_DIR / f"{ad}.png")
        shutil.copyfile(kaynak / LABELS_DIR / f"{ad}.txt", hedef / LABELS_DIR / f"{ad}.txt")


def _adlar(dizin: Path) -> list[str]:
    return sorted(p.stem for p in (dizin / IMAGES_DIR).glob("*.png"))


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="İP5 eğitim kümelerini kur")
    p.add_argument("--sentetik-orani", type=float, default=0.25,
                   help="karışık kümede sentetik payı (K1: %%25'e kadar kayıp yok)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cikti", default=CIKTI)
    args = p.parse_args(argv)

    ham = Path(HAM_GERCEK)
    eksik = [d for d in PARQUET.values() if not (ham / d).exists()]
    if eksik:
        print(f"gerçek veri yok: {ham} içinde {eksik}")
        return 1

    cikti = Path(args.cikti)
    if cikti.exists():
        shutil.rmtree(cikti)   # kalıntı dosya sessizce eğitime karışmasın

    # --- gerçek fotoğraflar: üç bölüm de ayrı ayrı dönüştürülür ---
    gercek = {}
    for bolum, dosya in PARQUET.items():
        hedef = cikti / "_gercek" / bolum
        gercek[bolum] = hf_parquet_disa_aktar(ham / dosya, hedef, onek=f"real{bolum[:2]}")
        print(f"gerçek/{bolum:5s} {gercek[bolum]:4d} görüntü  (kadranı etiketli olanlar)")

    # --- sentetik ---
    sentetik_dizin = cikti / "_sentetik"
    n_sentetik = sentetik_disa_aktar(SENTETIK, sentetik_dizin)
    print(f"sentetik      {n_sentetik:4d} görüntü")

    # --- karışım: gerçek eğitim kümesine hedef orana ulaşana kadar sentetik ekle ---
    n_gercek_train = gercek["train"]
    oran = args.sentetik_orani
    # n_syn / (n_real + n_syn) = oran  →  n_syn = oran·n_real / (1 − oran)
    n_karisim = min(n_sentetik, round(oran * n_gercek_train / (1 - oran))) if oran < 1 else n_sentetik

    rng = random.Random(args.seed)
    secilen = rng.sample(_adlar(sentetik_dizin), n_karisim)

    yapilandirmalar = {
        "sentetik": [(sentetik_dizin, _adlar(sentetik_dizin))],
        "gercek": [(cikti / "_gercek" / "train", _adlar(cikti / "_gercek" / "train"))],
        "karisik": [(cikti / "_gercek" / "train", _adlar(cikti / "_gercek" / "train")),
                    (sentetik_dizin, secilen)],
    }

    ozet = {
        "is_paketi": "IP5",
        "tarih": date.today().isoformat(),
        "gercek_kaynak": "Francesco/gauge-u2lwv (Roboflow-100, CC) — gerçek fotoğraf",
        "sentetik_kaynak": SENTETIK,
        "ortak_val": gercek["val"],
        "ortak_test": gercek["test"],
        "yapilandirmalar": {},
    }

    for ad, parcalar in yapilandirmalar.items():
        hedef = cikti / ad / "train"
        toplam = 0
        dagilim = {}
        for kaynak, adlar in parcalar:
            _tasi(kaynak, hedef, adlar)
            etiket = "sentetik" if "_sentetik" in str(kaynak) else "gercek"
            dagilim[etiket] = dagilim.get(etiket, 0) + len(adlar)
            toplam += len(adlar)

        veri_yaml_yaz(cikti / ad / "gauge.yaml",
                      train=hedef / IMAGES_DIR,
                      val=cikti / "_gercek" / "val" / IMAGES_DIR,
                      test=cikti / "_gercek" / "test" / IMAGES_DIR)

        sentetik_pay = dagilim.get("sentetik", 0) / toplam if toplam else 0.0
        ozet["yapilandirmalar"][ad] = {"train": toplam, "dagilim": dagilim,
                                       "sentetik_orani": round(sentetik_pay, 3)}
        print(f"{ad:9s} train={toplam:4d}  {dagilim}  sentetik oranı %{sentetik_pay*100:.0f}")

    yol = Path(OZET_YOLU)
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nözet: {yol}")
    print(f"ortak val={gercek['val']} · ortak test={gercek['test']} (üç yapılandırma da aynısını kullanır)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
