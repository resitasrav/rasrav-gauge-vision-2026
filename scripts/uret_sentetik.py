"""Sentetik gösterge veri setini üretir (İP3).

    python scripts/uret_sentetik.py                          # 100 görüntü, tohum 0
    python scripts/uret_sentetik.py --sayi 20 --seed 7
    python scripts/uret_sentetik.py --cikti data/synthetic/v1 --gosterge PT-101

Rapora konacak örnek ızgarası da burada üretilir (`--izgara`, varsayılan açık):
`outputs/figures/ip3_ornek_izgara.png`. Figürü üreten kod scriptte duruyor ki
rapordaki görsel her zaman yeniden üretilebilsin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.synth.generate import generate_dataset, load_labels

VARSAYILAN_CIKTI = "data/synthetic/v0"
IZGARA_YOLU = "outputs/figures/ip3_ornek_izgara.png"
OZET_YOLU = "outputs/metrics/ip3_sentetik_ozet.json"
IZGARA_SUTUN = 4
IZGARA_SATIR = 3
KUCUK_RESIM_PX = 220
ETIKET_YUKSEKLIK_PX = 26


def ornek_izgarasi(veri_dizini: Path, cikti: Path, sutun: int, satir: int) -> Path | None:
    """Veri setinden bir tutam örneği tek karede yan yana koyar.

    Amaç gözle denetim: 100 görüntüyü tek tek açmadan "çeşitlilik var mı,
    etiket değeri ibrenin gösterdiği yere uyuyor mu" bakılabilsin.
    """
    kayitlar = load_labels(veri_dizini)
    if not kayitlar:
        return None

    # Baştan sırayla değil, eşit aralıklarla seçiyoruz — ilk 12 kare
    # rastgele sıralı olsa da tüm setin temsilcisi olmayabilir.
    n = min(sutun * satir, len(kayitlar))
    secilen = [kayitlar[round(i * (len(kayitlar) - 1) / max(1, n - 1))] for i in range(n)]

    hucre_h = KUCUK_RESIM_PX + ETIKET_YUKSEKLIK_PX
    tuval = np.full((satir * hucre_h, sutun * KUCUK_RESIM_PX, 3), 255, dtype=np.uint8)

    for i, k in enumerate(secilen):
        img = cv2.imread(str(veri_dizini / k["file"]))
        if img is None:
            continue
        kucuk = cv2.resize(img, (KUCUK_RESIM_PX, KUCUK_RESIM_PX), interpolation=cv2.INTER_AREA)
        r, s = divmod(i, sutun)
        y, x = r * hucre_h, s * KUCUK_RESIM_PX
        tuval[y:y + KUCUK_RESIM_PX, x:x + KUCUK_RESIM_PX] = kucuk
        cv2.putText(tuval, f"{k['gauge_id']}  {k['value']:.1f}  {k['angle_deg']:+.0f}deg",
                    (x + 6, y + KUCUK_RESIM_PX + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)

    cikti.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cikti), tuval)
    return cikti


def olcum_ozeti(veri_dizini: Path, cikti: Path, ozet) -> Path:
    """Üretimin sayısal özetini yazar — günlük rapordaki ölçüm tablosunun kaynağı.

    "Ölçüm yoksa iş bitmiş sayılmaz": veri setinin gerçekten kadranın her yerini
    kapsadığı burada sayıyla gösteriliyor, gözle değil.
    """
    kayitlar = load_labels(veri_dizini)
    gostergeler: dict[str, dict] = {}
    for gid in ozet.per_gauge:
        degerler = [k["value"] for k in kayitlar if k["gauge_id"] == gid]
        acilar = [k["angle_deg"] for k in kayitlar if k["gauge_id"] == gid]
        gostergeler[gid] = {
            "adet": len(degerler),
            "deger_min": round(min(degerler), 2),
            "deger_max": round(max(degerler), 2),
            "aci_min_deg": round(min(acilar), 1),
            "aci_max_deg": round(max(acilar), 1),
        }

    ozet_json = {
        "is_paketi": "IP3",
        "veri_seti": str(veri_dizini).replace("\\", "/"),
        "tohum": ozet.seed,
        "toplam_goruntu": ozet.count,
        "etiketli_goruntu": len(kayitlar),
        "yatiklik_max_deg": round(max(abs(k["roll_deg"]) for k in kayitlar), 1),
        "gostergeler": gostergeler,
    }
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps(ozet_json, indent=2, ensure_ascii=False), encoding="utf-8")
    return cikti


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Sentetik gösterge veri seti üret (İP3)")
    p.add_argument("--sayi", type=int, default=100, help="üretilecek görüntü sayısı")
    p.add_argument("--seed", type=int, default=0, help="tohum — aynı tohum aynı veri seti")
    p.add_argument("--cikti", default=VARSAYILAN_CIKTI)
    p.add_argument("--gosterge", nargs="*", help="sadece bu gösterge id'leri (varsayılan: hepsi)")
    p.add_argument("--izgara", action=argparse.BooleanOptionalAction, default=True,
                   help="rapor için örnek ızgarası üret")
    args = p.parse_args(argv)

    ozet = generate_dataset(args.cikti, count=args.sayi, seed=args.seed,
                            gauge_ids=args.gosterge)

    print(f"{ozet.count} gorüntü üretildi  ·  tohum {ozet.seed}  ·  {ozet.out_dir}")
    for gid, n in ozet.per_gauge.items():
        print(f"   {gid:8s} {n:4d}")
    print(f"etiketler: {ozet.labels_path}")

    print(f"ölçüm özeti: {olcum_ozeti(Path(args.cikti), Path(OZET_YOLU), ozet)}")

    if args.izgara:
        yol = ornek_izgarasi(Path(args.cikti), Path(IZGARA_YOLU), IZGARA_SUTUN, IZGARA_SATIR)
        if yol:
            print(f"örnek ızgarası: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
