"""Dijital panel okuma doğruluğu — İP11'in bitti kriteri.

    python scripts/olc_ip11.py
    python scripts/olc_ip11.py --zor          # bozulma eksenleriyle birlikte

Ölçülen iki büyüklük:

    hane doğruluğu   doğru okunan hane / toplam hane   (karakter seviyesi)
    dizge doğruluğu  tamamı doğru okunan panel / toplam (tam eşleşme)

İkisi ayrı raporlanıyor çünkü ayrı şeyler söylüyorlar. Dört haneli bir panelde
%95 hane doğruluğu, dizgelerin ancak %81'inin tam doğru olması demektir
(0,95⁴). **Yayınlanan sayı dizgedir**, dolayısıyla asıl ölçüt ikincisidir; hane
doğruluğu ise hatanın nerede olduğunu gösterir.

Sönük segmentler ayrı bir eksen olarak taranıyor: gerçek panellerde sönük
segment görünür kalır ve okumanın asıl zorluğu budur ("1"in yanındaki sönük
segmentler "8"e benzer bir hayalet bırakır). Ablasyon, bu zorluğun ne kadarını
yöntemin kaldırdığını gösterir.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.read.digital import read_digital
from gauge_vision.synth.degrade import Bozulma, bozulmalar_uygula
from gauge_vision.synth.dial import DialTruth
from gauge_vision.synth.digital import bicimle, render_digital

METRIK_YOLU = "outputs/metrics/ip11_dijital.json"
FIGUR_YOLU = "outputs/figures/ip11_dijital.png"
TOHUM = 11


def _degerler(gauge, n: int, rng) -> list[float]:
    """Panelin aralığını kapsayan değerler + kenar durumları.

    Rastgele örnekleme tek başına yetmez: sıfır, negatif ve taşma sınırları
    nadiren rastgele düşer ama sahada sık görülür.
    """
    a = gauge.raw.get("range") or {}
    alt, ust = float(a.get("min", 0)), float(a.get("max", 100))
    ozel = [0.0, alt, ust, alt / 2, ust / 2]
    kalan = max(0, n - len(ozel))
    return ozel + list(rng.uniform(alt, ust, kalan))


def kosu(gauge, degerler, bozulma: Bozulma, sonuk_goster: bool, rng) -> dict:
    hane_dogru = hane_toplam = dizge_dogru = 0
    okunamayan = 0
    guvenler = []
    hatalar = []

    d = gauge.digits or {}
    count, decimals = int(d.get("count", 4)), int(d.get("decimals", 1))
    allow_minus = bool(d.get("allow_minus", True))

    for deger in degerler:
        img, truth = render_digital(gauge, deger, sonuk_goster=sonuk_goster)
        if bozulma.etkin:
            # Dijital panelde kadran geometrisi yok; bozulma modülü DialTruth
            # beklediği için asgari bir nesne veriliyor. Perspektif bu ölçümde
            # kullanılmıyor (panel düz varsayılıyor — yöntemin bilinen sınırı).
            sahte = DialTruth(gauge_id=gauge.id, value=deger, angle_deg=0.0,
                              roll_deg=0.0, angle_img_deg=0.0,
                              center_px=(img.shape[1] // 2, img.shape[0] // 2),
                              tip_px=(0, 0), radius_px=img.shape[0] // 3,
                              bbox_xyxy=truth.panel_bbox_xyxy)
            img, _ = bozulmalar_uygula(img, sahte, bozulma, rng)

        okuma = read_digital(img, gauge)
        guvenler.append(okuma.conf)

        beklenen = bicimle(deger, count, decimals, allow_minus)
        beklenen_haneler = [c for c in beklenen if c != "."]

        if okuma.value is None:
            okunamayan += 1
            hane_toplam += len(beklenen_haneler)
            continue

        okunan = bicimle(okuma.value, count, decimals, allow_minus)
        okunan_haneler = [c for c in okunan if c != "."]

        hane_toplam += len(beklenen_haneler)
        hane_dogru += sum(1 for a, b in zip(beklenen_haneler, okunan_haneler) if a == b)
        if okunan == beklenen:
            dizge_dogru += 1
        else:
            hatalar.append({"gercek": beklenen, "okunan": okunan})

    n = len(degerler)
    return {
        "panel": n,
        "hane_dogrulugu": round(hane_dogru / hane_toplam, 4) if hane_toplam else 0.0,
        "dizge_dogrulugu": round(dizge_dogru / n, 4) if n else 0.0,
        "okunamayan": okunamayan,
        "ortalama_guven": round(float(np.mean(guvenler)), 3) if guvenler else 0.0,
        "hatali_ornekler": hatalar[:8],
    }


def figur(rapor: dict, cikti: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kosullar = list(rapor["kosullar"].keys())
    hane = [rapor["kosullar"][k]["hane_dogrulugu"] * 100 for k in kosullar]
    dizge = [rapor["kosullar"][k]["dizge_dogrulugu"] * 100 for k in kosullar]
    x = np.arange(len(kosullar))

    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(kosullar)), 4.2))
    ax.bar(x - 0.2, hane, 0.4, label="hane doğruluğu", color="#2980b9")
    ax.bar(x + 0.2, dizge, 0.4, label="dizge doğruluğu (yayınlanan sayı)",
           color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(kosullar, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    ax.set_title("İP11 — dijital panel okuma doğruluğu")

    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=130)
    plt.close(fig)
    return cikti


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Dijital panel okuma doğruluğu (İP11)")
    p.add_argument("--panel", type=int, default=60, help="koşul başına panel sayısı")
    p.add_argument("--zor", action="store_true", help="bozulma eksenlerini de tara")
    p.add_argument("--figur", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    gauges = load_gauges()
    dijitaller = [g for g in gauges.values() if g.type == "digital"]
    if not dijitaller:
        print("envanterde dijital gösterge yok")
        return 1
    gauge = dijitaller[0]
    rng = np.random.default_rng(TOHUM)
    degerler = _degerler(gauge, args.panel, rng)

    kosullar: list[tuple[str, Bozulma, bool]] = [
        ("temiz", Bozulma(), True),
        ("sönük segment yok", Bozulma(), False),
    ]
    if args.zor:
        kosullar += [
            ("bulanık 9px", Bozulma(bulaniklik_px=9), True),
            ("bulanık 15px", Bozulma(bulaniklik_px=15), True),
            ("düşük ışık ×0.4", Bozulma(isik_kazanci=0.4), True),
            ("düşük ışık ×0.15", Bozulma(isik_kazanci=0.15), True),
            ("parlama %50", Bozulma(parlama=0.5), True),
            ("jpeg q25", Bozulma(jpeg_kalite=25), True),
        ]

    print(f"gösterge: {gauge.id} · {gauge.digits} · {len(degerler)} panel/koşul\n")
    print(f"{'koşul':>20s} {'hane':>8s} {'dizge':>8s} {'okunamayan':>11s} {'güven':>7s}")

    rapor = {"is_paketi": "IP11", "tarih": date.today().isoformat(),
             "gosterge": gauge.id, "panel_basina": len(degerler), "kosullar": {}}

    for ad, bozulma, sonuk in kosullar:
        o = kosu(gauge, degerler, bozulma, sonuk, np.random.default_rng(TOHUM))
        rapor["kosullar"][ad] = o
        print(f"{ad:>20s} {o['hane_dogrulugu']*100:>7.1f}% {o['dizge_dogrulugu']*100:>7.1f}% "
              f"{o['okunamayan']:>7d}/{o['panel']:<3d} {o['ortalama_guven']:>7.2f}")

    yol = Path(METRIK_YOLU)
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(rapor, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nölçüm: {yol}")

    ilk = rapor["kosullar"]["temiz"]
    if ilk["hatali_ornekler"]:
        print("hatalı örnekler (temiz koşul):")
        for h in ilk["hatali_ornekler"]:
            print(f"   gerçek '{h['gercek']}' → okunan '{h['okunan']}'")

    if args.figur:
        print(f"figür: {figur(rapor, Path(FIGUR_YOLU))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
