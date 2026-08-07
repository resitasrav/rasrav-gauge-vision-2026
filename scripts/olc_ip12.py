"""Lamba ve vana durum doğruluğu — İP12'nin bitti kriteri.

    python scripts/olc_ip12.py
    python scripts/olc_ip12.py --zor

Ölçülen: **durum doğruluğu** (doğru sınıflanan kare / toplam) ve **karışıklık
matrisi**. İkisi ayrı bilgi veriyor: toplam doğruluk "ne kadar iyi", matris
"hangi durumu hangisiyle karıştırıyor" der. Kırmızıyı yeşil okumak ile kırmızıyı
okuyamamak endüstride aynı şey değildir — biri sessiz bir arıza, diğeri görünür
bir eksiklik.

Vanada ayrıca **tolerans eğrisi** taranıyor: kol ideal açıdan sapmaya
başladığında okuma ne zaman kesiliyor? Envanterdeki "±20°" notu burada sayıya
dönüşüyor.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.read.state import read_state
from gauge_vision.synth.degrade import Bozulma, bozulmalar_uygula
from gauge_vision.synth.dial import DialTruth
from gauge_vision.synth.state import render_lamp, render_valve

METRIK_YOLU = "outputs/metrics/ip12_lamba_vana.json"
FIGUR_YOLU = "outputs/figures/ip12_lamba_vana.png"
TOHUM = 12


def _boz(img, bozulma: Bozulma, gauge_id: str, rng):
    if not bozulma.etkin:
        return img
    h, w = img.shape[:2]
    sahte = DialTruth(gauge_id=gauge_id, value=0.0, angle_deg=0.0, roll_deg=0.0,
                      angle_img_deg=0.0, center_px=(w // 2, h // 2), tip_px=(0, 0),
                      radius_px=min(h, w) // 3,
                      bbox_xyxy=(0, 0, w, h))
    return bozulmalar_uygula(img, sahte, bozulma, rng)[0]


def kosu_lamba(gauge, bozulma: Bozulma, tekrar: int, rng) -> dict:
    durumlar = gauge.state_names
    dogru = 0
    okunamayan = 0
    matris: dict[str, dict[str, int]] = {d: {} for d in durumlar}
    guvenler = []

    for durum in durumlar:
        for i in range(tekrar):
            # `off` durumunda mercek rengi değişir; ikisi de `off`tur.
            renk = durumlar[(i % max(1, len(durumlar) - 1))] if durum == "off" else None
            renk = renk if renk in ("red", "green", "yellow", "blue") else None
            img, truth = render_lamp(gauge, durum, renk=renk)
            img = _boz(img, bozulma, gauge.id, rng)

            okuma = read_state(img, gauge)
            guvenler.append(okuma.conf)
            tahmin = okuma.value if okuma.value is not None else "okunamadı"
            matris[durum][tahmin] = matris[durum].get(tahmin, 0) + 1
            if okuma.value == durum:
                dogru += 1
            elif okuma.value is None:
                okunamayan += 1

    toplam = len(durumlar) * tekrar
    return {
        "kare": toplam,
        "dogruluk": round(dogru / toplam, 4),
        "okunamayan": okunamayan,
        # Sessiz hata: okundu ama YANLIŞ durum. Asıl tehlikeli olan budur.
        "yanlis_sinif": toplam - dogru - okunamayan,
        "ortalama_guven": round(float(np.mean(guvenler)), 3),
        "karisiklik": matris,
    }


def kosu_vana(gauge, bozulma: Bozulma, tekrar: int, rng) -> dict:
    durumlar = gauge.state_names
    dogru = okunamayan = 0
    matris: dict[str, dict[str, int]] = {d: {} for d in durumlar}
    guvenler = []

    for durum in durumlar:
        for i in range(tekrar):
            # Küçük montaj sapmaları: gerçek vanalar tam yatay/dik durmaz.
            sapma = float(rng.uniform(-8.0, 8.0))
            img, _ = render_valve(gauge, durum, sapma_deg=sapma)
            img = _boz(img, bozulma, gauge.id, rng)

            okuma = read_state(img, gauge)
            guvenler.append(okuma.conf)
            tahmin = okuma.value if okuma.value is not None else "okunamadı"
            matris[durum][tahmin] = matris[durum].get(tahmin, 0) + 1
            if okuma.value == durum:
                dogru += 1
            elif okuma.value is None:
                okunamayan += 1

    toplam = len(durumlar) * tekrar
    return {
        "kare": toplam,
        "dogruluk": round(dogru / toplam, 4),
        "okunamayan": okunamayan,
        "yanlis_sinif": toplam - dogru - okunamayan,
        "ortalama_guven": round(float(np.mean(guvenler)), 3),
        "karisiklik": matris,
    }


def tolerans_egrisi(gauge) -> list[dict]:
    """Kol ideal açıdan saptıkça okuma ne zaman kesiliyor?

    Envanterdeki "±20° içindeyse o duruma sayılır" notu burada sayıya dönüşür.
    Arada kalan açıda okuma ÜRETİLMEMELİ: yarı açık bir vana gerçek bir
    durumdur ve "açık" diye yayınlanması tehlikelidir.
    """
    satirlar = []
    for sapma in range(0, 50, 5):
        img, _ = render_valve(gauge, "open", sapma_deg=float(sapma))
        okuma = read_state(img, gauge)
        satirlar.append({"sapma_deg": sapma,
                         "okunan": okuma.value,
                         "conf": round(okuma.conf, 3)})
    return satirlar


def figur(rapor: dict, cikti: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for ad, renk in (("lamba", "#c0392b"), ("vana", "#2980b9")):
        kosullar = list(rapor[ad].keys())
        dogruluk = [rapor[ad][k]["dogruluk"] * 100 for k in kosullar]
        x = np.arange(len(kosullar))
        ax1.plot(x, dogruluk, "o-", color=renk, label=ad)
    ax1.set_xticks(np.arange(len(rapor["lamba"])))
    ax1.set_xticklabels(list(rapor["lamba"].keys()), rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("durum doğruluğu %")
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_title("İP12 — koşullara göre doğruluk")

    egri = rapor["vana_tolerans"]
    sapmalar = [e["sapma_deg"] for e in egri]
    conf = [e["conf"] * 100 for e in egri]
    okundu = [100 if e["okunan"] else 0 for e in egri]
    ax2.plot(sapmalar, conf, "o-", color="#2980b9", label="güven %")
    ax2.step(sapmalar, okundu, where="post", color="#27ae60", ls="--",
             label="okuma üretildi mi")
    ax2.axvline(20, color="gray", ls=":", label="envanterdeki ±20°")
    ax2.set_xlabel("kolun ideal açıdan sapması (°)")
    ax2.set_ylabel("%")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title("Vana toleransı — nerede susuyor?")

    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=130)
    plt.close(fig)
    return cikti


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Lamba/vana durum doğruluğu (İP12)")
    p.add_argument("--tekrar", type=int, default=30, help="durum başına kare")
    p.add_argument("--zor", action="store_true")
    p.add_argument("--figur", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    gauges = load_gauges()
    lamba = next((g for g in gauges.values() if g.type == "lamp"), None)
    vana = next((g for g in gauges.values() if g.type == "valve"), None)
    if lamba is None or vana is None:
        print("envanterde lamba ya da vana yok")
        return 1

    kosullar: list[tuple[str, Bozulma]] = [("temiz", Bozulma())]
    if args.zor:
        kosullar += [
            ("düşük ışık ×0.4", Bozulma(isik_kazanci=0.4)),
            ("düşük ışık ×0.15", Bozulma(isik_kazanci=0.15)),
            ("parlama %50", Bozulma(parlama=0.5)),
            ("bulanık 9px", Bozulma(bulaniklik_px=9)),
            ("jpeg q25", Bozulma(jpeg_kalite=25)),
            ("eğik 25°", Bozulma(egiklik_deg=25)),
        ]

    rapor = {"is_paketi": "IP12", "tarih": date.today().isoformat(),
             "lamba_id": lamba.id, "vana_id": vana.id,
             "lamba": {}, "vana": {}}

    print(f"lamba {lamba.id}: {lamba.state_names} · vana {vana.id}: {vana.state_names}\n")
    print(f"{'koşul':>18s} {'lamba':>18s} {'vana':>18s}")
    print(f"{'':18s} {'doğru  yanlış  yok':>18s} {'doğru  yanlış  yok':>18s}")

    for ad, bozulma in kosullar:
        l = kosu_lamba(lamba, bozulma, args.tekrar, np.random.default_rng(TOHUM))
        v = kosu_vana(vana, bozulma, args.tekrar, np.random.default_rng(TOHUM))
        rapor["lamba"][ad] = l
        rapor["vana"][ad] = v
        print(f"{ad:>18s} "
              f"{l['dogruluk']*100:>5.1f}% {l['yanlis_sinif']:>6d} {l['okunamayan']:>4d} "
              f"{v['dogruluk']*100:>7.1f}% {v['yanlis_sinif']:>6d} {v['okunamayan']:>4d}")

    rapor["vana_tolerans"] = tolerans_egrisi(vana)
    print("\nvana tolerans eğrisi (open durumundan sapma):")
    for e in rapor["vana_tolerans"]:
        print(f"   {e['sapma_deg']:>3d}° → {str(e['okunan']):>12s}  conf {e['conf']:.2f}")

    print("\nkarışıklık matrisi (temiz, lamba):")
    for gercek, satir in rapor["lamba"]["temiz"]["karisiklik"].items():
        print(f"   {gercek:>8s} → {satir}")

    yol = Path(METRIK_YOLU)
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(rapor, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nölçüm: {yol}")

    if args.figur:
        print(f"figür: {figur(rapor, Path(FIGUR_YOLU))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
