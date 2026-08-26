r"""Buton paneli okuma doğruluğu — `keypad` tipinin sayısal bitti kriteri.

    python scripts\olc_keypad.py
    python scripts\olc_keypad.py --zor

**Ölçülen şey iki katmanlıdır ve ikisi de gereklidir:**

1. **Buton doğruluğu** — her butonun rengi tek tek doğru sınıflandı mı.
2. **Makine durumu doğruluğu** — bileşimden çıkan durum adı doğru mu.

İkisi ayrı ölçülür çünkü ayrı ayrı bozulabilirler: kurallar yalnız adı geçen
butonlara baktığı için, kuralın umursamadığı bir buton yanlış okunsa da makine
durumu doğru çıkar. O hata görünmez kalırsa gerçek pano gelince patlar.

**En önemli sütun `yanlış`tır, `doğru` değil.** Buton panelinde yanlış okumanın
bedeli bir sayı değil MAKİNENİN DURUMUDUR: duran bir hatta "çalışıyor" demek,
yanlış bir basınç değerinden tehlikelidir (3. kural). Bu yüzden ölçüm sessiz
hatayı ayrı sayar ve hedef `yanlış = 0`'dır.

**Kapsama ayrıca raporlanır.** `unreadable` bir başarısızlık değil, güvenli
davranıştır — ama kapsama sıfıra düşerse sistem işe yaramaz. İkisi birlikte
okunur (İP15'in kapsama/risk ödünleşmesiyle aynı ilke).

⚠ Sentetik panel gerçek panonun yerini tutmaz: buton kapağı çiziği, üstündeki
yazı, cam yansıması ve tozlanma burada yok. Bu tablo "yöntem oturuyor mu"
sorusunu cevaplar, "sahada ne olur"u değil.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from gauge_vision.config import Gauge, load_gauges
from gauge_vision.read.calibrate import DURUM_OKUNAMADI
from gauge_vision.read.keypad import read_keypad
from gauge_vision.synth.degrade import Bozulma, bozulmalar_uygula
from gauge_vision.synth.dial import DialTruth
from gauge_vision.synth.keypad import render_keypad

METRIK_YOLU = Path("outputs/metrics/keypad.json")
FIGUR_YOLU = Path("outputs/figures/keypad.png")
TOHUM = 7


def _bilesimler(gauge: Gauge, limit: int | None) -> list[dict[str, str]]:
    """Butonların TÜM durum bileşimleri (kartezyen çarpım).

    Örneklem değil tümü: dört butonun 2×2×2×2 = 16 bileşimi var ve hepsi
    fiziksel olarak mümkün. Rastgele örneklemek, envanterde kuralı olmayan
    bileşimleri kaçırabilirdi — oysa ölçümün asıl sorusu tam da onlarda:
    sistem bilmediği bir bileşimde susuyor mu, yoksa uyduruyor mu?
    """
    adlar = [b["id"] for b in gauge.buttons]
    listeler = [list(b.get("states") or []) for b in gauge.buttons]
    hepsi = [dict(zip(adlar, secim)) for secim in itertools.product(*listeler)]
    return hepsi[:limit] if limit else hepsi


def _boz(img: np.ndarray, bozulma: Bozulma, gauge_id: str, rng) -> np.ndarray:
    """Bozulmaları uygular. `bozulmalar_uygula` kadran ground truth'u istiyor.

    Buton panelinde kadran geometrisi yok; `olc_ip12.py`'daki lamba/vana
    çözümünün aynısı: yalnız kare boyutunu taşıyan bir yer tutucu verilir ve
    dönen ground truth atılır. Bozulma zaten görüntüye uygulanıyor, etiketi
    burada taşımıyoruz.
    """
    if not bozulma.etkin:
        return img
    h, w = img.shape[:2]
    yer_tutucu = DialTruth(gauge_id=gauge_id, value=0.0, angle_deg=0.0,
                           roll_deg=0.0, angle_img_deg=0.0,
                           center_px=(w // 2, h // 2), tip_px=(0, 0),
                           radius_px=min(h, w) // 3, bbox_xyxy=(0, 0, w, h))
    return bozulmalar_uygula(img, yer_tutucu, bozulma, rng)[0]


def kosu(gauge: Gauge, bozulma: Bozulma, tekrar: int,
         rng: np.random.Generator) -> dict:
    """Tek bir koşulda bütün bileşimleri `tekrar` kez okur."""
    dogru_durum = yanlis_durum = okunamayan = 0
    dogru_buton = toplam_buton = 0
    kuralsiz = 0

    for bilesim in _bilesimler(gauge, None):
        img, truth = render_keypad(gauge, bilesim)
        if truth.machine_state is None:
            # Envanterde kuralı olmayan bileşim: doğru davranış `unreadable`.
            kuralsiz += 1
        for _ in range(tekrar):
            kare = _boz(img, bozulma, gauge.id, rng)
            okuma = read_keypad(kare, gauge)

            okunan = okuma.extra.get("buttons", {})
            for bid, gercek in bilesim.items():
                toplam_buton += 1
                if okunan.get(bid) == gercek:
                    dogru_buton += 1

            if okuma.status == DURUM_OKUNAMADI or okuma.value is None:
                okunamayan += 1
            elif okuma.value == truth.machine_state:
                dogru_durum += 1
            else:
                # Bir durum yayınlandı ve YANLIŞ — sessiz hata.
                yanlis_durum += 1

    toplam = len(_bilesimler(gauge, None)) * tekrar
    return {
        "kare": toplam,
        "kuralsiz_bilesim": kuralsiz,
        "durum_dogru": dogru_durum,
        "durum_yanlis": yanlis_durum,
        "okunamayan": okunamayan,
        "durum_dogruluk": round(dogru_durum / max(toplam, 1), 4),
        "sessiz_hata_orani": round(yanlis_durum / max(toplam, 1), 4),
        "kapsama": round((toplam - okunamayan) / max(toplam, 1), 4),
        "buton_dogruluk": round(dogru_buton / max(toplam_buton, 1), 4),
    }


def _figur(rapor: dict, yol: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kosullar = list(rapor["kosullar"])
    dogru = [rapor["kosullar"][k]["durum_dogruluk"] * 100 for k in kosullar]
    kapsama = [rapor["kosullar"][k]["kapsama"] * 100 for k in kosullar]
    yanlis = [rapor["kosullar"][k]["sessiz_hata_orani"] * 100 for k in kosullar]

    x = np.arange(len(kosullar))
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(kosullar)), 4.5))
    ax.bar(x - 0.22, dogru, 0.36, label="doğru durum %", color="#2b8a3e")
    ax.bar(x + 0.22, kapsama, 0.36, label="kapsama %", color="#3b7dd8")
    ax.plot(x, yanlis, "o-", color="#d94f4f", label="SESSİZ HATA %")
    ax.set_xticks(x)
    ax.set_xticklabels(kosullar, rotation=20, ha="right")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(f"Buton paneli okuma — {rapor['gauge_id']}")
    ax.legend()
    fig.tight_layout()
    yol.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(yol, dpi=130)
    print(f"figür: {yol}")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Buton paneli durum doğruluğu")
    p.add_argument("--tekrar", type=int, default=10, help="bileşim başına kare")
    p.add_argument("--zor", action="store_true")
    p.add_argument("--figur", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    gauges = load_gauges()
    panel = next((g for g in gauges.values() if g.type == "keypad"), None)
    if panel is None:
        print("envanterde 'keypad' tipinde gösterge yok")
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

    bilesim_sayisi = len(_bilesimler(panel, None))
    kuralsiz = sum(1 for b in _bilesimler(panel, None)
                   if render_keypad(panel, b)[1].machine_state is None)
    print(f"panel {panel.id}: {panel.button_names}")
    print(f"bileşim {bilesim_sayisi} · envanterde kuralı olmayan {kuralsiz} "
          f"(bunlarda doğru davranış: unreadable)\n")

    rapor = {"is_paketi": "keypad", "tarih": date.today().isoformat(),
             "gauge_id": panel.id, "butonlar": panel.button_names,
             "bilesim_sayisi": bilesim_sayisi, "kuralsiz_bilesim": kuralsiz,
             "tekrar": args.tekrar, "kosullar": {}}

    print(f"{'koşul':>18s} {'durum':>8s} {'buton':>8s} {'kapsama':>8s} {'SESSİZ':>8s}")
    for ad, bozulma in kosullar:
        s = kosu(panel, bozulma, args.tekrar, np.random.default_rng(TOHUM))
        rapor["kosullar"][ad] = s
        print(f"{ad:>18s} {100*s['durum_dogruluk']:>7.1f}% "
              f"{100*s['buton_dogruluk']:>7.1f}% {100*s['kapsama']:>7.1f}% "
              f"{s['durum_yanlis']:>7d}")

    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    METRIK_YOLU.write_text(json.dumps(rapor, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nölçüm: {METRIK_YOLU}")
    if args.figur:
        _figur(rapor, FIGUR_YOLU)

    toplam_yanlis = sum(k["durum_yanlis"] for k in rapor["kosullar"].values())
    if toplam_yanlis:
        print(f"⚠ SESSİZ HATA: {toplam_yanlis} karede yanlış makine durumu "
              f"yayınlandı — 3. kural ihlali, düzeltilmeden ilerlenmez")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
