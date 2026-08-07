"""Güven eşiğini ÖLÇÜMDEN kalibre eder — İP15.

    python scripts/kalibre_ip15.py
    python scripts/kalibre_ip15.py --hedef 5.0

Girdi: `olc_ip14.py`'nin ürettiği (güven, hata) çiftleri. Yani eşik zor
koşulların tamamı üzerinde kalibre edilir — temiz görüntüde her eşik iyi görünür.

**Neden bu iş var.** `gauges.yaml`'daki `conf_threshold: 0.70` şu ana kadar bir
VARSAYIMDI; hiçbir ölçümden gelmiyordu. 3. kural ("yanlış okumaktansa okumamak")
ancak eşik sayıyla seçilirse bir mühendislik kararıdır, aksi hâlde temenni.

**Ölçülen üç büyüklük ve aralarındaki ödünleşme:**

    kapsama          kaç kare okundu / toplam        ↑ istenir
    sessiz_hata      kabul edilenler içinde hatası hedefi aşanların oranı  ↓ istenir
    kabul_hatasi     kabul edilenlerin ortalama hatası                     ↓ istenir

`sessiz_hata` belirleyicidir. Reddedilen kare zararsızdır: tur raporunda
"okunamadı" yazar, operatör bakar. Kabul edilmiş yanlış bir sayı ise sessizce
yanlış karar üretir — endüstride pahalı olan budur.

Eşik yükseldikçe kapsama düşer, sessiz hata da düşer. Seçim, sessiz hatayı
kabul edilebilir bir seviyeye indiren EN DÜŞÜK eşiktir: gereğinden yüksek eşik
bedavaya kapsama kaybettirir.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

CIFT_YOLU = "outputs/metrics/ip14_guven_hata_ciftleri.json"
METRIK_YOLU = "outputs/metrics/ip15_guven_esigi.json"
FIGUR_YOLU = "outputs/figures/ip15_guven_esigi.png"

# Bir okumanın "sessizce yanlış" sayılacağı hata sınırı. Projenin hedefi %5
# ortalama; tek bir okumanın hedefi aşması onu kullanılamaz yapar.
VARSAYILAN_HEDEF = 5.0
ADAYLAR = np.round(np.arange(0.0, 0.96, 0.05), 2)


def yukle(yol: Path) -> list[dict]:
    ciftler = json.loads(yol.read_text(encoding="utf-8"))
    # Tespit hiç olmadıysa güven zaten 0; eşik kararına katkısı yok ama
    # kapsama paydasında kalmalı — sahada o kare de okunamamış sayılır.
    return ciftler


def degerlendir(ciftler: list[dict], esik: float, hedef: float) -> dict:
    """Verilen eşikte kapsama, sessiz hata ve kabul edilenlerin hatası."""
    kabul = [c for c in ciftler if c["conf"] >= esik and c["hata_yuzde"] is not None]
    hatalar = np.array([c["hata_yuzde"] for c in kabul]) if kabul else np.array([])

    sessiz = int(np.count_nonzero(hatalar > hedef)) if hatalar.size else 0
    return {
        "esik": float(esik),
        "kapsama": round(len(kabul) / len(ciftler), 3),
        "kabul": len(kabul),
        "sessiz_hata_adet": sessiz,
        "sessiz_hata_orani": round(sessiz / len(kabul), 4) if kabul else 0.0,
        # Tüm kareler payda: sahada asıl önemli olan "kaç turda yanlış sayı yayınlandı".
        "sessiz_hata_tum_kareler": round(sessiz / len(ciftler), 4),
        "kabul_hatasi_ort": round(float(hatalar.mean()), 3) if hatalar.size else None,
        "kabul_hatasi_p95": round(float(np.percentile(hatalar, 95)), 3) if hatalar.size else None,
        "kabul_hatasi_max": round(float(hatalar.max()), 3) if hatalar.size else None,
    }


def sec(ciftler: list[dict], satirlar: list[dict], hedef_sessiz: float,
        hedef: float, eksen_siniri: float) -> tuple[dict, bool]:
    """Ölçütü sağlayan EN DÜŞÜK eşik. `(satır, ölçüt_sağlandı_mı)` döner.

    **İki kısıt, ikisi de gerekli:**

    1. Genel sessiz hata oranı ≤ `hedef_sessiz`
    2. **Her bozulma ekseninde ayrı ayrı** ≤ `eksen_siniri`

    İkincisi olmadan seçim yanıltıcı olur: 07.08 ölçümünde eşik 0,40 genel
    sessiz hatayı %1,47'ye indiriyordu ama eğiklik ekseninde oran %7,75'ti.
    Ortalamanın altına saklanan bir arıza sahada tek bir gösterge tipinde
    sürekli yanlış okuma demektir — ortalama iyi göründüğü için de fark
    edilmesi zordur.

    Daha yükseği bedavaya kapsama kaybettirir; ölçüt sağlandıktan sonra eşiği
    büyütmenin faydası yok.
    """
    for s in satirlar:
        if s["kabul"] == 0 or s["sessiz_hata_orani"] > hedef_sessiz:
            continue
        eksenler = eksen_bazli(ciftler, s["esik"], hedef)
        en_kotu = max((e["sessiz_hata_orani"] for e in eksenler.values()), default=1.0)
        if en_kotu <= eksen_siniri:
            return s, True

    # Hiçbir eşik ölçütü sağlamıyor: en iyisini döndür ve bunu ÇAĞIRANA bildir.
    # Sessizce "seçildi" demek, sağlanmamış bir güvenceyi sağlanmış göstermek olur.
    return min(satirlar, key=lambda s: (s["sessiz_hata_orani"], -s["kapsama"])), False


def eksen_bazli(ciftler: list[dict], esik: float, hedef: float) -> dict:
    """Seçilen eşik her bozulma ekseninde ne yapıyor?

    Tek bir toplam sayı, eşiğin bir eksende çuvalladığını gizleyebilir.
    """
    cikti = {}
    for eksen in sorted({c["eksen"] for c in ciftler}):
        alt = [c for c in ciftler if c["eksen"] == eksen]
        cikti[eksen] = degerlendir(alt, esik, hedef)
    return cikti


def figur(satirlar: list[dict], secilen: dict, hedef: float, cikti: Path) -> Path:
    """Kapsama–sessiz hata ödünleşmesi. Eşik seçimi bu eğriden okunur."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    esikler = [s["esik"] for s in satirlar]
    kapsama = [s["kapsama"] * 100 for s in satirlar]
    sessiz = [s["sessiz_hata_orani"] * 100 for s in satirlar]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(esikler, kapsama, "o-", color="#2980b9", label="kapsama %")
    ax.plot(esikler, sessiz, "s-", color="#c0392b",
            label=f"kabul edilenler içinde hata > %{hedef:g} olanlar")
    ax.axvline(secilen["esik"], color="#27ae60", ls="--", lw=1.6,
               label=f"seçilen eşik {secilen['esik']:.2f}")
    ax.set_xlabel("güven eşiği")
    ax.set_ylabel("%")
    ax.set_title("İP15 — eşik yükseldikçe kapsama düşer, sessiz hata da düşer")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=130)
    plt.close(fig)
    return cikti


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Güven eşiği kalibrasyonu (İP15)")
    p.add_argument("--ciftler", default=CIFT_YOLU)
    p.add_argument("--hedef", type=float, default=VARSAYILAN_HEDEF,
                   help="bir okumanın 'sessizce yanlış' sayılacağı hata sınırı (%%)")
    p.add_argument("--sessiz-sinir", type=float, default=0.005,
                   help="kabul edilenler içinde izin verilen sessiz hata oranı")
    p.add_argument("--eksen-sinir", type=float, default=0.02,
                   help="tek bir bozulma ekseninde izin verilen sessiz hata oranı")
    p.add_argument("--figur", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    yol = Path(args.ciftler)
    if not yol.exists():
        print(f"çiftler yok: {yol} — önce scripts/olc_ip14.py")
        return 1

    ciftler = yukle(yol)
    print(f"{len(ciftler)} kare · hedef: tek okuma hatası ≤ %{args.hedef:g} · "
          f"izin verilen sessiz hata ≤ %{args.sessiz_sinir*100:g}\n")

    satirlar = [degerlendir(ciftler, e, args.hedef) for e in ADAYLAR]
    print(f"{'eşik':>6s} {'kapsama':>9s} {'kabul':>7s} {'sessiz':>8s} "
          f"{'sessiz%':>9s} {'ort':>7s} {'p95':>7s} {'max':>7s}")
    for s in satirlar:
        print(f"{s['esik']:>6.2f} {s['kapsama']*100:>8.1f}% {s['kabul']:>7d} "
              f"{s['sessiz_hata_adet']:>8d} {s['sessiz_hata_orani']*100:>8.2f}% "
              f"{str(s['kabul_hatasi_ort']):>7s} {str(s['kabul_hatasi_p95']):>7s} "
              f"{str(s['kabul_hatasi_max']):>7s}")

    secilen, saglandi = sec(ciftler, satirlar, args.sessiz_sinir, args.hedef,
                            args.eksen_sinir)
    print(f"\n>>> SEÇİLEN EŞİK: {secilen['esik']:.2f}")
    print(f"    kapsama %{secilen['kapsama']*100:.1f} · "
          f"sessiz hata %{secilen['sessiz_hata_orani']*100:.2f} "
          f"({secilen['sessiz_hata_adet']}/{secilen['kabul']} kabul) · "
          f"kabul edilenlerin ortalama hatası %{secilen['kabul_hatasi_ort']}")
    if not saglandi:
        print(f"    ⚠ HİÇBİR EŞİK ÖLÇÜTÜ SAĞLAMIYOR (genel ≤%{args.sessiz_sinir*100:g}, "
              f"eksen ≤%{args.eksen_sinir*100:g}). Yukarıdaki en iyi olan, güvenceli olan değil.")

    eksenler = eksen_bazli(ciftler, secilen["esik"], args.hedef)
    print(f"\n{'eksen':>12s} {'kapsama':>9s} {'sessiz%':>9s} {'ort hata':>9s}")
    for ad, e in eksenler.items():
        print(f"{ad:>12s} {e['kapsama']*100:>8.1f}% {e['sessiz_hata_orani']*100:>8.2f}% "
              f"{str(e['kabul_hatasi_ort']):>9s}")

    rapor = {
        "is_paketi": "IP15", "tarih": date.today().isoformat(),
        "kare_sayisi": len(ciftler),
        "hedef_hata_yuzde": args.hedef,
        "izin_verilen_sessiz_hata": args.sessiz_sinir,
        "izin_verilen_eksen_sessiz_hata": args.eksen_sinir,
        "olcut_saglandi": saglandi,
        "secilen_esik": secilen,
        "tarama": satirlar,
        "eksen_bazli": eksenler,
    }
    myol = Path(METRIK_YOLU)
    myol.parent.mkdir(parents=True, exist_ok=True)
    myol.write_text(json.dumps(rapor, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nölçüm: {myol}")

    if args.figur:
        print(f"figür: {figur(satirlar, secilen, args.hedef, Path(FIGUR_YOLU))}")

    print(f"\nEnvantere işlemek için: configs/gauges.yaml → defaults.conf_threshold: "
          f"{secilen['esik']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
