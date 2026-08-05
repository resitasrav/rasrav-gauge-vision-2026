"""İbre açısı hatasını ölçer ve rapor figürlerini üretir (İP6).

    python scripts/olc_ip6.py                      # tam ölçüm + figürler
    python scripts/olc_ip6.py --veri data/synthetic/v0 --yontem polar
    python scripts/olc_ip6.py --no-cozunurluk --no-merkez

Üç ayrı soruyu cevaplar:

1. **Ana ölçüm (İP6 bitti kriteri):** iki yöntemin sentetik sette ortalama açı
   hatası — K3 kıyasının sayısı.
2. **Çözünürlük (U6):** kadran çapı 60/80/120/200 piksele düşürüldüğünde hata ne
   oluyor. Bedirhan'ın 640×480 yayınında kadran ~60-80 piksel çapa düşecek;
   ekipten çözünürlük artışı istenip istenmeyeceğine bu tablo karar verdirir.
3. **Merkez hatası:** açı ölçümü kadranın merkezini doğru bildiğini varsayıyor.
   İP5'in kutusu kusurlu olacağından merkez 2/4/8 piksel kaydırılıp hata
   yeniden ölçülür — İP8'e girmeden önce bu bağımlılığın büyüklüğü bilinmeli.

Çıktılar: `outputs/metrics/ip6_aci_hatasi.json` ·
`outputs/figures/ip6_hata_dagilimi.png` · `outputs/figures/ip6_ornek_okuma.png`
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.read.evaluate import NeedleResult, error_stats, read_dataset
from gauge_vision.read.needle import METHODS

VARSAYILAN_VERI = "data/synthetic/v0"
METRIK_YOLU = "outputs/metrics/ip6_aci_hatasi.json"
DAGILIM_YOLU = "outputs/figures/ip6_hata_dagilimi.png"
ORNEK_YOLU = "outputs/figures/ip6_ornek_okuma.png"

# U6: Bedirhan'ın yayınında kadranın düşeceği tahmini çap 60-80 piksel.
COZUNURLUK_CAPLARI = (60, 80, 120, 200)
# publisher.py:24 → QUALITY = 80. Sayı oradan alındı, tahmin değil.
YAYIN_JPEG_Q = 80
MERKEZ_SARSINTILARI = (2.0, 4.0, 8.0)

ORNEK_SUTUN, ORNEK_SATIR = 3, 2
ORNEK_PX = 240
ORNEK_ETIKET_PX = 30


def hata_ozeti(sonuclar: list[NeedleResult]) -> dict:
    """Bir koşunun tüm sayıları: genel, gösterge bazlı, güven ve süre."""
    okunan = [s for s in sonuclar if s.ok]
    hatalar = [abs(s.angle_error_deg) for s in okunan]

    gostergeler = {}
    for gid in sorted({s.gauge_id for s in sonuclar}):
        g_hatalar = [abs(s.angle_error_deg) for s in okunan if s.gauge_id == gid]
        gostergeler[gid] = error_stats(g_hatalar).as_dict()

    return {
        "goruntu": len(sonuclar),
        "okunan": len(okunan),
        "okunamayan": len(sonuclar) - len(okunan),
        "aci_hatasi_deg": error_stats(hatalar).as_dict(),
        "gosterge_bazli_deg": gostergeler,
        "ortalama_guven": round(float(np.mean([s.confidence for s in okunan])), 3) if okunan else 0.0,
        "goruntu_basina_ms": round(float(np.mean([s.elapsed_ms for s in sonuclar])), 2),
    }


def cozunurluk_taramasi(veri: Path, yontem: str, caplar, jpeg: int | None = None) -> dict:
    """U6'nın sayısal dayanağı: kadran çapı düştükçe açı hatası.

    `jpeg` verilirse kare yayının sıkıştırmasından da geçirilir; U6'daki iki
    iddia (çözünürlük ve q80 artefaktları) böylece ayrı ayrı ölçülür.
    """
    tablo = {}
    for cap in caplar:
        sonuclar = read_dataset(veri, method=yontem, dial_diameter_px=cap, jpeg_quality=jpeg)
        okunan = [s for s in sonuclar if s.ok]
        tablo[f"{cap}px"] = {
            "okunan": len(okunan),
            "okunamayan": len(sonuclar) - len(okunan),
            **error_stats([abs(s.angle_error_deg) for s in okunan]).as_dict(),
        }
    return tablo


def merkez_hatasi_taramasi(veri: Path, yontem: str, sarsintilar) -> dict:
    """Merkez bilinmediğinde ne kaybediyoruz — İP5'e olan bağımlılığın ölçüsü."""
    tablo = {}
    for px in sarsintilar:
        sonuclar = read_dataset(veri, method=yontem, center_jitter_px=px)
        okunan = [s for s in sonuclar if s.ok]
        tablo[f"{px:g}px"] = {
            "okunan": len(okunan),
            **error_stats([abs(s.angle_error_deg) for s in okunan]).as_dict(),
        }
    return tablo


def dagilim_figuru(kosular: dict[str, list[NeedleResult]], cikti: Path) -> Path:
    """Hata dağılımı: kümülatif eğri + gösterge bazlı ortalama.

    Kümülatif eğri seçildi çünkü rapor okuyucusunun sorusu "ortalama kaç" değil
    **"kaç kare şu eşiğin altında"**dır; hedef metrik de bir eşik cümlesidir.
    """
    import matplotlib
    matplotlib.use("Agg")   # başsız ortamda da çalışsın (CI, uzak makine)
    import matplotlib.pyplot as plt

    fig, (sol, sag) = plt.subplots(1, 2, figsize=(11, 4.2))

    for yontem, sonuclar in kosular.items():
        hatalar = np.sort([abs(s.angle_error_deg) for s in sonuclar if s.ok])
        if hatalar.size == 0:
            continue
        oran = np.arange(1, hatalar.size + 1) / hatalar.size * 100
        sol.plot(hatalar, oran, marker=".", ms=3, lw=1.4, label=yontem)

    sol.set_xscale("log")
    sol.set_xlabel("mutlak açı hatası (derece, log)")
    sol.set_ylabel("bu hatanın altındaki kare (%)")
    sol.set_title("Hata dağılımı — kümülatif")
    sol.grid(alpha=0.3)
    sol.legend()

    gostergeler = sorted({s.gauge_id for s in next(iter(kosular.values()))})
    genislik = 0.8 / max(1, len(kosular))
    for i, (yontem, sonuclar) in enumerate(kosular.items()):
        ortalamalar = [
            error_stats([abs(s.angle_error_deg) for s in sonuclar
                         if s.ok and s.gauge_id == gid]).mean
            for gid in gostergeler
        ]
        x = np.arange(len(gostergeler)) + i * genislik
        sag.bar(x, ortalamalar, width=genislik, label=yontem)

    sag.set_xticks(np.arange(len(gostergeler)) + genislik * (len(kosular) - 1) / 2)
    sag.set_xticklabels(gostergeler)
    sag.set_ylabel("ortalama açı hatası (derece)")
    sag.set_title("Gösterge bazlı ortalama")
    sag.grid(alpha=0.3, axis="y")
    sag.legend()

    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=130)
    plt.close(fig)
    return cikti


def ornek_figuru(veri: Path, sonuclar: list[NeedleResult], cikti: Path) -> Path | None:
    """Gözle denetim: ölçülen ibre kırmızı, gerçek yeşil çizilir.

    Sayı tablosu yöntemin çalıştığını söyler ama nasıl yanıldığını söylemez;
    hatanın en büyük olduğu kareler burada gözle görülebilsin diye seçiliyor.
    """
    okunan = sorted((s for s in sonuclar if s.ok),
                    key=lambda s: abs(s.angle_error_deg), reverse=True)
    if not okunan:
        return None

    n = min(ORNEK_SUTUN * ORNEK_SATIR, len(okunan))
    secilen = okunan[:n]   # en kötü n kare

    hucre = ORNEK_PX + ORNEK_ETIKET_PX
    tuval = np.full((ORNEK_SATIR * hucre, ORNEK_SUTUN * ORNEK_PX, 3), 255, dtype=np.uint8)
    kayitlar = {k["file"]: k for k in _etiketler(veri)}

    for i, s in enumerate(secilen):
        img = cv2.imread(str(veri / s.file))
        if img is None:
            continue
        k = kayitlar[s.file]
        c = tuple(k["center_px"])
        r = k["radius_px"]

        # Kalınlıklar bilerek farklı: en kötü karede bile sapma ~1 piksel olduğu
        # için eşit kalınlıkta çizilirse ölçülen, gerçeği tamamen örtüyor ve
        # figür "hiç ölçüm yapılmamış" gibi görünüyor. Kalın yeşilin içinden
        # geçen ince kırmızı, örtüşmenin kendisini görünür kılıyor.
        _isin(img, c, r, s.truth_angle_img_deg, (60, 170, 60), 5)      # gerçek — yeşil
        _isin(img, c, r, s.measured_angle_deg, (40, 40, 220), 1)       # ölçülen — kırmızı

        kucuk = cv2.resize(img, (ORNEK_PX, ORNEK_PX), interpolation=cv2.INTER_AREA)
        sat, sut = divmod(i, ORNEK_SUTUN)
        y, x = sat * hucre, sut * ORNEK_PX
        tuval[y:y + ORNEK_PX, x:x + ORNEK_PX] = kucuk
        cv2.putText(tuval, f"{s.gauge_id}  hata {abs(s.angle_error_deg):.2f}deg  "
                           f"conf {s.confidence:.2f}",
                    (x + 6, y + ORNEK_PX + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (40, 40, 40), 1, cv2.LINE_AA)

    cikti.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cikti), tuval)
    return cikti


def _etiketler(veri: Path) -> list[dict]:
    from gauge_vision.synth.generate import load_labels
    return load_labels(veri)


def _isin(img, center, radius, aci_deg: float, renk, kalinlik: int) -> None:
    import math
    rad = math.radians(aci_deg)
    uc = (round(center[0] + radius * 0.92 * math.cos(rad)),
          round(center[1] - radius * 0.92 * math.sin(rad)))
    cv2.line(img, tuple(center), uc, renk, kalinlik, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="İbre açısı hatasını ölç (İP6)")
    p.add_argument("--veri", default=VARSAYILAN_VERI)
    p.add_argument("--yontem", nargs="*", choices=METHODS, default=list(METHODS))
    p.add_argument("--cozunurluk", action=argparse.BooleanOptionalAction, default=True,
                   help="U6 için çözünürlük taraması")
    p.add_argument("--merkez", action=argparse.BooleanOptionalAction, default=True,
                   help="merkez kayması duyarlılığı")
    p.add_argument("--figur", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    veri = Path(args.veri)
    if not (veri / "labels.jsonl").exists():
        print(f"veri seti yok: {veri} — önce scripts/uret_sentetik.py çalıştır")
        return 1

    kosular: dict[str, list[NeedleResult]] = {}
    ozet: dict = {
        "is_paketi": "IP6",
        "tarih": date.today().isoformat(),
        "veri_seti": str(veri).replace("\\", "/"),
        "yontemler": {},
    }

    for yontem in args.yontem:
        sonuclar = read_dataset(veri, method=yontem)
        kosular[yontem] = sonuclar
        ozet["yontemler"][yontem] = hata_ozeti(sonuclar)

        o = ozet["yontemler"][yontem]["aci_hatasi_deg"]
        print(f"{yontem:6s}  ortalama {o['ortalama']:6.3f}°  medyan {o['medyan']:6.3f}°  "
              f"p95 {o['p95']:6.3f}°  max {o['max']:6.3f}°  "
              f"okunamayan {ozet['yontemler'][yontem]['okunamayan']}")

    en_iyi = min(ozet["yontemler"],
                 key=lambda y: ozet["yontemler"][y]["aci_hatasi_deg"]["ortalama"])
    ozet["secilen_yontem"] = en_iyi
    print(f"\nseçilen yöntem (İP7'ye girdi): {en_iyi}")

    if args.cozunurluk:
        print("\nçözünürlük taraması (U6) — sol: temiz, sağ: yayın sıkıştırması q80:")
        temiz = cozunurluk_taramasi(veri, en_iyi, COZUNURLUK_CAPLARI)
        sikistirilmis = cozunurluk_taramasi(veri, en_iyi, COZUNURLUK_CAPLARI, jpeg=YAYIN_JPEG_Q)
        ozet["cozunurluk_u6"] = {"temiz": temiz, f"jpeg_q{YAYIN_JPEG_Q}": sikistirilmis}
        for cap in temiz:
            t, s = temiz[cap], sikistirilmis[cap]
            print(f"   kadran çapı {cap:>6s}   temiz {t['ortalama']:7.3f}° (p95 {t['p95']:6.3f}°)"
                  f"   q{YAYIN_JPEG_Q} {s['ortalama']:7.3f}° (p95 {s['p95']:6.3f}°)"
                  f"   okunamayan {s['okunamayan']}")

    if args.merkez:
        print("\nmerkez kayması duyarlılığı:")
        ozet["merkez_hatasi"] = merkez_hatasi_taramasi(veri, en_iyi, MERKEZ_SARSINTILARI)
        for px, satir in ozet["merkez_hatasi"].items():
            print(f"   kayma {px:>5s}  ortalama {satir['ortalama']:7.3f}°  "
                  f"p95 {satir['p95']:7.3f}°")

    metrik = Path(METRIK_YOLU)
    metrik.parent.mkdir(parents=True, exist_ok=True)
    metrik.write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nölçüm: {metrik}")

    if args.figur:
        print(f"dağılım figürü: {dagilim_figuru(kosular, Path(DAGILIM_YOLU))}")
        yol = ornek_figuru(veri, kosular[en_iyi], Path(ORNEK_YOLU))
        if yol:
            print(f"örnek okuma figürü: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
