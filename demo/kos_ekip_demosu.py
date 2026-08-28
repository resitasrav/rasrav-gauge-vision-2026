"""demo/kos_ekip_demosu.py — bir klasördeki HER videoyu üç panelli demodan geçirir.

    python demo/kos_ekip_demosu.py
    python demo/kos_ekip_demosu.py --sadece araba karasel

Her video için üç çıktı:

  1. `<ad>_ekip.mp4`  — GÖSTERGE | ALGILAMA | ANOMALİ yan yana, gözle inceleme
  2. `<ad>_ekip.json` — üç modülün kendi ölçüm özeti (gözle görülemeyenler)
  3. `_ozet.json`     — hepsinin tek tabloda karşılaştırması

Videolar TEK TEK işlenir, birleştirilmez: her videonun kendi ANOMALİ referansı
var (kendi ilk kareleri) ve ALGILAMA'nın iz kimlikleri (`track_id`) video
sınırında sıfırlanmalı — hepsini tek akışta işlemek iki modülü de bozardı.

Alt süreç olarak çağrılıyor, `main()` import edilip döngüye sokulmuyor: bir
videoda çöken model (CUDA bellek hatası, bozuk dosya) diğer 15'ini
düşürmesin. Çöken video raporda `hata` ile görünür, koşu devam eder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
STAJ_DIR = DEMO_DIR.parent
VARSAYILAN_KLASOR = DEMO_DIR / "girdi" / "video"
VARSAYILAN_CIKTI = DEMO_DIR / "cikti" / "ekip"
UZANTILAR = (".mp4", ".avi", ".mov", ".mkv")


def _anomali_satiri(an: dict) -> str:
    """ANOMALİ özetini tek satıra çevirir — EKSİK ANAHTARA 0 UYDURMADAN.

    Bu fonksiyon bir hatadan doğdu: konsol satırı `anomali_orani` okuyordu ama
    Özgür'ün motoru `alarm_orani` yayınlıyor. Eksik anahtar `or 0` ile sıfıra
    çevrilince 17 videonun hepsi "%0 anomali" göründü — oysa JSON'da 666 alarm
    yazıyordu. Ölçüm doğruydu, GÖSTERİM yalan söylüyordu.

    Ders bu depoda zaten var (3. kural): bilinmeyen için sayı uydurulmaz.
    Aynısı ekrana basılan özet için de geçerli.
    """
    if "hata" in an:
        return f"HATA ({an['hata'][:40]})"
    if "alarm_orani" in an:                      # Özgür'ün MOG2 motoru
        return f"%{100 * an['alarm_orani']:.0f} alarm"
    if "anomali_orani" in an:                    # PaDiM yedeği
        oran = an["anomali_orani"]
        return "?" if oran is None else f"%{100 * oran:.0f} anomali"
    return "?"


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--klasor", type=Path, default=VARSAYILAN_KLASOR)
    p.add_argument("--cikti", type=Path, default=VARSAYILAN_CIKTI)
    p.add_argument("--sadece", nargs="+", default=None, metavar="AD",
                   help="yalniz bu adlar (uzantisiz)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args(argv)

    videolar = sorted(y for y in a.klasor.iterdir() if y.suffix.lower() in UZANTILAR)
    if a.sadece:
        istenen = {s.lower().removesuffix(".mp4") for s in a.sadece}
        videolar = [y for y in videolar if y.stem.lower() in istenen]
    if not videolar:
        raise SystemExit(f"video yok: {a.klasor}")
    a.cikti.mkdir(parents=True, exist_ok=True)

    print(f"{len(videolar)} video -> {a.cikti}\n")
    raporlar = []
    for n, yol in enumerate(videolar, 1):
        cikti = a.cikti / f"{yol.stem}_ekip.mp4"
        komut = [sys.executable, str(DEMO_DIR / "run_demo.py"),
                 "--video", str(yol), "--out", str(cikti),
                 "--no-show", "--conf", str(a.conf)]
        if a.max_frames:
            komut += ["--max-frames", str(a.max_frames)]

        print(f"[{n}/{len(videolar)}] {yol.name} ...", end=" ", flush=True)
        t0 = time.time()
        sonuc = subprocess.run(komut, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        rapor_yolu = cikti.with_suffix(".json")
        if sonuc.returncode != 0 or not rapor_yolu.exists():
            son = (sonuc.stderr or sonuc.stdout).strip().splitlines()[-1:] or ["?"]
            print(f"HATA ({time.time()-t0:.0f}s): {son[0][:110]}")
            raporlar.append({"video": yol.name, "hata": son[0][:400]})
            continue

        r = json.loads(rapor_yolu.read_text(encoding="utf-8"))
        raporlar.append(r)
        g, al, an = r["GOSTERGE"], r["ALGILAMA"], r["ANOMALI"]
        print(f"{time.time()-t0:.0f}s | "
              f"G: {g.get('analog_okunan', '-')}/{g.get('analog_kutu', '-')} okuma, "
              f"flip {g.get('flip_180', '-')} | "
              f"A: {al.get('hedefli_kare', '-')} hedefli kare | "
              f"AN: {_anomali_satiri(an)}")

    (a.cikti / "_ozet.json").write_text(
        json.dumps(raporlar, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nozet: {a.cikti / '_ozet.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
