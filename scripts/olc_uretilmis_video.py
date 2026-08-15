r"""Üretilmiş (AI) fabrika videolarında TESPİT genellemesini ölçer.

    python scripts\olc_uretilmis_video.py --klasor data\real

**Bu script okuma doğruluğunu ÖLÇEMEZ ve ölçmeye çalışmaz.** Sebebi 2. kuraldır:
bir kadranın değeri, envanterdeki kalibrasyonundan (min/max, angle_min/angle_max)
türetilir. Üretilmiş videodaki kadran hiç var olmadı — ibresinin "gerçekte" kaç
bar gösterdiği diye bir olgu yok. Envantere uydurma bir satır ekleyip sayı
üretmek, ground truth uydurmak olurdu; bu projede reddedilen şeyin ta kendisi.

**Ölçebildiği şey gerçekten değerli:** model kendi sentetik üretecinin dışında,
hiç görmediği fotogerçekçi bir endüstriyel sahnede göstergeyi bulabiliyor mu.
Raporlardaki "dijital/lamba/vana %100'leri genelleme değil, model kendi
üretecinin çıktısında ölçülüyor" sınırının doğrudan cevabı budur.

**Ölçülen sayı KAPSAMA değil VARLIKTIR.** Ground truth kutu yok, dolayısıyla
duyarlılık/kesinlik hesaplanamaz: kaçırılan kadran görünmez. Raporlanan şey
"karede en az bir tespit var mı", "kare başına kaç tespit", "güven dağılımı" ve
sınıf dağılımıdır. Sınıf dağılımı yanlış pozitifi ELE VERİR — sahnede vana
yokken `valve` sayısı sıfırdan büyükse, o sayı doğrudan hatadır.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

UZANTILAR = (".mp4", ".mov", ".avi", ".mkv")
METRIK_YOLU = "outputs/metrics/uretilmis_video_tespit.json"
FIGUR_KLASORU = "outputs/figures"
COK_SINIF_AGIRLIK = Path("runs/detect/models/ip5/cok_sinif/weights/best.pt")

# Sahnede FİZİKSEL olarak ne var. Yalnızca yanlış pozitifi görünür kılmak için;
# ölçüme girmiyor, tabloda "beklenmeyen" sütunu olarak raporlanıyor.
BEKLENEN_SINIFLAR = {
    "A_slow_steady_handheld_walk_t": {"gauge"},
    "Static_camera_slowly_pushing": {"digital", "lamp", "gauge"},
    "Slow_lateral_camera_dolly_alon": {"valve"},
}


def kareleri_oku(yol: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(yol))
    if not cap.isOpened():
        # Dosya kopyalanırken açılmaya çalışılırsa moov atomu henüz yazılmamış
        # olur ve OpenCV -1x-1 döner. Bu bir kodek sorunu DEĞİLDİR; ölçümü
        # sessizce boş geçmemek için ayrı raporlanıyor.
        return []
    kareler = []
    while True:
        ok, kare = cap.read()
        if not ok:
            break
        kareler.append(kare)
    cap.release()
    return kareler


def olc(kareler: list[np.ndarray], model, conf: float) -> dict:
    sinif_sayim: Counter = Counter()
    kare_basi, confler, bos = [], [], 0
    for kare in kareler:
        kutular = model.predict(kare, conf=conf, verbose=False)[0].boxes
        if len(kutular) == 0:
            bos += 1
        kare_basi.append(len(kutular))
        for i in range(len(kutular)):
            sinif_sayim[model.names[int(kutular.cls[i])]] += 1
            confler.append(float(kutular.conf[i]))
    n = max(1, len(kareler))
    return {
        "kare": len(kareler),
        "tespitli_kare": n - bos,
        "kare_orani": round((n - bos) / n, 3),
        "kare_basi_ortalama": round(float(np.mean(kare_basi)) if kare_basi else 0.0, 2),
        "kare_basi_maks": int(max(kare_basi)) if kare_basi else 0,
        "ortalama_guven": round(float(np.mean(confler)) if confler else 0.0, 3),
        "sinif_dagilimi": dict(sinif_sayim),
    }


def figur_yaz(kareler: list[np.ndarray], model, conf: float, yol: Path) -> None:
    """Sayı "kaç tane" der, figür "nereye" der. Dar/kaymış kutu ve yanlış
    pozitif ancak gözle görülür; ikisi de sayıda iyi görünür."""
    secili = []
    for i in range(6):
        kare = kareler[int(len(kareler) * i / 6)].copy()
        kutular = model.predict(kare, conf=conf, verbose=False)[0].boxes
        for j in range(len(kutular)):
            x1, y1, x2, y2 = (int(v) for v in kutular.xyxy[j])
            cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(kare, f"{model.names[int(kutular.cls[j])]} {float(kutular.conf[j]):.2f}",
                        (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        cv2.putText(kare, f"kare {int(len(kareler)*i/6)} · {len(kutular)} tespit",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        secili.append(cv2.resize(kare, (480, 270)))
    yol.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(yol), np.vstack([np.hstack(secili[:3]), np.hstack(secili[3:6])]))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--klasor", type=Path, default=Path("data/real"))
    ap.add_argument("--agirlik", type=Path, default=COK_SINIF_AGIRLIK)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--metrik", type=Path, default=Path(METRIK_YOLU))
    ap.add_argument("--figur-klasoru", type=Path, default=Path(FIGUR_KLASORU))
    args = ap.parse_args()

    videolar = sorted(p for p in args.klasor.iterdir()
                      if p.suffix.lower() in UZANTILAR)
    if not videolar:
        print(f"HATA: {args.klasor} içinde video yok.")
        return 1

    from ultralytics import YOLO
    model = YOLO(str(args.agirlik))
    print(f"{len(videolar)} video · tespit: {args.agirlik}\n")

    sonuc = {}
    for yol in videolar:
        kareler = kareleri_oku(yol)
        if not kareler:
            print(f"{yol.name}: AÇILAMADI (kopyalama bitmemiş olabilir)")
            sonuc[yol.stem] = {"hata": "video acilamadi"}
            continue

        olcum = olc(kareler, model, args.conf)
        beklenen = BEKLENEN_SINIFLAR.get(yol.stem)
        if beklenen:
            olcum["beklenmeyen_sinif"] = {a: n for a, n in olcum["sinif_dagilimi"].items()
                                          if a not in beklenen}
        sonuc[yol.stem] = olcum
        figur_yaz(kareler, model, args.conf,
                  args.figur_klasoru / f"tespit_{yol.stem}.png")

        print(f"{yol.name}")
        print(f"   {olcum['tespitli_kare']}/{olcum['kare']} karede tespit "
              f"(%{olcum['kare_orani']*100:.0f}) · kare başı {olcum['kare_basi_ortalama']} "
              f"(maks {olcum['kare_basi_maks']}) · ort güven {olcum['ortalama_guven']}")
        print(f"   sınıflar: {olcum['sinif_dagilimi']}")
        if olcum.get("beklenmeyen_sinif"):
            print(f"   ⚠ YANLIŞ POZİTİF: {olcum['beklenmeyen_sinif']} "
                  f"(sahnede bu tip yok)")
        print()

    args.metrik.parent.mkdir(parents=True, exist_ok=True)
    args.metrik.write_text(json.dumps(
        {"agirlik": str(args.agirlik), "conf": args.conf, "videolar": sonuc},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.metrik}\n{args.figur_klasoru}/tespit_*.png")
    print("\nNOT: bu sayılar VARLIK ölçer, kapsama değil — ground truth kutu yok, "
          "kaçırılan gösterge görünmez. Okuma doğruluğu bu videolarda ÖLÇÜLEMEZ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
