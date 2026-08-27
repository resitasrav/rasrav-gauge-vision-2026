"""Bir klasördeki bütün videoları GÖSTERGE zincirinden geçirir.

    python scripts/isle_video_kumesi.py --klasor ../demo/girdi/video
    python scripts/isle_video_kumesi.py --klasor ... --adim 2 --max-kare 400

Her video için iki çıktı üretir:

  1. İşaretli video (`<ad>_cikti.mp4`) — gözle inceleme içindir.
  2. JSON rapor (`<ad>.json`) — gözle görülemeyen hataları yakalar.

İkisi ayrı işler. Videoya bakarak "ibre doğru mu" sorusuna cevap veremezsin,
çünkü internetten alınmış bir manometrenin gerçek değeri bilinmiyor. Ama
BAKMADAN yakalanabilecek iki hata sınıfı var ve JSON tam olarak onları ölçüyor:

  * **180° sıçrama**: ibre iki ardışık karede 180° dönemez. Fizik yasağı
    olduğu için gerçek değeri bilmeye gerek yok — sıçramanın kendisi hatadır.
  * **Çözünürlük yetersizliği**: kadran yarıçapı çok küçükse polar tarama
    yeterli örnek alamaz. Okuma "başarılı" görünür ama gürültüdür.

KİMLİKSİZ MOD: bu videolar envanterde değil, bu yüzden hiçbir kutuya
kalibrasyon uygulanmıyor — yalnız geometri (çember + ibre açısı) üretiliyor.
Kimlik beyan etmek sessiz yanlış değer üretir (26.08 ölçümü: devir saatine
PT-101 denince "0,8 bar ok" yayınlanıyordu). 3. kural: yanlış okumaktansa
okumamak.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

VARSAYILAN_AGIRLIK = KOK / "runs/detect/models/ip5/keypad5/weights/best.pt"
VARSAYILAN_KLASOR = KOK.parent / "demo" / "girdi" / "video"
VARSAYILAN_CIKTI = KOK.parent / "demo" / "cikti" / "video"

# Yazılan videonun genişlik tavanı. ÇÖZÜMLEME HAM KARE ÜZERİNDE yapılır;
# burası yalnız dosya boyutu içindir - 4K çıktı 14 video için gereksiz yere
# gigabaytlara çıkıyor ve gözle inceleme 1080p'de zaten yapılabiliyor.
CIKTI_EN_TAVAN = 1920

# İbre 25 fps'te iki kare arasında en fazla birkaç derece döner. 170-190°
# aralığı ibrenin ucu ile kuyruğunun karıştığı bilinen hatadır (İP6 duyarlılık
# tablosu: merkez hatası buna yol açıyor).
FLIP_ALT, FLIP_UST = 170.0, 190.0
# Sıçrama sayılan ama flip olmayan oynama: kadran sabitken bu kadar dönmemeli.
KARARSIZ_ESIK = 30.0
# Bunun altındaki yarıçapta polar tarama yeterli örnek alamıyor (İP6: 40 px
# altında açı hatası hızla büyüyor).
MIN_GUVENLI_YARICAP = 40.0

RENK = {
    "gauge": (0, 165, 255), "digital": (255, 180, 0),
    "lamp": (0, 220, 220), "valve": (200, 120, 255),
    "keypad": (120, 255, 120),
}


def _model(yol: Path):
    from ultralytics import YOLO
    if not yol.exists():
        raise SystemExit(f"agirlik yok: {yol}")
    return YOLO(str(yol))


def _ciz(kare, tespitler, okumalar) -> None:
    """Tespit kutuları + kimliksiz analog geometrisi. Değer/birim YAZILMAZ."""
    for t in tespitler:
        x1, y1, x2, y2 = (int(v) for v in t.box_xyxy)
        renk = RENK.get(t.sinif, (150, 150, 150))
        cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)
        cv2.putText(kare, f"{t.sinif} {t.conf:.2f}", (x1 + 3, max(y1 - 7, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, renk, 2, cv2.LINE_AA)
    for o in okumalar:
        if not o.ok:
            continue
        cv2.circle(kare, o.center_px, int(o.radius_px), (0, 165, 255), 2, cv2.LINE_AA)
        cv2.line(kare, o.center_px, o.needle.tip_px, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(kare, o.center_px, 4, (255, 255, 255), -1, cv2.LINE_AA)
        x1, y1 = int(o.box_xyxy[0]), int(o.box_xyxy[1])
        cv2.putText(kare, f"aci {o.needle.angle_img_deg:.0f} deg  r={o.radius_px:.0f}px",
                    (x1 + 3, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 165, 255), 2, cv2.LINE_AA)


def _aci_farki(a: float, b: float) -> float:
    """İki açı arasındaki en kısa fark, 0-180 aralığında."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def video_isle(yol: Path, model, cikti_kok: Path, conf: float,
               adim: int, max_kare: int | None) -> dict:
    from gauge_vision.pipeline import detect_objects, read_all_analog

    cap = cv2.VideoCapture(str(yol))
    if not cap.isOpened():
        return {"video": yol.name, "hata": "acilamadi"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    en = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    boy = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    olcek = min(1.0, CIKTI_EN_TAVAN / en)
    c_en, c_boy = int(en * olcek), int(boy * olcek)
    cikti_kok.mkdir(parents=True, exist_ok=True)
    kare_kok = cikti_kok / "kareler" / yol.stem
    yazici = cv2.VideoWriter(str(cikti_kok / f"{yol.stem}_cikti.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(adim, 1), (c_en, c_boy))

    sinif_sayim: dict[str, int] = {}
    sinif_guven: dict[str, list] = {}
    yaricaplar: list[float] = []
    analog_kutu = analog_okundu = 0
    aci_izi: list[tuple[int, float]] = []   # (kare no, en buyuk kadranin acisi)
    kareler_islenen = 0
    okunamama: dict[str, int] = {}

    i = -1
    while True:
        ok, kare = cap.read()
        if not ok:
            break
        i += 1
        if i % adim:
            continue
        if max_kare and kareler_islenen >= max_kare:
            break
        kareler_islenen += 1

        tespitler = detect_objects(kare, model, conf=conf)
        okumalar = read_all_analog(kare, model, tespitler=tespitler)

        for t in tespitler:
            sinif_sayim[t.sinif] = sinif_sayim.get(t.sinif, 0) + 1
            sinif_guven.setdefault(t.sinif, []).append(t.conf)
        analog_kutu += len(okumalar)
        for o in okumalar:
            if o.ok:
                analog_okundu += 1
                yaricaplar.append(float(o.radius_px))
            else:
                okunamama[o.reason or "?"] = okunamama.get(o.reason or "?", 0) + 1

        basarili = [o for o in okumalar if o.ok]
        if basarili:
            en_buyuk = max(basarili, key=lambda o: o.radius_px)
            aci_izi.append((i, float(en_buyuk.needle.angle_img_deg)))

        cizili = kare.copy()
        _ciz(cizili, tespitler, okumalar)
        if olcek < 1.0:
            cizili = cv2.resize(cizili, (c_en, c_boy), interpolation=cv2.INTER_AREA)
        cv2.putText(cizili, f"{yol.name}  kare {i}/{toplam}  kimlik beyani yok - deger uretilmiyor",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        yazici.write(cizili)

    cap.release()
    yazici.release()

    # --- gözle görülemeyen hatalar ---
    fliplar, kararsizlar = [], []
    for (k1, a1), (k2, a2) in zip(aci_izi, aci_izi[1:]):
        d = _aci_farki(a1, a2)
        if FLIP_ALT <= d <= FLIP_UST:
            fliplar.append({"kare": k2, "onceki_aci": round(a1, 1),
                            "aci": round(a2, 1), "fark": round(d, 1)})
        elif d > KARARSIZ_ESIK:
            kararsizlar.append({"kare": k2, "fark": round(d, 1)})

    # Şüpheli kareleri diske yaz: gözle bakılacak olan bunlar, 4000 kare değil.
    if fliplar:
        kare_kok.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(yol))
        for f in fliplar[:6]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f["kare"])
            ok, kare = cap.read()
            if not ok:
                continue
            tespitler = detect_objects(kare, model, conf=conf)
            _ciz(kare, tespitler, read_all_analog(kare, model, tespitler=tespitler))
            if kare.shape[1] > CIKTI_EN_TAVAN:
                o = CIKTI_EN_TAVAN / kare.shape[1]
                kare = cv2.resize(kare, (CIKTI_EN_TAVAN, int(kare.shape[0] * o)))
            cv2.imwrite(str(kare_kok / f"flip_{f['kare']:06d}.png"), kare)
        cap.release()

    rapor = {
        "video": yol.name,
        "cozunurluk": f"{en}x{boy}",
        "kare_toplam": toplam,
        "kare_islenen": kareler_islenen,
        "tespit": {
            s: {"sayi": n, "ort_guven": round(float(np.mean(sinif_guven[s])), 3)}
            for s, n in sorted(sinif_sayim.items(), key=lambda kv: -kv[1])
        },
        "analog": {
            "kutu": analog_kutu,
            "okunan": analog_okundu,
            "kapsam": round(analog_okundu / analog_kutu, 3) if analog_kutu else None,
            "okunamama_sebepleri": okunamama,
            "yaricap_px": {
                "medyan": round(statistics.median(yaricaplar), 1) if yaricaplar else None,
                "min": round(min(yaricaplar), 1) if yaricaplar else None,
                "max": round(max(yaricaplar), 1) if yaricaplar else None,
                "esik_alti_oran": round(
                    sum(r < MIN_GUVENLI_YARICAP for r in yaricaplar) / len(yaricaplar), 3)
                if yaricaplar else None,
            },
        },
        "zamansal": {
            "acili_kare": len(aci_izi),
            "flip_180": len(fliplar),
            "flip_orani": round(len(fliplar) / max(len(aci_izi) - 1, 1), 4),
            "kararsiz_30deg_ustu": len(kararsizlar),
            "ornek_flipler": fliplar[:6],
        },
    }
    (cikti_kok / f"{yol.stem}.json").write_text(
        json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    return rapor


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--klasor", type=Path, default=VARSAYILAN_KLASOR)
    p.add_argument("--cikti", type=Path, default=VARSAYILAN_CIKTI)
    p.add_argument("--agirlik", type=Path, default=VARSAYILAN_AGIRLIK)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--adim", type=int, default=1, help="her N. kare islensin")
    p.add_argument("--max-kare", type=int, default=None)
    p.add_argument("--sadece", nargs="+", default=None, metavar="AD",
                   help="yalniz bu adlar islensin (uzantisiz), or: --sadece araba karasel")
    a = p.parse_args(argv)

    videolar = sorted([y for y in a.klasor.iterdir()
                       if y.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")])
    if a.sadece:
        istenen = {s.lower().removesuffix(".mp4") for s in a.sadece}
        videolar = [y for y in videolar if y.stem.lower() in istenen]
        eksik = istenen - {y.stem.lower() for y in videolar}
        if eksik:
            raise SystemExit(f"bulunamadi: {sorted(eksik)}")
    if not videolar:
        raise SystemExit(f"video yok: {a.klasor}")
    model = _model(a.agirlik)
    print(f"agirlik: {a.agirlik.name} ({len(model.names)} sinif: {list(model.names.values())})")
    print(f"{len(videolar)} video -> {a.cikti}\n")

    raporlar = []
    for n, yol in enumerate(videolar, 1):
        t0 = time.time()
        print(f"[{n}/{len(videolar)}] {yol.name} ...", end=" ", flush=True)
        r = video_isle(yol, model, a.cikti, a.conf, a.adim, a.max_kare)
        raporlar.append(r)
        tespit = ", ".join(f"{s}:{d['sayi']}" for s, d in r.get("tespit", {}).items()) or "TESPIT YOK"
        print(f"{time.time() - t0:.0f}s | {tespit} | flip {r['zamansal']['flip_180']}")

    # Kismi kosu butun kumenin ozetini EZMEZ: `_ozet.json` "14 videonun hali"
    # diye okunuyor ve iki videoluk bir kosuyla ustune yazilirsa karsilastirma
    # zemini sessizce kaybolur.
    ozet_yol = a.cikti / ("_ozet_secim.json" if a.sadece else "_ozet.json")
    ozet_yol.write_text(
        json.dumps(raporlar, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nozet: {ozet_yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
