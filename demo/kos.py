"""demo/kos.py — girdi klasöründeki videoyu üç modülden BİRLİKTE geçirir.

    python demo\\kos.py                      # girdi/video/yeni içindeki videoyu isler
    python demo\\kos.py --video <yol>        # belirli bir dosya
    python demo\\kos.py --goster             # canli pencere de acilsin

Otomatik akış:

  girdi klasoru (tek video)
      -> letterbox kirpma
      -> GOSTERGE + ALGILAMA + ANOMALI ayni karede, birbirinden haberdar
      -> cikti/<ad>_birlesik.mp4      (tek pencere goruntusu, H.264)
      -> cikti/<ad>_birlesik.json     (kare kare zaman cizelgesi + ozet)
      -> cikti/<ad>_birlesik_paylas.mp4  (gonderilebilir kucuk kopya)

Girdi klasörü TEK video ile çalışır (istek böyleydi). Birden fazla varsa en
yenisi alınır ve bu açıkça yazılır — sessizce birini seçmek, hangi videonun
işlendiğini bilmeden çıktıya bakmaya yol açar.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
if (DEMO_DIR.parent / "src" / "gauge_vision").is_dir():
    GOSTERGE_REPO = DEMO_DIR.parent
    STAJ_DIR = GOSTERGE_REPO.parent
else:
    STAJ_DIR = DEMO_DIR.parent
    GOSTERGE_REPO = STAJ_DIR / "rasrav-gauge-vision-2026"

def _venv_ile_yeniden_calis() -> None:
    """Yanlış yorumlayıcıyla başlatıldıysa depo venv'ine geçer.

    Bu proje bu tuzağa iki kez düştü. PowerShell'de `python demo\\kos.py`
    yazınca PATH'teki sistem Python'u (3.14) çalışıyor; `ultralytics`, `torch`
    ve `cv2` orada YOK, hepsi depo venv'inde (3.13). Hata mesajı
    "No module named 'ultralytics'" oluyor ve kurulum bozuk sanılıyor —
    oysa kurulum sağlam, sadece yanlış python.

    Kullanıcıya uzun yolu ezberletmek yerine script kendini düzeltiyor.
    """
    if importlib.util.find_spec("ultralytics") is not None:
        return                                   # zaten doğru ortamdayız
    venv_py = GOSTERGE_REPO / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = GOSTERGE_REPO / ".venv" / "bin" / "python"
    if not venv_py.exists() or Path(sys.executable).resolve() == venv_py.resolve():
        return                                   # venv yok ya da zaten oradayız
    print(f"[BILGI] yanlis python ({Path(sys.executable).name}) - "
          f"depo venv'ine geciliyor")
    raise SystemExit(subprocess.call([str(venv_py), str(Path(__file__).resolve()),
                                      *sys.argv[1:]]))


_venv_ile_yeniden_calis()

sys.path.insert(0, str(GOSTERGE_REPO / "src"))
sys.path.insert(0, str(GOSTERGE_REPO / "scripts"))
sys.path.insert(0, str(DEMO_DIR))

# Bu importlar venv kontrolunden SONRA: cv2/torch/ultralytics sistem
# Python'unda yok ve yukarida import edilirse kendi kendini duzeltme
# mekanizmasi hic calisamadan ImportError ile duserdi.
import cv2  # noqa: E402
import anomali_ozgur  # noqa: E402
import ciz_birlesik  # noqa: E402
from birlesik import BirlesikZincir  # noqa: E402
from paylasilabilir_yap import cevir, ffmpeg_yolu  # noqa: E402
from video_yazici import yazici_ac  # noqa: E402

GIRDI_KLASORU = DEMO_DIR / "girdi" / "video" / "yeni"
CIKTI_KLASORU = DEMO_DIR / "cikti" / "birlesik"
AGIRLIK = GOSTERGE_REPO / "runs/detect/models/ip5/keypad5/weights/best.pt"
ALGILAMA_AGIRLIK = GOSTERGE_REPO / "yolov8n.pt"
UZANTILAR = (".mp4", ".avi", ".mov", ".mkv")
# Cikti tuvali SABIT. Kirpma degisince pencere boyutu degisirse VideoWriter
# kalan kareleri sessizce atar (31.08'de 2108 kareden 916'si yazildi).
CIKTI_GORUNTU_EN = 1200
CIKTI_BOY = 900
PAYLAS_HEDEF_MB = 16.0


def video_sec(klasor: Path) -> Path:
    videolar = sorted((y for y in klasor.iterdir() if y.suffix.lower() in UZANTILAR),
                      key=lambda y: y.stat().st_mtime, reverse=True)
    if not videolar:
        raise SystemExit(f"girdi klasorunde video yok: {klasor}")
    if len(videolar) > 1:
        print(f"[UYARI] klasorde {len(videolar)} video var, EN YENISI isleniyor: "
              f"{videolar[0].name}")
        print(f"        digerleri: {', '.join(y.name for y in videolar[1:])}")
    return videolar[0]


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--girdi", type=Path, default=GIRDI_KLASORU)
    p.add_argument("--cikti", type=Path, default=CIKTI_KLASORU)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-kare", type=int, default=None)
    p.add_argument("--goster", action="store_true", help="canli pencere ac")
    p.add_argument("--paylas-atla", action="store_true")
    a = p.parse_args(argv)

    video = a.video or video_sec(a.girdi)
    a.cikti.mkdir(parents=True, exist_ok=True)
    print(f"video   : {video.name}")

    from ultralytics import YOLO
    print("modeller: GOSTERGE + ALGILAMA + ANOMALI yukleniyor...")
    gmodel = YOLO(str(AGIRLIK))
    amodel = YOLO(str(ALGILAMA_AGIRLIK))
    kok = anomali_ozgur.kok_bul(STAJ_DIR)
    if kok is None:
        raise SystemExit("Ozgur'un anomali_motor.py'si bulunamadi")
    motor = anomali_ozgur.OzgurMotoru(kok)
    print(f"ANOMALI : {kok.name} (Ozgur'un kendi modulu)")

    zincir = BirlesikZincir(gmodel, None, amodel, motor, conf=a.conf)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cikti_yolu = a.cikti / f"{video.stem}_birlesik.mp4"
    yazici = None
    olay_gecmisi: deque = deque(maxlen=40)
    zaman_cizelgesi: list[dict] = []
    tum_olaylar: list[dict] = []
    kirpimlar: list[tuple[int, tuple]] = []

    i = -1
    t0 = time.perf_counter()
    try:
        while True:
            ok, ham = cap.read()
            if not ok:
                break
            i += 1
            if a.max_kare and i >= a.max_kare:
                break
            tk = time.perf_counter()
            kare, tespitler, okumalar, algilama, nesneler, ham_an, s = \
                zincir.isle(ham, i, i / fps)
            if not kirpimlar or kirpimlar[-1][1] != s.kirpim:
                kirpimlar.append((i, s.kirpim))

            olay_gecmisi.extend(s.olaylar)
            for o in s.olaylar:
                if o.kaynak == "ANOMALI" or "beklenen" in o.metin:
                    tum_olaylar.append({"kare": o.kare, "sn": round(o.saniye, 2),
                                        "kaynak": o.kaynak, "metin": o.metin})
            zaman_cizelgesi.append({
                "kare": i, "sn": round(s.saniye, 2),
                "G_tespit": s.gosterge["tespit"],
                "G_okuma": s.gosterge["analog_okunan"],
                "A_kutu": s.algilama["kutu"],
                "AN_alarm": s.anomali["alarm"],
                "AN_aciklanan": s.anomali["aciklanan"]})

            cizili = ciz_birlesik.kareyi_ciz(kare, tespitler, okumalar, algilama, nesneler)
            hud = ciz_birlesik.hud_ciz(CIKTI_BOY, s, olay_gecmisi,
                                       1.0 / max(time.perf_counter() - tk, 1e-6),
                                       video.name)
            pencere = ciz_birlesik.birlestir(cizili, hud, CIKTI_GORUNTU_EN, CIKTI_BOY)

            if yazici is None:
                h, w = pencere.shape[:2]
                yazici, kodek = yazici_ac(cikti_yolu, fps, (w, h))
                print(f"cikti   : {cikti_yolu.name}  ({w}x{h}, {kodek})")
            yazici.write(pencere)

            if a.goster:
                cv2.imshow("Birlesik demo (q = cik)", pencere)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if i % 200 == 0:
                print(f"  kare {i}/{toplam}", flush=True)
    finally:
        cap.release()
        if yazici is not None:
            yazici.release()
        cv2.destroyAllWindows()

    # Yazilan kare sayisi islenen ile TUTMALI. Tutmuyorsa video girdiden
    # kisadir ve bu sessizce olur - o yuzden burada acikca sinaniyor.
    if yazici is not None and yazici.yazilan != len(zaman_cizelgesi):
        print(f"[HATA] islenen {len(zaman_cizelgesi)} kare, yazilan "
              f"{yazici.yazilan} - cikti EKSIK")

    sure = time.perf_counter() - t0
    alarmli = sum(1 for k in zaman_cizelgesi if k["AN_alarm"] > 0)
    aciklanan = sum(k["AN_aciklanan"] for k in zaman_cizelgesi)
    rapor = {
        "video": video.name,
        "kare": len(zaman_cizelgesi),
        "sure_sn": round(sure, 1),
        "kirpim_degisimleri": [{"kare": k, "kutu": list(b)} for k, b in kirpimlar],
        "GOSTERGE": {
            "tespit": _sinif_topla(zaman_cizelgesi),
            "analog_okuma": sum(k["G_okuma"] for k in zaman_cizelgesi)},
        "ALGILAMA": {
            "hedefli_kare": sum(1 for k in zaman_cizelgesi if k["A_kutu"] > 0),
            "ort_kutu": round(statistics.mean(k["A_kutu"] for k in zaman_cizelgesi), 2)
            if zaman_cizelgesi else 0},
        "ANOMALI": {
            "alarmli_kare": alarmli,
            "alarm_orani": round(alarmli / max(len(zaman_cizelgesi), 1), 3),
            # Capraz bilginin olculebilir karsiligi: diger iki modulun
            # aciklayabildigi, dolayisiyla alarma donusmeyen on plan sayisi.
            "diger_modullerce_aciklanan": aciklanan},
        "olaylar": tum_olaylar[:400],
        "zaman_cizelgesi": zaman_cizelgesi,
    }
    rapor_yolu = cikti_yolu.with_suffix(".json")
    rapor_yolu.write_text(json.dumps(rapor, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    print(f"\n{len(zaman_cizelgesi)} kare · {sure:.0f} sn")
    print(f"  GOSTERGE : {rapor['GOSTERGE']['tespit']} · "
          f"{rapor['GOSTERGE']['analog_okuma']} analog okuma")
    print(f"  ALGILAMA : {rapor['ALGILAMA']['hedefli_kare']} hedefli kare")
    print(f"  ANOMALI  : {alarmli} alarmli kare (%{100*alarmli/max(len(zaman_cizelgesi),1):.0f}) · "
          f"{aciklanan} on plan digerlerince aciklandi")
    print(f"  rapor    : {rapor_yolu.name}")

    if not a.paylas_atla:
        paylas = cikti_yolu.with_name(f"{cikti_yolu.stem}_paylas.mp4")
        ffmpeg = ffmpeg_yolu()
        crf = 23
        cevir(cikti_yolu, paylas, ffmpeg, crf)
        while paylas.stat().st_size / 1048576 > PAYLAS_HEDEF_MB and crf < 34:
            crf += 3
            cevir(cikti_yolu, paylas, ffmpeg, crf)
        print(f"  paylas   : {paylas.name}  "
              f"({cikti_yolu.stat().st_size/1048576:.0f} -> "
              f"{paylas.stat().st_size/1048576:.1f} MB, CRF {crf})")
    return 0


def _sinif_topla(cizelge) -> dict:
    toplam: dict[str, int] = {}
    for k in cizelge:
        for s, n in k["G_tespit"].items():
            toplam[s] = toplam.get(s, 0) + n
    return dict(sorted(toplam.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
