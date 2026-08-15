"""demo/run_demo.py — GÖSTERGE + ALGILAMA + ANOMALİ modüllerini tek pencerede
yan yana (3 panel) gösteren birleşik demo.

    python demo/run_demo.py --video <yol>

Hiçbir modülün kaynak dosyası bu script tarafından DEĞİŞTİRİLMEZ. Yalnızca:
  - GÖSTERGE (Reşit'in kendi modülü): gauge_vision paketi ve scripts/canli_oku.py
    import edilip ÇALIŞTIRILIR (kod değişmez).
  - ALGILAMA (Bedirhan) ve ANOMALİ (Özgür): neden doğrudan import edilip
    kare-bazlı çağrılamadıkları demo/uyusmazliklar/RAPOR.md'de maddeleniyor.
    ALGILAMA panelinde, live_detector.py'nin ÇAĞIRDIĞI aynı kütüphaneyle
    (ultralytics YOLO.track, aynı varsayılan model + eşik) demo tarafında ayrı
    bir sarmalayıcı çalıştırılır — bu bir "yeniden yazım" değil, dosyasının tek
    fonksiyona ayrılmamış olmasının pratik çözümüdür (RAPOR.md madde 1).
    ANOMALİ panelinde gerçek bir çağrı denenmez (anomali_test.py bir eğitim
    scriptidir, MVTec-AD indirir ve saatlerce eğitir — bir video karesiyle
    hiçbir ilgisi yoktur); panel sabit bir HATA mesajı gösterir.
"""

from __future__ import annotations

import argparse
import sys
import time
import textwrap
from pathlib import Path

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent

# Bu script iki yerde yaşayabilir ve ikisi de geçerlidir:
#   STAJ/demo/run_demo.py                        (çalışma alanı kopyası)
#   STAJ/rasrav-gauge-vision-2026/demo/run_demo.py   (depoya gömülü, sürümlü)
# Depoya gömülen kopya sürüm takibi için var — çalışma alanı git deposu değil ve
# bu dosya orada hiçbir yerde kayıtlı değildi. Kök, "src/gauge_vision var mı"
# diye BAKILARAK bulunuyor; sabit isimle aranırsa gömülü kopya kendi adını bir
# kez daha ekleyip `.../rasrav-gauge-vision-2026/rasrav-gauge-vision-2026` arar.
if (DEMO_DIR.parent / "src" / "gauge_vision").is_dir():
    GOSTERGE_REPO = DEMO_DIR.parent            # depoya gömülü kopya
    STAJ_DIR = GOSTERGE_REPO.parent
else:
    STAJ_DIR = DEMO_DIR.parent                 # çalışma alanı kopyası
    GOSTERGE_REPO = STAJ_DIR / "rasrav-gauge-vision-2026"
ALGILAMA_REPO = STAJ_DIR / "ORTAK" / "OrtakProjeler" / "Bedirhangok_Akilli_Fabrika"
ANOMALI_REPO = STAJ_DIR / "ORTAK" / "OrtakProjeler" / "OzgurKotbas_Akilli_Fabrika"

sys.path.insert(0, str(GOSTERGE_REPO / "src"))
sys.path.insert(0, str(GOSTERGE_REPO / "scripts"))

PANEL_W, PANEL_H = 480, 360
TITLE_H = 30
FOOTER_H = 34
RENK_BASLIK_BG = (60, 60, 60)
RENK_YAZI = (255, 255, 255)
RENK_HATA = (0, 0, 220)
RENK_HATA_BG = (30, 30, 30)


# ───────────────────────── ortak panel yardımcıları ─────────────────────────

def _letterbox(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    olcek = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * olcek)), max(1, int(fh * olcek))
    kucuk = cv2.resize(frame, (nw, nh))
    tuval = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    tuval[y0:y0 + nh, x0:x0 + nw] = kucuk
    return tuval


def _hata_paneli(kaynak_kare: np.ndarray, mesaj: str) -> np.ndarray:
    panel = _letterbox(kaynak_kare, PANEL_W, PANEL_H)
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W, PANEL_H), RENK_HATA_BG, -1)
    panel = cv2.addWeighted(overlay, 0.55, panel, 0.45, 0)
    satirlar = textwrap.wrap("HATA: " + mesaj, width=34)
    y = PANEL_H // 2 - (len(satirlar) * 20) // 2
    for satir in satirlar:
        cv2.putText(panel, satir, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    RENK_HATA, 2, cv2.LINE_AA)
        y += 24
    return panel


def _basliklandir(panel: np.ndarray, baslik: str) -> np.ndarray:
    tuval = np.zeros((PANEL_H + TITLE_H, PANEL_W, 3), dtype=np.uint8)
    tuval[:TITLE_H] = RENK_BASLIK_BG
    cv2.putText(tuval, baslik, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                RENK_YAZI, 2, cv2.LINE_AA)
    tuval[TITLE_H:] = panel
    return tuval


# ───────────────────────── GÖSTERGE (Reşit) ─────────────────────────

def gosterge_hazirla(gosterge_id: str, agirlik_yolu: Path):
    from gauge_vision.config import load_gauges
    from ultralytics import YOLO

    gauges = load_gauges(str(GOSTERGE_REPO / "configs" / "gauges.yaml"))
    if gosterge_id not in gauges:
        raise RuntimeError(
            f"envanterde yok: {gosterge_id} — mevcutlar: {list(gauges)}")
    gauge = gauges[gosterge_id]
    if not agirlik_yolu.exists():
        raise RuntimeError(f"ağırlık dosyası yok: {agirlik_yolu}")
    model = YOLO(str(agirlik_yolu))
    return gauge, model


def gosterge_isle(frame: np.ndarray, model, gauge, conf: float) -> np.ndarray:
    from gauge_vision.pipeline import detect_objects, read_gauge
    import canli_oku  # scripts/canli_oku.py — DEĞİŞTİRİLMEDEN import edilip çizim fonksiyonu çağrılıyor

    sonuc = read_gauge(frame, model, gauge, detect_conf=conf)
    kare = frame.copy()

    # Önce karedeki BÜTÜN göstergeler tipiyle (gri, "okunmuyor"), sonra
    # beyan edilen göstergenin okuması üstüne. Sıra önemli: okuma çizimi
    # üstte kalmalı, gri kutular onu örtmemeli.
    #
    # Bunun sebebi ölçülmüş bir yanlış anlaşılma: karede iki kadran varken
    # ekranda tek kutu görünüyor ve GÖSTERGE "yalnız birini buluyor" sanılıyordu.
    # Tespit hepsini buluyor; okuma tek göstergeye bakıyor çünkü kalibrasyon
    # göstergeye özeldir ve envanterden gelir (2. kural).
    canli_oku.tespitleri_ciz(kare, detect_objects(frame, model, conf=conf),
                             okunan_kutu=sonuc.box_xyxy)
    canli_oku.kareyi_ciz(kare, sonuc, gauge)  # "okunamadı" / değer yazımı bu fonksiyondan gelir (3. kural burada uygulanıyor)
    return _letterbox(kare, PANEL_W, PANEL_H)


# ───────────────────────── ALGILAMA (Bedirhan) ─────────────────────────
# vision/live_detector.py'de tek-kare fonksiyonu yok (mantık main()'in while
# döngüsü içinde) — bkz RAPOR.md madde 1. Aynı model + aynı .track() çağrısı
# burada demo tarafında tekrarlanıyor; Bedirhan'ın DOSYASI çalıştırılmıyor,
# import da edilmiyor.

def algilama_hazirla(agirlik_yolu: Path):
    from ultralytics import YOLO
    if not agirlik_yolu.exists():
        raise RuntimeError(f"ağırlık dosyası yok: {agirlik_yolu}")
    return YOLO(str(agirlik_yolu))


def algilama_isle(frame: np.ndarray, model, conf: float = 0.4) -> np.ndarray:
    kare = frame.copy()
    fh, fw = kare.shape[:2]
    sonuclar = model.track(source=frame, conf=conf, persist=True, verbose=False)

    en_iyi, en_yuksek_conf = None, -1.0
    if len(sonuclar) > 0 and sonuclar[0].boxes is not None and sonuclar[0].boxes.id is not None:
        kutular = sonuclar[0].boxes
        for i, kutu in enumerate(kutular):
            x1, y1, x2, y2 = map(int, kutu.xyxy[0])
            conf_i = float(kutu.conf[0])
            cls_id = int(kutu.cls[0])
            cls_ad = model.names[cls_id]
            iz_id = int(kutular.id[i]) if kutular.id is not None else -1
            if conf_i > en_yuksek_conf:
                en_yuksek_conf = conf_i
                en_iyi = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls_ad, "id": iz_id}
            cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(kare, f"ID:{iz_id} {cls_ad}:{conf_i:.2f}", (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if en_iyi is not None:
        cx, cy = (en_iyi["x1"] + en_iyi["x2"]) // 2, (en_iyi["y1"] + en_iyi["y2"]) // 2
        fcx, fcy = fw // 2, fh // 2
        dx, dy = cx - fcx, cy - fcy
        cv2.circle(kare, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(kare, (fcx, fcy), (cx, cy), (255, 0, 0), 2)
        cv2.putText(kare, f"dx:{dx} dy:{dy}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    else:
        cv2.putText(kare, "hedef yok", (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2)

    return _letterbox(kare, PANEL_W, PANEL_H)


# ───────────────────────── ANOMALİ (Özgür) ─────────────────────────
# anomali_test.py bir eğitim + toplu-tahmin scriptidir (anomalib Engine.fit /
# predict, MVTec-AD "bottle" kategorisi). Tek kare alan bir fonksiyon yok,
# kaydedilmiş bir ağırlık da yok (repo taraması: .ckpt/.pt/.pth bulunamadı).
# Gerçek bir çağrı denemek MVTec-AD indirip eğitime başlar, bir video karesiyle
# ilgisi olmaz. Bu yüzden burada BİLE ÇAĞRI YAPILMIYOR, sabit hata üretiliyor.

ANOMALI_HATA_MESAJI = (
    "anomali_test.py tek kare almiyor: MVTec-AD 'bottle' uzerinde egitim/"
    "topluca tahmin scripti, kayitli agirlik da yok (bkz RAPOR.md madde 2)"
)


def anomali_isle(frame: np.ndarray) -> np.ndarray:
    raise RuntimeError(ANOMALI_HATA_MESAJI)


# ───────────────────────── ana akış ─────────────────────────

def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Ekip modülleri — birleşik demo")
    p.add_argument("--video", required=True, help="işlenecek video dosyası")
    p.add_argument("--gosterge", default="PT-101", help="GÖSTERGE envanterindeki gauge_id")
    p.add_argument("--gosterge-agirlik",
                    default=str(GOSTERGE_REPO / "runs/detect/models/ip5/karisik/weights/best.pt"))
    p.add_argument("--algilama-agirlik", default=str(GOSTERGE_REPO / "yolov8n.pt"),
                    help="ALGILAMA panelinde kullanılacak YOLO ağırlığı (Bedirhan'ın varsayılanıyla aynı: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="GÖSTERGE tespit güven eşiği")
    p.add_argument("--out", default=str(DEMO_DIR / "cikti" / "demo.mp4"))
    p.add_argument("--no-show", action="store_true", help="canlı pencere açma, sadece dosyaya yaz")
    p.add_argument("--max-frames", type=int, default=None,
                    help="yalnızca ilk N kareyi işle (hızlı deneme için)")
    args = p.parse_args(argv)

    video_yolu = Path(args.video)
    if not video_yolu.exists():
        print(f"[HATA] video bulunamadı: {video_yolu}")
        return 1

    cap = cv2.VideoCapture(str(video_yolu))
    if not cap.isOpened():
        print(f"[HATA] video açılamadı: {video_yolu}")
        return 1
    kaynak_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print("[BİLGİ] GÖSTERGE modülü yükleniyor...")
    try:
        gauge, gmodel = gosterge_hazirla(args.gosterge, Path(args.gosterge_agirlik))
        gosterge_hata = None
    except Exception as e:
        gauge = gmodel = None
        gosterge_hata = str(e)
        print(f"[UYARI] GÖSTERGE hazırlanamadı: {e}")

    print("[BİLGİ] ALGILAMA modülü (demo sarmalayıcı) yükleniyor...")
    try:
        amodel = algilama_hazirla(Path(args.algilama_agirlik))
        algilama_hata = None
    except Exception as e:
        amodel = None
        algilama_hata = str(e)
        print(f"[UYARI] ALGILAMA hazırlanamadı: {e}")

    cikti_yolu = Path(args.out)
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    pencere = "Ekip Demo — GOSTERGE | ALGILAMA | ANOMALI (kapatmak icin q)"

    kare_idx = 0
    t_basla = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            kare_idx += 1
            if args.max_frames is not None and kare_idx > args.max_frames:
                kare_idx -= 1
                break
            t0 = time.perf_counter()

            try:
                if gosterge_hata is not None:
                    raise RuntimeError(gosterge_hata)
                p1 = gosterge_isle(frame, gmodel, gauge, args.conf)
            except Exception as e:
                p1 = _hata_paneli(frame, str(e))
            p1 = _basliklandir(p1, "GOSTERGE (Resit)")

            try:
                if algilama_hata is not None:
                    raise RuntimeError(algilama_hata)
                p2 = algilama_isle(frame, amodel)
            except Exception as e:
                p2 = _hata_paneli(frame, str(e))
            p2 = _basliklandir(p2, "ALGILAMA (Bedirhan)")

            try:
                p3 = anomali_isle(frame)
                p3 = _basliklandir(p3, "ANOMALI (Ozgur)")
            except Exception as e:
                p3 = _hata_paneli(frame, str(e))
                p3 = _basliklandir(p3, "ANOMALI (Ozgur)")

            birlesik = np.hstack([p1, p2, p3])
            gecen = time.perf_counter() - t0
            fps = 1.0 / gecen if gecen > 0 else 0.0

            altbilgi = np.zeros((FOOTER_H, birlesik.shape[1], 3), dtype=np.uint8)
            cv2.putText(altbilgi, f"kare {kare_idx} · {fps:.1f} FPS (kare basi)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RENK_YAZI, 1, cv2.LINE_AA)
            birlesik = np.vstack([birlesik, altbilgi])

            if writer is None:
                h, w = birlesik.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(cikti_yolu), fourcc, kaynak_fps, (w, h))
            writer.write(birlesik)

            if not args.no_show:
                cv2.imshow(pencere, birlesik)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[BİLGİ] kullanıcı tarafından durduruldu")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    toplam = time.perf_counter() - t_basla
    print(f"[BİLGİ] {kare_idx} kare işlendi, {toplam:.1f} sn · çıktı: {cikti_yolu}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
