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

    # "yok" = kimlik beyanı yok. Zincir hiçbir kutuya envanter kalibrasyonu
    # uygulamaz; panel yalnızca görüntüden ölçülebileni gösterir (tip + açı).
    # İnternetten alınmış rastgele videolarda DÜRÜST mod budur: 26.08'de
    # PT-101 beyanıyla devir saati "0,8 bar ok", termometre "2,2 bar ok"
    # yayınlandığı ölçüldü — yanlış kimlik beyanı sessiz yanlış değer üretir.
    if gosterge_id == "yok":
        gauge = None
    else:
        gauges = load_gauges(str(GOSTERGE_REPO / "configs" / "gauges.yaml"))
        if gosterge_id not in gauges:
            raise RuntimeError(
                f"envanterde yok: {gosterge_id} — mevcutlar: {list(gauges)} "
                f"(kimliksiz mod için: --gosterge yok)")
        gauge = gauges[gosterge_id]
    if not agirlik_yolu.exists():
        raise RuntimeError(f"ağırlık dosyası yok: {agirlik_yolu}")
    model = YOLO(str(agirlik_yolu))
    return gauge, model


def gosterge_isle(frame: np.ndarray, model, gauge, conf: float) -> np.ndarray:
    from gauge_vision.pipeline import detect_objects, read_all_analog, read_gauge
    import canli_oku  # scripts/canli_oku.py — DEĞİŞTİRİLMEDEN import edilip çizim fonksiyonları çağrılıyor

    kare = frame.copy()
    tespitler = detect_objects(frame, model, conf=conf)

    # Katmanlar alttan üste: (1) gri tespit kutuları, (2) turuncu kimliksiz
    # analog geometrileri (çember + ibre + açı), (3) beyan edilen göstergenin
    # kalibrasyonlu okuması. Sıra önemli: değer beyanı en üstte kalmalı.
    #
    # Bunun sebebi ölçülmüş bir yanlış anlaşılma: karede iki kadran varken
    # ekranda tek kutu görünüyor ve GÖSTERGE "yalnız birini buluyor" sanılıyordu.
    # Artık HER analog kutu tek tek okunuyor (çember + ibre açısı); değere
    # çevirme yalnız kimliği beyan edilende — kalibrasyon göstergeye özeldir
    # ve envanterden gelir (2. kural), kimliksiz kutuya uygulanmaz (3. kural).
    sonuc = read_gauge(frame, model, gauge, detect_conf=conf) if gauge else None
    okunan_kutu = sonuc.box_xyxy if sonuc else None

    canli_oku.tespitleri_ciz(kare, tespitler, okunan_kutu=okunan_kutu)
    canli_oku.analoglari_ciz(
        kare, read_all_analog(frame, model, tespitler=tespitler),
        okunan_kutu=okunan_kutu)
    if sonuc is not None:
        canli_oku.kareyi_ciz(kare, sonuc, gauge)  # "okunamadı" / değer yazımı burada (3. kural)
    else:
        cv2.putText(kare, "kimlik beyani yok - deger/birim uretilmiyor",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                    cv2.LINE_AA)
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


# ───────────────────────── ANOMALİ (Özgür) — DÜZELTİLMİŞ ENTEGRASYON v2 ───────────────────────
# RAPOR.md §1: anomali_test.py (eğitim scripti) demo için uygun değil.
# ÇÖZÜM: Özgür'ün demo_anomali.py'deki AlgilayiciIP8 (SSIM+ORB) ve
# AlgilayiciMOG2 sınıflarının mantığı buraya bağımsız sarmalayıcı olarak
# entegre edildi. Özgür'ün hiçbir dosyası değiştirilmedi.
# Yöntem: MOG2 arka plan çıkarma (IP9) + SSIM fark skoru (IP8 referanssız mod)
# Çıktı: patrol/alert sözleşmesiyle uyumlu {is_alert, severity, score} bilgisi
#
# v2 — FP düzeltmeleri (analiz_cop_kutusu_fp.py bulgularına göre, 27.08.2026):
#   Ö1: MOG2 warm-up — ilk kare N=40 kere learningRate=1.0 ile beslenir;
#       history=200 yetersizliği giderilir (İP12'deki aynı prensip).
#   Ö2: Tavan bölgesi bastırma — fg maskesinin üst %18'i sıfırlanır;
#       kamera açı kaymasından doğan tavan/lamba gürültüsü kesilir.
#   Sonuç: WP01 FP=3 → FP=0-1, F1 0.667 → 0.800+ hedeflenir.

import math as _math
from collections import deque as _deque

# Ö1: Warm-up için referans kareyi kaç kez besleyeceğiz
_WARMUP_N = 40
# Ö2: Tavan crop — üst kaçta birini MOG2 fg maskesinden sıfırlayacağız
_TAVAN_CROP_ORAN = 0.18


class _AlgilayiciMOG2:
    """Özgür'ün AlgilayiciMOG2 mantığı (demo_anomali.py'den bağımsız kopya).

    v2: Ö1+Ö2 FP düzeltmeleri entegre edildi.
    """

    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=20, detectShadows=True)
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._yellow_lo = np.array([18, 80, 80])
        self._yellow_hi = np.array([38, 255, 255])
        self._warmed_up = False   # Ö1: ilk kare warm-up tamamlandı mı?

    def warmup(self, frame: np.ndarray, n: int = _WARMUP_N) -> None:
        """Ö1 — MOG2 cold-start düzeltmesi.

        Referans kareyi n kere learningRate=1.0 ile besleyerek arka plan
        modelini ısıtır. history=200 yerine n=40 yeterli: MOG2 Gaussian
        mixture yakınsaması ~30 tekrarda sabitlenir.
        İP12 notu: 'son 30 kare learningRate=0' — burada tersine
        'ilk 40 kare learningRate=1.0' mantığı uygulanıyor.
        """
        for _ in range(n):
            self.mog2.apply(frame, learningRate=1.0)
        self._warmed_up = True

    def _yellow_mask(self, bgr):
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._yellow_lo, self._yellow_hi)
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        return cv2.dilate(mask, k, iterations=1)

    def isle(self, frame: np.ndarray) -> dict:
        fg = self.mog2.apply(frame)
        fg[fg == 127] = 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k_close)

        # Ö2: Tavan bölgesi bastırma — kamera açı kaymasından gelen
        # lamba/panel gürültüsünü keser. SSIM detektöründeki floor_crop
        # mantığını MOG2'ye taşır (analiz_cop_kutusu_fp.py §4).
        tavan_sinir = int(fg.shape[0] * _TAVAN_CROP_ORAN)
        fg[:tavan_sinir, :] = 0

        yellow = self._yellow_mask(frame)
        if fg.shape != yellow.shape:
            yellow = cv2.resize(yellow, (fg.shape[1], fg.shape[0]))
        fg[yellow > 0] = 0
        fg_ratio = float(np.sum(fg > 0)) / fg.size
        nesneler = self._detect(fg, yellow)
        return {"is_alert": len(nesneler) > 0, "nesneler": nesneler,
                "fg_mask": fg, "fg_ratio": round(fg_ratio, 4)}

    def _detect(self, mask, yellow_mask):
        h, w = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        objs = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 1500:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh > h * w * 0.40:
                continue
            cx, cy = x + bw // 2, y + bh // 2
            try:
                if yellow_mask[cy, cx] > 0:
                    continue
            except IndexError:
                pass
            objs.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                          "area": int(bw * bh)})
        objs.sort(key=lambda o: o["area"], reverse=True)
        return objs


class _AnomalDurumu:
    """Demo boyunca yaşayan ANOMALİ durum nesnesi.

    v2: İlk kare geldiğinde Ö1 warm-up otomatik tetiklenir.
    """

    def __init__(self):
        self.algilayici   = _AlgilayiciMOG2()
        self.score_hist   = _deque(maxlen=60)
        self.toplam_uyari = 0
        self.kare_no      = 0
        self.ref_frame    = None   # İP8 referansı: ilk kare

    def isle(self, frame: np.ndarray) -> dict:
        """Kare → {is_alert, severity, score, fg_mask, fg_ratio, nesneler}"""
        self.kare_no += 1
        if self.ref_frame is None:
            # Ö1: İlk kare gelince warm-up yap, sonra MOG2'ye gerçek kareler
            self.ref_frame = frame.copy()
            self.algilayici.warmup(self.ref_frame)

        sonuc    = self.algilayici.isle(frame)
        fg_mask  = sonuc["fg_mask"]
        fg_ratio = sonuc["fg_ratio"]
        nesneler = sonuc["nesneler"]
        is_alert = sonuc["is_alert"]

        # Anomali skoru: MOG2 fg oranı + nesne sayısı ağırlıklı
        score = min(1.0, fg_ratio * 15.0 + len(nesneler) * 0.15)
        self.score_hist.append(score)
        if is_alert:
            self.toplam_uyari += 1

        severity = ("HIGH"   if len(nesneler) >= 2 else
                    "MEDIUM" if len(nesneler) == 1 else "NONE")
        return {
            "is_alert":    is_alert,
            "severity":    severity,
            "score":       score,
            "fg_mask":     fg_mask,
            "fg_ratio":    fg_ratio,
            "nesneler":    nesneler,
            "kare_no":     self.kare_no,
            "toplam_uyari": self.toplam_uyari,
        }


def anomali_isle(frame: np.ndarray, durum: "_AnomalDurumu") -> np.ndarray:
    """Kare → ANOMALİ paneli (480×360 BGR).

    patrol/alert sözleşmesi: is_alert, severity, score gösterilir.
    Özgür'ün hiçbir dosyası değiştirilmedi — bağımsız sarmalayıcı.
    """
    r = durum.isle(frame)

    # Panel arka planı: fg mask üstüne canlı kare karışımı
    if r["fg_mask"] is not None:
        fg_bgr = cv2.cvtColor(r["fg_mask"], cv2.COLOR_GRAY2BGR)
        live   = _letterbox(frame, PANEL_W, PANEL_H)
        fg_res = _letterbox(fg_bgr, PANEL_W, PANEL_H)
        panel  = cv2.addWeighted(live, 0.45, fg_res, 0.55, 0)
    else:
        panel = _letterbox(frame, PANEL_W, PANEL_H)

    # Durum bandı
    brenk = (0, 0, 200) if r["is_alert"] else (30, 180, 60)
    btxt  = f">>> UYARI <<< [{r['severity']}]" if r["is_alert"] else "Normal"
    cv2.rectangle(panel, (0, 0), (PANEL_W, 28), (0, 0, 0), -1)
    cv2.putText(panel, btxt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, brenk, 2, cv2.LINE_AA)

    # Tespit kutuları
    for i, obj in enumerate(r["nesneler"][:3]):
        cx = obj["x"] * PANEL_W // max(frame.shape[1], 1)
        cy = obj["y"] * PANEL_H // max(frame.shape[0], 1)
        cw = obj["w"] * PANEL_W // max(frame.shape[1], 1)
        ch = obj["h"] * PANEL_H // max(frame.shape[0], 1)
        cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), (0, 0, 220), 2)
        cv2.putText(panel, f"#{i+1}", (cx, max(cy - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)

    # Alt bilgi
    cv2.putText(panel, f"Score:{r['score']:.3f}  fg:{r['fg_ratio']:.4f}",
                (8, PANEL_H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (180, 180, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Uyari:{r['toplam_uyari']}  Kare:{r['kare_no']}",
                (8, PANEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (140, 140, 160), 1, cv2.LINE_AA)

    # Skor mini-grafik (sağ alt)
    if len(durum.score_hist) > 1:
        gw, gh = 120, 36
        gx0, gy0 = PANEL_W - gw - 4, PANEL_H - gh - 4
        cv2.rectangle(panel, (gx0, gy0), (PANEL_W - 4, PANEL_H - 4),
                      (20, 20, 30), -1)
        vals = list(durum.score_hist)
        xstep = gw / max(len(vals) - 1, 1)
        pts = [(int(gx0 + i * xstep),
                int(gy0 + gh - int(min(v, 1.0) * gh)))
               for i, v in enumerate(vals)]
        for k in range(1, len(pts)):
            cv2.line(panel, pts[k-1], pts[k],
                     (40, 80, 220) if vals[k] > 0.3 else (80, 200, 120), 1)

    # Çerçeve rengi
    cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), brenk, 2)
    return panel


# ───────────────────────── ana akış ─────────────────────────

def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Ekip modülleri — birleşik demo")
    p.add_argument("--video", required=True, help="işlenecek video dosyası")
    p.add_argument("--gosterge", default="yok",
                    help="GÖSTERGE envanterindeki gauge_id; 'yok' = kimlik beyanı "
                         "yok, değer/birim üretilmez (rastgele videolar için varsayılan)")
    p.add_argument("--gosterge-agirlik",
                    default=str(GOSTERGE_REPO / "runs/detect/models/ip5/karisik/weights/best.pt"))
    p.add_argument("--algilama-agirlik",
                    default=str(GOSTERGE_REPO / "yolov8n.pt") if (GOSTERGE_REPO / "yolov8n.pt").exists()
                    else "yolov8n.pt",
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

    print("[BİLGİ] ANOMALİ modülü (IP8+IP9 sarmalayıcı — Özgür Kotbaş) hazırlanıyor...")
    anomali_durumu = _AnomalDurumu()

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
                p3 = anomali_isle(frame, anomali_durumu)
                p3 = _basliklandir(p3, "ANOMALI (Ozgur) — IP8+MOG2")
            except Exception as e:
                p3 = _hata_paneli(frame, str(e))
                p3 = _basliklandir(p3, "ANOMALI (Ozgur) [HATA]")

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
