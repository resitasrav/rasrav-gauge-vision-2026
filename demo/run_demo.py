# -*- coding: utf-8 -*-
"""demo/run_demo.py — GÖSTERGE + ALGILAMA + ANOMALİ modüllerini tek pencerede
yan yana (3 panel) gösteren birleşik demo.

    python demo/run_demo.py --video <yol>

DOKÜMAN DAYANAĞI
================
Bu dosya aşağıdaki 6 proje dokümanında tanımlanan şartları karşılamak için
yazılmıştır (DOKUMANLAR/):

  Bedirhan_Gok_proje.md      → İP4, İP6, İP10, İP12, İP13 şartları
  Bedirhan_is_paketleri.md   → vision/target_offset ≥15 Hz; fabrika sınıfları
  Resit_Asrav_proje.md       → inspect/reading MQTT; unreadable bayrağı
  Resit_is_paketleri.md      → İP10, İP15 şartları
  Ozgur_Kotbas_proje.md      → patrol/alert MQTT; PDF/MD rapor; alarm oranı
  Ozgur_is_paketleri.md      → İP10, İP13, İP14, İP16 şartları

ENTEGRASYON KARARLARI
=====================
1. MQTT (İP10 — 3 modül): paho-mqtt varsa gerçek broker; yoksa JSONL dosyası.
   • inspect/reading   → Reşit İP10
   • vision/target_offset → Bedirhan İP4/İP10
   • patrol/alert      → Özgür İP10
2. KKD sınıf haritası (Bedirhan İP13): yolov8n.pt COCO sınıflarını fabrika
   kategorilerine map eder. Özel SH17 modeli (--algilama-agirlik) yüklenirse
   o sınıf adları doğrudan kullanılır.
3. Kapalı çevrim simülasyonu (Bedirhan İP12): gerçek servo yok; PID çıktısı
   terminal ve MD raporda "ne gönderilirdi" olarak loglanır.
4. Demo sonu rapor (Özgür İP13/İP16): tur bittikten sonra MD rapor üretilir.
5. Öncelik 1 (robot hareketi filtresi) ve Öncelik 4 (anomali kategorisi) korunur.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEMO_DIR  = Path(__file__).resolve().parent
REPO_DIR  = DEMO_DIR.parent          # rasrav_gauge_repo
STAJ_DIR  = REPO_DIR.parent          # akilli_fabrika_staj-2026 (Opsiyonel, varsa)

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "scripts"))
sys.path.insert(0, str(STAJ_DIR))

# ── MQTT (opsiyonel — paho-mqtt yoksa dosya modu) ──────────────────────────
try:
    import paho.mqtt.client as _mqtt_mod
    MQTT_AVAILABLE = True
except ImportError:
    _mqtt_mod      = None
    MQTT_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# GENEL SABİTLER
# ═══════════════════════════════════════════════════════════════════════════
PANEL_W, PANEL_H = 480, 360
TITLE_H  = 30
FOOTER_H = 40

RENK_BG_BASLIK = (50, 50, 50)
RENK_YAZI      = (240, 240, 240)
RENK_UYARI     = (0, 0, 220)
RENK_OK        = (30, 200, 80)
RENK_HATA_BG   = (20, 20, 20)

# Bedirhan İP13: COCO → Fabrika KKD sınıf haritası
# Kendi SH17 modeliniz yüklenirse bu map atlanır.
COCO_SINIF_HARITASI: dict[str, str] = {
    "person":     "insan",
    "forklift":   "forklift",
    "truck":      "agir_arac",
    "car":        "arac",
    "motorcycle": "arac",
    "bicycle":    "arac",
    "fire extinguisher": "yangin_tupü",
    "helmet":     "baret",
    "hard hat":   "baret",
    "vest":       "is_yelegi",
}

# Bedirhan İP13 kritik sınıflar (SH17 etiket isimleri de dahil)
KKD_KRITIK: set[str] = {
    "Fall-Detected", "NO-Hardhat", "NO-Safety Vest",
    "NO-Gloves", "NO-Goggles", "NO-Mask",
    "insan_dusme", "baret_yok", "yelek_yok",
}

# Bedirhan İP12 — PID simülasyon parametreleri (gerçek servo yok)
PID_KP, PID_KI, PID_KD = 0.08, 0.001, 0.02
PID_MAX_CIKTI = 30.0   # derece/sn (simüle — servo limiti)

# ═══════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════

def _letterbox(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    olcek = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * olcek)), max(1, int(fh * olcek))
    kucuk  = cv2.resize(frame, (nw, nh))
    tuval  = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    tuval[y0:y0 + nh, x0:x0 + nw] = kucuk
    return tuval


def _hata_paneli(kaynak: np.ndarray, mesaj: str) -> np.ndarray:
    panel = _letterbox(kaynak, PANEL_W, PANEL_H)
    ov    = panel.copy()
    cv2.rectangle(ov, (0, 0), (PANEL_W, PANEL_H), RENK_HATA_BG, -1)
    panel = cv2.addWeighted(ov, 0.60, panel, 0.40, 0)
    for i, satir in enumerate(textwrap.wrap("HATA: " + mesaj, width=32)):
        cv2.putText(panel, satir, (14, PANEL_H // 2 - 30 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, RENK_UYARI, 1, cv2.LINE_AA)
    return panel


def _basliklandir(panel: np.ndarray, baslik: str) -> np.ndarray:
    tuval = np.zeros((PANEL_H + TITLE_H, PANEL_W, 3), dtype=np.uint8)
    tuval[:TITLE_H] = RENK_BG_BASLIK
    cv2.putText(tuval, baslik, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, RENK_YAZI, 1, cv2.LINE_AA)
    tuval[TITLE_H:] = panel
    return tuval


# ═══════════════════════════════════════════════════════════════════════════
# MQTT YAYINCISI — 3 MODÜL İÇİN ORTAKLAŞTIRILMIŞ (İP10 ×3)
# ═══════════════════════════════════════════════════════════════════════════

class _MqttYayinci:
    """
    Üç modülün (GÖSTERGE, ALGILAMA, ANOMALİ) MQTT yayınını ortak yönetir.

    Reşit İP10  → inspect/reading
    Bedirhan İP10 → vision/target_offset
    Özgür İP10  → patrol/alert
    """

    def __init__(self, broker: str, port: int):
        self._broker   = broker
        self._port     = port
        self._client   = None
        self._cikti_d  = DEMO_DIR / "cikti"
        self._cikti_d.mkdir(parents=True, exist_ok=True)
        self._dosyalar: dict[str, Path] = {}
        self._gonderilen = 0

        if MQTT_AVAILABLE:
            try:
                c = _mqtt_mod.Client(client_id=f"ekip_demo_{int(time.time())%10000}")
                c.connect(broker, port, keepalive=30)
                c.loop_start()
                self._client = c
                print(f"  [MQTT] Broker bağlantısı: {broker}:{port}")
            except Exception as e:
                print(f"  [MQTT] Broker bağlanamadı ({e}) — dosya moduna geçildi.")
        else:
            print("  [MQTT] paho-mqtt kurulu değil — dosya modunda çalışılıyor.")

    def yayinla(self, topic: str, mesaj: dict) -> None:
        """topic'e JSON yayını; broker yoksa JSONL dosyasına yazar."""
        mesaj.setdefault("ts", datetime.now().isoformat())
        payload = json.dumps(mesaj, ensure_ascii=False)
        self._gonderilen += 1

        if self._client is not None:
            try:
                self._client.publish(topic, payload, qos=0)
                return
            except Exception:
                pass  # dosyaya düş

        # Dosya modu
        if topic not in self._dosyalar:
            temiz = topic.replace("/", "_")
            self._dosyalar[topic] = self._cikti_d / f"mqtt_{temiz}.jsonl"
        with open(self._dosyalar[topic], "a", encoding="utf-8") as f:
            f.write(payload + "\n")

    def kapat(self) -> None:
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# GÖSTERGE PANELİ — Reşit Asrav (İP5-İP10, İP15)
# ═══════════════════════════════════════════════════════════════════════════

def gosterge_hazirla(gosterge_id: str, agirlik: Path):
    from gauge_vision.config import load_gauges
    from ultralytics import YOLO
    gauge = None
    if gosterge_id != "yok":
        gauges = load_gauges(str(REPO_DIR / "configs" / "gauges.yaml"))
        if gosterge_id in gauges:
            gauge = gauges[gosterge_id]
        else:
            raise RuntimeError(f"Gösterge ID bulunamadı: {gosterge_id}")
    if not agirlik.exists():
        raise RuntimeError(f"Ağırlık dosyası yok: {agirlik}")
    return gauge, YOLO(str(agirlik))


def gosterge_isle(frame: np.ndarray, model, gauge, conf: float,
                   mqtt: _MqttYayinci, kare_no: int) -> tuple[np.ndarray, dict]:
    """
    Reşit İP5-İP8: YOLO tespit → ibre açısı → değer (kalibrasyonlu)
    Reşit İP15: conf < eşik → status: unreadable (yanlış okumaktansa okuyamadım)
    Reşit İP10: inspect/reading MQTT yayını (her 15 karede bir ~1 Hz)
    """
    from gauge_vision.pipeline import detect_objects, read_all_analog, read_gauge
    import canli_oku

    kare   = frame.copy()
    tespitler = detect_objects(frame, model, conf=conf)
    sonuc     = read_gauge(frame, model, gauge, detect_conf=conf) if gauge else None
    okunan_kutu = sonuc.box_xyxy if sonuc else None

    canli_oku.tespitleri_ciz(kare, tespitler, okunan_kutu=okunan_kutu)
    canli_oku.analoglari_ciz(
        kare, read_all_analog(frame, model, tespitler=tespitler),
        okunan_kutu=okunan_kutu)
    if sonuc is not None:
        canli_oku.kareyi_ciz(kare, sonuc, gauge)
    else:
        cv2.putText(kare, "kimlik beyani yok — deger/birim uretilmiyor",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

    # Reşit İP15: unreadable bayrağı — conf_threshold gauges.yaml'da 0.70
    # NOT: FrameResult'ın value alanı reading.value üzerinden alınır.
    okuma_durumu = "ok"
    deger_str    = "—"
    if sonuc is not None:
        # reading alt nesnesi: GaugeReading dataclass → .value, .conf alanları
        okuma = getattr(sonuc, "reading", None)
        deger = getattr(okuma, "value", None) if okuma else None
        okuma_conf = getattr(okuma, "conf", getattr(sonuc, "detect_conf", 0.0))
        if deger is not None:
            deger_str = f"{deger:.2f}"
        # İP15: conf < eşik → unreadable (yanlış okumaktansa okuyamadım)
        esik = getattr(gauge, "conf_threshold", 0.70) if gauge else 0.70
        if okuma_conf < esik or deger is None:
            okuma_durumu = "unreadable"
            deger_str    = "unreadable" if deger is None else f"unreadable({deger:.2f})"

    # Reşit İP10: MQTT yayını — her 15 karede bir (~1 Hz @15fps)
    if kare_no % 15 == 0:
        okuma = getattr(sonuc, "reading", None) if sonuc else None
        mqtt.yayinla("inspect/reading", {
            "schema": 1,
            "gauge_id": gauge.id if gauge else "unknown",
            "type":     "analog",
            "value":    getattr(okuma, "value", None) if okuma else None,
            "unit":     gauge.unit if gauge else None,
            "status":   okuma_durumu,
            "conf":     round(getattr(okuma, "conf", 0.0), 3) if okuma else 0.0,
        })

    okuma_ozet = {
        "gauge_id": gauge.id if gauge else "unknown",
        "deger": deger_str,
        "durum": okuma_durumu,
    }
    return _letterbox(kare, PANEL_W, PANEL_H), okuma_ozet


# ═══════════════════════════════════════════════════════════════════════════
# ALGILAMA PANELİ — Bedirhan Gök (İP6, İP10, İP12, İP13)
# ═══════════════════════════════════════════════════════════════════════════

class _PidSimulasyonu:
    """
    Bedirhan İP12 — Kapalı Çevrim Simülasyonu.

    Gerçek pan-tilt servo bağlantısı olmadığı için PID çıktısı hesaplanır
    ve loglanır. Gerçek sistemde bu değer servo sürücüsüne gönderilir.
    """
    def __init__(self):
        self._int_x = 0.0
        self._int_y = 0.0
        self._prev_dx = 0.0
        self._prev_dy = 0.0

    def hesapla(self, dx: float, dy: float, dt: float = 1/25) -> dict:
        self._int_x = max(-200, min(200, self._int_x + dx * dt))
        self._int_y = max(-200, min(200, self._int_y + dy * dt))
        d_dx = (dx - self._prev_dx) / max(dt, 1e-6)
        d_dy = (dy - self._prev_dy) / max(dt, 1e-6)
        self._prev_dx, self._prev_dy = dx, dy

        u_pan  = PID_KP * dx + PID_KI * self._int_x + PID_KD * d_dx
        u_tilt = PID_KP * dy + PID_KI * self._int_y + PID_KD * d_dy
        u_pan  = max(-PID_MAX_CIKTI, min(PID_MAX_CIKTI, u_pan))
        u_tilt = max(-PID_MAX_CIKTI, min(PID_MAX_CIKTI, u_tilt))
        return {"pan_deg_s": round(u_pan, 2), "tilt_deg_s": round(u_tilt, 2)}


def algilama_hazirla(agirlik: Path):
    from ultralytics import YOLO
    if not agirlik.exists():
        raise RuntimeError(f"Ağırlık dosyası yok: {agirlik}")
    return YOLO(str(agirlik))


def algilama_isle(frame: np.ndarray, model, conf: float,
                  pid: _PidSimulasyonu, mqtt: _MqttYayinci,
                  kare_no: int) -> tuple[np.ndarray, dict]:
    """
    Bedirhan İP6: insan/nesne tespiti (YOLO)
    Bedirhan İP9: dx,dy ofset hesabı
    Bedirhan İP13: COCO sınıflarını fabrika KKD kategorilerine map et
    Bedirhan İP12: PID simülasyon (kapalı çevrim logu)
    Bedirhan İP10: vision/target_offset MQTT yayını ≥15 Hz
    """
    kare = frame.copy()
    fh, fw = kare.shape[:2]
    sonuclar = model.track(source=frame, conf=conf, persist=True, verbose=False)

    dx, dy      = 0, 0
    cls_ad      = ""
    cls_fabrika = ""
    en_iyi      = None
    en_yuksek_conf = -1.0

    if (sonuclar and sonuclar[0].boxes is not None
            and sonuclar[0].boxes.id is not None):
        kutular = sonuclar[0].boxes
        for i, kutu in enumerate(kutular):
            x1, y1, x2, y2 = map(int, kutu.xyxy[0])
            conf_i  = float(kutu.conf[0])
            cls_id  = int(kutu.cls[0])
            cls_raw = model.names[cls_id]
            # Bedirhan İP13: fabrika sınıfı haritası
            cls_fab = COCO_SINIF_HARITASI.get(cls_raw.lower(), cls_raw)
            iz_id   = int(kutular.id[i]) if kutular.id is not None else -1

            renk = RENK_UYARI if cls_raw in KKD_KRITIK else (50, 220, 50)
            cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)
            cv2.putText(kare,
                        f"ID:{iz_id} {cls_fab}:{conf_i:.2f}",
                        (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, renk, 1)

            if conf_i > en_yuksek_conf:
                en_yuksek_conf = conf_i
                en_iyi = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cls": cls_raw, "cls_fabrika": cls_fab,
                    "iz_id": iz_id, "conf": conf_i,
                }

    if en_iyi is not None:
        cx, cy  = (en_iyi["x1"] + en_iyi["x2"]) // 2, (en_iyi["y1"] + en_iyi["y2"]) // 2
        fcx, fcy = fw // 2, fh // 2
        dx, dy   = cx - fcx, cy - fcy
        cls_ad   = en_iyi["cls"]
        cls_fabrika = en_iyi["cls_fabrika"]
        cv2.circle(kare, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(kare, (fcx, fcy), (cx, cy), (255, 100, 0), 2)
        cv2.putText(kare, f"dx:{dx}  dy:{dy}",
                    (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(kare, "hedef yok", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Bedirhan İP12: PID simülasyon logu
    pid_cikti = pid.hesapla(dx, dy)
    cv2.putText(kare,
                f"PID pan:{pid_cikti['pan_deg_s']:+.1f} tilt:{pid_cikti['tilt_deg_s']:+.1f} deg/s",
                (8, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (200, 200, 100), 1, cv2.LINE_AA)

    # Bedirhan İP10: vision/target_offset MQTT ≥15 Hz (her kare)
    mqtt.yayinla("vision/target_offset", {
        "dx":   int(dx),
        "dy":   int(dy),
        "class": cls_ad,
        "class_fabrika": cls_fabrika,
        "conf": round(en_yuksek_conf, 3) if en_yuksek_conf > 0 else 0.0,
        "pid":  pid_cikti,
        "kare": kare_no,
    })

    robot_durum = {
        "dx": dx, "dy": dy,
        "class": cls_ad, "class_fabrika": cls_fabrika,
        "conf": round(en_yuksek_conf, 3) if en_yuksek_conf > 0 else 0.0,
        "pid": pid_cikti,
    }
    return _letterbox(kare, PANEL_W, PANEL_H), robot_durum


# ═══════════════════════════════════════════════════════════════════════════
# ANOMALİ PANELİ — Özgür Kotbaş (İP9, İP10, İP13, Öncelik 1/2/4)
# ═══════════════════════════════════════════════════════════════════════════

class _AnomalDurumu:
    """
    Özgür İP9: MOG2+ORB+RANSAC ensemble anomali tespiti
    Öncelik 1: Robot hareket halindeyken analiz durdurulur
    Öncelik 2: Geçişte kritik YOLO sınıfı → HIGH uyarı
    Öncelik 4: Kategori tabanlı anomali skoru
    """

    def __init__(self):
        try:
            from scripts.core import anomali_hizalamali as hz
            self._algilayici = hz.AkisAlgilayici()
            self._hazir = True
        except ImportError as e:
            print(f"  [ANOMALİ] Özgür modülü yüklenemedi: {e}")
            self._algilayici = None
            self._hazir = False

        self.score_hist   = deque(maxlen=60)
        self.toplam_uyari = 0
        self.kare_no      = 0
        self._gecis_uyarilari: list[dict] = []
        
        # Kamera Hareketi Tespiti (Odometri Simülasyonu)
        self.prev_gray = None
        self.p0 = None
        self.is_moving = False

    def _kamera_hareketli_mi(self, frame: np.ndarray) -> bool:
        """Optik akış ile kameranın (robotun) fiziksel olarak ilerleyip ilerlemediğini ölçer."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5) # Hızlandırmak için küçült
        if self.prev_gray is None:
            self.prev_gray = gray
            self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
            return True
            
        hareket = False
        if self.p0 is not None and len(self.p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None)
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]
                if len(good_new) > 10:
                    distances = np.linalg.norm(good_new - good_old, axis=1)
                    mean_dist = np.mean(distances)
                    hareket = mean_dist > 1.5  # 1.5 px'den fazla kayma varsa robot yürüyor demektir
        
        # Özellik noktalarını (köşeleri) yenile
        if self.kare_no % 5 == 0 or self.p0 is None or len(self.p0) < 50:
            self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
        else:
            self.p0 = good_new.reshape(-1, 1, 2) if 'good_new' in locals() and len(good_new) > 0 else None
            
        self.prev_gray = gray
        # Hareketi yumuşat (1 karelik titremeleri engelle)
        self.is_moving = hareket
        return self.is_moving

    def isle(self, frame: np.ndarray, robot_durum: dict) -> dict:
        self.kare_no += 1
        dx       = robot_durum.get("dx", 0)
        dy       = robot_durum.get("dy", 0)
        cls_ad   = robot_durum.get("class", "")
        cls_fab  = robot_durum.get("class_fabrika", cls_ad)
        kritik   = cls_ad in KKD_KRITIK or cls_fab in KKD_KRITIK

        # Öncelik 1: Robot hareket halinde → analiz atla
        hareketli = self._kamera_hareketli_mi(frame)
        if (abs(dx) > 20 or abs(dy) > 20 or hareketli) and not kritik:
            self.score_hist.append(0.0)
            return self._bos_sonuc("HAREKET (Yuruyus) — MOG2 uykuya alindi")

        # Öncelik 2: Kritik KKD geçiş uyarısı
        if kritik:
            self.toplam_uyari += 1
            self.score_hist.append(1.0)
            return {
                "is_alert": True, "severity": "HIGH", "score": 1.0,
                "fg_mask": None, "fg_ratio": 0.0,
                "nesneler": [{"x": 0, "y": 0, "w": 0, "h": 0,
                              "kategori": "kkd_ihlali"}],
                "kare_no": self.kare_no, "toplam_uyari": self.toplam_uyari,
                "karar_aciklama": f"KKD İHLALİ: {cls_ad or cls_fab}",
            }

        if not self._hazir:
            self.score_hist.append(0.0)
            return self._bos_sonuc("Modül yüklenemedi")

        # Normal analiz
        r        = self._algilayici.isle(frame)
        nesneler = r.get("nesneler", [])
        is_alert = len(nesneler) > 0
        severity = ("HIGH" if len(nesneler) >= 2
                    else "MEDIUM" if len(nesneler) == 1
                    else "NONE")
        score    = min(1.0, len(nesneler) * 0.5)

        # Öncelik 4: Kategori tabanlı sınıflandırma
        for o in nesneler:
            if "kategori" not in o:
                roi = frame[max(0, o["y"]):o["y"]+o["h"],
                            max(0, o["x"]):o["x"]+o["w"]]
                if roi.size > 0:
                    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    v_ort = hsv[..., 2].mean()
                    s_ort = hsv[..., 1].mean()
                    bw, bh = o["w"], o["h"]
                    if v_ort < 80 and s_ort < 60 and bw > bh * 1.5:
                        o["kategori"] = "zemin_sizintisi"
                    elif bh > bw * 1.5:
                        o["kategori"] = "yapi_anomalisi"
                    else:
                        o["kategori"] = "yabanci_sabit_nesne"
                else:
                    o["kategori"] = "yabanci_sabit_nesne"

        if is_alert:
            self.toplam_uyari += 1
        self.score_hist.append(score)

        kategoriler = [o.get("kategori", "bilinmiyor") for o in nesneler]
        ana_kat     = kategoriler[0] if kategoriler else "normal"
        return {
            "is_alert": is_alert, "severity": severity, "score": score,
            "fg_mask":  r.get("fg_mask"), "fg_ratio": r.get("fg_ratio", 0.0),
            "nesneler": nesneler,
            "kare_no":  self.kare_no, "toplam_uyari": self.toplam_uyari,
            "karar_aciklama": f"MOG2: {ana_kat.upper()}" if is_alert else "",
        }

    def _bos_sonuc(self, aciklama: str = "") -> dict:
        return {
            "is_alert": False, "severity": "NONE", "score": 0.0,
            "fg_mask": None, "fg_ratio": 0.0, "nesneler": [],
            "kare_no": self.kare_no, "toplam_uyari": self.toplam_uyari,
            "karar_aciklama": aciklama,
        }


def anomali_isle(frame: np.ndarray, durum: _AnomalDurumu,
                 robot_durum: dict, mqtt: _MqttYayinci,
                 kare_no: int, wp_sayisi: int) -> np.ndarray:
    """
    Özgür İP9: Ensemble anomali tespiti kare-bazlı
    Özgür İP10: patrol/alert MQTT yayını
    FP DÜZELTME: fg_ratio=0 iken (warmup sürüyor) veya robot hareket halinde
    uyarı basılmaz.
    """
    r = durum.isle(frame, robot_durum)

    # FP Düzeltme: MOG2 warmup (ilk 40 kare) süresince fg_ratio=0 → alert iptal
    if r["is_alert"] and r.get("fg_ratio", 0.0) == 0.0 and kare_no < 45:
        r = dict(r)  # frozendict değil; güncellenebilir kopyası
        r["is_alert"] = False
        r["severity"] = "NONE"
        r["karar_aciklama"] = f"Warmup ({kare_no}/40) — atlandı"

    # Özgür İP10: patrol/alert yayını — her alert veya 30 karede bir
    if r["is_alert"] or kare_no % 30 == 0:
        mqtt.yayinla("patrol/alert", {
            "type":      "patrol_alert",
            "severity":  r["severity"],
            "waypoint":  f"WP_demo_{wp_sayisi:02d}",
            "score":     round(r["score"], 3),
            "det_count": len(r["nesneler"]),
            "img_ref":   "",
            "is_alert":  r["is_alert"],
            "degisiklik_tipi": (r["nesneler"][0].get("kategori", "bilinmiyor")
                                if r["nesneler"] else "normal"),
            "karar_aciklama": r.get("karar_aciklama", ""),
        })

    # Görselleştirme
    if r["fg_mask"] is not None:
        fg_bgr = cv2.cvtColor(r["fg_mask"], cv2.COLOR_GRAY2BGR)
        panel  = cv2.addWeighted(_letterbox(frame, PANEL_W, PANEL_H), 0.45,
                                 _letterbox(fg_bgr, PANEL_W, PANEL_H), 0.55, 0)
    else:
        panel = _letterbox(frame, PANEL_W, PANEL_H)

    brenk = RENK_UYARI if r["is_alert"] else RENK_OK
    btxt  = (f">>> UYARI [{r['severity']}]" if r["is_alert"] else "Normal")
    cv2.rectangle(panel, (0, 0), (PANEL_W, 30), (0, 0, 0), -1)
    cv2.putText(panel, btxt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, brenk, 2, cv2.LINE_AA)

    if r.get("karar_aciklama"):
        cv2.putText(panel, r["karar_aciklama"], (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 220) if r["is_alert"] else (100, 180, 255),
                    1, cv2.LINE_AA)

    for i, obj in enumerate(r["nesneler"][:3]):
        cx = obj["x"] * PANEL_W // max(frame.shape[1], 1)
        cy = obj["y"] * PANEL_H // max(frame.shape[0], 1)
        cw = obj["w"] * PANEL_W // max(frame.shape[1], 1)
        ch = obj["h"] * PANEL_H // max(frame.shape[0], 1)
        cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), RENK_UYARI, 2)
        cv2.putText(panel, f"#{i+1}", (cx, max(cy - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, RENK_UYARI, 1)

    cv2.putText(panel, f"Score:{r['score']:.3f}  fg:{r['fg_ratio']:.4f}",
                (8, PANEL_H - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (180, 180, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Uyari:{r['toplam_uyari']}  Kare:{r['kare_no']}",
                (8, PANEL_H - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (140, 140, 160), 1, cv2.LINE_AA)

    # Skor mini-grafik
    if len(durum.score_hist) > 1:
        gw, gh = 100, 32
        gx0    = PANEL_W - gw - 4
        gy0    = PANEL_H - gh - 4
        cv2.rectangle(panel, (gx0, gy0), (PANEL_W - 4, PANEL_H - 4), (15, 15, 25), -1)
        vals  = list(durum.score_hist)
        xstep = gw / max(len(vals) - 1, 1)
        pts   = [(int(gx0 + i * xstep),
                  int(gy0 + gh - int(min(v, 1.0) * gh))) for i, v in enumerate(vals)]
        for k in range(1, len(pts)):
            cv2.line(panel, pts[k-1], pts[k],
                     RENK_UYARI if vals[k] > 0.3 else RENK_OK, 1)

    cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), brenk, 2)
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# DEMO SONU MD RAPORU — Özgür İP13/İP16
# ═══════════════════════════════════════════════════════════════════════════

def _rapor_uret(oturum: dict, cikti_yolu: Path) -> Path:
    """
    Özgür İP13: Demo oturumunu Markdown rapor olarak yazar.
    Şiddet sıralı (HIGH → NORMAL), MQTT dosyaları referanslanır.
    """
    ts    = oturum["baslangic"]
    rapor = cikti_yolu.parent / f"demo_rapor_{ts[:10].replace('-','')}.md"
    satirlar = [
        f"# Ekip Demo Raporu\n",
        f"**Tarih:** {ts[:19]}  ",
        f"**Video:** `{oturum.get('video', '?')}`  ",
        f"**Toplam kare:** {oturum['kare_sayisi']}  ",
        f"**Süre:** {oturum['sure_s']:.1f} sn  \n",
        "---\n",
        "## Özgür — ANOMALİ Özeti",
        f"- Toplam uyarı: **{oturum['anomali_uyari']}**",
        f"- Alarm/kare oranı: **{oturum['anomali_uyari']/max(oturum['kare_sayisi'],1):.4f}**\n",
        "## Bedirhan — ALGILAMA Özeti",
        f"- Tespit edilen benzersiz hedefler: {oturum['bedirhan_tespit']}",
        f"- KKD kritik alarm: **{oturum['bedirhan_kritik']}**",
        f"- PID simüle (max pan): {oturum['pid_max_pan']:.1f} deg/s\n",
        "## Reşit — GÖSTERGE Özeti",
        f"- Okunan kare: {oturum['resit_okunan']}",
        f"- unreadable: {oturum['resit_unreadable']}\n",
        "---\n",
        "## MQTT Çıktı Dosyaları",
        f"- `cikti/mqtt_inspect_reading.jsonl`  ← Reşit İP10",
        f"- `cikti/mqtt_vision_target_offset.jsonl`  ← Bedirhan İP10",
        f"- `cikti/mqtt_patrol_alert.jsonl`  ← Özgür İP10\n",
        "## Kapalı Çevrim (Bedirhan İP12 — Simülasyon)",
        "> PID çıktısı yukarıdaki değerlere göre hesaplanmış ve loglanmıştır.",
        "> Gerçek servo bağlantısı kurulduğunda bu değerler servo sürücüsüne gider.\n",
        "---",
        f"*Rapor otomatik üretildi — run_demo.py v2 (demo ortamı)*",
    ]
    rapor.write_text("\n".join(satirlar), encoding="utf-8")
    return rapor


# ═══════════════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Ekip Demo — GÖSTERGE | ALGILAMA | ANOMALİ")
    p.add_argument("--video", required=True)
    p.add_argument("--gosterge", default="yok",
                   help="gauges.yaml'daki gauge_id; 'yok' = kimlik beyanı yok")
    p.add_argument("--gosterge-agirlik",
                   default=str(REPO_DIR / "runs/detect/models/ip5/karisik/weights/best.pt"))
    p.add_argument("--algilama-agirlik",
                   default=str(REPO_DIR / "yolov8n.pt"),
                   help="Bedirhan'ın YOLO ağırlığı (SH17 veya COCO)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--mqtt-broker", default="localhost")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--out", default=str(DEMO_DIR / "cikti" / "demo.mp4"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args(argv)

    video_yolu = Path(args.video)
    if not video_yolu.exists():
        print(f"[HATA] Video bulunamadı: {video_yolu}")
        return 1

    cap = cv2.VideoCapture(str(video_yolu))
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_yolu}")
        return 1
    kaynak_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # MQTT başlat (İP10 ×3)
    print("[BİLGİ] MQTT yayıncısı başlatılıyor...")
    mqtt = _MqttYayinci(args.mqtt_broker, args.mqtt_port)

    # GÖSTERGE hazırla (Reşit)
    print("[BİLGİ] GÖSTERGE modülü yükleniyor...")
    try:
        gauge, gmodel = gosterge_hazirla(args.gosterge,
                                         Path(args.gosterge_agirlik))
        gosterge_hata = None
    except Exception as e:
        gauge = gmodel = None
        gosterge_hata = str(e)
        print(f"  [UYARI] GÖSTERGE hazırlanamadı: {e}")

    # ALGILAMA hazırla (Bedirhan)
    print("[BİLGİ] ALGILAMA modülü yükleniyor (Bedirhan)...")
    try:
        amodel       = algilama_hazirla(Path(args.algilama_agirlik))
        algilama_hata = None
    except Exception as e:
        amodel        = None
        algilama_hata = str(e)
        print(f"  [UYARI] ALGILAMA hazırlanamadı: {e}")

    pid = _PidSimulasyonu()

    # ANOMALİ hazırla (Özgür)
    print("[BİLGİ] ANOMALİ modülü hazırlanıyor (Özgür)...")
    anomali_durumu = _AnomalDurumu()

    # Video çıktısı
    cikti_yolu = Path(args.out)
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    # Oturum istatistikleri (rapor için)
    oturum = {
        "baslangic": datetime.now().isoformat(),
        "video": str(video_yolu),
        "kare_sayisi": 0,
        "sure_s": 0.0,
        "anomali_uyari": 0,
        "bedirhan_tespit": 0,
        "bedirhan_kritik": 0,
        "pid_max_pan": 0.0,
        "resit_okunan": 0,
        "resit_unreadable": 0,
    }
    wp_sayisi = 0

    kare_idx  = 0
    t_basla   = time.perf_counter()
    pencere   = "Ekip Demo — GOSTERGE | ALGILAMA | ANOMALI (Q=cikis)"

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
            oturum["kare_sayisi"] = kare_idx

            # ── GÖSTERGE PANELİ (Reşit) ─────────────────────────────────
            try:
                if gosterge_hata:
                    raise RuntimeError(gosterge_hata)
                p1, okuma_ozet = gosterge_isle(
                    frame, gmodel, gauge, args.conf, mqtt, kare_idx)
                if okuma_ozet["durum"] == "unreadable":
                    oturum["resit_unreadable"] += 1
                else:
                    oturum["resit_okunan"] += 1
            except Exception as e:
                p1 = _hata_paneli(frame, str(e))
                okuma_ozet = {"gauge_id": "?", "deger": "?", "durum": "hata"}
            p1 = _basliklandir(p1,
                f"GOSTERGE (Resit) — {okuma_ozet.get('deger','?')} | {okuma_ozet.get('durum','?')}")

            # ── ALGILAMA PANELİ (Bedirhan) ───────────────────────────────
            try:
                if algilama_hata:
                    raise RuntimeError(algilama_hata)
                p2, robot_durum = algilama_isle(
                    frame, amodel, args.conf, pid, mqtt, kare_idx)
                pd  = robot_durum.get("pid", {})
                oturum["pid_max_pan"] = max(
                    oturum["pid_max_pan"], abs(pd.get("pan_deg_s", 0)))
                if robot_durum.get("class"):
                    oturum["bedirhan_tespit"] += 1
                if robot_durum.get("class") in KKD_KRITIK:
                    oturum["bedirhan_kritik"] += 1
            except Exception as e:
                p2 = _hata_paneli(frame, str(e))
                robot_durum = {"dx": 0, "dy": 0, "class": "", "class_fabrika": ""}
            p2 = _basliklandir(p2,
                f"ALGILAMA (Bedirhan) — dx:{robot_durum.get('dx',0)} dy:{robot_durum.get('dy',0)}")

            # ── ANOMALİ PANELİ (Özgür) ──────────────────────────────────
            try:
                p3 = anomali_isle(
                    frame, anomali_durumu, robot_durum, mqtt, kare_idx, wp_sayisi)
                oturum["anomali_uyari"] = anomali_durumu.toplam_uyari
            except Exception as e:
                p3 = _hata_paneli(frame, str(e))
            p3 = _basliklandir(p3,
                f"ANOMALI (Ozgur) — Uyari:{anomali_durumu.toplam_uyari}")

            # ── BİRLEŞİK EKRAN ──────────────────────────────────────────
            birlesik = np.hstack([p1, p2, p3])
            gecen    = time.perf_counter() - t0
            fps_an   = 1.0 / gecen if gecen > 0 else 0.0

            altbilgi = np.zeros((FOOTER_H, birlesik.shape[1], 3), dtype=np.uint8)
            cv2.putText(altbilgi,
                        f"Kare {kare_idx} · {fps_an:.1f} FPS  |  "
                        f"MQTT:{mqtt._gonderilen}  |  "
                        f"Uyari:{anomali_durumu.toplam_uyari}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, RENK_YAZI, 1, cv2.LINE_AA)
            birlesik = np.vstack([birlesik, altbilgi])

            if writer is None:
                h, w  = birlesik.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(cikti_yolu), fourcc, kaynak_fps, (w, h))
            writer.write(birlesik)

            if not args.no_show:
                cv2.imshow(pencere, birlesik)
                tus = cv2.waitKey(1) & 0xFF
                if tus in (ord("q"), 27):
                    break
                if tus == ord("n"):
                    wp_sayisi += 1  # manuel waypoint geçişi
    except KeyboardInterrupt:
        print("\n[BİLGİ] Kullanıcı tarafından durduruldu.")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    oturum["sure_s"] = time.perf_counter() - t_basla

    # Özgür İP13/İP16: Demo sonu MD raporu
    rapor_yolu = _rapor_uret(oturum, cikti_yolu)
    print(f"\n[BİLGİ] {kare_idx} kare işlendi, {oturum['sure_s']:.1f} sn")
    print(f"[BİLGİ] Video çıktısı: {cikti_yolu}")
    print(f"[BİLGİ] Devriye raporu: {rapor_yolu}")
    print(f"[BİLGİ] MQTT yayını: {mqtt._gonderilen} mesaj (cikti/mqtt_*.jsonl)")

    mqtt.kapat()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
