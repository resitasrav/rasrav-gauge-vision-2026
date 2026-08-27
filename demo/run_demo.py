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
import json
import statistics
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
sys.path.insert(0, str(DEMO_DIR))          # anomali_demo.py yanımızda duruyor

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


def gosterge_isle(frame: np.ndarray, model, gauge, conf: float) -> tuple[np.ndarray, dict]:
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

    okumalar = read_all_analog(frame, model, tespitler=tespitler)
    canli_oku.tespitleri_ciz(kare, tespitler, okunan_kutu=okunan_kutu)
    canli_oku.analoglari_ciz(kare, okumalar, okunan_kutu=okunan_kutu)
    if sonuc is not None:
        canli_oku.kareyi_ciz(kare, sonuc, gauge)  # "okunamadı" / değer yazımı burada (3. kural)
    else:
        cv2.putText(kare, "kimlik beyani yok - deger/birim uretilmiyor",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                    cv2.LINE_AA)

    sinif_sayim: dict[str, int] = {}
    for t in tespitler:
        sinif_sayim[t.sinif] = sinif_sayim.get(t.sinif, 0) + 1
    olcum = {
        "tespit": sinif_sayim,
        "analog_kutu": len(okumalar),
        "analog_okunan": sum(1 for o in okumalar if o.ok),
        # En büyük kadranın açısı: JSON'dan 180° sıçrama aranabilsin diye.
        "aci": next((round(float(o.needle.angle_img_deg), 1) for o in
                     sorted((o for o in okumalar if o.ok),
                            key=lambda o: -o.radius_px)), None),
    }
    return _letterbox(kare, PANEL_W, PANEL_H), olcum


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


def algilama_isle(frame: np.ndarray, model, conf: float = 0.4) -> tuple[np.ndarray, dict]:
    kare = frame.copy()
    fh, fw = kare.shape[:2]
    sonuclar = model.track(source=frame, conf=conf, persist=True, verbose=False)

    sinif_sayim: dict[str, int] = {}
    en_iyi, en_yuksek_conf = None, -1.0
    if len(sonuclar) > 0 and sonuclar[0].boxes is not None and sonuclar[0].boxes.id is not None:
        kutular = sonuclar[0].boxes
        for i, kutu in enumerate(kutular):
            x1, y1, x2, y2 = map(int, kutu.xyxy[0])
            conf_i = float(kutu.conf[0])
            cls_id = int(kutu.cls[0])
            cls_ad = model.names[cls_id]
            iz_id = int(kutular.id[i]) if kutular.id is not None else -1
            sinif_sayim[cls_ad] = sinif_sayim.get(cls_ad, 0) + 1
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
        # ALGILAMA'nın yayınladığı büyüklük budur: `vision/target_offset`.
        olcum = {"tespit": sinif_sayim, "hedef": en_iyi["cls"],
                 "iz_id": en_iyi["id"], "dx": dx, "dy": dy}
    else:
        cv2.putText(kare, "hedef yok", (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2)
        olcum = {"tespit": sinif_sayim, "hedef": None, "iz_id": -1,
                 "dx": None, "dy": None}

    return _letterbox(kare, PANEL_W, PANEL_H), olcum


# ───────────────────────── ANOMALİ (Özgür) ─────────────────────────
# anomali_test.py bir eğitim + toplu-tahmin scriptidir (anomalib Engine.fit /
# predict, MVTec-AD "bottle" kategorisi). Tek kare alan bir fonksiyon yok,
# kaydedilmiş bir ağırlık da yok (repo taraması: .ckpt/.pt/.pth bulunamadı).
# Gerçek bir çağrı denemek MVTec-AD indirip eğitime başlar, bir video karesiyle
# ilgisi olmaz — bu tespit hâlâ geçerli (RAPOR.md madde 1-2).
#
# 27.08'e kadar panel bu yüzden sabit HATA gösteriyordu. Artık ALGILAMA'ya
# uygulanan çözümün aynısı burada da uygulanıyor: modülün DOSYASI değil
# YÖNTEMİ demo tarafında koşturuluyor (`demo/anomali_demo.py`, PaDiM).
# Künye panelin başlığında duruyor ki kimse bunu Özgür'ün kodunun çıktısı
# sanmasın.

import anomali_demo  # noqa: E402  (demo klasörü sys.path'e main() içinde eklenir)


def anomali_hazirla(video_yolu: Path):
    return anomali_demo.uyumla(anomali_demo.uyum_kareleri(str(video_yolu)))


def anomali_isle(frame: np.ndarray, model) -> tuple[np.ndarray, dict]:
    cizili, olcum = anomali_demo.anomali_isle(frame, model)
    return _letterbox(cizili, PANEL_W, PANEL_H), olcum


# ───────────────────────── modül özetleri ─────────────────────────
# Üç modülün çıktısı ÜÇ FARKLI büyüklük (RAPOR.md §0: farklı görevler, farklı
# şemalar). Ortak bir "reading" özetine zorlamak yanlış olurdu — her modül
# kendi yayınladığı büyüklüğün özetini üretiyor.

def _sinif_topla(iz: list[dict]) -> dict[str, int]:
    toplam: dict[str, int] = {}
    for k in iz:
        for s, n in k.get("tespit", {}).items():
            toplam[s] = toplam.get(s, 0) + n
    return dict(sorted(toplam.items(), key=lambda kv: -kv[1]))


def _gosterge_ozeti(iz: list[dict], hata: str | None) -> dict:
    if hata:
        return {"hata": hata}
    kutu = sum(k["analog_kutu"] for k in iz)
    okunan = sum(k["analog_okunan"] for k in iz)
    # 180° sıçrama: ibre iki ardışık karede 180° dönemez (fizik yasağı).
    # Ground truth olmadan ölçülebilen tek hata sinyali budur.
    acilar = [(i, k["aci"]) for i, k in enumerate(iz) if k["aci"] is not None]
    flip = 0
    for (_, a1), (_, a2) in zip(acilar, acilar[1:]):
        d = abs(a1 - a2) % 360.0
        d = d if d <= 180.0 else 360.0 - d
        if 170.0 <= d <= 190.0:
            flip += 1
    return {"tespit": _sinif_topla(iz), "analog_kutu": kutu,
            "analog_okunan": okunan,
            "kapsam": round(okunan / kutu, 3) if kutu else None,
            "acili_kare": len(acilar), "flip_180": flip}


def _algilama_ozeti(iz: list[dict], hata: str | None) -> dict:
    if hata:
        return {"hata": hata}
    hedefli = [k for k in iz if k["hedef"] is not None]
    return {"tespit": _sinif_topla(iz), "hedefli_kare": len(hedefli),
            "hedefsiz_kare": len(iz) - len(hedefli),
            "izlenen_id_sayisi": len({k["iz_id"] for k in hedefli if k["iz_id"] >= 0}),
            "ort_sapma_px": (
                round(statistics.mean(abs(k["dx"]) + abs(k["dy"]) for k in hedefli), 1)
                if hedefli else None)}


def _anomali_ozeti(iz: list[dict], hata: str | None, esik: float | None) -> dict:
    if hata:
        return {"hata": hata}
    skorlar = [k["skor"] for k in iz]
    isaretli = [i for i, k in enumerate(iz) if k["anomali"]]
    return {"yontem": "PaDiM (demo sarmalayici, torchvision ResNet18)",
            "referans": "videonun ilk kareleri",
            "esik": esik,
            "skor": {"medyan": round(statistics.median(skorlar), 2),
                     "min": round(min(skorlar), 2),
                     "maks": round(max(skorlar), 2)} if skorlar else None,
            "anomali_kare": len(isaretli),
            "anomali_orani": round(len(isaretli) / len(iz), 3) if iz else None,
            "ilk_anomali_kareleri": isaretli[:10]}


# ───────────────────────── ana akış ─────────────────────────

def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Ekip modülleri — birleşik demo")
    p.add_argument("--video", required=True, help="işlenecek video dosyası")
    p.add_argument("--gosterge", default="yok",
                    help="GÖSTERGE envanterindeki gauge_id; 'yok' = kimlik beyanı "
                         "yok, değer/birim üretilmez (rastgele videolar için varsayılan)")
    p.add_argument("--gosterge-agirlik",
                    default=str(GOSTERGE_REPO / "runs/detect/models/ip5/keypad5/weights/best.pt"))
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

    # ANOMALİ "normal" referansını videonun kendi başından çıkarır, bu yüzden
    # hazırlık video yolunu bilmek zorunda — diğer iki modülden farkı budur.
    print("[BİLGİ] ANOMALİ modülü (demo sarmalayıcı, PaDiM) uyumlanıyor...")
    try:
        anmodel = anomali_hazirla(video_yolu)
        anomali_hata = None
        print(f"[BİLGİ] ANOMALİ eşiği {anmodel.esik:.1f} "
              f"({anmodel.uyum_kare_sayisi} kare referans)")
    except Exception as e:
        anmodel = None
        anomali_hata = str(e)
        print(f"[UYARI] ANOMALİ hazırlanamadı: {e}")
    gosterge_izi: list[dict] = []
    algilama_izi: list[dict] = []
    anomali_izi: list[dict] = []

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
                p1, gosterge_olcum = gosterge_isle(frame, gmodel, gauge, args.conf)
                gosterge_izi.append(gosterge_olcum)
            except Exception as e:
                p1 = _hata_paneli(frame, str(e))
            p1 = _basliklandir(p1, "GOSTERGE (Resit)")

            try:
                if algilama_hata is not None:
                    raise RuntimeError(algilama_hata)
                p2, algilama_olcum = algilama_isle(frame, amodel)
                algilama_izi.append(algilama_olcum)
            except Exception as e:
                p2 = _hata_paneli(frame, str(e))
            p2 = _basliklandir(p2, "ALGILAMA (Bedirhan) - demo sarmalayici")

            try:
                if anomali_hata is not None:
                    raise RuntimeError(anomali_hata)
                p3, anomali_olcum = anomali_isle(frame, anmodel)
                anomali_izi.append(anomali_olcum)
            except Exception as e:
                p3 = _hata_paneli(frame, str(e))
            p3 = _basliklandir(p3, "ANOMALI (Ozgur) - demo sarmalayici")

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

    # Her modül KENDİ incelemesinin özetini yazar. Video gözle bakmak içindir;
    # gözle görülemeyen şeyler (180° sıçrama, kaç karede hedef vardı, anomali
    # skorunun zaman içindeki seyri) buradan okunur.
    rapor = {
        "video": video_yolu.name,
        "kare_islenen": kare_idx,
        "sure_sn": round(toplam, 1),
        "GOSTERGE": _gosterge_ozeti(gosterge_izi, gosterge_hata),
        "ALGILAMA": _algilama_ozeti(algilama_izi, algilama_hata),
        "ANOMALI": _anomali_ozeti(anomali_izi, anomali_hata,
                                  anmodel.esik if anmodel else None),
    }
    rapor_yolu = cikti_yolu.with_suffix(".json")
    rapor_yolu.write_text(json.dumps(rapor, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"[BİLGİ] {kare_idx} kare işlendi, {toplam:.1f} sn · çıktı: {cikti_yolu}")
    print(f"[BİLGİ] rapor: {rapor_yolu}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
