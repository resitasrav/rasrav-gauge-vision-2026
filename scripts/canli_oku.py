"""Zinciri uçtan uca çalıştırır: görüntü → tespit → kırp → açı → değer (İP5+İP6+İP7).

    python scripts/canli_oku.py --kaynak 0 --gosterge PT-101          # kamera
    python scripts/canli_oku.py --kaynak data/synthetic/v0/images/0004_PT-101.png --gosterge PT-101
    python scripts/canli_oku.py --kaynak 0 --gosterge PT-101 --kaydet

Bu script yeni bir yöntem getirmez; üç iş paketinin çıktısını birbirine bağlar.
Ölçüm scriptleri (`olc_ip6.py`, `olc_ip7.py`) kırpımı ve merkezi **etiketten** alır;
burada ikisi de **tespitten** gelir. Aradaki fark, zincirin gerçek hatasıdır.

**Bilinçli sınırlar — demoda ekrana da yazılır, izleyen yanılmasın:**

1. **Hangi gösterge olduğu elle veriliyor** (`--gosterge`). Tespit "burada bir gösterge
   var" der, "bu PT-101'dir" demez. Gerçek sistemde bunu robotun durağı (waypoint)
   söyleyecektir. Yanlış gösterge seçilirse sayı sessizce yanlış çıkar — bu yüzden
   seçilen kimlik kareye yazılır.
2. **Yatıklık (roll) sıfır kabul edilir.** Sentetik veride etiketten geliyordu; gerçek
   görüntüde kadranın kendi geometrisinden çıkarılması gerekir (İP8, K2 ile birlikte).
   Kamera yatıksa okuma yatıklık kadar kayar.
3. **Merkez kutudan türetilir.** Ölçülen sapma kadran çapının ~%4'üdür ve zincire
   ~%3 hata sokar (05.08). Demo bunu gizlemez; iyileştirme sıradaki iştir.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from gauge_vision.config import load_gauges
from gauge_vision.read.calibrate import DURUM_OK, DURUM_OKUNAMADI, read_value
from gauge_vision.read.needle import read_needle_angle

VARSAYILAN_AGIRLIK = "runs/detect/models/ip5/karisik/weights/best.pt"
CIKTI_DIZINI = "outputs/figures"

# Kadran yüzünün yarıçapı kutudan türetilir. Kutu bezeli de içerdiği için ham
# yarının tamamı alınmaz: sentetik üreteçte dış yarıçap = kadran yarıçapı × 1,07
# (BEZEL_WIDTH_RATIO). Fazla büyük bir yarıçap tarama halkasını kadranın dışına
# taşırır ve ana çizgiler ibre sanılabilir.
KUTU_YARICAP_ORANI = 1 / 1.07

# Kutu kare değilse (açılı bakış, kısmi örtme) kısa kenar esas alınır: uzun kenara
# göre alınan yarıçap kadranın dışını tarar.
MIN_YARICAP_PX = 12

RENK_OK = (60, 200, 60)
RENK_UYARI = (40, 40, 220)
RENK_BILGI = (40, 40, 40)


def kutudan_kadran(kutu_xyxy) -> tuple[tuple[int, int], float]:
    """Tespit kutusundan kadran merkezi ve yarıçapı."""
    x1, y1, x2, y2 = kutu_xyxy
    merkez = (round((x1 + x2) / 2), round((y1 + y2) / 2))
    yaricap = min(x2 - x1, y2 - y1) / 2 * KUTU_YARICAP_ORANI
    return merkez, yaricap


def kareyi_oku(kare, model, gauge, *, conf_esik: float):
    """Tek karede tespit → açı → değer. Bulunamazsa None döner."""
    sonuc = model.predict(kare, conf=conf_esik, verbose=False)[0]
    if len(sonuc.boxes) == 0:
        return None

    # En güvenli kutu: karede birden çok gösterge olabilir, demo tek gösterge okur.
    en_iyi = int(sonuc.boxes.conf.argmax())
    kutu = sonuc.boxes.xyxy[en_iyi].tolist()
    tespit_guveni = float(sonuc.boxes.conf[en_iyi])

    merkez, yaricap = kutudan_kadran(kutu)
    if yaricap < MIN_YARICAP_PX:
        return {"kutu": kutu, "tespit_guveni": tespit_guveni, "okuma": None,
                "sebep": f"kadran çok küçük ({yaricap:.0f} px)"}

    aci = read_needle_angle(kare, merkez, yaricap, method="polar")
    if aci is None:
        return {"kutu": kutu, "tespit_guveni": tespit_guveni, "okuma": None,
                "sebep": "ibre bulunamadı"}

    # Tespit güveni ile açı güveni çarpılıyor: zincirin güveni en zayıf halkasından
    # yüksek olamaz. İP15'in eşiği bu birleşik sayıya uygulanacak.
    okuma = read_value(gauge, aci.angle_img_deg, roll_deg=0.0,
                       confidence=aci.confidence * tespit_guveni)
    return {"kutu": kutu, "tespit_guveni": tespit_guveni, "aci": aci, "okuma": okuma,
            "merkez": merkez, "yaricap": yaricap}


def kareyi_ciz(kare, sonuc, gauge) -> None:
    """Sonucu kareye yazar. Okunamadıysa değer YAZILMAZ (3. kural)."""
    yuksek = 30

    def yaz(metin, renk=RENK_BILGI, buyuk=False):
        nonlocal yuksek
        cv2.putText(kare, metin, (12, yuksek), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 if buyuk else 0.55, renk, 2 if buyuk else 1, cv2.LINE_AA)
        yuksek += 34 if buyuk else 24

    if sonuc is None:
        yaz("gosterge bulunamadi", RENK_UYARI)
        return

    x1, y1, x2, y2 = (int(v) for v in sonuc["kutu"])
    okuma = sonuc.get("okuma")
    kutu_rengi = RENK_OK if okuma and okuma.status == DURUM_OK else RENK_UYARI
    cv2.rectangle(kare, (x1, y1), (x2, y2), kutu_rengi, 2)

    if okuma is None:
        yaz(f"okunamadi: {sonuc['sebep']}", RENK_UYARI)
        return

    # Ölçülen ibreyi çiz — sayının nereden geldiği gözle denetlenebilsin.
    cv2.line(kare, sonuc["merkez"], sonuc["aci"].tip_px, RENK_UYARI, 2, cv2.LINE_AA)
    cv2.circle(kare, sonuc["merkez"], 4, RENK_UYARI, -1, cv2.LINE_AA)

    if okuma.value is None:
        yaz(f"{gauge.id}: DEGER YOK", RENK_UYARI, buyuk=True)
        yaz(f"status: {okuma.status}   conf: {okuma.conf:.2f}", RENK_UYARI)
    else:
        yaz(f"{gauge.id}: {okuma.value:g} {gauge.unit}", kutu_rengi, buyuk=True)
        yaz(f"status: {okuma.status}   conf: {okuma.conf:.2f}"
            f"   ham aci: {okuma.raw_angle:+.1f} deg", RENK_BILGI)

    yaz(f"tespit {sonuc['tespit_guveni']:.2f} · aci {sonuc['aci'].confidence:.2f}"
        f" · kadran capi {2*sonuc['yaricap']:.0f} px", RENK_BILGI)
    # Sınırlar ekranda: demoyu izleyen neyin varsayım olduğunu bilsin.
    yaz("gosterge kimligi ELLE verildi · yatiklik duzeltmesi YOK", RENK_BILGI)


def dosyadan(yol: Path, model, gauge, conf_esik: float, kaydet: bool) -> int:
    kare = cv2.imread(str(yol))
    if kare is None:
        print(f"görüntü okunamadı: {yol}")
        return 1

    sonuc = kareyi_oku(kare, model, gauge, conf_esik=conf_esik)
    kareyi_ciz(kare, sonuc, gauge)

    okuma = sonuc.get("okuma") if sonuc else None
    if okuma and okuma.value is not None:
        print(f"{yol.name}: {okuma.value:g} {gauge.unit}  "
              f"[{okuma.status}] conf={okuma.conf:.2f} ham_aci={okuma.raw_angle:+.1f}")
    else:
        print(f"{yol.name}: okuma üretilemedi")

    if kaydet:
        cikti = Path(CIKTI_DIZINI) / f"canli_{yol.stem}.png"
        cikti.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cikti), kare)
        print(f"kaydedildi: {cikti}")
    else:
        cv2.imshow("Gosterge okuma (kapatmak icin q)", kare)
        while cv2.waitKey(50) & 0xFF != ord("q"):
            if cv2.getWindowProperty("Gosterge okuma (kapatmak icin q)",
                                     cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
    return 0


def kameradan(kaynak: int, model, gauge, conf_esik: float, kaydet: bool) -> int:
    cap = cv2.VideoCapture(kaynak)
    if not cap.isOpened():
        print(f"kamera açılamadı: {kaynak}")
        return 1

    pencere = "Gosterge okuma (kapatmak icin q)"
    print("kamera açık — kapatmak için görüntü penceresindeyken 'q'")
    sayac = 0
    try:
        while True:
            ok, kare = cap.read()
            if not ok:
                print("kare okunamadı")
                break

            t0 = time.perf_counter()
            sonuc = kareyi_oku(kare, model, gauge, conf_esik=conf_esik)
            gecen = (time.perf_counter() - t0) * 1000
            kareyi_ciz(kare, sonuc, gauge)
            cv2.putText(kare, f"{1000/max(gecen,1e-6):.1f} FPS ({gecen:.0f} ms)",
                        (12, kare.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, RENK_BILGI, 1, cv2.LINE_AA)

            cv2.imshow(pencere, kare)
            tus = cv2.waitKey(1) & 0xFF
            if tus == ord("q"):
                break
            if kaydet and tus == ord("s"):
                sayac += 1
                cikti = Path(CIKTI_DIZINI) / f"canli_kare_{sayac:03d}.png"
                cikti.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cikti), kare)
                print(f"kaydedildi: {cikti}")
    except KeyboardInterrupt:
        print("\ndurduruldu")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Uçtan uca gösterge okuma (İP5+İP6+İP7)")
    p.add_argument("--kaynak", default="0", help="kamera indeksi (0) veya görüntü dosyası")
    p.add_argument("--gosterge", required=True, help="envanterdeki gauge_id (örn. PT-101)")
    p.add_argument("--agirlik", default=VARSAYILAN_AGIRLIK)
    p.add_argument("--conf", type=float, default=0.25, help="tespit güven eşiği")
    p.add_argument("--kaydet", action="store_true",
                   help="dosya modunda kaydet; kamera modunda 's' ile kare yakala")
    args = p.parse_args(argv)

    gauges = load_gauges()
    if args.gosterge not in gauges:
        print(f"envanterde yok: {args.gosterge} — mevcutlar: {list(gauges)}")
        return 1
    gauge = gauges[args.gosterge]
    if gauge.type != "analog":
        print(f"{gauge.id} analog değil ({gauge.type}) — bu script analog kadran okur")
        return 1

    agirlik = Path(args.agirlik)
    if not agirlik.exists():
        print(f"ağırlık yok: {agirlik}\nönce: python scripts/egit_ip5.py")
        return 1

    from ultralytics import YOLO
    model = YOLO(str(agirlik))
    print(f"gösterge: {gauge.id} ({gauge.name}) · kadran {gauge.scale.min:g}-"
          f"{gauge.scale.max:g} {gauge.unit} · ağırlık: {agirlik}")

    if args.kaynak.isdigit():
        return kameradan(int(args.kaynak), model, gauge, args.conf, args.kaydet)
    return dosyadan(Path(args.kaynak), model, gauge, args.conf, args.kaydet)


if __name__ == "__main__":
    raise SystemExit(main())
