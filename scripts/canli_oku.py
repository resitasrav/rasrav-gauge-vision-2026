"""Zinciri uçtan uca çalıştırır: görüntü → tespit → kırp → açı → değer (İP5+İP6+İP7).

    python scripts/canli_oku.py --kaynak 0 --gosterge PT-101          # kamera
    python scripts/canli_oku.py --kaynak data/synthetic/v0/images/0004_PT-101.png --gosterge PT-101
    python scripts/canli_oku.py --kaynak 0 --gosterge PT-101 --kaydet

Bu script yeni bir yöntem getirmez; üç iş paketinin çıktısını birbirine bağlar.
Ölçüm scriptleri (`olc_ip6.py`, `olc_ip7.py`) kırpımı ve merkezi **etiketten** alır;
burada ikisi de **tespitten** gelir. Aradaki fark, zincirin gerçek hatasıdır.

**Bilinçli sınır — demoda ekrana da yazılır, izleyen yanılmasın:**

**Hangi gösterge olduğu elle veriliyor** (`--gosterge`). Tespit "burada bir gösterge
var" der, "bu PT-101'dir" demez. Gerçek sistemde bunu robotun durağı (waypoint)
söyleyecektir (U11). Yanlış gösterge seçilirse değer sessizce yanlış çıkar; tek
otomatik işaret yatıklık kestiriminin susmasıdır (`read/roll.py` uyum kapısı).

06.08'de kapanan iki sınır — ekrandaki `merkez` ve `yatiklik` alanları bunları
gösterir, ikisi de artık GÖRÜNTÜDEN geliyor ve etiket kullanılmıyor:
  · merkez kutudan değil kadran çemberinden (`detect/refine.py`)
  · yatıklık 0 kabul edilmiyor, kadranın çizgilerinden kestiriliyor (`read/roll.py`)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from gauge_vision.config import load_gauges
from gauge_vision.pipeline import read_frame
from gauge_vision.read.calibrate import DURUM_OK

VARSAYILAN_AGIRLIK = "runs/detect/models/ip5/karisik/weights/best.pt"
CIKTI_DIZINI = "outputs/figures"

RENK_OK = (60, 200, 60)
RENK_UYARI = (40, 40, 220)
RENK_BILGI = (40, 40, 40)


def kareyi_ciz(kare, sonuc, gauge) -> None:
    """`FrameResult`'ı kareye yazar. Okunamadıysa değer YAZILMAZ (3. kural).

    Yazılar kadranın üstüne binmesin diye üstten değil ALTTAN yukarı doğru
    diziliyor: kadran genelde karenin ortasında durur ve sayının kaynağını
    gözle denetlemek için ibrenin görünür kalması gerekir.
    """
    satirlar: list[tuple[str, tuple[int, int, int], bool]] = []
    okuma = sonuc.reading

    if sonuc.box_xyxy is not None:
        x1, y1, x2, y2 = (int(v) for v in sonuc.box_xyxy)
        kutu_rengi = RENK_OK if okuma and okuma.status == DURUM_OK else RENK_UYARI
        cv2.rectangle(kare, (x1, y1), (x2, y2), kutu_rengi, 2)
    else:
        kutu_rengi = RENK_UYARI

    if okuma is None:
        satirlar.append((f"okunamadi: {sonuc.reason}", RENK_UYARI, True))
    else:
        # Ölçülen ibreyi çiz — sayının nereden geldiği gözle denetlenebilsin.
        cv2.line(kare, sonuc.center_px, sonuc.needle.tip_px, RENK_UYARI, 2, cv2.LINE_AA)
        cv2.circle(kare, sonuc.center_px, 4, RENK_UYARI, -1, cv2.LINE_AA)

        if okuma.value is None:
            satirlar.append((f"{gauge.id}: DEGER YOK", RENK_UYARI, True))
            satirlar.append((f"status: {okuma.status}  conf: {okuma.conf:.2f}", RENK_UYARI, False))
        else:
            satirlar.append((f"{gauge.id}: {okuma.value:g} {gauge.unit}", kutu_rengi, True))
            satirlar.append((f"status: {okuma.status}  conf: {okuma.conf:.2f}"
                             f"  ham aci: {okuma.raw_angle:+.1f} deg", RENK_BILGI, False))
        satirlar.append((f"tespit {sonuc.detect_conf:.2f} · aci {sonuc.needle.confidence:.2f}"
                         f" · kadran capi {2*sonuc.radius_px:.0f} px", RENK_BILGI, False))

        # Merkezin ve yatıklığın NEREDEN geldiği ekranda: ikisi de zincirin
        # doğruluğunu belirliyor ve ikisi de sessizce başarısız olabilir.
        merkez_kaynak = "cemberden" if sonuc.center_refined else "KUTUDAN (rafine yok)"
        yatiklik = (f"{sonuc.roll_deg:+.1f} deg (uyum {sonuc.roll.match:.2f})"
                    if sonuc.roll else "KESTIRILEMEDI, 0 kabul edildi")
        satirlar.append((f"merkez {merkez_kaynak} · yatiklik {yatiklik}", RENK_BILGI, False))

    # Kalan sınır ekranda: demoyu izleyen neyin varsayım olduğunu bilsin.
    satirlar.append(("gosterge kimligi ELLE verildi", RENK_BILGI, False))

    y = 30
    for metin, renk, buyuk in satirlar:
        cv2.putText(kare, metin, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 if buyuk else 0.55, renk, 2 if buyuk else 1, cv2.LINE_AA)
        y += 34 if buyuk else 24


def dosyadan(yol: Path, model, gauge, conf_esik: float, kaydet: bool) -> int:
    kare = cv2.imread(str(yol))
    if kare is None:
        print(f"görüntü okunamadı: {yol}")
        return 1

    sonuc = read_frame(kare, model, gauge, detect_conf=conf_esik)
    kareyi_ciz(kare, sonuc, gauge)

    if sonuc.ok:
        o = sonuc.reading
        print(f"{yol.name}: {o.value:g} {gauge.unit}  "
              f"[{o.status}] conf={o.conf:.2f} ham_aci={o.raw_angle:+.1f}")
    else:
        print(f"{yol.name}: okuma üretilemedi — {sonuc.reason or 'düşük güven'}")

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
            sonuc = read_frame(kare, model, gauge, detect_conf=conf_esik)
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
