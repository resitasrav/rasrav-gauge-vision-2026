r"""Gömülü hedefe taşınabilirlik: okuma katmanının SAF CPU maliyeti.

    python scripts\olc_gomulu.py
    python scripts\olc_gomulu.py --tekrar 100 --hedef-carpan 4.0

**Neden bu ölçüm var.** Sistem sahada bir Orange Pi 5 (RK3588) + NPU
hızlandırıcılı kartta koşacak. Bugüne kadarki tek hız sayısı **95,6 ms/kare**
idi (1080p, RTX 4050: YOLO 28,6 ms + okuma 67,0 ms) ve o sayı gömülü hedef
hakkında hiçbir şey söylemiyor: orada tespit NPU'ya gider, okuma ise
**CPU'da kalır.**

Bu script okuma katmanını GPU'suz ölçer ve hedef karta izdüşürür. Ölçülen şey
doğruluk değil **maliyettir**; doğruluk sayıları kendi ölçüm scriptlerinde.

**Mimari not — taşınabilirlik zaten var.** `src/gauge_vision/` içinde tek bir
`torch` / `ultralytics` / `cuda` importu yoktur; kütüphane saf NumPy + OpenCV +
PyYAML'dır. Tespit modeli zincire **dışarıdan** verilir ve yalnız şu arayüzü
ister:

    sonuc = model.predict(image, conf=..., verbose=False)[0]
    sonuc.boxes.xyxy / .conf / .cls / len()
    sonuc.names                      # {sınıf_id: ad}

Bu yüzden ONNX/RKNN/HEF ile derlenmiş bir modeli sarmalamak, bu arayüzü taklit
eden küçük bir sınıf yazmaktan ibarettir — `scripts/canli_oku.py::_TespitYok`
o sarmalayıcının çalışan bir örneğidir.

**⚠ İZDÜŞÜM TAHMİNDİR, ÖLÇÜM DEĞİL.** `--hedef-carpan` bu makinenin CPU'su ile
hedef kartın tek çekirdek farkını temsil eder. Gerçek sayı ancak kartın
üstünde koşturularak alınır; bu script oraya kopyalanıp aynen çalıştırılabilir
(bağımlılığı yalnız numpy + opencv).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.detect.refine import refine_dial
from gauge_vision.pipeline import dial_from_box
from gauge_vision.read.digital import read_digital
from gauge_vision.read.keypad import read_keypad
from gauge_vision.read.needle import read_needle_angle
from gauge_vision.read.roll import estimate_roll
from gauge_vision.read.state import read_state
from gauge_vision.synth.dial import render_analog
from gauge_vision.synth.digital import render_digital
from gauge_vision.synth.keypad import render_keypad
from gauge_vision.synth.state import render_lamp, render_valve

METRIK_YOLU = Path("outputs/metrics/gomulu_maliyet.json")

# Gerçek zincirde okuyucuya giden şey tam kare değil, tespit kutusunun
# kırpımıdır. Kırpım boyutu göstergenin karedeki büyüklüğüne bağlı; 320 px
# devriye karesinde makul bir kadran boyudur (İP8 fotoğraflarında 200-400 px).
KESIT_PX = 320
# Kamera karesi. Analog aşamalar bu boyuttaki kareye veriliyor — zincirin
# gerçekte yaptığı bu. 27.08'de `read_needle_angle`'a ROI kırpması eklendikten
# sonra maliyet kare boyutundan büyük ölçüde bağımsız hâle geldi (1080p'de
# 18,13 → 3,25 ms), ama `estimate_roll` hâlâ tam kareye bakıyor.
KARE_W, KARE_H = 1920, 1080


def _sure(fn, tekrar: int) -> float:
    """`fn`'i `tekrar` kez koşup kare başına MEDYAN ms döner.

    Medyan, ortalama değil: tek bir işletim sistemi kesintisi ortalamayı
    savurur ve gömülü hedefte bütçe planlarken yanıltır.
    """
    fn()  # ısınma — ilk çağrıda OpenCV kendi tablolarını kuruyor
    olculer = []
    for _ in range(tekrar):
        t0 = time.perf_counter()
        fn()
        olculer.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(olculer))


def olc(tekrar: int) -> dict:
    gauges = load_gauges()
    analog = next(g for g in gauges.values() if g.type == "analog")
    dijital = next(g for g in gauges.values() if g.type == "digital")
    lamba = next(g for g in gauges.values() if g.type == "lamp")
    vana = next(g for g in gauges.values() if g.type == "valve")
    panel = next((g for g in gauges.values() if g.type == "keypad"), None)

    # Analog aşamalar TAM KAREYE veriliyor — zincir de öyle yapıyor
    # (`read_frame` kırpmadan geçirir; kırpmayı her okuyucu kendi içinde
    # yapar). 320'lik bir kesitle ölçmek maliyeti olduğundan küçük gösterirdi.
    kadran, _ = render_analog(analog, (analog.scale.min + analog.scale.max) / 2,
                              size=KESIT_PX)
    kare = np.full((KARE_H, KARE_W, 3), 95, np.uint8)
    kx, ky = (KARE_W - KESIT_PX) // 2, (KARE_H - KESIT_PX) // 2
    kare[ky:ky + KESIT_PX, kx:kx + KESIT_PX] = kadran
    merkez, yaricap = dial_from_box((kx, ky, kx + KESIT_PX, ky + KESIT_PX))
    dij, _ = render_digital(dijital, 123.4)
    dij = cv2.resize(dij, (KESIT_PX, int(KESIT_PX * dij.shape[0] / dij.shape[1])))
    lmp, _ = render_lamp(lamba, "green", size=KESIT_PX)
    vlv, _ = render_valve(vana, "open", size=KESIT_PX)

    adimlar: dict[str, float] = {
        "merkez rafinesi (refine_dial)": _sure(
            lambda: refine_dial(kare, merkez, yaricap), tekrar),
        "yatıklık kestirimi (estimate_roll)": _sure(
            lambda: estimate_roll(kare, merkez, yaricap, analog), tekrar),
        "ibre açısı (read_needle_angle)": _sure(
            lambda: read_needle_angle(kare, merkez, yaricap, method="polar"), tekrar),
        "dijital panel (read_digital)": _sure(
            lambda: read_digital(dij, dijital), tekrar),
        "lamba (read_state)": _sure(lambda: read_state(lmp, lamba), tekrar),
        "vana (read_state)": _sure(lambda: read_state(vlv, vana), tekrar),
    }
    if panel is not None:
        bilesim = {b["id"]: (b.get("states") or ["off"])[0] for b in panel.buttons}
        pnl, _ = render_keypad(panel, bilesim)
        adimlar["buton paneli (read_keypad)"] = _sure(
            lambda: read_keypad(pnl, panel), tekrar)

    # Analog zincir = rafine + yatıklık + ibre (tespit HARİÇ — o NPU'ya gider).
    analog_toplam = sum(adimlar[k] for k in adimlar if k.startswith(
        ("merkez", "yatıklık", "ibre")))
    return {"adimlar": adimlar, "analog_okuma_toplam": analog_toplam}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Okuma katmanının CPU maliyeti (gömülü hedef)")
    p.add_argument("--tekrar", type=int, default=50)
    p.add_argument("--hedef-carpan", type=float, default=4.0,
                   help="hedef kartın bu makineye göre tek çekirdek yavaşlık katsayısı")
    p.add_argument("--hedef-ad", default="Orange Pi 5 (RK3588, Cortex-A76)")
    # Aşağıdaki üçü VARSAYIMDIR ve bilerek dışarıdan verilir: kart elde
    # olmadan ölçülemezler, ama bütçenin nasıl dağıldığını görmek için
    # sayıya ihtiyaç var. Kart gelince gerçek değerlerle koşturulur.
    p.add_argument("--on-son-ms", type=float, default=1.7,
                   help="tespitin CPU'da kalan kısmı (letterbox+NMS), bu makinede ölçüldü")
    p.add_argument("--npu-ms", type=float, default=6.0,
                   help="NPU'da çıkarım süresi (Hailo-8 ~3-6 ms, RK3588 NPU ~8-15 ms)")
    p.add_argument("--yakalama-ms", type=float, default=8.0,
                   help="kamera karesinin yakalanma + kod çözme maliyeti")
    p.add_argument("--gosterge-sayisi", type=int, default=1,
                   help="karede kaç gösterge okunuyor (okuma maliyeti tekrarlanır)")
    args = p.parse_args(argv)

    # OpenCV'nin kendi iş parçacıkları ölçümü kirletir: gömülü hedefte de
    # devriye döngüsü tek çekirdek varsayımıyla planlanmalı.
    cv2.setNumThreads(1)

    print(f"makine: {platform.processor() or platform.machine()} · "
          f"OpenCV {cv2.__version__} · tek iş parçacığı")
    print(f"kare: {KARE_W}×{KARE_H} px · kadran: {KESIT_PX} px · "
          f"tekrar: {args.tekrar}\n")

    sonuc = olc(args.tekrar)
    adimlar = sonuc["adimlar"]

    print(f"{'adım':>36s} {'bu makine':>11s} {'hedef (×' + str(args.hedef_carpan) + ')':>14s}")
    for ad, ms in adimlar.items():
        print(f"{ad:>36s} {ms:>9.2f}ms {ms * args.hedef_carpan:>12.1f}ms")

    at = sonuc["analog_okuma_toplam"]
    print(f"\n{'ANALOG OKUMA TOPLAMI (tespit hariç)':>36s} "
          f"{at:>9.2f}ms {at * args.hedef_carpan:>12.1f}ms")

    # --- Kare bütçesi ---
    # NPU yalnız ÇIKARIMI devralır. Kalan üç kalem CPU'da kalır ve kare hızını
    # asıl onlar belirler:
    #   1. yakalama/kod çözme  — kamera akışı (MJPEG/H.264) çözülmesi
    #   2. ön+son işleme       — letterbox ve NMS
    #   3. okuma               — bu scriptin ölçtüğü kısım
    cpu_ms = args.on_son_ms + at
    hedef_cpu = cpu_ms * args.hedef_carpan
    toplam = hedef_cpu + args.npu_ms + args.yakalama_ms
    print(f"\n--- HEDEF KARTTA KARE BÜTÇESİ (gösterge başına 1 okuma) ---")
    print(f"{'yakalama/kod çözme (varsayım)':>36s} {args.yakalama_ms:>9.1f}ms")
    print(f"{'tespit ön+son işleme (CPU)':>36s} "
          f"{args.on_son_ms * args.hedef_carpan:>9.1f}ms")
    print(f"{'tespit çıkarımı (NPU, varsayım)':>36s} {args.npu_ms:>9.1f}ms")
    print(f"{'okuma (CPU)':>36s} {at * args.hedef_carpan:>9.1f}ms")
    print(f"{'TOPLAM':>36s} {toplam:>9.1f}ms  →  **{1000 / toplam:.0f} kare/s**")

    if args.gosterge_sayisi > 1:
        cok = hedef_cpu + args.npu_ms + args.yakalama_ms + \
            at * args.hedef_carpan * (args.gosterge_sayisi - 1)
        print(f"\n{args.gosterge_sayisi} gösterge birden okunursa: "
              f"{cok:.0f} ms → {1000 / cok:.0f} kare/s "
              f"(okuma maliyeti gösterge başına tekrarlanır)")

    print(f"\n⚠ Devriye döngüsü saniyede **1 okuma** ister — gösterge saniyede 25 kez "
          f"değişmiyor.\n   Kare hızını 30'a çıkarmak yanlış eksende iyileştirmedir; "
          f"artan bütçe\n   daha yüksek çözünürlüğe ya da zamansal ortalamaya "
          f"harcanmalıdır (180° sorunu).")

    rapor = {
        "tarih": date.today().isoformat(),
        "makine": platform.processor() or platform.machine(),
        "opencv": cv2.__version__,
        "kare_px": [KARE_W, KARE_H],
        "kadran_px": KESIT_PX,
        "tekrar": args.tekrar,
        "tek_is_parcacigi": True,
        "adimlar_ms": {k: round(v, 3) for k, v in adimlar.items()},
        "analog_okuma_toplam_ms": round(at, 3),
        "izdusum": {
            "hedef": args.hedef_ad,
            "carpan": args.hedef_carpan,
            "analog_okuma_ms": round(at * args.hedef_carpan, 1),
            "varsayimlar_ms": {"on_son": args.on_son_ms, "npu": args.npu_ms,
                               "yakalama": args.yakalama_ms},
            "kare_toplam_ms": round(toplam, 1),
            "kare_hizi": round(1000 / toplam, 1),
            "not": "TAHMİN — gerçek sayı kartın üstünde koşturularak alınmalı; "
                   "bu script bağımlılığı yalnız numpy+opencv olduğu için "
                   "karta olduğu gibi kopyalanabilir",
        },
    }
    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    METRIK_YOLU.write_text(json.dumps(rapor, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nölçüm: {METRIK_YOLU}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
