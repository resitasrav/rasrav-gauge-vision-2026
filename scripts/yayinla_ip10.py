"""Okuma zincirini `inspect/reading` konusuna bağlar — İP10.

    python scripts/yayinla_ip10.py                       # sentetik turu yayınla
    python scripts/yayinla_ip10.py --broker localhost    # gerçek brokera
    python scripts/yayinla_ip10.py --dogrula outputs/mqtt/xxx.jsonl

Bir **devriye turu** simüle eder: envanterdeki her göstergeyi sırayla okur ve
mesajı yayınlar. Analog, dijital, lamba ve vana — dört tipin dördü de aynı
konudan, aynı gövdeyle akar.

**Broker gerekmez.** `paho-mqtt` yoksa ya da broker ayakta değilse mesajlar
JSONL olarak `outputs/mqtt/` altına yazılır ve `--dogrula` ile denetlenir.
Ekibin kayıt aracı `inspect/reading`'i henüz kaydetmiyor (U5); yayının
doğruluğu bu bağımlılık çözülmeden de gösterilebilmeli.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.publish.reading import (
    ReadingPublisher,
    SemaHatasi,
    mesaj_dogrula,
)
from gauge_vision.read.calibrate import read_value
from gauge_vision.read.digital import read_digital
from gauge_vision.read.state import read_state
from gauge_vision.synth.digital import render_digital
from gauge_vision.synth.state import render_lamp, render_valve

TOHUM = 10


def tur_oku(gauges: dict, rng) -> list:
    """Envanterdeki her göstergeyi bir kez okur — bir devriye turu.

    Gerçek turda görüntü kameradan gelir; burada sentetik olarak üretiliyor.
    Ölçülen şey görüntü kalitesi değil, **yayın hattının dört gösterge tipini
    de aynı sözleşmeyle taşıyıp taşımadığıdır.**
    """
    okumalar = []
    for gauge in gauges.values():
        if gauge.type == "analog":
            # Analog tarafta zincir zaten ölçüldü (İP8 provası); burada açıyı
            # doğrudan veriyoruz — konu yayın, tespit değil.
            deger = float(rng.uniform(gauge.scale.min, gauge.scale.max))
            aci = gauge.scale.angle_for_value(deger)
            okuma = read_value(gauge, aci, confidence=float(rng.uniform(0.5, 0.99)))
        elif gauge.type == "digital":
            a = gauge.raw.get("range") or {}
            deger = float(rng.uniform(float(a.get("min", 0)), float(a.get("max", 100))))
            img, _ = render_digital(gauge, deger)
            okuma = read_digital(img, gauge)
        elif gauge.type == "lamp":
            durum = gauge.state_names[int(rng.integers(0, len(gauge.state_names)))]
            img, _ = render_lamp(gauge, durum)
            okuma = read_state(img, gauge)
        elif gauge.type == "valve":
            durum = gauge.state_names[int(rng.integers(0, len(gauge.state_names)))]
            img, _ = render_valve(gauge, durum, sapma_deg=float(rng.uniform(-10, 10)))
            okuma = read_state(img, gauge)
        else:
            continue
        okumalar.append(okuma)
    return okumalar


def dogrula_dosya(yol: Path) -> int:
    """Kaydedilmiş JSONL'i şemaya karşı denetler."""
    satirlar = [s for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]
    hatali = 0
    for i, s in enumerate(satirlar, 1):
        try:
            mesaj_dogrula(json.loads(s))
        except (SemaHatasi, json.JSONDecodeError) as e:
            hatali += 1
            print(f"   satır {i}: {e}")
    print(f"{len(satirlar)} mesaj · {len(satirlar) - hatali} geçerli · {hatali} hatalı")
    return 0 if hatali == 0 else 1


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="inspect/reading yayını (İP10)")
    p.add_argument("--broker", default=None, help="MQTT broker adresi (yoksa dosya)")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--tur", type=int, default=3, help="kaç devriye turu")
    p.add_argument("--dogrula", default=None, help="JSONL dosyasını denetle ve çık")
    args = p.parse_args(argv)

    if args.dogrula:
        return dogrula_dosya(Path(args.dogrula))

    gauges = load_gauges()
    rng = np.random.default_rng(TOHUM)

    yayinci = ReadingPublisher(host=args.broker or "localhost", port=args.port,
                               source="gauge-vision",
                               zorla_dosya=args.broker is None)
    mod = yayinci.baglan()
    print(f"yayın modu: {mod}" + (f" · {args.broker}:{args.port}" if mod == "mqtt"
                                  else f" · {yayinci._dosya}"))
    print(f"konu: {yayinci.topic}\n")

    sayim: dict[str, int] = {}
    for tur in range(args.tur):
        for okuma in tur_oku(gauges, rng):
            try:
                m = yayinci.yayinla(okuma, img_ref=f"frames/tur{tur:02d}.jpg")
            except SemaHatasi as e:
                print(f"   ✗ {okuma.gauge_id}: şema hatası — {e}")
                continue
            sayim[m["status"]] = sayim.get(m["status"], 0) + 1
            deger = m["value"] if m["value"] is not None else "—"
            birim = f" {m['unit']}" if m["unit"] and m["value"] is not None else ""
            print(f"   {m['gauge_id']:8s} {m['type']:8s} {str(deger):>8s}{birim:6s} "
                  f"[{m['status']:12s}] conf {m['conf']:.2f}")
        print()

    yayinci.kapat()
    print(f"gönderilen {yayinci.gonderilen} · reddedilen {yayinci.reddedilen}")
    print("durum dağılımı: " + " · ".join(f"{k} {v}" for k, v in sorted(sayim.items())))

    if mod == "dosya" and yayinci._dosya:
        print(f"\ndoğrulama: python scripts/yayinla_ip10.py --dogrula {yayinci._dosya}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
