r"""Kadranı ekranda tam ekran gösterir; telefonla fotoğraflanacak (İP8 · A yolu).

    python scripts\ekran_kadran.py --gosterge PT-101 --adet 12

Tuşlar:  BOŞLUK / → sonraki · ← önceki · q çıkış

**Neden bu düzenek.** İP8 "gerçek gösterge fotoğraflarında uçtan uca hata
tablosu" istiyor. Etiketli açık veri kalmadı (İP1: A1/A2 erişilemez, A5
etiketsiz) ve gerçek manometre alıp elle etiketlemek 2-3 gün. Ekrandan çekimde
görüntü **gerçek mercekten, gerçek ışıktan, gerçek sensörden** geçiyor; değer
ise birebir biliniyor çünkü kareyi biz ürettik. Gerçek manometrenin yerini
tutmaz — cam yansıması, metal doku ve tozlanma yok — ama sentetik ile gerçek
arasındaki basamaktır ve elle etiketleme gerektirmez. (docs/SORULAR.md · S1)

**Neden zamanlayıcı değil elle ilerletme.** Fotoğrafçının kareyi kurması,
odaklaması ve açıyı seçmesi gerekiyor; sabit süreli geçiş bulanık kareler
üretir. Bulanıklığın etkisi zaten İP14'te ayrı bir eksen olarak ölçüldü, burada
ölçülmek istenen o değil.

**Neden ekranda büyük bir sıra numarası var.** Eşleştirme çekim sırasına göre
yapılıyor, ama sıraya **güvenilmiyor**: bir kare atlanırsa ya da iki kez
çekilirse tüm eşleşme bir kayar ve hata tablosu sessizce anlamsızlaşır —
üstelik sayılar makul görünmeye devam eder, çünkü komşu kareler birbirine
yakın değerlerdir. Numara, o kaymayı **gözle** yakalanabilir kılıyor:
`olc_ip8.py` bütün fotoğrafları atadığı değerle birlikte tek bir kontak
sayfasına diziyor, numaralar sırayla gitmiyorsa on saniyede görülüyor.

Numarayı fotoğraftan otomatik okumak denenmedi: fotoğraflanmış bir yazıyı
çözmek kendi başına bir OCR işi ve ölçüm aracının kendisi ölçtüğü şey kadar
hataya açık hâle gelirdi. Sayım denetimi (fotoğraf sayısı = kare sayısı) +
gözle doğrulama, aynı hatayı daha ucuza yakalıyor.

Değerler `manifest.json`'a yazılır; ölçüm onu okur.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.synth.dial import DialLook, render_analog

PENCERE = "IP8 - ekran kadrani"
# Sıra numarası şeridi: kadranın ALTINDA, kadrana değmeyen ayrı bir bant.
# Kadranın üstüne yazılsaydı okuma zincirine gürültü katardı — ölçtüğümüz şeyin
# içine ölçüm aracını karıştırmak olurdu.
SERIT_ORANI = 0.13


def degerler(gauge, adet: int) -> list[float]:
    """Skalayı uçtan uca tarayan değerler.

    Uç noktalar (min ve max) MUTLAKA var: kadranın iki ucu, açı→değer
    dönüşümünün işaret ve yön hatalarının en görünür olduğu yerdir. Aradakiler
    eşit aralıklı — rastgele seçim, hangi bölgenin ölçülmediğini gizler.
    """
    lo, hi = gauge.scale.min, gauge.scale.max
    if adet < 2:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (adet - 1) for i in range(adet)]


def kare_uret(gauge, deger: float, sira: int, toplam: int,
              boyut: int) -> tuple[np.ndarray, dict]:
    """Bir gösterim karesi + o karenin ground truth kaydı."""
    kadran, truth = render_analog(gauge, deger, size=boyut, look=DialLook())

    serit_h = int(boyut * SERIT_ORANI)
    tuval = np.full((boyut + serit_h, boyut, 3), 255, np.uint8)
    tuval[:boyut] = kadran

    etiket = f"#{sira:02d}"
    olcek = serit_h / 40.0
    (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, olcek, 3)
    cv2.putText(tuval, etiket, ((boyut - tw) // 2, boyut + (serit_h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, olcek, (0, 0, 0), 3, cv2.LINE_AA)

    kayit = {
        "sira": sira,
        "gauge_id": gauge.id,
        "value": round(float(deger), 4),
        "angle_deg": round(float(truth.angle_deg), 4),
        "toplam": toplam,
    }
    return tuval, kayit


def main() -> int:
    ap = argparse.ArgumentParser(description="İP8 ekran kadranı gösterimi")
    ap.add_argument("--gosterge", default="PT-101")
    ap.add_argument("--adet", type=int, default=12)
    ap.add_argument("--boyut", type=int, default=900, help="kadran kenarı (piksel)")
    ap.add_argument("--manifest", type=Path,
                    default=Path("outputs/metrics/ip8_ekran_manifest.json"))
    ap.add_argument("--tam-ekran", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    gauge = load_gauges()[args.gosterge]
    if gauge.type != "analog":
        print(f"HATA: {gauge.id} analog değil (tip: {gauge.type}) — "
              f"ekran kadranı yalnızca analog gösterge için")
        return 1

    hedefler = degerler(gauge, args.adet)
    kareler, kayitlar = [], []
    for i, d in enumerate(hedefler, start=1):
        kare, kayit = kare_uret(gauge, d, i, len(hedefler), args.boyut)
        kareler.append(kare)
        kayitlar.append(kayit)

    manifest = {
        "olusturuldu": datetime.now().isoformat(timespec="seconds"),
        "gauge_id": gauge.id,
        "unit": gauge.unit,
        "boyut_px": args.boyut,
        "kareler": kayitlar,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    print(f"{gauge.id} · {len(kareler)} kare · manifest: {args.manifest}")
    print("Tuşlar: BOŞLUK/→ sonraki · ← önceki · q çıkış")
    print("Her kareyi telefonla çekin; ekrandaki #NN numarası kareye GİRSİN.\n")
    for k in kayitlar:
        print(f"  #{k['sira']:02d}  {k['value']:g} {gauge.unit}")

    cv2.namedWindow(PENCERE, cv2.WINDOW_NORMAL)
    if args.tam_ekran:
        cv2.setWindowProperty(PENCERE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    i = 0
    while True:
        cv2.imshow(PENCERE, kareler[i])
        tus = cv2.waitKey(0) & 0xFF
        if tus in (ord("q"), 27):
            break
        if tus in (ord(" "), 83, ord("d")):        # 83 = sağ ok
            i = min(i + 1, len(kareler) - 1)
        elif tus in (81, ord("a")):                # 81 = sol ok
            i = max(i - 1, 0)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
