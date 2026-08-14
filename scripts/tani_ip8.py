r"""Gerçek fotoğrafta zincirin HANGİ adımının çöktüğünü ölçer (İP8 teşhisi).

    python scripts\tani_ip8.py --fotograflar data\real\PT-101_ekran ^
        --agirlik runs\detect\models\ip5\karisik\weights\best.pt

**Neden ayrı bir araç.** `olc_ip8.py` "hata ne kadar" diye sorar; bu script
"hata NEREDE doğuyor" diye sorar. İlk gerçek çekimde 12 karenin 12'sinde hem
merkez rafinesi hem yatıklık kestirimi sessizce devre dışı kaldı ve zincir kaba
kutu merkeziyle, yatıklık 0 varsayarak okumaya devam etti. Dışarıdan görünen
tek şey "hata %6" idi; hangi adımın çöktüğü görünmüyordu.

**Kapıları geçici olarak açar.** Reddedilen bir adımın iç sayıları normalde
kaybolur (`None` döner). Burada eşikler sonsuza çekilip **reddedilme sebebinin
sayısı** okunuyor: rafinede artık/yayılma oranı, yatıklıkta uyum ve tepe oranı.
Eşik değiştirmek için değil, eşiğin nerede kaldığını görmek için.

**Bu script eşik ÖNERMEZ.** Sayılar gösterdi ki gerçek fotoğrafta yatıklık uyumu
sentetikteki GÜRÜLTÜ seviyesiyle aynı bölgede. Böyle bir durumda eşiği indirmek
kanıt üretmez, yalnızca gürültüyü kabul eder ve 3. kuralı çiğner. Eşik ancak
"gerçek dağılım" ile "gürültü dağılımı" ölçülüp ARALARI açık olduğunda taşınır.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

import gauge_vision.detect.refine as R
import gauge_vision.read.roll as RL
from gauge_vision.config import load_gauges
from gauge_vision.detect.perspective import duzlestir
from gauge_vision.pipeline import KUTU_YARICAP_ORANI, dial_from_box

UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp")
METRIK_YOLU = "outputs/metrics/ip8_tani.json"
# Sentetikte ölçülen referans dağılımlar (06.08 · roll.py başlığı):
SENTETIK_UYUM_MIN = 0.629     # gerçek kadranda en düşük uyum
SENTETIK_GURULTU_MAX = 0.176  # rastgele gürültüde en yüksek uyum


def _roll_ic_sayilari(image, merkez, yaricap, gauge) -> dict | None:
    """`estimate_roll`'un kapıya takılmadan önceki iç sayıları.

    Hesap `roll.roll_evidence`'tan geliyor — eskiden burada bir kopyası vardı
    ve roll.py'daki her değişiklikte sessizce bayatlayacaktı (13.08'de ikinci
    tepe araması tüm çembere genişleyince bayatladı da). Kopya silindi.
    """
    kanit = RL.roll_evidence(image, merkez, yaricap, gauge)
    if kanit is None:
        return None
    return {
        "uyum": round(kanit.match, 4),
        "ayriklik": round(kanit.separation, 4),
        "roll_deg": round(kanit.roll_deg, 2),
        "kuresel_tepe_deg": round(kanit.global_best_deg, 1),
        "kontrast": round(kanit.contrast, 1),
    }


def tani(yollar: list[Path], gauge, model, conf: float) -> list[dict]:
    # Kapılar KAPATILIYOR — reddedilme sebebinin sayısını görebilmek için.
    # Süreç boyunca modül düzeyinde değiştiriliyor; bu script tek seferlik bir
    # teşhis aracı, üretim yolunda çalışmıyor.
    R.MAX_KAYMA_ORANI = R.MAX_ARTIK_ORANI = R.MAX_YAYILMA_ORANI = 9e9
    R.MEDYAN_MIN, R.MEDYAN_MAX = 0.0, 9e9

    satirlar = []
    for p in yollar:
        img = cv2.imread(str(p))
        if img is None:
            satirlar.append({"dosya": p.name, "asama": "goruntu acilamadi"})
            continue

        sonuc = model.predict(img, conf=conf, verbose=False)[0]
        if len(sonuc.boxes) == 0:
            satirlar.append({"dosya": p.name, "asama": "tespit yok"})
            continue

        i = int(sonuc.boxes.conf.argmax())
        kutu = tuple(float(v) for v in sonuc.boxes.xyxy[i].tolist())
        satir = {"dosya": p.name, "tespit_guveni": round(float(sonuc.boxes.conf[i]), 3)}

        merkez, yaricap = dial_from_box(kutu)
        dz = duzlestir(img, merkez, yaricap)
        satir["duzlestirme"] = dz is not None
        if dz is not None:
            img = dz.image
            merkez = dz.center_px
            yaricap = dz.radius_px * KUTU_YARICAP_ORANI

        daire = R.refine_dial(img, merkez, yaricap)
        if daire is None:
            satir["rafine"] = "hesaplanamadi"
        else:
            satir.update({
                "rafine_kayma_orani": round(daire.shift_px / max(yaricap, 1e-6), 4),
                "rafine_artik_orani": round(daire.residual_ratio, 4),
                "rafine_yayilma_orani": round(daire.spread_ratio, 4),
                "rafine_destek": daire.support,
            })
            merkez, yaricap = daire.center_px, daire.radius_px
        satir["yaricap_px"] = round(float(yaricap), 1)

        ic = _roll_ic_sayilari(img, merkez, yaricap, gauge)
        satir["roll"] = ic if ic else "profil uretilemedi"
        satirlar.append(satir)
    return satirlar


def main() -> int:
    ap = argparse.ArgumentParser(description="İP8 zincir teşhisi")
    ap.add_argument("--fotograflar", required=True, type=Path)
    ap.add_argument("--agirlik", required=True, type=Path)
    ap.add_argument("--gosterge", default="PT-101")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cikti", type=Path, default=Path(METRIK_YOLU))
    args = ap.parse_args()

    from ultralytics import YOLO

    gauge = load_gauges()[args.gosterge]
    model = YOLO(str(args.agirlik))
    yollar = sorted((p for p in args.fotograflar.iterdir()
                     if p.suffix.lower() in UZANTILAR),
                    key=lambda p: (int(m[0]) if (m := re.findall(r"\d+", p.stem)) else 0))

    satirlar = tani(yollar, gauge, model, args.conf)

    print(f"{gauge.id} · {len(satirlar)} kare\n")
    print(f"Esikler — rafine artik <= {0.05} · yayilma <= {0.05} · "
          f"roll uyum >= {RL.MIN_UYUM}\n")
    print("| dosya | duzlestirme | rafine artik | rafine yayilma | roll uyum | roll |")
    print("|---|---|---|---|---|---|")
    for s in satirlar:
        r = s.get("roll") if isinstance(s.get("roll"), dict) else {}
        print(f"| {s['dosya']} | {s.get('duzlestirme', '—')} | "
              f"{s.get('rafine_artik_orani', '—')} | {s.get('rafine_yayilma_orani', '—')} | "
              f"{r.get('uyum', '—')} | {r.get('roll_deg', '—')} |")

    uyumlar = [s["roll"]["uyum"] for s in satirlar
               if isinstance(s.get("roll"), dict)]
    ozet = {
        "gauge_id": gauge.id,
        "kare": len(satirlar),
        "esikler": {"rafine_artik": 0.05, "rafine_yayilma": 0.05,
                    "roll_uyum": RL.MIN_UYUM},
        "sentetik_referans": {"gercek_kadran_uyum_min": SENTETIK_UYUM_MIN,
                              "gurultu_uyum_max": SENTETIK_GURULTU_MAX},
        "satirlar": satirlar,
    }
    if uyumlar:
        ozet["roll_uyum"] = {"min": round(min(uyumlar), 4),
                             "ortanca": round(float(np.median(uyumlar)), 4),
                             "max": round(max(uyumlar), 4)}
        print(f"\nroll uyumu — min {min(uyumlar):.3f} · ortanca "
              f"{np.median(uyumlar):.3f} · max {max(uyumlar):.3f}")
        print(f"Sentetikte: gercek kadran >= {SENTETIK_UYUM_MIN} · "
              f"gurultu <= {SENTETIK_GURULTU_MAX}")
        if max(uyumlar) < RL.MIN_UYUM:
            print("\nSONUC: hicbir kare esigi gecmiyor ve uyum sentetikteki "
                  "GURULTU bolgesinde. Esik indirilmemeli — kanit yok, "
                  "indirmek gurultuyu kabul etmek olur.")

    args.cikti.parent.mkdir(parents=True, exist_ok=True)
    args.cikti.write_text(json.dumps(ozet, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nTani: {args.cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
