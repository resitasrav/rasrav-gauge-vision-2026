r"""Ekrandan çekilmiş fotoğraflarda uçtan uca hata tablosu (İP8).

    python scripts\ekran_kadran.py --gosterge PT-101 --adet 12   # once goster+cek
    python scripts\olc_ip8.py --fotograflar data\real\PT-101_ekran

Zincirin **gerçek optik yoldan geçmiş** görüntüdeki hatasını ölçer. Sentetikte
%0,19 çıkan sayının gerçek mercek, gerçek ışık ve gerçek sensör altında ne
olduğunu gösterir. Ground truth `ekran_kadran.py`'ın yazdığı manifestten gelir;
elle etiketleme yok, dolayısıyla etiket hatası da yok (STAJ/SORULAR.md · S1 · A).

**Eşleştirme çekim sırasına göre, ama sıraya güvenilmiyor.** Fotoğraf sayısı
manifestteki kare sayısına eşit değilse ölçüm **yapılmıyor**: bir kare atlanmış
ya da iki kez çekilmişse tüm eşleşme kayar ve tablo sessizce anlamsızlaşır.
Sayılar yine makul görünür — komşu kareler birbirine yakın değerlerdir — bu
yüzden kaymanın kendi kendini ele vermesi beklenemez. Ek olarak kontak sayfası
üretiliyor: her fotoğrafın üstünde atanan değer yazılı, ekrandaki #NN ile
karşılaştırmak on saniye sürüyor.

**Neyi ölçmez.** Ekranda cam yansıması, metal doku, tozlanma ve gerçek sanayi
aydınlatması yok. Bu tablo "sahada ne olur"u değil, "gerçek optik yol zincire ne
kadar hata katıyor"u söyler. Gerçek manometre hâlâ B seçeneğidir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.pipeline import read_gauge
from gauge_vision.read.evaluate import error_stats

UZANTILAR = (".jpg", ".jpeg", ".png", ".heic", ".bmp")
VARSAYILAN_MANIFEST = "outputs/metrics/ip8_ekran_manifest.json"
METRIK_YOLU = "outputs/metrics/ip8_ekran_hatasi.json"
KONTAK_YOLU = "outputs/figures/ip8_kontak_sayfasi.png"
# Sentetik ölçümler — gerçek sayı bunların yanına konmadan anlam kazanmıyor.
SENTETIK_ZINCIR_YUZDE = 0.19      # 07.08, data/synthetic/v1
KONTAK_SUTUN = 4
KONTAK_HUCRE = 320


class _TumKare:
    """Tespit yerine tüm kareyi kutu veren yer tutucu.

    Ekran fotoğrafında kadran karenin çoğunu kaplar; YOLO'nun sentetik kadran
    üzerinde eğitilmiş modeli ekran görüntüsünde henüz sınanmadı. `--tespitsiz`
    ile tespit atlanıp yalnızca OKUMA zincirinin hatası ölçülebilsin diye var —
    tespit hatası ile okuma hatası aynı sayıya karışmasın.
    """

    class _Kutular:
        def __init__(self, h, w):
            self.xyxy = np.array([[0.0, 0.0, float(w), float(h)]])
            self.conf = np.array([1.0])

        def __len__(self):
            return 1

    class _Sonuc:
        def __init__(self, h, w):
            self.boxes = _TumKare._Kutular(h, w)

    def predict(self, image, **_):
        h, w = image.shape[:2]
        return [_TumKare._Sonuc(h, w)]


def fotograflari_bul(klasor: Path) -> list[Path]:
    """Klasördeki fotoğraflar, çekim sırasına göre.

    Sıralama önce dosya adına göre: telefonlar artan numara verir (IMG_0041,
    IMG_0042...) ve bu, değişiklik zamanından daha güvenilirdir — kopyalama
    işlemleri zaman damgasını bozar, adı bozmaz.
    """
    return sorted(p for p in klasor.iterdir()
                  if p.suffix.lower() in UZANTILAR and p.is_file())


def kontak_sayfasi(yollar: list[Path], kayitlar: list[dict], birim: str,
                   cikti: Path) -> None:
    """Her fotoğrafı atanan değeriyle birlikte tek sayfada dizer.

    Kaymayı gözle yakalamak için: fotoğrafın içindeki #NN ile buradaki sıra
    numarası uyuşmuyorsa eşleşme bozuktur ve tablo çöpe gider.
    """
    satir = (len(yollar) + KONTAK_SUTUN - 1) // KONTAK_SUTUN
    sayfa = np.full((satir * KONTAK_HUCRE, KONTAK_SUTUN * KONTAK_HUCRE, 3), 40, np.uint8)

    for i, (yol, k) in enumerate(zip(yollar, kayitlar)):
        img = cv2.imread(str(yol))
        if img is None:
            continue
        h, w = img.shape[:2]
        olcek = (KONTAK_HUCRE - 40) / max(h, w)
        kucuk = cv2.resize(img, (int(w * olcek), int(h * olcek)))
        r, s = divmod(i, KONTAK_SUTUN)
        y0, x0 = r * KONTAK_HUCRE, s * KONTAK_HUCRE
        sayfa[y0:y0 + kucuk.shape[0], x0:x0 + kucuk.shape[1]] = kucuk
        cv2.putText(sayfa, f"#{k['sira']:02d}  {k['value']:g} {birim}",
                    (x0 + 8, y0 + KONTAK_HUCRE - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cikti.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cikti), sayfa)


def olc(yollar: list[Path], kayitlar: list[dict], gauge, model, conf: float) -> list[dict]:
    """Her fotoğrafı okur, ground truth ile karşılaştırır."""
    aralik = gauge.scale.max - gauge.scale.min
    satirlar = []

    for yol, k in zip(yollar, kayitlar):
        img = cv2.imread(str(yol))
        if img is None:
            satirlar.append({**k, "dosya": yol.name, "durum": "okunamadi",
                             "sebep": "goruntu acilamadi"})
            continue

        sonuc = read_gauge(img, model, gauge, detect_conf=conf)
        okuma = sonuc.reading
        satir = {**k, "dosya": yol.name}

        if okuma is None or okuma.value is None:
            # Okunamayan kare hata ortalamasına GİRMEZ ama kapsama düşer.
            # İkisini karıştırmak, eşiği yükselterek "hata"yı iyileştirmeyi
            # mümkün kılardı; o bir ölçüm değil, ölçümün kandırılmasıdır.
            satir.update({"durum": "unreadable",
                          "conf": round(float(okuma.conf), 4) if okuma else 0.0,
                          "sebep": getattr(sonuc, "reason", None)})
        else:
            hata = abs(float(okuma.value) - k["value"])
            satir.update({"durum": okuma.status,
                          "okunan": round(float(okuma.value), 4),
                          "conf": round(float(okuma.conf), 4),
                          "hata": round(hata, 4),
                          "hata_yuzde": round(100.0 * hata / aralik, 4)})
        satirlar.append(satir)
    return satirlar


def main() -> int:
    ap = argparse.ArgumentParser(description="İP8 — ekrandan çekim ölçümü")
    ap.add_argument("--fotograflar", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=Path(VARSAYILAN_MANIFEST))
    ap.add_argument("--agirlik", type=Path, default=None,
                    help="YOLO ağırlığı; verilmezse tespit atlanır")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cikti", type=Path, default=Path(METRIK_YOLU))
    ap.add_argument("--kontak", type=Path, default=Path(KONTAK_YOLU))
    args = ap.parse_args()

    if not args.fotograflar.is_dir():
        print(f"HATA: klasör yok — {args.fotograflar}")
        return 1
    if not args.manifest.exists():
        print(f"HATA: manifest yok — {args.manifest}\n"
              f"Önce: python scripts\\ekran_kadran.py --gosterge <ID>")
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    kayitlar = manifest["kareler"]
    yollar = fotograflari_bul(args.fotograflar)

    if len(yollar) != len(kayitlar):
        # Sessizce kırpıp devam etmek en tehlikeli seçenek olurdu: eşleşme
        # kayar, sayılar makul görünür, tablo yanlış olur.
        print(f"HATA: {len(yollar)} fotoğraf var, manifest {len(kayitlar)} kare "
              f"bekliyor.\nEşleştirme sıraya dayanıyor; sayılar tutmadan ölçüm "
              f"yapılmaz. Fazla/eksik kareyi düzeltip tekrar koşturun.")
        for i, y in enumerate(yollar, start=1):
            print(f"  {i:02d}  {y.name}")
        return 1

    gauge = load_gauges()[manifest["gauge_id"]]
    if args.agirlik:
        from ultralytics import YOLO
        model, tespit = YOLO(str(args.agirlik)), "YOLO"
    else:
        model, tespit = _TumKare(), "yok (tum kare)"

    print(f"{gauge.id} · {len(yollar)} fotoğraf · tespit: {tespit}\n")
    satirlar = olc(yollar, kayitlar, gauge, model, args.conf)
    kontak_sayfasi(yollar, kayitlar, gauge.unit or "", args.kontak)

    okunan = [s for s in satirlar if "hata_yuzde" in s]
    hatalar = np.array([s["hata_yuzde"] for s in okunan]) if okunan else np.array([])

    print("| # | gercek | okunan | hata % | conf | durum |")
    print("|---|---|---|---|---|---|")
    for s in satirlar:
        print(f"| {s['sira']:02d} | {s['value']:g} | "
              f"{s.get('okunan', '—')} | {s.get('hata_yuzde', '—')} | "
              f"{s.get('conf', '—')} | {s['durum']} |")

    ozet = {
        "gauge_id": gauge.id,
        "kaynak": str(args.fotograflar),
        "tespit": tespit,
        "kare": len(satirlar),
        "okunan": len(okunan),
        "kapsama": round(len(okunan) / len(satirlar), 4) if satirlar else 0.0,
        "sentetik_referans_yuzde": SENTETIK_ZINCIR_YUZDE,
        "satirlar": satirlar,
    }
    if hatalar.size:
        from dataclasses import asdict
        ozet["hata"] = {k: round(float(v), 4)
                        for k, v in asdict(error_stats(hatalar)).items()}

    print(f"\nKapsama: {ozet['okunan']}/{ozet['kare']} "
          f"(%{100 * ozet['kapsama']:.1f})")
    if hatalar.size:
        print(f"Ortalama hata: %{hatalar.mean():.3f} · p95 %{np.percentile(hatalar, 95):.3f} "
              f"· en kotu %{hatalar.max():.3f}")
        print(f"Sentetik referans (07.08): %{SENTETIK_ZINCIR_YUZDE}")
    else:
        print("Hicbir kare okunamadi — hata istatistigi uretilmedi.")

    args.cikti.parent.mkdir(parents=True, exist_ok=True)
    args.cikti.write_text(json.dumps(ozet, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nOlcum: {args.cikti}\nKontak sayfasi: {args.kontak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
