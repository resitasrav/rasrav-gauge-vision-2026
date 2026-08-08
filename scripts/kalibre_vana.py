r"""Vana kol açılarını etiketli fotoğraflardan ölçer ve envanter satırını yazar.

    python scripts\kalibre_vana.py --klasor data\real\VL-601 --gosterge VL-601

**Neyi çözüyor.** `configs/gauges.yaml`'daki `lever_angle` değerleri şu an
VARSAYIM (STAJ/SORULAR.md · S2): "kol yatay = açık" kabul edildi. Gerçek montajda
ters olabilir ve bu **sessizce** yanlış durum üretir — vana kapalıyken "açık"
yayınlanır. Hiçbir birim testi yakalayamaz, çünkü kod ile sentetik üreteç aynı
varsayımı paylaşır. Varsayımı kıracak tek şey **sahadan gelen etiketli görüntü**.

**Nasıl.** Dosya adının başındaki durum adı etikettir (`open_01.jpg`,
`closed_03.png`). Her görüntüde kolun açısı ölçülür, durum başına açılar
toplanır ve dairesel ortanca alınır. Çıktı, YAML'a yapıştırılacak satırlardır.

**Neden ortanca ve neden dairesel.** Kol iki uçludur: 179° ile 1° aynı duruştur,
aradaki fark 178 değil 2 derecedir. Düz ortalama bu sarmalamayı göremez ve iki
ölçümün ortasını 90°'ye — yani tam ters duruma — koyar. Doğrusu açıları
iki katına çıkarıp birim vektör toplamak, sonra yarıya bölmektir. Ortanca değil
yön ortalaması kullanılıyor ama kırpma ortancayla yapılıyor: tek bir bozuk
karenin (kol yerine gölgeyi bulmuş) sonucu kaydırmaması için.

**Bu bir "model eğitimi" değil, tek parametreli kestirimdir** ve kasıtlı olarak
öyle: durum başına öğrenilecek tek bir sayı var. Daha büyük bir öğrenici, üç
fotoğrafla bunu daha iyi yapamaz — sadece neden öyle karar verdiğini
söyleyemez hâle gelir. Görüntü koşulları çeşitlenip geometrik kestirim
yetmezse (ıslak/paslı kol, kısmi kapanma) doğru adım burayı büyütmek değil,
İP12'ye sınıflandırıcı eklemektir; o zaman bu script etiketli kümenin
sağlamasını yapan araç olarak kalır.

Ölçüm `outputs/metrics/vana_kalibrasyon.json` dosyasına da yazılır (4. kural).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.read.state import VANA_MIN_UZAMA, _kol_acisi

UZANTILAR = (".png", ".jpg", ".jpeg", ".bmp")
# Bir durumun ölçümleri bu kadar dağınıksa beyan güvenilmez: ya etiketler
# karışmış ya da kol her karede farklı yerde. Sayı gözle değil ölçümle
# gelmeli — sahada koşulduğunda gerçek dağılıma göre güncellenecek.
MAX_YAYILMA_DEG = 12.0


def _durum_adi(yol: Path) -> str:
    """Dosya adının ilk parçası etikettir: `open_01.jpg` → `open`."""
    return yol.stem.split("_")[0].lower()


def _yon_ortalamasi(acilar: list[float]) -> tuple[float, float]:
    """180° modunda yön ortalaması ve yayılma. `(açı, yayılma_deg)` döner.

    Açılar iki katına çıkarılıp birim vektör toplanıyor: bu, 179°/1°
    sarmalamasını doğru ele alan tek yoldur. Yayılma, bileşke vektörün
    boyundan geliyor — 1'e yakınsa ölçümler örtüşüyor, 0'a yakınsa dağınık.
    """
    ikili = np.radians(np.asarray(acilar, dtype=np.float64) * 2.0)
    c, s = float(np.cos(ikili).mean()), float(np.sin(ikili).mean())
    uzunluk = math.hypot(c, s)
    ortalama = math.degrees(math.atan2(s, c)) / 2.0 % 180.0
    # Dairesel standart sapma; 2θ uzayında hesaplanıp yarıya bölünüyor.
    yayilma = math.degrees(math.sqrt(-2.0 * math.log(max(uzunluk, 1e-9)))) / 2.0
    return ortalama, yayilma


def olc(klasor: Path) -> dict[str, list[float]]:
    """Klasördeki her görüntüde kol açısını ölçer, duruma göre gruplar."""
    olculen: dict[str, list[float]] = defaultdict(list)
    atlanan: list[str] = []

    for yol in sorted(p for p in klasor.iterdir() if p.suffix.lower() in UZANTILAR):
        img = cv2.imread(str(yol))
        if img is None:
            atlanan.append(f"{yol.name}: okunamadı")
            continue
        sonuc = _kol_acisi(img)
        if sonuc is None:
            atlanan.append(f"{yol.name}: kol bulunamadı")
            continue
        aci, uzama = sonuc
        if uzama < VANA_MIN_UZAMA:
            # Uzun-ince değil: bulunan şey kol değil. Sessizce dahil edilirse
            # kalibrasyonu bozar; kalibrasyon hatası her okumaya yayılır.
            atlanan.append(f"{yol.name}: şekil kol değil (uzama {uzama:.1f})")
            continue
        olculen[_durum_adi(yol)].append(aci)

    for satir in atlanan:
        print(f"  atlandı — {satir}")
    return dict(olculen)


def rapor(olculen: dict[str, list[float]], gauge, tolerans: float) -> dict:
    """Ölçümü özetler, çelişkileri söyler, YAML satırlarını üretir."""
    izinli = set(gauge.state_names)
    ozet, uyarilar = {}, []

    for ad, acilar in sorted(olculen.items()):
        if ad not in izinli:
            uyarilar.append(f"'{ad}' envanterde tanımlı değil — dosya adı yanlış olabilir")
            continue
        ortalama, yayilma = _yon_ortalamasi(acilar)
        ozet[ad] = {"n": len(acilar), "lever_angle": round(ortalama, 1),
                    "yayilma_deg": round(yayilma, 2)}
        if yayilma > MAX_YAYILMA_DEG:
            uyarilar.append(f"'{ad}' ölçümleri dağınık (yayılma {yayilma:.1f}°) — "
                            f"etiketler karışmış olabilir")

    eksik = izinli - set(ozet)
    if eksik:
        uyarilar.append(f"şu durumlar için hiç görüntü yok: {sorted(eksik)}")

    # Envanterin kendi kuralı burada da geçerli: iki durum toleranslarıyla
    # çakışıyorsa beyan kullanılamaz. config.py bunu zaten reddediyor; kullanıcı
    # YAML'a yapıştırmadan ÖNCE bilsin diye burada da kontrol ediliyor.
    adlar = list(ozet)
    for i in range(len(adlar)):
        for j in range(i + 1, len(adlar)):
            a, b = ozet[adlar[i]]["lever_angle"], ozet[adlar[j]]["lever_angle"]
            fark = abs(a - b) % 180.0
            fark = min(fark, 180.0 - fark)
            if fark < 2 * tolerans:
                uyarilar.append(f"'{adlar[i]}' ({a:.0f}°) ve '{adlar[j]}' ({b:.0f}°) "
                                f"±{tolerans:.0f}° toleransla ayırt edilemez")

    # Mevcut beyanla karşılaştırma: asıl merak edilen "varsayım tuttu mu".
    mevcut = gauge.state_angles
    for ad, s in ozet.items():
        if ad in mevcut:
            fark = abs(s["lever_angle"] - mevcut[ad]) % 180.0
            s["mevcut_beyan"] = mevcut[ad]
            s["sapma_deg"] = round(min(fark, 180.0 - fark), 1)

    return {"gauge_id": gauge.id, "tolerance_deg": tolerans,
            "durumlar": ozet, "uyarilar": uyarilar}


def main() -> int:
    ap = argparse.ArgumentParser(description="Vana kol açısı kalibrasyonu")
    ap.add_argument("--klasor", required=True, type=Path,
                    help="etiketli görüntüler (dosya adı: <durum>_NN.jpg)")
    ap.add_argument("--gosterge", default="VL-601")
    ap.add_argument("--cikti", type=Path,
                    default=Path("outputs/metrics/vana_kalibrasyon.json"))
    args = ap.parse_args()

    if not args.klasor.is_dir():
        print(f"HATA: klasör yok — {args.klasor}")
        return 1

    gauge = load_gauges()[args.gosterge]
    if gauge.type != "valve":
        print(f"HATA: {gauge.id} bir vana değil (tip: {gauge.type})")
        return 1

    print(f"{gauge.id} · {args.klasor}")
    olculen = olc(args.klasor)
    if not olculen:
        print("Hiçbir karede kol ölçülemedi — kalibrasyon yapılamadı.")
        return 1

    sonuc = rapor(olculen, gauge, gauge.tolerance_deg)

    print("\n| durum | kare | ölçülen açı | yayılma | mevcut beyan | sapma |")
    print("|---|---|---|---|---|---|")
    for ad, s in sonuc["durumlar"].items():
        print(f"| {ad} | {s['n']} | {s['lever_angle']:.1f}° | {s['yayilma_deg']:.2f}° | "
              f"{s.get('mevcut_beyan', '—')} | {s.get('sapma_deg', '—')} |")

    for u in sonuc["uyarilar"]:
        print(f"\n⚠ {u}")

    print("\nconfigs/gauges.yaml · states bloğuna yapıştır:\n")
    for ad, s in sonuc["durumlar"].items():
        print(f"      - name: {ad}")
        print(f"        lever_angle: {s['lever_angle']:.0f}")

    args.cikti.parent.mkdir(parents=True, exist_ok=True)
    args.cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nÖlçüm yazıldı: {args.cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
