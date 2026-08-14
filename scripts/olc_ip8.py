r"""Ekrandan çekilmiş fotoğraflarda uçtan uca hata tablosu — dört tip (İP8).

    python scripts\ekran_kadran.py                     # once goster+cek (28 kare)
    python scripts\olc_ip8.py --fotograflar data\real\ip8_ekran

Zincirin **gerçek optik yoldan geçmiş** görüntüdeki hatasını ölçer; artık dört
gösterge tipinde birden. Sentetikte çıkan sayıların (analog %0,19 · dijital
%93,3 · lamba/vana %100/%100) gerçek mercek, gerçek ışık ve gerçek sensör
altında ne olduğunu gösterir. Ground truth `ekran_kadran.py`'ın yazdığı
manifestten gelir; elle etiketleme yok, dolayısıyla etiket hatası da yok
(STAJ/SORULAR.md · S1 · A).

Tipe göre ölçülen şey farklıdır ve tek sayıya indirgenmez:

    analog   → tam skala hata yüzdesi (İP8'in asıl hedef metriği)
    digital  → dizge doğruluğu (İP11 ile aynı ölçüt: hane hane tam eşleşme)
    lamp     → durum doğruluğu
    valve    → durum doğruluğu (ara konum karesinde DOĞRU cevap unreadable'dır)

**Eşleştirme çekim sırasına göre, ama sıraya güvenilmiyor.** Fotoğraf sayısı
manifestteki kare sayısına eşit değilse ölçüm **yapılmıyor**: bir kare atlanmış
ya da iki kez çekilmişse tüm eşleşme kayar ve tablo sessizce anlamsızlaşır.
Sayılar yine makul görünür — komşu kareler birbirine yakın değerlerdir — bu
yüzden kaymanın kendi kendini ele vermesi beklenemez. Ek olarak kontak sayfası
üretiliyor: her fotoğrafın üstünde atanan değer yazılı, ekrandaki #NN ile
karşılaştırmak on saniye sürüyor.

**Neyi ölçmez.** Ekranda cam yansıması, metal doku, tozlanma ve gerçek sanayi
aydınlatması yok. Bu tablo "sahada ne olur"u değil, "gerçek optik yol zincire ne
kadar hata katıyor"u söyler. Ayrıca dijital/lamba/vana modelleri kendi sentetik
üretecimizin çıktısıyla eğitildi; ekrandaki görüntü de o üreteçten geliyor, yani
bu ölçüm "başka marka bir panele genelleme"yi DEĞİL, optik yolun katkısını
ölçer. Gerçek panel/lamba fotoğrafı hâlâ ayrı bir iştir (S1 · B seçeneği).
"""

from __future__ import annotations

import sys
import argparse
import json
import re
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
SENTETIK_REFERANS = {
    "analog": {"zincir_yuzde": 0.19,           # 07.08, data/synthetic/v1
               "benzetilmis_cekim_yuzde": 0.697},  # 19.08, İP8 benzetilmiş 12 kare
    "digital": {"dizge_dogrulugu": 0.933},     # İP11
    "lamp": {"dogruluk": 1.0},                 # İP12
    "valve": {"dogruluk": 1.0},                # İP12
}
KONTAK_SUTUN = 4
KONTAK_HUCRE = 320
# İP5+ dört sınıflı model (13.08). Varsayılan tespit bu: analogda merkez
# rafinesi kutu hatasını düzeltebiliyor ama dijital/lamba/vanada öyle bir
# geometrik düzeltme yok — panelin fotoğraf içindeki yerini yalnız tespit
# bulabilir. Zaten İP8 "uçtan uca" diyor; tespit zincirin parçasıdır.
COK_SINIF_AGIRLIK = Path("runs/detect/models/ip5/cok_sinif/weights/best.pt")


class _TumKare:
    """Tespit yerine tüm kareyi kutu veren yer tutucu.

    Ekran fotoğrafında gösterge karenin çoğunu kaplar; YOLO'nun sentetik veri
    üzerinde eğitilmiş modeli ekran görüntüsünde henüz sınanmadı. `--agirlik`
    verilmeyince tespit atlanıp yalnızca OKUMA zincirinin hatası ölçülsün diye
    var — tespit hatası ile okuma hatası aynı sayıya karışmasın.
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


def fotograflari_bul(klasor: Path) -> tuple[list[Path], list[str]]:
    """Klasördeki fotoğraflar, çekim sırasına göre. `(yollar, uyarılar)`.

    **Alfabetik sıralama BURADA ÇALIŞMAZ** ve bunu ilk gerçek çekim gösterdi:
    telefondan gelen adlar `IMG-1 … IMG-12` gibi **sıfır dolgusuz** olduğu için
    alfabetik sıra `IMG-1, IMG-10, IMG-11, IMG-12, IMG-2 …` veriyor. On iki
    karenin on biri yanlış değere eşleşti; sayım denetimi bunu yakalayamadı
    çünkü sayı doğruydu, **sıra** yanlıştı. (Kontak sayfası yakaladı.)

    Doğrusu, addaki SAYIYA göre sıralamak. Her ad tam bir tam sayı içeriyorsa ve
    bu sayılar benzersizse sıra ondan kurulur — `IMG-3` ile `IMG3` arasındaki
    biçim farkı da böylece önemsizleşir. Koşul sağlanmazsa alfabetiğe düşülüyor
    ama **sessizce değil**: uyarı basılıyor, çünkü sessiz bir sıra hatası
    tablonun tamamını çöpe çevirir ve sayılar makul görünmeye devam eder.
    """
    yollar = [p for p in klasor.iterdir()
              if p.suffix.lower() in UZANTILAR and p.is_file()]
    uyarilar: list[str] = []

    sayilar = {}
    for p in yollar:
        bulunan = re.findall(r"\d+", p.stem)
        if len(bulunan) == 1:
            sayilar[p] = int(bulunan[0])

    if len(sayilar) == len(yollar) and len(set(sayilar.values())) == len(yollar):
        return sorted(yollar, key=lambda p: sayilar[p]), uyarilar

    uyarilar.append(
        "Dosya adlarindan sira numarasi cikarilamadi (her adda tam bir benzersiz "
        "sayi olmali) — ALFABETIK siraya dusuldu. Kontak sayfasindaki #NN "
        "numaralari sirayla gitmiyorsa esleme yanlistir.")
    return sorted(yollar), uyarilar


def beklenen_metni(k: dict, gauges: dict) -> str:
    """Karenin ground truth'unun insan-okur hâli (kontak sayfası + tablo)."""
    tip = k.get("type", "analog")
    if tip == "analog":
        birim = gauges[k["gauge_id"]].unit or ""
        return f"{k['value']:g} {birim}".strip()
    if tip == "digital":
        return k["text"].strip() if "text" in k else f"{k['value']:g}"
    return k["beklenen"]


def kontak_sayfasi(yollar: list[Path], kayitlar: list[dict], gauges: dict,
                   cikti: Path) -> None:
    """Her fotoğrafı atanan ground truth'uyla birlikte tek sayfada dizer.

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
        cv2.putText(sayfa, f"#{k['sira']:02d}  {beklenen_metni(k, gauges)}",
                    (x0 + 8, y0 + KONTAK_HUCRE - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cikti.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cikti), sayfa)


def _degerlendir(satir: dict, okuma, k: dict, gauge) -> None:
    """Okumayı ground truth ile tipe göre karşılaştırır, `satir`'ı doldurur.

    Neden tek "hata %" sütunu yok: dizgede ve durumda "yüzde hata" tanımsızdır.
    Analogda sürekli hata ölçülür; diğer üçünde ikili doğru/yanlış — İP11 ve
    İP12'nin ölçütleriyle aynı kalsın ki sentetik sayılarla kıyas anlamlı olsun.
    """
    tip = k.get("type", "analog")
    okunamadi = okuma is None or okuma.value is None

    if okunamadi:
        satir.update({"durum": "unreadable",
                      "conf": round(float(okuma.conf), 4) if okuma else 0.0})
        # Ara konum vanası gibi karelerde doğru cevap okumamaktır (6. kural).
        if k.get("beklenen") == "unreadable":
            satir["dogru"] = True
        elif tip != "analog":
            satir["dogru"] = False
        return

    satir.update({"durum": okuma.status, "conf": round(float(okuma.conf), 4)})

    if tip == "analog":
        aralik = gauge.scale.max - gauge.scale.min
        hata = abs(float(okuma.value) - k["value"])
        satir.update({"okunan": round(float(okuma.value), 4),
                      "hata": round(hata, 4),
                      "hata_yuzde": round(100.0 * hata / aralik, 4)})
    elif tip == "digital":
        # İP11'in ölçütü: hane hane TAM eşleşme. 0,1 birim sapmış bir okuma
        # "yaklaşık doğru" değil yanlıştır — panelde öyle bir sayı yazmıyor.
        decimals = int((gauge.digits or {}).get("decimals", 1))
        okunan_metin = f"{float(okuma.value):.{decimals}f}"
        satir.update({"okunan": okunan_metin,
                      "dogru": okunan_metin == k["text"].strip()})
    else:  # lamp / valve
        satir.update({"okunan": str(okuma.value),
                      "dogru": (k["beklenen"] != "unreadable"
                                and str(okuma.value) == k["beklenen"])})


def olc(yollar: list[Path], kayitlar: list[dict], gauges: dict, model,
        conf: float, *, perspektif: bool = False) -> list[dict]:
    """Her fotoğrafı tipine göre okur, ground truth ile karşılaştırır.

    `perspektif` bir ablasyon anahtarıdır ve yalnız analog dalını etkiler.
    Ekrandan çekimde kamera göstergeye dik duramıyor — ekran masada, fotoğrafçı
    ayakta — dolayısıyla eğiklik burada sentetikteki gibi bir "eksen" değil,
    işin doğasında var. Düzeltmenin gerçek fotoğrafta ne kazandırdığı ancak
    açık/kapalı iki koşu yan yana konunca görülür.
    """
    satirlar = []
    for yol, k in zip(yollar, kayitlar):
        img = cv2.imread(str(yol))
        satir = {**k, "dosya": yol.name}
        if img is None:
            satir.update({"durum": "okunamadi", "sebep": "goruntu acilamadi"})
            satirlar.append(satir)
            continue

        gauge = gauges[k["gauge_id"]]
        sonuc = read_gauge(img, model, gauge, detect_conf=conf,
                           perspektif=perspektif)
        _degerlendir(satir, sonuc.reading, k, gauge)
        if satir.get("durum") == "unreadable" and getattr(sonuc, "reason", ""):
            satir["sebep"] = sonuc.reason
        satirlar.append(satir)
    return satirlar


def ozetle(satirlar: list[dict]) -> dict:
    """Tip bazında özet istatistikler — her tip kendi ölçütüyle."""
    tipler: dict[str, dict] = {}

    analog = [s for s in satirlar if s.get("type", "analog") == "analog"]
    if analog:
        okunan = [s for s in analog if "hata_yuzde" in s]
        hatalar = np.array([s["hata_yuzde"] for s in okunan])
        blok = {"kare": len(analog), "okunan": len(okunan),
                "kapsama": round(len(okunan) / len(analog), 4),
                "referans": SENTETIK_REFERANS["analog"]}
        if hatalar.size:
            from dataclasses import asdict
            blok["hata"] = {k: round(float(v), 4)
                            for k, v in asdict(error_stats(hatalar)).items()}
        tipler["analog"] = blok

    for tip in ("digital", "lamp", "valve"):
        grup = [s for s in satirlar if s.get("type") == tip]
        if not grup:
            continue
        dogru = sum(1 for s in grup if s.get("dogru"))
        tipler[tip] = {"kare": len(grup), "dogru": dogru,
                       "dogruluk": round(dogru / len(grup), 4),
                       "referans": SENTETIK_REFERANS[tip]}
    return tipler


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="İP8 — ekrandan çekim ölçümü (dört tip)")
    ap.add_argument("--fotograflar", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=Path(VARSAYILAN_MANIFEST))
    ap.add_argument("--agirlik", type=Path, default=None,
                    help=f"YOLO ağırlığı (varsayılan: {COK_SINIF_AGIRLIK} varsa o)")
    ap.add_argument("--tespitsiz", action="store_true",
                    help="tespiti atla, tüm kareyi kutu say — tespit hatası ile "
                         "okuma hatasını ayırmak için ablasyon")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--perspektif", action=argparse.BooleanOptionalAction,
                    default=False, help="elips→daire düzleştirme (İP14, analog)")
    ap.add_argument("--cikti", type=Path, default=Path(METRIK_YOLU))
    ap.add_argument("--kontak", type=Path, default=Path(KONTAK_YOLU))
    args = ap.parse_args()

    if not args.fotograflar.is_dir():
        print(f"HATA: klasör yok — {args.fotograflar}")
        return 1
    if not args.manifest.exists():
        print(f"HATA: manifest yok — {args.manifest}\n"
              f"Önce: python scripts\\ekran_kadran.py")
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    kayitlar = manifest["kareler"]
    # Eski (tek gösterge, yalnız analog) manifest de okunabilsin: tip ve
    # gauge_id kare kaydında yoksa üst seviyeden gelir.
    for k in kayitlar:
        k.setdefault("gauge_id", manifest.get("gauge_id"))
        k.setdefault("type", "analog")

    yollar, uyarilar = fotograflari_bul(args.fotograflar)
    for u in uyarilar:
        print(f"UYARI: {u}\n")

    # --- Eşleştirme: SIRA değil, dosya adındaki KARE NUMARASI ---
    #
    # Sıraya dayalı eşleştirme, eksik kare olduğunda bir kayar ve komşu kareler
    # birbirine yakın değerler taşıdığı için **sayılar makul görünmeye devam
    # eder**. Bu yüzden eskiden sayım tutmuyorsa ölçüm hiç yapılmıyordu; ama o
    # da 23 geçerli kareyi 5 eksik yüzünden çöpe atmak demekti.
    #
    # Doğrusu: her fotoğrafın adı taşıdığı kare numarasına göre manifestteki
    # kaydına bağlanır. Eksik kare artık **eksik kalır**, kaymaya yol açmaz;
    # hangilerinin eksik olduğu tabloya ve JSON'a yazılır. Ad numara taşımıyorsa
    # eski katı davranış sürer — orada kayma riski gerçekten var.
    numaralar = {}
    for p in yollar:
        bulunan = re.findall(r"\d+", p.stem)
        if len(bulunan) == 1:
            numaralar[p] = int(bulunan[0])

    kayit_no = {int(k.get("sira", i + 1)): k for i, k in enumerate(kayitlar)}
    ada_gore = len(numaralar) == len(yollar) and set(numaralar.values()) <= set(kayit_no)

    if ada_gore:
        yollar = sorted(yollar, key=lambda p: numaralar[p])
        kayitlar = [kayit_no[numaralar[p]] for p in yollar]
        eksik = sorted(set(kayit_no) - set(numaralar.values()))
        if eksik:
            print(f"NOT: {len(yollar)}/{len(kayit_no)} kare var. Eksik: "
                  f"{', '.join(f'#{n:02d}' for n in eksik)}\n"
                  f"Eşleştirme dosya adındaki numaraya göre yapıldı, sıraya "
                  f"göre değil — eksik kareler kaymaya yol açmaz.\n")
    elif len(yollar) != len(kayitlar):
        print(f"HATA: {len(yollar)} fotoğraf var, manifest {len(kayitlar)} kare "
              f"bekliyor ve dosya adlarından kare numarası çıkarılamadı.\n"
              f"Eşleştirme sıraya düşüyor; sayılar tutmadan ölçüm yapılmaz — "
              f"bir kare atlanırsa eşleşme kayar ve sayılar makul görünmeye "
              f"devam eder.\nDosyaları kare numarasıyla adlandırın (01.jpg, "
              f"02.jpg …) ya da eksiği tamamlayın.")
        for i, y in enumerate(yollar, start=1):
            print(f"  {i:02d}  {y.name}")
        return 1

    gauges = load_gauges()
    agirlik = args.agirlik or (COK_SINIF_AGIRLIK if COK_SINIF_AGIRLIK.exists()
                               else None)
    if args.tespitsiz or agirlik is None:
        model, tespit = _TumKare(), "yok (tum kare)"
        if agirlik is None and not args.tespitsiz:
            print(f"UYARI: {COK_SINIF_AGIRLIK} bulunamadı — tespitsiz ölçülüyor. "
                  f"Dijital/lamba/vana kareleri sıkı çekilmediyse okunamayabilir.\n")
    else:
        from ultralytics import YOLO
        model, tespit = YOLO(str(agirlik)), f"YOLO ({agirlik})"

    print(f"{len(yollar)} fotoğraf · tespit: {tespit}\n")
    satirlar = olc(yollar, kayitlar, gauges, model, args.conf,
                   perspektif=args.perspektif)
    kontak_sayfasi(yollar, kayitlar, gauges, args.kontak)

    print("| # | gosterge | beklenen | okunan | sonuc | conf | durum |")
    print("|---|---|---|---|---|---|---|")
    for s in satirlar:
        if s.get("type", "analog") == "analog":
            sonuc = f"%{s['hata_yuzde']:g}" if "hata_yuzde" in s else "—"
        else:
            sonuc = {True: "dogru", False: "YANLIS"}.get(s.get("dogru"), "—")
        print(f"| {s['sira']:02d} | {s['gauge_id']} | "
              f"{beklenen_metni(s, gauges)} | {s.get('okunan', '—')} | "
              f"{sonuc} | {s.get('conf', '—')} | {s['durum']} |")

    tipler = ozetle(satirlar)
    ozet = {
        "kaynak": str(args.fotograflar),
        "tespit": tespit,
        "kare": len(satirlar),
        "tipler": tipler,
        "satirlar": satirlar,
    }

    print()
    for tip, blok in tipler.items():
        if tip == "analog":
            h = blok.get("hata")
            print(f"analog : kapsama {blok['okunan']}/{blok['kare']}"
                  + (f" · ort %{h['mean']:.3f} · p95 %{h['p95']:.3f} "
                     f"· en kotu %{h['max']:.3f}" if h else " · hic kare okunamadi")
                  + f"  [sentetik %{blok['referans']['zincir_yuzde']} · "
                    f"benzetilmis %{blok['referans']['benzetilmis_cekim_yuzde']}]")
        else:
            ref = next(iter(blok["referans"].values()))
            print(f"{tip:<7}: {blok['dogru']}/{blok['kare']} dogru "
                  f"(%{100 * blok['dogruluk']:.1f})  [sentetik %{100 * ref:.1f}]")

    args.cikti.parent.mkdir(parents=True, exist_ok=True)
    args.cikti.write_text(json.dumps(ozet, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nOlcum: {args.cikti}\nKontak sayfasi: {args.kontak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
