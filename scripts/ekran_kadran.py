r"""Göstergeleri ekranda tam ekran gösterir; telefonla fotoğraflanacak (İP8 · A yolu).

    python scripts\ekran_kadran.py                          # dört tip, varsayılan plan (28 kare)
    python scripts\ekran_kadran.py --gosterge PT-101 --adet 12   # tek gösterge

Tuşlar:  BOŞLUK / → sonraki · ← önceki · q çıkış

**Neden bu düzenek.** İP8 "gerçek gösterge fotoğraflarında uçtan uca hata
tablosu" istiyor. Etiketli açık veri kalmadı (İP1: A1/A2 erişilemez, A5
etiketsiz) ve gerçek manometre alıp elle etiketlemek 2-3 gün. Ekrandan çekimde
görüntü **gerçek mercekten, gerçek ışıktan, gerçek sensörden** geçiyor; değer
ise birebir biliniyor çünkü kareyi biz ürettik. Gerçek manometrenin yerini
tutmaz — cam yansıması, metal doku ve tozlanma yok — ama sentetik ile gerçek
arasındaki basamaktır ve elle etiketleme gerektirmez. (STAJ/SORULAR.md · S1)

**Neden dört tip birden.** İlk sürüm yalnız analog kadranı gösteriyordu; oysa
dijital panel, lamba ve vana da ekranda gösterilip fotoğraflanabilir ve
"%93,3 / %100 / %100" sayılarının tamamı sentetik veriden geliyor — yani model
kendi üretecimizin çıktısında ölçüldü. Ekrandan çekim, dört tipin hepsinde
gerçek optik yolun katkısını aynı ucuz yöntemle ölçer.

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

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.pipeline import KIRPIM_PAYI
from gauge_vision.synth.dial import DialLook, render_analog
from gauge_vision.synth.digital import CANVAS_H, CANVAS_W, CERCEVE_BGR, render_digital
from gauge_vision.synth.state import LAMBA_RENKLERI, PANO_BGR, render_lamp, render_valve

PENCERE = "IP8 - ekran gostergesi"
# Sıra numarası şeridi: göstergenin ALTINDA, göstergeye değmeyen ayrı bir bant.
# Göstergenin üstüne yazılsaydı okuma zincirine gürültü katardı — ölçtüğümüz
# şeyin içine ölçüm aracını karıştırmak olurdu.
SERIT_ORANI = 0.13
# İçerik ile şerit arasındaki tampon: içeriğin KENDİ zemin renginde bir bant.
# Zincir tespit kutusunu KIRPIM_PAYI kadar payla kırpıyor; tampon olmasaydı o
# pay şeride taşar ve #NN yazısının koyu pikselleri okumaya karışırdı — düzenek
# testi bunu ölçtü: vananın PCA kol açısı yazı yüzünden bozuldu, dört karenin
# dördü okunamadı. Tampon paydan biraz geniş ki kutu birkaç piksel kaysa da
# yazı kırpıma girmesin. (pipeline.KIRPIM_PAYI'dan türetiliyor — ikinci kopya
# yazılsaydı oradaki değişiklik burayı sessizce geçersiz kılardı.)
TAMPON_ORANI = KIRPIM_PAYI + 0.04

# Varsayılan plan: envanterdeki dört tipin temsilcileri, bu sırayla.
# Analog başta çünkü en çok karesi olan o — çekim yarıda kalırsa en azından
# İP8'in asıl hedef metriği (analog okuma hatası) ölçülebilsin.
VARSAYILAN_PLAN = ["PT-101", "DP-401", "LM-501", "VL-601"]

# Vanada tolerans-içi sapma ve ara konum kareleri (derece).
# 12° < tolerance_deg(20): kol sapmış ama durum hâlâ "open" okunmalı.
# 45° tam ortada: iki duruma da uymuyor → beklenen cevap unreadable.
# "Yanlış okumaktansa okumamak" kuralı gerçek optik yolda da geçerli mi,
# ancak böyle bir kareyle ölçülür.
VANA_SAPMA_ICI_DEG = 12.0
VANA_ARA_KONUM_DEG = 45.0


def analog_degerler(gauge, adet: int) -> list[float]:
    """Skalayı uçtan uca tarayan değerler.

    Uç noktalar (min ve max) MUTLAKA var: kadranın iki ucu, açı→değer
    dönüşümünün işaret ve yön hatalarının en görünür olduğu yerdir. Aradakiler
    eşit aralıklı — rastgele seçim, hangi bölgenin ölçülmediğini gizler.
    """
    lo, hi = gauge.scale.min, gauge.scale.max
    if adet < 2:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (adet - 1) for i in range(adet)]


def dijital_degerler(gauge, adet: int) -> list[float]:
    """Panelin fiziksel aralığını (range) uçtan uca tarayan değerler.

    `range` yoksa 0'dan hane kapasitesine kadar gidilir. Uç noktalar burada da
    zorunlu: negatif uç, eksi işaretinin gerçek fotoğrafta çözülüp
    çözülemediğini tek başına test eden karedir (S3 bunun için açık).
    """
    d = gauge.digits or {}
    decimals = int(d.get("decimals", 1))
    aralik = gauge.raw.get("range") or {}
    lo = float(aralik.get("min", 0.0))
    hi = float(aralik.get("max", 10 ** max(1, int(d.get("count", 4)) - decimals) - 1))
    if adet < 2:
        return [round((lo + hi) / 2, decimals)]
    return [round(lo + (hi - lo) * i / (adet - 1), decimals) for i in range(adet)]


def lamba_plani(gauge) -> list[dict]:
    """Lamba kareleri: her renkli durum + sönük hâller.

    Sönük kare iki kez var: renksiz mercek VE renkli durumların ilkinin
    sönük hâli. İkincisi bilinçli zorluk — sönük kırmızı lamba "koyu kırmızı"
    görünür ve okuyucunun onu "red" değil "off" sayması gerekir (İP12'nin
    sentetikte %100 verdiği ayrım; gerçek ekran+kamera yolunda ilk kez burada).
    """
    adlar = [s["name"] for s in (gauge.states or [])]
    renkliler = [a for a in adlar if a in LAMBA_RENKLERI]
    plan: list[dict] = []
    if "off" in adlar:
        plan.append({"beklenen": "off", "renk": None})
        if renkliler:
            plan.append({"beklenen": "off", "renk": renkliler[0]})
    for ad in renkliler:
        plan.append({"beklenen": ad, "renk": None})
    return plan


def vana_plani(gauge) -> list[dict]:
    """Vana kareleri: her durum nominal açıda + tolerans-içi sapma + ara konum."""
    adlar = [s["name"] for s in (gauge.states or [])]
    plan = [{"beklenen": ad, "sapma_deg": 0.0} for ad in adlar]
    if adlar:
        plan.append({"beklenen": adlar[0], "sapma_deg": VANA_SAPMA_ICI_DEG})
        # Ara konum: hiçbir duruma sayılmamalı. `beklenen` alanına durum adı
        # değil "unreadable" yazılır; ölçüm scripti bunu "okumama başarısı"
        # olarak değerlendirir.
        plan.append({"beklenen": "unreadable", "sapma_deg": VANA_ARA_KONUM_DEG,
                     "cizilen_durum": adlar[0]})
    return plan


def ekran_boyutu(varsayilan: tuple[int, int] = (1920, 1080)) -> tuple[int, int]:
    """Fiziksel ekran çözünürlüğü (en, boy). Bulunamazsa varsayılan döner."""
    try:
        import ctypes
        u = ctypes.windll.user32           # type: ignore[attr-defined]
        u.SetProcessDPIAware()             # ölçeklenmiş ekranda gerçek piksel
        en, boy = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
        return (en, boy) if en > 0 and boy > 0 else varsayilan
    except Exception:
        return varsayilan


def ekrana_gom(kare: np.ndarray, ekran_en: int, ekran_boy: int) -> np.ndarray:
    """Kareyi ekran oranına **en-boy koruyarak** gömer (letterbox).

    Neden gerekli: `cv2.WINDOW_NORMAL` tam ekranda görüntüyü pencereye
    **yayar**, oranı korumaz. 900×1010'luk kare 1920×1080'lik ekrana
    taşındığında yatayda 2,00 kat geriliyordu — yani kadran, fotoğraf daha
    çekilmeden ekranda elipse dönüşüyordu. Bu, İP8'in ölçtüğü şeyin ta
    kendisini bozar: açı okuma dairesellik varsayar, 2 kat gerilmiş bir
    kadranda gerçek açı θ, görüntüde atan(tan(θ)/2) olarak görünür.

    Belirti olarak 19.08 çekiminde de görüldü ama yanlış teşhis edildi
    ("uzak ve eğik çekim"): merkez rafinesi 12 karenin 11'inde kanıt
    kapısından döndü ve yatıklık uyumu gürültü bölgesinde kaldı — ikisi de
    dairesel olmayan bir kadranın beklenen sonucudur.

    Kenar boşluğu, karenin kendi zemin rengiyle değil **beyazla** doldurulur:
    ekranın kendi kenarı zaten oradadır, ayrıca tespit için içeriğin sınırının
    belli olması iyidir.
    """
    h, w = kare.shape[:2]
    olcek = min(ekran_en / w, ekran_boy / h)
    yeni = (max(1, int(round(w * olcek))), max(1, int(round(h * olcek))))
    kucuk = cv2.resize(kare, yeni, interpolation=cv2.INTER_AREA
                       if olcek < 1 else cv2.INTER_CUBIC)

    tuval = np.full((ekran_boy, ekran_en, 3), 255, np.uint8)
    y0 = (ekran_boy - yeni[1]) // 2
    x0 = (ekran_en - yeni[0]) // 2
    tuval[y0:y0 + yeni[1], x0:x0 + yeni[0]] = kucuk
    return tuval


def _serit_ekle(icerik: np.ndarray, sira: int, boyut: int) -> np.ndarray:
    """İçeriğin altına #NN şeridini ekler."""
    serit_h = int(boyut * SERIT_ORANI)
    h, w = icerik.shape[:2]
    tuval = np.full((h + serit_h, w, 3), 255, np.uint8)
    tuval[:h] = icerik

    etiket = f"#{sira:02d}"
    olcek = serit_h / 40.0
    (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, olcek, 3)
    cv2.putText(tuval, etiket, ((w - tw) // 2, h + (serit_h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, olcek, (0, 0, 0), 3, cv2.LINE_AA)
    return tuval


def kare_uret(gauge, spec: dict, sira: int, toplam: int,
              boyut: int) -> tuple[np.ndarray, dict]:
    """Bir gösterim karesi + o karenin ground truth kaydı.

    `spec` tipe göre değişir: analog/dijitalde `{"value": ...}`,
    lambada `{"beklenen", "renk"}`, vanada `{"beklenen", "sapma_deg"}`.

    Kayda `icerik_bbox` de yazılır: göstergenin kare içindeki gerçek kutusu
    (tampon ve şerit hariç). Düzenek testi "ideal tespit"i buradan kurar;
    tahminle ikinci kez yazılsaydı kare düzeni değişince test sessizce yanlış
    bölgeyi ölçerdi.
    """
    kayit = {"sira": sira, "gauge_id": gauge.id, "type": gauge.type,
             "toplam": toplam}
    tampon_bgr = None

    if gauge.type == "analog":
        icerik, truth = render_analog(gauge, spec["value"], size=boyut,
                                      look=DialLook())
        kayit.update(value=round(float(spec["value"]), 4),
                     angle_deg=round(float(truth.angle_deg), 4))

    elif gauge.type == "digital":
        # Panel kendi en-boy oranında, ekran genişliğine ölçekli çizilir —
        # sonradan büyütme yok, segment kenarları keskin kalsın.
        icerik, truth = render_digital(
            gauge, spec["value"], size=(boyut, int(boyut * CANVAS_H / CANVAS_W)))
        kayit.update(value=round(float(spec["value"]), 4), text=truth.text)
        tampon_bgr = CERCEVE_BGR

    elif gauge.type == "lamp":
        durum = spec["beklenen"]
        icerik, _ = render_lamp(gauge, durum, size=boyut, renk=spec.get("renk"))
        kayit.update(beklenen=durum, renk=spec.get("renk"))
        tampon_bgr = PANO_BGR

    elif gauge.type == "valve":
        cizilen = spec.get("cizilen_durum", spec["beklenen"])
        icerik, truth = render_valve(gauge, cizilen, size=boyut,
                                     sapma_deg=spec.get("sapma_deg", 0.0))
        kayit.update(beklenen=spec["beklenen"],
                     sapma_deg=spec.get("sapma_deg", 0.0),
                     lever_angle_deg=round(float(truth.lever_angle_deg), 2))
        tampon_bgr = PANO_BGR
    else:
        raise ValueError(f"{gauge.id}: bilinmeyen tip {gauge.type}")

    h_icerik, w_icerik = icerik.shape[:2]
    kayit["icerik_bbox"] = [0, 0, w_icerik, h_icerik]

    # Analogda tampon yok: kadranın okunan bölgesi (tarama halkası) merkezde,
    # kırpım payının şeride taşması onu etkilemiyor — benzetilmiş 12 karelik
    # koşu (%0,697) bu düzenle alındı, düzen değişirse o referans da geçersizleşir.
    if tampon_bgr is not None:
        bant = np.full((int(h_icerik * TAMPON_ORANI), w_icerik, 3),
                       tampon_bgr, np.uint8)
        icerik = np.vstack([icerik, bant])

    return _serit_ekle(icerik, sira, boyut), kayit


def plan_kur(gauges: dict, istenen: str, adet: int, adet_dijital: int) -> list[tuple]:
    """Gösterilecek (gauge, spec) listesi. `istenen` "hepsi" ya da ID listesi."""
    idler = VARSAYILAN_PLAN if istenen == "hepsi" else [
        s.strip() for s in istenen.split(",") if s.strip()]
    cizelge: list[tuple] = []
    for gid in idler:
        gauge = gauges[gid]
        if gauge.type == "analog":
            cizelge += [(gauge, {"value": v}) for v in analog_degerler(gauge, adet)]
        elif gauge.type == "digital":
            cizelge += [(gauge, {"value": v})
                        for v in dijital_degerler(gauge, adet_dijital)]
        elif gauge.type == "lamp":
            cizelge += [(gauge, s) for s in lamba_plani(gauge)]
        elif gauge.type == "valve":
            cizelge += [(gauge, s) for s in vana_plani(gauge)]
    return cizelge


def _kayit_ozeti(k: dict, gauges: dict) -> str:
    """Terminal listesi için tek satırlık insan-okur özet."""
    g = gauges[k["gauge_id"]]
    if k["type"] == "analog":
        return f"{k['value']:g} {g.unit or ''}".strip()
    if k["type"] == "digital":
        return f"[{k['text']}] {g.unit or ''}".strip()
    if k["type"] == "lamp":
        renk = f" (sönük {k['renk']})" if k.get("renk") else ""
        return f"{k['beklenen']}{renk}"
    sapma = f" (sapma {k['sapma_deg']:g}°)" if k.get("sapma_deg") else ""
    return f"{k['beklenen']}{sapma}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="İP8 ekran gösterimi (dört tip)")
    ap.add_argument("--gosterge", default="hepsi",
                    help='"hepsi" ya da virgüllü ID listesi (örn. PT-101,DP-401)')
    ap.add_argument("--adet", type=int, default=12, help="analog kare sayısı")
    ap.add_argument("--adet-dijital", type=int, default=8,
                    help="dijital panel kare sayısı")
    ap.add_argument("--boyut", type=int, default=900, help="gösterge eni (piksel)")
    ap.add_argument("--manifest", type=Path,
                    default=Path("outputs/metrics/ip8_ekran_manifest.json"))
    ap.add_argument("--tam-ekran", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    gauges = load_gauges()
    cizelge = plan_kur(gauges, args.gosterge, args.adet, args.adet_dijital)
    if not cizelge:
        print("HATA: plan boş — gösterge ID'lerini kontrol edin")
        return 1

    kareler, kayitlar = [], []
    for i, (gauge, spec) in enumerate(cizelge, start=1):
        kare, kayit = kare_uret(gauge, spec, i, len(cizelge), args.boyut)
        kareler.append(kare)
        kayitlar.append(kayit)

    manifest = {
        "olusturuldu": datetime.now().isoformat(timespec="seconds"),
        "gostergeler": sorted({k["gauge_id"] for k in kayitlar}),
        "boyut_px": args.boyut,
        "kareler": kayitlar,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    print(f"{len(kareler)} kare · {', '.join(manifest['gostergeler'])} · "
          f"manifest: {args.manifest}")
    print("Tuşlar: BOŞLUK/→ sonraki · ← önceki · q çıkış")
    print("Her kareyi telefonla çekin; ekrandaki #NN numarası kareye GİRSİN.\n")
    for k in kayitlar:
        print(f"  #{k['sira']:02d}  {k['gauge_id']:<7} {_kayit_ozeti(k, gauges)}")

    cv2.namedWindow(PENCERE, cv2.WINDOW_NORMAL)
    if args.tam_ekran:
        cv2.setWindowProperty(PENCERE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Kareler ekran oranına GÖMÜLÜYOR, esnetilmiyor. Bkz. `ekrana_gom`:
        # WINDOW_NORMAL tam ekranda görüntüyü pencereye yayar ve 900×1010'luk
        # kareyi 1920×1080'e taşırken yatayda 2 kat geriyor — kadran ekranda
        # zaten elips oluyor, fotoğraf daha çekilmeden.
        ekran = ekran_boyutu()
        kareler = [ekrana_gom(k, *ekran) for k in kareler]

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
