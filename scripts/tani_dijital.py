r"""Gerçek fotoğrafta dijital panelin NEDEN okunamadığını ölçer (İP8 teşhisi).

    python scripts\tani_dijital.py --fotograflar data\real\ip8_ekran

**Neden ayrı bir araç.** İlk gerçek çekimde dijital panel 5/5 `unreadable`
verdi. `olc_ip8.py` bunu görür ama sebebini göstermez; `tani_ip8.py` ise analog
zincirin adımlarına (merkez rafinesi, yatıklık, ibre) bakar — dijital kolun
hiçbirini kullanmaz. Bu script dijital kolun kendi adımlarını ayırır:

    tespit → hane kutusu bulma → segment eşiği → desen çözümü

**Ölçülen sonuç: çöken adım HANE KUTUSU BULMADIR.** Tespit beş karede de
0,93-0,97 güvenle paneli tam yerinden buluyor; kutunun gerçek bir hanenin
üstüne düştüğü her yerde rakam 1,000 güvenle DOĞRU çözülüyor (ölçülen: 0, 1, 2,
4, 6, 8, 9). Bilgi görüntüde var; kaybolan şey hanelerin nerede olduğu.

**Sebep `_segment_maskesi`'nin küresel varsayımıdır.** Maske üç parlaklık
seviyesini (zemin / sönük segment / yanık segment) iki kademeli Otsu ile ayırır
ve bu, panel boyunca zeminin SABİT olmasını gerektirir. Ekrandan çekilen gerçek
fotoğrafta panelin üstünde bir yansıma gradyanı var: sol üçte birin zemin
medyanı 60, sağ üçte birinki 39 (#13). Bu 1,5 katlık gradyan, zemin ile sönük
segment arasındaki farktan büyük — küresel eşik ikisini ayıramaz. Sönük
segmentler maskeden düşünce, yalnızca orta çubuğu yanan "-" ya da az segmentli
haneler tam boy kutu üretemez ve `beklenen` sayıda kutu bulunamaz.

**Denenip ÖLÇÜMLE ELENEN üç düzeltme.** Hiçbiri koda girmedi; sayıları burada
duruyor ki aynı yol ikinci kez denenmesin:

    duzlestir_gauss   büyük sigmalı bulanığı zemin sayıp çıkarmak.
                      Haneler bulanığa karışıyor (bir hane panel boyunun
                      yarısı kadar); 5/5 → 0/5, üstelik ham hâlde çözülen
                      haneleri de bozuyor.
    zemin_sutun       zemini her sütunun %15'lik yüzdeliğinden kestirmek.
                      Gradyanı gerçekten düzeltiyor ama hane kutusu sorunu
                      gradyandan bağımsız sürüyor; 0/5.
    renklilik         segment renkli, parlama renksiz → max-min kanal farkı.
                      En umutlusu: #13'ün "-" işaretini ve #15'in tüm dizgesini
                      DOĞRU çözüyor. Yine de 0/5 eşiği geçemiyor ve **#18'de
                      yanlış bir haneyi ("5", doğrusu "6") 1,000 güvenle
                      üretiyor.** Sessiz hata üreten bir düzeltme, düzeltme
                      değildir (3. kural) — bu yüzden reddedildi.

**Kalıcı çözüm eşik ayarı değildir.** İki iş gerekiyor ve ikisi de dijital kola
bugün yok: (1) panelin dörtgen köşelerinden perspektif düzeltmesi —
`detect/perspective.py` yalnız dairesel kadranı düzeltiyor, dikdörtgen paneli
değil; (2) hane ızgarasının görüntüden değil TESPİT kutusundan + envanterdeki
hane sayısından kurulması. `read/digital.py` bu ikincisini zaten kendi
yorumunda kalıcı çözüm olarak işaret ediyor.

**Bu script eşik ÖNERMEZ.** Beş karenin beşi de `unreadable` döndü, yani zincir
yanlış sayı yayınlamadı — başarısızlık GÜVENLİ tarafta. Eşiği indirmek bu
güvenliği, karşılığında hiçbir kanıt almadan satmak olurdu.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from gauge_vision.config import load_gauges
from gauge_vision.pipeline import KIRPIM_PAYI, read_gauge
from gauge_vision.read import digital as D

UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp")
METRIK_YOLU = "outputs/metrics/ip8_dijital_tani.json"
FIGUR_YOLU = "outputs/figures/ip8_dijital_tani.png"
COK_SINIF_AGIRLIK = Path("runs/detect/models/ip5/cok_sinif/weights/best.pt")


# ----------------------------------------------------- elenen dönüşümler ----
#
# Kodda DEĞİL burada duruyorlar: hiçbiri okuma yoluna girmedi. Amaç bir sonraki
# denemenin sıfırdan başlamaması — "bunu denedim mi?" sorusunun cevabı ölçülü.

def _duzlestir_gauss(bgr: np.ndarray) -> np.ndarray:
    gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(3.0, min(gri.shape) / 12.0)
    zemin = cv2.GaussianBlur(gri.astype(np.float32), (0, 0), k)
    return np.clip(gri - zemin + float(zemin.mean()), 0, 255).astype(np.uint8)


def _zemin_sutun(bgr: np.ndarray) -> np.ndarray:
    """Zemin kestirimi sütun başına: rakamlar bir sütunu baştan sona doldurmaz,
    dolayısıyla düşük yüzdelik o sütunun zeminidir."""
    gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    p = np.percentile(gri, 15, axis=0)
    p = cv2.GaussianBlur(p.reshape(1, -1), (0, 0), max(1.0, len(p) / 40)).ravel()
    return np.clip(gri - p[None, :] + float(p.mean()), 0, 255).astype(np.uint8)


def _renklilik(bgr: np.ndarray) -> np.ndarray:
    """Segment renkli (yeşil/kırmızı/amber), parlama renksiz — kanal açıklığı
    ikisini ayırır. LCD gibi renksiz panellerde bu ayrım YOKTUR."""
    f = bgr.astype(np.float32)
    c = f.max(axis=2) - f.min(axis=2)
    return cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


DONUSUMLER = {
    "ham": lambda b: cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
    "duzlestir_gauss": _duzlestir_gauss,
    "zemin_sutun": _zemin_sutun,
    "renklilik": _renklilik,
}


def _panel_kirpimi(img, model, conf: float):
    """Tespit kutusu + zincirin kırpım payı — okuma yolunun gördüğü bölge."""
    sonuc = model.predict(img, conf=conf, verbose=False)[0].boxes
    if len(sonuc) == 0:
        return None, 0.0
    i = int(np.argmax(sonuc.conf.cpu().numpy()))
    x1, y1, x2, y2 = (float(v) for v in sonuc.xyxy[i])
    dw, dh = (x2 - x1) * KIRPIM_PAYI, (y2 - y1) * KIRPIM_PAYI
    h, w = img.shape[:2]
    kutu = (max(0, int(x1 - dw)), max(0, int(y1 - dh)),
            min(w, int(x2 + dw)), min(h, int(y2 + dh)))
    return kutu, float(sonuc.conf[i])


def _coz(gri: np.ndarray, adet: int) -> dict:
    """Hane bulma + desen çözümünü adım adım açar."""
    kutular = D._haneleri_bul(gri, adet)
    egim = D._egim_kestir(D._segment_maskesi(gri), kutular)
    sonuk, yanik = D._panel_seviyeleri(gri)

    haneler = []
    for k in kutular:
        parlaklik = D._segment_parlakliklari(gri, k, egim)
        ch, guven = D._haneyi_coz(parlaklik, sonuk, yanik)
        esik = (sonuk + yanik) / 2.0
        haneler.append({
            "kutu": [int(v) for v in k],
            "en": int(k[2] - k[0]), "boy": int(k[3] - k[1]),
            "karakter": ch, "guven": round(guven, 3),
            "yanik": "".join(s for s in "abcdefg" if parlaklik[s] > esik),
        })

    tam = all(h["karakter"] is not None for h in haneler) and len(haneler) == adet
    return {
        "hane_sayisi": len(haneler),
        "egim": round(float(egim), 3),
        "sonuk_ref": round(float(sonuk), 1), "yanik_ref": round(float(yanik), 1),
        "haneler": haneler,
        "dizge": (D._noktayi_yerlestir([h["karakter"] for h in haneler], 1)
                  if tam else None),
        "guven": round(min((h["guven"] for h in haneler), default=0.0), 3),
    }


def _zemin_gradyani(gri: np.ndarray) -> dict:
    """Küresel eşiğin dayandığı varsayımın ölçüsü: zemin panel boyunca sabit mi?"""
    w = gri.shape[1]
    sol = float(np.median(gri[:, : w // 3]))
    sag = float(np.median(gri[:, 2 * w // 3:]))
    return {"sol_medyan": round(sol, 1), "sag_medyan": round(sag, 1),
            "oran": round(sol / max(sag, 1.0), 2)}


def figur_yaz(kayitlar: list[dict], yol: Path) -> None:
    """Maske + bulunan hane kutuları. Sayı tablosu neyin çöktüğünü söyler,
    figür NEDEN çöktüğünü gösterir — sönük segmentler maskede yok."""
    satirlar = []
    for k in kayitlar:
        gri, kutular = k.pop("_gri"), k.pop("_kutular")
        goster = cv2.cvtColor(gri, cv2.COLOR_GRAY2BGR)
        goster[D._segment_maskesi(gri) > 0] = (0, 0, 255)
        for kutu in kutular:
            cv2.rectangle(goster, kutu[:2], kutu[2:], (0, 255, 255), 3)
        cv2.putText(goster, f"#{k['sira']:02d} bek {k['beklenen']}", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        satirlar.append(cv2.resize(goster, (900, 380)))
    yol.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(yol), np.vstack(satirlar))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fotograflar", type=Path, default=Path("data/real/ip8_ekran"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("outputs/metrics/ip8_ekran_manifest.json"))
    ap.add_argument("--agirlik", type=Path, default=COK_SINIF_AGIRLIK)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--metrik", type=Path, default=Path(METRIK_YOLU))
    ap.add_argument("--figur", type=Path, default=Path(FIGUR_YOLU))
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    kayitlar = [k for k in manifest.get("kareler", manifest.get("kayitlar", []))
                if k.get("type") == "digital"]
    if not kayitlar:
        print(f"HATA: {args.manifest} içinde dijital kare yok.")
        return 1

    from ultralytics import YOLO
    model = YOLO(str(args.agirlik))
    gauges = load_gauges()

    print(f"{len(kayitlar)} dijital kare · tespit: {args.agirlik}\n")
    cikti, figur_girdisi = [], []
    for kayit in kayitlar:
        yol = next((p for p in args.fotograflar.iterdir()
                    if p.suffix.lower() in UZANTILAR
                    and p.stem.lstrip("0") == str(kayit["sira"])), None)
        if yol is None:
            print(f"#{kayit['sira']:02d} fotoğraf yok, atlandı")
            continue

        img = cv2.imread(str(yol))
        gauge = gauges[kayit["gauge_id"]]
        adet = int((gauge.digits or {}).get("count", 4))

        kutu, tespit_guveni = _panel_kirpimi(img, model, args.conf)
        if kutu is None:
            cikti.append({"sira": kayit["sira"], "tespit_guveni": 0.0,
                          "coken_adim": "tespit"})
            print(f"#{kayit['sira']:02d} TESPİT yok")
            continue

        kirpim = img[kutu[1]:kutu[3], kutu[0]:kutu[2]]
        zincir = read_gauge(img, model, gauge, detect_conf=args.conf).reading

        denemeler = {ad: _coz(fn(kirpim), adet) for ad, fn in DONUSUMLER.items()}
        ham = denemeler["ham"]
        cozulen = [h["karakter"] for h in ham["haneler"] if h["karakter"] is not None]

        kayit_cikti = {
            "sira": kayit["sira"], "beklenen": kayit["text"],
            "tespit_guveni": round(tespit_guveni, 3),
            "zincir_durumu": zincir.status,
            "zincir_guveni": round(float(zincir.conf), 3),
            "zemin_gradyani": _zemin_gradyani(cv2.cvtColor(kirpim, cv2.COLOR_BGR2GRAY)),
            # Ayrım burada kuruluyor: tespit sağlamsa ve çözülen haneler
            # doğruysa, çöken adım hane KUTUSUDUR — desen tablosu değil.
            "coken_adim": ("hane_kutusu" if tespit_guveni >= 0.5 and cozulen
                           else "tespit" if tespit_guveni < 0.5 else "desen"),
            "denemeler": denemeler,
        }
        cikti.append(kayit_cikti)
        figur_girdisi.append({
            "sira": kayit["sira"], "beklenen": kayit["text"],
            "_gri": cv2.cvtColor(kirpim, cv2.COLOR_BGR2GRAY),
            "_kutular": [h["kutu"] for h in ham["haneler"]],
        })

        print(f"#{kayit['sira']:02d} bek {kayit['text']:>7s} · tespit "
              f"{tespit_guveni:.3f} · zincir {zincir.status} ({zincir.conf:.3f}) · "
              f"zemin sol/sağ {kayit_cikti['zemin_gradyani']['oran']}×")
        for ad, d in denemeler.items():
            print(f"     {ad:16s} {str(d['dizge'] or '—'):>8s}  guven {d['guven']:.2f}  "
                  f"{[(h['karakter'], h['guven']) for h in d['haneler']]}")

    if figur_girdisi:
        figur_yaz(figur_girdisi, args.figur)

    ozet = {
        "kaynak": str(args.fotograflar),
        "agirlik": str(args.agirlik),
        "kare_sayisi": len(cikti),
        "tespit_ortalama": round(float(np.mean([k["tespit_guveni"] for k in cikti])), 3),
        "coken_adim_dagilimi": {a: sum(1 for k in cikti if k.get("coken_adim") == a)
                                for a in ("tespit", "hane_kutusu", "desen")},
        "sessiz_hata": sum(1 for k in cikti if k.get("zincir_durumu") == "ok"),
        "kareler": cikti,
    }
    args.metrik.parent.mkdir(parents=True, exist_ok=True)
    args.metrik.write_text(json.dumps(ozet, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print(f"\nTespit ortalaması : {ozet['tespit_ortalama']:.3f}")
    print(f"Çöken adım        : {ozet['coken_adim_dagilimi']}")
    print(f"Sessiz hata       : {ozet['sessiz_hata']} "
          f"(zincir yanlış sayı yayınlamadı → başarısızlık güvenli tarafta)")
    print(f"\n{args.metrik}\n{args.figur}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
