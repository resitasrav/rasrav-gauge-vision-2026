"""Birleşik zincirin tek pencerelik görüntüsü.

Üç panel yok: TEK kare, üstünde üç modülün katmanları ve sağda ortak durum
şeridi. Katman sırası bilinçli — üstteki alttakini örtebilir, bu yüzden en
önemli olan en üstte:

  1. ALGILAMA kutuları (yeşil, ince)     — bağlam
  2. GÖSTERGE kutuları + ibre geometrisi — okunan şeyler
  3. AÇIKLANMIŞ anomali (gri, kesikli)   — "beklenen hareket", bastırılmış
  4. ALARM (kırmızı, kalın)              — hiçbir modülün açıklayamadığı

Bir anomali kutusu gri çizildiyse bu bir bastırma DEĞİL saklama değildir:
kutusu ekranda durur ve neden bastırıldığı yanına yazılır. Sessizce
düşürülseydi demoya bakan kişi modülün onu hiç görmediğini sanardı.
"""
from __future__ import annotations

import cv2
import numpy as np

from birlesik import HUD_EN, OLAY_SATIRI, RENK, RENK_ALARM, RENK_ALGILAMA, RENK_BEKLENEN

YAZI = cv2.FONT_HERSHEY_SIMPLEX
HUD_BG = (28, 28, 28)
HUD_YAZI = (235, 235, 235)
HUD_SOLUK = (150, 150, 150)


def _kesikli_dikdortgen(img, p1, p2, renk, kalinlik=2, adim=12):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, adim * 2):
        cv2.line(img, (x, y1), (min(x + adim, x2), y1), renk, kalinlik)
        cv2.line(img, (x, y2), (min(x + adim, x2), y2), renk, kalinlik)
    for y in range(y1, y2, adim * 2):
        cv2.line(img, (x1, y), (x1, min(y + adim, y2)), renk, kalinlik)
        cv2.line(img, (x2, y), (x2, min(y + adim, y2)), renk, kalinlik)


def kareyi_ciz(kare, tespitler, okumalar, algilama, nesneler) -> np.ndarray:
    ciz = kare.copy()

    for a in algilama:
        x1, y1, x2, y2 = (int(v) for v in a["kutu"])
        cv2.rectangle(ciz, (x1, y1), (x2, y2), RENK_ALGILAMA, 1)
        cv2.putText(ciz, f"{a['sinif']}#{a['id']}", (x1 + 2, max(y1 - 5, 12)),
                    YAZI, 0.45, RENK_ALGILAMA, 1, cv2.LINE_AA)

    for t in tespitler:
        x1, y1, x2, y2 = (int(v) for v in t.box_xyxy)
        renk = RENK.get(t.sinif, (180, 180, 180))
        cv2.rectangle(ciz, (x1, y1), (x2, y2), renk, 2)
        cv2.putText(ciz, f"{t.sinif} {t.conf:.2f}", (x1 + 3, max(y1 - 6, 14)),
                    YAZI, 0.5, renk, 2, cv2.LINE_AA)
    for o in okumalar:
        if not o.ok:
            continue
        cv2.circle(ciz, o.center_px, int(o.radius_px), RENK["gauge"], 2, cv2.LINE_AA)
        cv2.line(ciz, o.center_px, o.needle.tip_px, RENK["gauge"], 2, cv2.LINE_AA)
        cv2.circle(ciz, o.center_px, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(ciz, f"aci {o.needle.angle_img_deg:.0f}",
                    (o.center_px[0] - 30, o.center_px[1] + int(o.radius_px) + 18),
                    YAZI, 0.5, RENK["gauge"], 2, cv2.LINE_AA)

    for n in nesneler:
        x1, y1, x2, y2 = (int(v) for v in n["kutu"])
        if n["alarm"]:
            cv2.rectangle(ciz, (x1, y1), (x2, y2), RENK_ALARM, 3)
            cv2.putText(ciz, "TANIMSIZ NESNE", (x1, max(y1 - 8, 14)),
                        YAZI, 0.55, RENK_ALARM, 2, cv2.LINE_AA)
        else:
            # Bastirilan kutu da GORUNUR kalir, sebebiyle birlikte.
            _kesikli_dikdortgen(ciz, (x1, y1), (x2, y2), RENK_BEKLENEN, 1)
            cv2.putText(ciz, n["aciklama"], (x1, max(y1 - 6, 12)),
                        YAZI, 0.45, RENK_BEKLENEN, 1, cv2.LINE_AA)
    return ciz


def _satir(tuval, y, metin, renk=HUD_YAZI, boyut=0.48, kalin=1):
    cv2.putText(tuval, metin, (14, y), YAZI, boyut, renk, kalin, cv2.LINE_AA)
    return y + int(22 * boyut / 0.48)


def hud_ciz(yukseklik: int, s, olay_gecmisi, fps: float, video_ad: str) -> np.ndarray:
    t = np.full((yukseklik, HUD_EN, 3), HUD_BG, np.uint8)
    y = 30
    y = _satir(t, y, video_ad[:34], (255, 255, 255), 0.58, 2)
    y = _satir(t, y + 4, f"kare {s.kare}  ·  {s.saniye:5.1f} sn  ·  {fps:4.1f} FPS", HUD_SOLUK)
    kx1, ky1, kx2, ky2 = s.kirpim
    y = _satir(t, y, f"kirpim {kx2-kx1}x{ky2-ky1} @({kx1},{ky1})", HUD_SOLUK)

    y += 14
    cv2.line(t, (14, y), (HUD_EN - 14, y), (70, 70, 70), 1)
    y += 24
    y = _satir(t, y, "GOSTERGE (Resit)", RENK["gauge"], 0.52, 2)
    tespit = s.gosterge["tespit"]
    y = _satir(t, y, "  " + (", ".join(f"{k}:{v}" for k, v in tespit.items()) or "tespit yok"))
    y = _satir(t, y, f"  analog okuma {s.gosterge['analog_okunan']}/{s.gosterge['analog_kutu']}")

    y += 12
    y = _satir(t, y, "ALGILAMA (Bedirhan)", RENK_ALGILAMA, 0.52, 2)
    y = _satir(t, y, f"  {s.algilama['kutu']} kutu · " +
               (", ".join(s.algilama["siniflar"][:4]) or "hedef yok"))

    y += 12
    y = _satir(t, y, "ANOMALI (Ozgur)", (90, 140, 255), 0.52, 2)
    an = s.anomali
    y = _satir(t, y, f"  on plan {an['nesne']} · aciklanan {an['aciklanan']}")
    y = _satir(t, y, f"  ALARM {an['alarm']}",
               RENK_ALARM if an["alarm"] else (0, 200, 0), 0.52, 2)
    if an["donme"]:
        y = _satir(t, y, "  donme - tespit askida", (0, 165, 255))

    y += 14
    cv2.line(t, (14, y), (HUD_EN - 14, y), (70, 70, 70), 1)
    y += 22
    y = _satir(t, y, "ortak olay akisi", HUD_SOLUK, 0.5, 1)
    for o in list(olay_gecmisi)[-OLAY_SATIRI:]:
        renk = {"GOSTERGE": RENK["gauge"], "ALGILAMA": RENK_ALGILAMA,
                "ANOMALI": (90, 140, 255)}.get(o.kaynak, HUD_SOLUK)
        y = _satir(t, y, f"{o.saniye:5.1f} {o.kaynak[:4]} {o.metin}"[:44], renk, 0.44)
    return t


def birlestir(cizili: np.ndarray, hud: np.ndarray,
              goruntu_en: int, goruntu_boy: int) -> np.ndarray:
    """Çizili kareyi SABİT boyutlu bir tuvale oturtup HUD ile birleştirir.

    Sol panel neden sabit boyutlu: kırpma geometrisi video ortasında değişiyor
    (dikey klip → tam genişlik). İlk sürüm pencereyi kırpmanın en-boyuna göre
    kuruyordu ve pencere boyutu kırpmayla birlikte değişiyordu.

    `cv2.VideoWriter` açıldığı boyuttan farklı bir kare gelince onu SESSİZCE
    ATAR — hata vermez, `write()` bir şey döndürmez. Sonuç ölçüldü: 2108 karelik
    video işlendi, JSON'a 2108 kare yazıldı, mp4'e yalnız ilk 916 kare girdi
    (kırpmanın değiştiği kare). 70 saniyelik girdi 30 saniyelik çıktı verdi ve
    hiçbir yerde hata görünmedi.

    Çıktı geometrisi girdi kırpmasına BAĞLI OLMAMALI. Kare artık oranı
    korunarak sabit tuvale oturtuluyor; kenarlarda siyah kalması normaldir.
    """
    h, w = cizili.shape[:2]
    olcek = min(goruntu_en / w, goruntu_boy / h)
    yeni_en, yeni_boy = max(int(w * olcek), 1), max(int(h * olcek), 1)
    kucuk = cv2.resize(cizili, (yeni_en, yeni_boy), interpolation=cv2.INTER_AREA)

    tuval = np.zeros((goruntu_boy, goruntu_en, 3), np.uint8)
    x0, y0 = (goruntu_en - yeni_en) // 2, (goruntu_boy - yeni_boy) // 2
    tuval[y0:y0 + yeni_boy, x0:x0 + yeni_en] = kucuk
    return np.hstack([tuval, hud])
