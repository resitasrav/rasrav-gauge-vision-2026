"""demo/anomali_demo.py — ANOMALİ paneli için demo-tarafı sarmalayıcı (PaDiM).

⚠ **Bu Özgür'ün modülü DEĞİLDİR.** `OzgurKotbas_Akilli_Fabrika/anomali_test.py`
çalıştırılmıyor, import da edilmiyor, değiştirilmiyor. Neden çağrılamadığı
`demo/uyusmazliklar/RAPOR.md` madde 1-2'de yazılı ve hâlâ geçerli:

  * dosya bir EĞİTİM scripti (`Engine.fit` + `Engine.predict`), tek kare alan
    bir fonksiyonu yok;
  * kaydedilmiş bir ağırlık (`.ckpt`) yok, yani her çağrı sıfırdan eğitim;
  * eğitim verisi MVTec-AD `bottle` — fabrika koridoruyla konu bakımından
    ilgisiz.

ALGILAMA panelinde uygulanan çözümün aynısı buraya da uygulandı: modülün
YÖNTEMİ demo tarafında tekrar çalıştırılıyor. Fark şu ve dürüstçe yazılmalı —
ALGILAMA'da aynı KÜTÜPHANE çağrılabiliyordu (`ultralytics`), burada
`anomalib` bu sanal ortamda kurulu değil ve kurmak çalışan `torch 2.13+cu126`
kurulumunu riske atıyor. Bu yüzden PaDiM'in kendisi (Defard ve ark., 2020)
torchvision ResNet18 üstünde uygulandı: kütüphane bir kolaylık, yöntem şu üç
adımdır ve üçü de burada aynen var.

  1. Önceden eğitilmiş CNN'in ara katman haritaları birleştirilir (layer1+2+3).
  2. Her yama konumu için normal veriden çok değişkenli Gauss (ortalama +
     kovaryans) çıkarılır.
  3. Yeni karede her yamanın Mahalanobis uzaklığı = anomali haritası.

**"Normal" ne sayılıyor?** MVTec `bottle` bu videolarla ilgisiz olduğu için
referans, VİDEONUN KENDİ ilk kareleridir. Yani panel "bu videonun başına göre
ne değişti" sorusunu cevaplıyor — devriye senaryosunda doğru soru budur
(aynı durak, aynı çerçeve, zamanla değişen sahne). Bunun bir VARSAYIM olduğu
panelde de yazıyor; sahnenin başı zaten anormalse ölçüm yanıltır.

Eşik ölçülüyor, tahmin edilmiyor (depo kuralı): referans bölge ikiye bölünür,
kovaryans birinci yarıdan çıkar, eşik uyuma GİRMEYEN ikinci yarının p99'undan
alınır. Bunun neden önemli olduğu `uyumla`'nın docstring'inde ölçümüyle yazılı.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# PaDiM'in özgün kurulumu: rastgele seçilmiş boyut alt kümesi (makale 100
# boyutun yettiğini ölçüyor; tam boyut 448 hem kovaryansı tekilleştirir hem
# yavaşlatır).
#
# ÖLÇÜLDÜ (27.08, ozgur.mp4 koridor çekimi) — ilk sürümün iki kusuru:
#
# 1. BOYUT 64 iken uyum kümesi de 64 kareydi. Kovaryans o zaman tekil olur ve
#    her uyum karesi kendi dağılımına TAM oturur: Mahalanobis ≈ sqrt(D) = 8.
#    17 videonun 17'sinde de eşik 7,7-7,9 çıktı — birbirinden tamamen farklı
#    videolarda aynı sayı. Düz çıkan tablo sonuç değil uyarıdır.
# 2. Eşik uyum kümesinin KENDİ skorlarından alınıyordu. O skorlar tanım gereği
#    en düşüktür; aynı sahnenin uyumda yer almayan karesi 45 kat yüksek
#    (medyan 7,46 → 337,74). Yani eşik yanlış popülasyonda ölçülmüştü —
#    bu deponun kendi kuralı: eşik İKİ kümenin dağılımı ölçülüp aralarına konur.
#
# Tarama (ayrık normal = aynı sahne, uyumda yok · uzak = koridorun ilerisi):
#
#   D    N     ic-orneklem   ayrik normal   uzak    ayrim
#   64   64        7,46         137,26     345,72   2,52x
#   64  240       11,64          84,63     165,66   1,96x
#   32  240        9,89          67,16     136,87   2,04x
#    8  240        7,92          20,07      89,97   4,48x   <-- secilen
#
# Boyut düştükçe ayrım artıyor: yüksek boyutta uzaklık gürültüyle şişiyor ve
# normal ile anormal birlikte yükseliyor.
BOYUT = 8
UYUM_ORAN = 0.25        # referans bölge: videonun ilk %25'i
UYUM_TAVAN = 240        # ondan fazlası ölçümde kazanç getirmedi, süre getirdi
UYUM_TABAN = 40         # bunun altında kovaryans anlamsız — raporda işaretlenir
KALIBRASYON_ORAN = 0.2  # referansın bu kadarı uyuma GİRMEZ, eşik onda ölçülür
GIRDI = 256             # ResNet girdisi (kare)
DUZENLEME = 0.01        # kovaryans köşegenine eklenen pay (tekillik koruması)
ESIK_YUZDELIK = 99.0    # AYRIK normal kümenin bu yüzdeliği "normalin tavanı"


class _Cikarici:
    """ResNet18'in layer1/2/3 haritalarını tek tensöre birleştirir."""

    def __init__(self, cihaz: str):
        from torchvision.models import ResNet18_Weights, resnet18
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(cihaz).eval()
        self.cihaz = cihaz
        # Boyut alt kümesi TOHUMLU: aynı video iki kez koşturulduğunda aynı
        # sayı çıkmalı, yoksa "anomali skoru" tekrar üretilemez olur.
        g = torch.Generator().manual_seed(1)
        self.idx = torch.randperm(64 + 128 + 256, generator=g)[:BOYUT].to(cihaz)
        self.ort = torch.tensor([0.485, 0.456, 0.406], device=cihaz).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=cihaz).view(1, 3, 1, 1)

    def hazirla(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(cv2.resize(bgr, (GIRDI, GIRDI)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).to(self.cihaz).permute(2, 0, 1).float().div_(255.0)
        return (t.unsqueeze(0) - self.ort) / self.std

    @torch.no_grad()
    def __call__(self, bgr: np.ndarray) -> torch.Tensor:
        x = self.hazirla(bgr)
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        h1 = n.layer1(x)
        h2 = n.layer2(h1)
        h3 = n.layer3(h2)
        # Üst katmanlar layer1 ızgarasına büyütülür (PaDiM'in "embedding
        # concatenation" adımı) — konum bilgisi layer1'in çözünürlüğünde tutulur.
        boy = h1.shape[-2:]
        h = torch.cat([h1,
                       F.interpolate(h2, size=boy, mode="nearest"),
                       F.interpolate(h3, size=boy, mode="nearest")], dim=1)
        return h[:, self.idx]            # (1, D, H, W)


@dataclass
class AnomaliModeli:
    ortalama: torch.Tensor      # (P, D)
    ters_kov: torch.Tensor      # (P, D, D)
    izgara: tuple[int, int]     # (h, w) yama ızgarası
    cikarici: _Cikarici
    uyum_kare_sayisi: int
    kalibrasyon_kare_sayisi: int
    esik: float = 0.0
    uyum_skorlari: list[float] = field(default_factory=list)
    kalibrasyon_skorlari: list[float] = field(default_factory=list)
    zayif_referans: bool = False


@torch.no_grad()
def _harita(model: AnomaliModeli, bgr: np.ndarray):
    E = model.cikarici(bgr)                          # (1, D, H, W)
    D = E.shape[1]
    H, W = model.izgara
    v = E.permute(0, 2, 3, 1).reshape(H * W, D) - model.ortalama
    m2 = torch.einsum("pi,pij,pj->p", v, model.ters_kov, v).clamp_min_(0)
    harita = m2.sqrt().reshape(H, W)
    return harita, float(harita.max())


def uyumla(referans: list[np.ndarray], cihaz: str | None = None) -> AnomaliModeli:
    """Referans karelerden yama başına Gauss çıkarır ve eşiği kalibre eder.

    Referans İKİYE bölünür ve bu bölme yöntemin can alıcı noktasıdır:

      * **uyum kümesi** — kovaryans buradan çıkar;
      * **kalibrasyon kümesi** — uyuma GİRMEZ, eşik burada ölçülür.

    Eşiği uyum kümesinde ölçmek ilk sürümün hatasıydı: o kareler tanım gereği
    kendi dağılımının merkezindedir ve skorları en düşüktür. Aynı sahnenin
    uyumda yer almayan bir karesi 45 kat yüksek çıkıyordu (7,46 → 337,74),
    yani eşik "normal" diye ölçtüğü şeyin ne olduğunu bilmiyordu. Kalibrasyon
    kümesi de normaldir ama uyumda YOKTUR — aranan popülasyon budur.
    """
    cihaz = cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
    kalib_n = max(int(len(referans) * KALIBRASYON_ORAN), 1)
    uyum, kalibrasyon = referans[:-kalib_n], referans[-kalib_n:]
    if not uyum:                       # çok kısa video: bölünemiyor
        uyum, kalibrasyon = referans, referans

    cik = _Cikarici(cihaz)
    E = torch.cat([cik(k) for k in uyum], 0)         # (N, D, H, W)
    N, D, H, W = E.shape
    E = E.permute(0, 2, 3, 1).reshape(N, H * W, D)   # (N, P, D)

    ortalama = E.mean(0)                             # (P, D)
    ort_cikmis = (E - ortalama).double()              # kovaryans çift duyarlıkta
    # Yama başına kovaryans: (P, D, D). einsum tek seferde, döngü yok.
    kov = torch.einsum("npi,npj->pij", ort_cikmis, ort_cikmis) / max(N - 1, 1)

    # DÜZENLEME PAYI GLOBAL İZDEN ÖLÇEKLENİR, yamanın kendi izinden DEĞİL.
    # İlk sürüm yamanın kendi izini kullanıyordu ve sabit kamerada çöküyordu:
    # hiç değişmeyen bir yamada iz 0'dır, dolayısıyla pay da 0 olur ve matris
    # tekil kalır. 27.08'de dört videoda (karasel, 6, 5s, 10) linalg.inv tam
    # olarak bunu söyleyerek patladı. Global iz her yamaya bir TABAN verir;
    # hiç değişmemiş bir yama artık sonsuz değil, BÜYÜK ama sonlu uzaklık
    # üretir — ki doğrusu budur: referansta hiç oynamamış bir bölge oynarsa
    # bu gerçekten güçlü bir anomali kanıtıdır.
    iz_global = torch.diagonal(kov, dim1=1, dim2=2).mean()
    birim = torch.eye(D, device=cihaz, dtype=kov.dtype).unsqueeze(0)
    kov = kov + DUZENLEME * iz_global * birim

    try:
        # Cholesky, inv'den hem daha kararlı hem daha hızlı (kovaryans simetrik
        # pozitif tanımlı olmalı); başarısız olması matrisin gerçekten bozuk
        # olduğunu söyler ve bunu YUTMAK sessiz hata olurdu.
        L = torch.linalg.cholesky(kov)
        ters = torch.cholesky_inverse(L).float()
    except RuntimeError as e:
        raise RuntimeError(
            f"kovaryans tersi alinamadi ({N} uyum karesi, D={D}): {e}. "
            "Referans bolgesi cok tekduze olabilir.") from e

    model = AnomaliModeli(ortalama, ters, (H, W), cik,
                          len(uyum), len(kalibrasyon))
    model.uyum_skorlari = [float(_harita(model, k)[1]) for k in uyum]
    model.kalibrasyon_skorlari = [float(_harita(model, k)[1]) for k in kalibrasyon]
    model.esik = float(np.percentile(model.kalibrasyon_skorlari, ESIK_YUZDELIK))
    # Kovaryans örnek sayısı boyut sayısına yaklaşırsa tekilleşir ve bütün
    # skorlar sqrt(D) civarına çöker. Raporda görünsün diye işaretleniyor.
    model.zayif_referans = len(uyum) < UYUM_TABAN
    return model


def anomali_isle(bgr: np.ndarray, model: AnomaliModeli) -> tuple[np.ndarray, dict]:
    """Kareyi puanlar ve ısı haritası bindirilmiş görüntüyü döndürür."""
    harita, skor = _harita(model, bgr)
    h, w = bgr.shape[:2]
    hm = F.interpolate(harita[None, None], size=(h, w), mode="bilinear",
                       align_corners=False)[0, 0].cpu().numpy()

    # Görselleştirme EŞİĞE göre normalize edilir, karenin kendi min-max'ına
    # göre değil: min-max normalizasyon tamamen normal bir kareyi de rengarenk
    # gösterir ve bakan kişi her karede anomali sanır.
    norm = np.clip(hm / max(model.esik, 1e-6), 0.0, 2.0) / 2.0
    isi = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cizili = cv2.addWeighted(bgr, 0.6, isi, 0.4, 0)

    anomali = skor > model.esik
    oran = skor / max(model.esik, 1e-6)
    renk = (0, 0, 220) if anomali else (0, 200, 0)
    etiket = "ANOMALI" if anomali else "normal"
    cv2.putText(cizili, f"{etiket}  skor {skor:.1f} / esik {model.esik:.1f} (x{oran:.2f})",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, renk, 2, cv2.LINE_AA)
    not_ = (f"referans: ilk {model.uyum_kare_sayisi} kare"
            f" (+{model.kalibrasyon_kare_sayisi} kalibrasyon)")
    if model.zayif_referans:
        not_ += "  ZAYIF"
    cv2.putText(cizili, not_, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    return cizili, {"skor": round(skor, 2), "esik": round(model.esik, 2),
                    "oran": round(oran, 3), "anomali": bool(anomali)}


def uyum_kareleri(video_yolu: str) -> list[np.ndarray]:
    """Videonun BAŞINDAN referans kareler ('normal' kabul edilen bölge).

    Sabit bir kare sayısı değil, videonun ilk %25'i alınıyor (tavanı
    `UYUM_TAVAN`): kısa bir videoda 240 kare zaten videonun tamamı olurdu ve
    o zaman "referans" ile "ölçülen" aynı şey olur, ölçüm anlamsızlaşır.
    """
    cap = cv2.VideoCapture(video_yolu)
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    hedef = min(UYUM_TAVAN, max(int(toplam * UYUM_ORAN), 1)) if toplam else UYUM_TAVAN
    kareler = []
    while len(kareler) < hedef:
        ok, k = cap.read()
        if not ok:
            break
        kareler.append(k)
    cap.release()
    if not kareler:
        raise RuntimeError(f"referans karesi okunamadi: {video_yolu}")
    return kareler
