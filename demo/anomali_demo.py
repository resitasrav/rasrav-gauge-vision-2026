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

Eşik ölçülüyor, tahmin edilmiyor (depo kuralı): uyum kümesinin kendi skor
dağılımının p99'u alınır. Altındaki kareler "normal", üstündekiler "anomali".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# PaDiM'in özgün kurulumu: rastgele seçilmiş boyut alt kümesi. Tam boyut (448)
# hem kovaryansı tekilleştirir hem yavaşlatır; makale 100 boyutun yettiğini
# ölçüyor. Burada 64 seçildi çünkü uyum kümesi 64 kare — kovaryansın kararlı
# olması için örnek sayısı boyut sayısından KÜÇÜK OLMAMALI.
BOYUT = 64
UYUM_KARE = 64          # "normal" kabul edilen kare sayısı (videonun başı)
GIRDI = 256             # ResNet girdisi (kare)
DUZENLEME = 0.01        # kovaryans köşegenine eklenen pay (tekillik koruması)
ESIK_YUZDELIK = 99.0    # uyum kümesinin bu yüzdeliği "normalin tavanı"


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
    esik: float = 0.0
    uyum_skorlari: list[float] = field(default_factory=list)


@torch.no_grad()
def _harita(model: AnomaliModeli, bgr: np.ndarray):
    E = model.cikarici(bgr)                          # (1, D, H, W)
    D = E.shape[1]
    H, W = model.izgara
    v = E.permute(0, 2, 3, 1).reshape(H * W, D) - model.ortalama
    m2 = torch.einsum("pi,pij,pj->p", v, model.ters_kov, v).clamp_min_(0)
    harita = m2.sqrt().reshape(H, W)
    return harita, float(harita.max())


def uyumla(kareler: list[np.ndarray], cihaz: str | None = None) -> AnomaliModeli:
    """Verilen 'normal' karelerden yama başına Gauss çıkarır."""
    cihaz = cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
    cik = _Cikarici(cihaz)
    E = torch.cat([cik(k) for k in kareler], 0)      # (N, D, H, W)
    N, D, H, W = E.shape
    E = E.permute(0, 2, 3, 1).reshape(N, H * W, D)   # (N, P, D)

    ortalama = E.mean(0)                             # (P, D)
    ort_cikmis = E - ortalama
    # Yama başına kovaryans: (P, D, D). einsum tek seferde, döngü yok.
    kov = torch.einsum("npi,npj->pij", ort_cikmis, ort_cikmis) / max(N - 1, 1)
    iz = torch.diagonal(kov, dim1=1, dim2=2).mean(1).view(-1, 1, 1)
    kov = kov + DUZENLEME * iz * torch.eye(D, device=cihaz).unsqueeze(0)

    model = AnomaliModeli(ortalama, torch.linalg.inv(kov), (H, W), cik, len(kareler))
    # Eşik uyum kümesinin KENDİ skorlarından: bu kareler tanım gereği normal,
    # dolayısıyla "normalin tavanı" onların dağılımıdır. Sabit sayı yazmak
    # depoda üç kez elenen hata sınıfıdır (mutlak eşik).
    model.uyum_skorlari = [float(_harita(model, k)[1]) for k in kareler]
    model.esik = float(np.percentile(model.uyum_skorlari, ESIK_YUZDELIK))
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
    cv2.putText(cizili, f"referans: videonun ilk {model.uyum_kare_sayisi} karesi",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return cizili, {"skor": round(skor, 2), "esik": round(model.esik, 2),
                    "oran": round(oran, 3), "anomali": bool(anomali)}


def uyum_kareleri(video_yolu: str, sayi: int = UYUM_KARE) -> list[np.ndarray]:
    """Videonun BAŞINDAN ardışık kareler — 'normal' referans kümesi."""
    cap = cv2.VideoCapture(video_yolu)
    kareler = []
    while len(kareler) < sayi:
        ok, k = cap.read()
        if not ok:
            break
        kareler.append(k)
    cap.release()
    if not kareler:
        raise RuntimeError(f"uyum karesi okunamadi: {video_yolu}")
    return kareler
