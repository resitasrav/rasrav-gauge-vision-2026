# Açık Veri Seti Değerlendirmesi (İP1)

**Hazırlayan:** Reşit Asrav · **Tarih:** 31.07.2026 · **Modül:** GÖSTERGE

**Amaç:** İP5 (YOLO ile gösterge tespiti) ve İP6 (ibre açısı) için kullanılabilecek açık
veri setlerinin taranması. İP4 bulgusu K1 uyarınca sentetik verinin tek başına
kullanılması uygun olmadığından, tarama bir formalite değil İP5'in ön koşuludur.

**Değerlendirme ölçütü:** Bir setin bizim için değeri, **hangi ground truth'u içerdiğine**
bağlıdır:

| Etiket türü | Neyi mümkün kılar |
|---|---|
| Sınırlayıcı kutu (bbox) | Yalnızca İP5 — göstergenin karede bulunması |
| Keypoint (merkez + ibre ucu) | İP6 — ibre açısının doğrudan hesaplanması |
| Keypoint (+ skala min/max) | İP7 — açının değere çevrilmesi için kadran çapaları |
| Okunan değer / açı | **İP8** — gerçek görüntüde uçtan uca hata ölçümü |

---

## 1. Aday Setler — Öncelik Sırasıyla

| # | Set | Kaynak | Boyut | Etiket | Lisans | Erişim | Hedef İP |
|:--:|---|---|---|---|---|---|---|
| **A1** | **SyntheticGauges** | Cambridge (Howells & Cipolla, CVPRW 2021) | 10.000 eğitim + 1.000 test, 1024×1024 | COCO formatında bbox **+ keypoint**: perspektif noktaları, skala min, skala max, **ibre merkezi, ibre ucu** | CC BY-NC 4.0 | **Google Drive — anahtar gerekmiyor** | İP5, İP6, İP7 |
| **A2** | **RealGauges** | Aynı çalışma | 6 gösterge; her biri için 36 fotoğraf + 3×5 sn video | Tespit: merkez koordinatı · Poz: düzleme normal · **Okuma: ibre değeri ve açısı** | CC BY-NC 4.0 | **Google Drive — anahtar gerekmiyor** | **İP8** |
| **A3** | Analog Meter (Roboflow) | Roboflow Universe | ~7.700 görüntü | Sınıflar: `Center`, `Gauge`, `Max`, `Min`, `Pointer base/end/middle/start/tip` — keypoint niteliğinde | Sette değişken | **Roboflow API anahtarı** | İP5, İP6 |
| **A4** | Synthetic Data for Precision Gauge Reading | Kaggle (Endava) | Belirtilmemiş | Segmentasyon maskesi, keypoint, hesaplanmış değer | Kaggle sayfasında | **Kaggle API token** | İP5, İP6 |
| **A5** | Pressure Gauge Reader Data | Kaggle (Aalborg atıksu pompa istasyonları) | Belirtilmemiş | Gerçek saha görüntüleri/videoları | Kaggle sayfasında | **Kaggle API token** | İP8, İP14 |
| A6 | Detect-and-read-meters | GitHub (shuyansy) | Belirtilmemiş | Tespit (COCO) + tanıma (Labelme): açı, kadran, değer | MIT | Depo bağlantısı | İP5, İP11 |
| A7 | NRC-GAMMA | GitHub (NRC Kanada) | 28.883 tam görüntü + 57.766 kırpım | Rakam kadranı etiketleri | Açık, ticari kullanım dahil | Doğrudan | Sınırlı — rakamlı gaz sayacı, ibreli gösterge değil |
| A8 | SyncG | Nature Sci. Data 2026 | 20.000, 145 ortam | Tespit, keypoint, segmentasyon, OCR | Doğrulanamadı | Yayın sayfası kimlik doğrulama istiyor | İP5, İP6 |

---

## 2. Öne Çıkan Bulgu: A1 + A2 İkilisi

Cambridge çalışmasının yayımladığı ikili, modülün üç ayrı ihtiyacını aynı anda
karşılamaktadır:

- **A1 (SyntheticGauges)** yalnızca gösterge kutusunu değil, **ibre merkezi ve ibre ucu**
  ile **skala min/max** keypoint'lerini de içermektedir. Bu, açı hesabı için gereken tüm
  geometriyi sağlamaktadır; kendi sentetik üretecimizin ürettiği etiketin birebir karşılığıdır.
- **A2 (RealGauges)**, gerçek göstergelerin videolarında **ibre değeri ve açısı etiketli**
  olarak yer almaktadır. Bu, İP8'in (gerçek görüntüde uçtan uca hata ölçümü) şimdiye kadar
  karşılanamayan ground truth ihtiyacını kısmen kapatmaktadır.
- Erişim **Google Drive üzerinden açık** olduğundan API anahtarı gerektirmemektedir.
- Lisans **CC BY-NC 4.0**'tır: atıfla kullanıma ve uyarlamaya izin verir, **ticari kullanımı
  yasaklar**. Staj kapsamında uygundur; kaynak gösterimi zorunludur.

**Yöntem tarafındaki referans değeri:** Aynı çalışma, yalnızca sentetik veriyle eğitilen iki
CNN ile **1 dereceden küçük ibre açısı hatası** ve mobil cihazda 25 fps bildirmektedir.
İP6'nın hedef eşiği belirlenirken bu değer referans alınabilir.

### Açık soru — iki kaynak arasında görünür çelişki

İP4'te SyncG çalışmasından, eğitim kümesinin %100'ü sentetik olduğunda **belirgin domain
gap** oluştuğu kaydedilmişti. Cambridge çalışması ise **yalnızca sentetik veriyle** eğitip
gerçek göstergelerde 1°'nin altında hata bildirmektedir. İki bulgu ilk bakışta çelişmektedir.

Olası açıklama, görevlerin farklı olmasıdır: SyncG bütün sahnede tespit/tanıma ölçmekte,
Cambridge çalışması ise kırpılmış gösterge üzerinde keypoint regresyonu yapmakta ve ayrıca
perspektif noktalarını da sentetik olarak üretip poz düzeltmesi uygulamaktadır. Bu açıklama
şu an bir varsayımdır; her iki çalışmanın tam metni okunmadan kesinleştirilmemelidir.

**Modüle etkisi:** K1 kararı (sentetik veri İP5'te tek başına kullanılmayacak) şimdilik
korunmaktadır; çelişki çözülene kadar temkinli tarafta kalınması tercih edilmiştir.

---

## 3. Planlanan Kullanım

| İş paketi | Kullanılacak veri | Gerekçe |
|---|---|---|
| İP5 — tespit | A1 + A3 + kendi sentetik verimiz (karışık eğitim) | K1 kararı: sentetik tek başına yetersiz |
| İP6 — ibre açısı | Önce kendi sentetik verimiz (yöntem oturtma), sonra A1 (bağımsız doğrulama) | Kendi verimizde ground truth tam kontrolümüzde |
| İP7 — açı→değer | Kendi sentetik verimiz + A1'in skala min/max keypoint'leri | Kadran çapaları gerekli |
| İP8 — gerçek test | **A2** + (varsa) A5 | Gerçek görüntüde etiketli değer yalnızca burada |
| İP11 — dijital OCR | A6 | Analog ve dijital paneli birlikte ele alan tek referans |
| İP14 — zor koşullar | A5 (saha görüntüleri) | Gerçek saha aydınlatması ve kirlilik |

---

## 4. Erişim Durumu ve İhtiyaçlar

| Erişim | Setler | Durum |
|---|---|---|
| Anahtarsız (Google Drive) | A1, A2 | **Hemen indirilebilir** |
| Anahtarsız (GitHub) | A6, A7 | Hemen indirilebilir |
| Kaggle API token | A4, A5 | **Talep edildi** |
| Roboflow API anahtarı | A3 | **Talep edildi** |
| Doğrulanamadı | A8 | Yayın sayfası kimlik doğrulama istiyor |

---

## 5. Açık Kalan Hususlar

- A4 ve A5'in görüntü sayısı, çözünürlüğü ve lisansı Kaggle sayfaları JavaScript ile
  oluşturulduğundan otomatik olarak okunamamıştır; token sağlandığında doğrudan
  meta verisinden teyit edilecektir.
- A1/A2'nin CC BY-NC lisansı ticari kullanımı kısıtlamaktadır. Staj sonrasında proje
  ticari bir ürüne dönüşürse bu setlerle eğitilen ağırlıklar kullanılamaz; kayıt altına
  alınmıştır.
- A7 (NRC-GAMMA) rakam kadranlı gaz sayaçlarından oluşmaktadır; ibreli gösterge okuma
  görevine doğrudan katkısı sınırlıdır. Yalnızca İP11'e dolaylı fayda sağlayabilir.

---

## 6. Kaynaklar

- [Real-time analogue gauge transcription on mobile phone](http://jjcvision.com/projects/gauge_reading.html) — Howells & Cipolla, CVPRW 2021 · SyntheticGauges + RealGauges · CC BY-NC 4.0
- [Synthetic Data for Precision Gauge Reading](https://www.kaggle.com/datasets/endava/synthetic-data-for-precision-gauge-reading) — Kaggle, Endava
- [Pressure Gauge Reader Data](https://www.kaggle.com/datasets/juliusgrassme/pressure-gauge-reader-data) — Kaggle
- [Detect-and-read-meters](https://github.com/shuyansy/Detect-and-read-meters) — MIT lisansı
- [NRC-GAMMA](https://github.com/nrc-cnrc/NRC-GAMMA) — açık erişim · [makale](https://arxiv.org/abs/2111.06827)
- [Roboflow Universe — analog gauge setleri](https://universe.roboflow.com/search?q=class%3Agauge)
