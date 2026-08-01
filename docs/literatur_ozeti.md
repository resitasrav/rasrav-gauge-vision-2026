# Mini Literatür Özeti — Gösterge Okuma (İP4)

**Hazırlayan:** Reşit Asrav · **Tarih:** 31.07.2026 · **Modül:** GÖSTERGE

**Tarama yöntemi:** Başlangıç noktası olarak `ZZZHANG-jx/Awesome-Image-based-Meter-Recognition-Reading`
derlemesi kullanılmıştır (20 makale ve 12 açık veri seti künyesi). Derlemenin üzerine,
29–30.07 günlük raporlarında "araştırılacak" olarak kaydedilen dört tasarım sorusu için
hedefli literatür araması yapılmıştır.

> **Okuma derinliği:** ⭐ işaretli çalışmalar özet veya tam metin düzeyinde incelenmiştir.
> İşaretsiz çalışmalar derleme listesinden künye düzeyinde (başlık, yıl, yöntem ailesi,
> görev) alınmıştır. İlgili olanlar İP6 ve İP11 aşamasında derinlemesine incelenecektir.

---

## 1. Özet Tablo

| # | Çalışma | Yıl | Yöntem ailesi | Veri | Modüle katkısı |
|:--:|---|:--:|---|---|---|
| 1 ⭐ | **Learning to Read Analog Gauges from Synthetic Data** (WACV) | 2023 | İki aşamalı CNN; yapısal bileşen tespiti + açı çıktısı | Sentetik eğitim + 4.813 el ile ayıklanmış gerçek görüntü | Sentetik-önce stratejisinin literatürdeki en yakın karşılığı. Ortalama hatada 4.55 iyileşme (%52 göreli) |
| 2 ⭐ | **SyncG — A Large-Scale Synthetic Benchmark for Robust Analog Gauge Reading** (Sci. Data) | 2026 | Blender tabanlı fotogerçekçi sentetik üretim | 20.000 görüntü, 145 farklı ortam; tespit, keypoint, segmentasyon ve OCR etiketi | Karışık eğitim oranı bulgusunun kaynağı (bkz. S1) |
| 3 ⭐ | **Computer vision and deep transfer learning for automatic gauge reading** (Sci. Rep.) | 2024 | Hough çember → kutupsal dönüşüm → transfer öğrenme | 1.011 görüntü, 9 sınıf, 128×128; SyntheticGauges+RealGauges | Klasik ve derin yöntemin melez kullanımına somut örnek. DenseNet169 F1 %97,6. Sınırı: okuma 9 sınıfa ayrılmakta, sürekli değer üretilmemektedir |
| 4 ⭐ | **Automatic analogue gauge reading using smartphones for industrial scenarios** (ICMLT) | 2023 | Düşük çözünürlükte tespit → yüksek çözünürlüklü kırpımda perspektif düzeltme → keypoint | Endüstriyel senaryo | U6 numaralı çözünürlük uyuşmazlığının doğrudan karşılığı |
| 5 ⭐ | **Detect-and-read-meters** (arXiv 2302.14323) | 2023 | YOLOv5 → STN hizalama → uçtan uca açı ve değer ağı | Tespit (COCO) ve tanıma (Labelme) veri setleri yayımlanmış | Analog ve dijital paneli birlikte ele alan tek referans uygulama. MIT lisanslı |
| 6 | A Robust Pointer Meter Reading Based on TransUNet + Perspective Transformation Correction (Electronics) | 2024 | Segmentasyon + perspektif düzeltme | — | Elips-daire düzeltmesinin güncel uygulaması |
| 7 | Perspective deformation correction for circular pointer meter (Measurement) | 2024 | Kadran yapısından perspektif düzeltme | — | Düzeltmenin kadranın kendi çizgilerinden türetilmesi |
| 8 | A pointer meter reading method based on human-like reading sequence + keypoint detection (Measurement) | 2025 | Keypoint | — | İP9 kapsamındaki alternatif yöntem için güncel referans |
| 9 | Vector Detection Network: Robots Reading Analog Meters in the Wild (IEEE TAI) | 2021 | Vektör tespiti | Saha verisi | Robot senaryosu — çatı projeye en yakın kurulum |
| 10 | It's About Time: Analog Clock Reading in the Wild (CVPR) | 2022 | Sentetik veri + bozunum artırma | Saha görüntüleri | İbre okumada sentetik veri kullanımının kanıtı |
| 11 | A High-Precision Automatic Pointer Meter Reading System in Low-Light Environment (Sensors) | 2021 | Düşük ışık ön işleme | — | İP14 (zor koşullar) için hazır referans |
| 12 | A pointer meter recognition method based on virtual sample generation (Measurement) | 2020 | Sanal örnek üretimi | — | Sentetik-önce yaklaşımının erken örneği |
| 13 | Detecting and recognizing seven segment digits using deep learning (ITM Conf.) | 2024 | YOLO tabanlı 7-segment tespiti | — | İP11 referansı. YOLOv8l @640 → mAP@50 0,979; YOLOv8n @320 → 0,786 |

**Derlemede listelenen açık veri setleri (İP1'e girdi):** SyntheticGauges+RealGauges (11K, 2021) ·
Pointer-10K (10K, 2021) · NRC-GAMMA (2,8K, 2021) · PMIs (1,8K, 2022) ·
UFPR-ADMR-v1/v2 (2K/5K) · SCUT-WMN (5K) · WMeter5K (5K, 2024)

---

## 2. Araştırma Sorularının Bulguları

### S1 — Sentetik veride oturan yöntemin gerçek veride performans kaybı (domain gap)

Bulgu sayısaldır. SyncG çalışmasında eğitim kümesindeki sentetik oranı değiştirilerek
gerçek veri üzerindeki doğruluk ölçülmüştür:

| Eğitimde sentetik oranı | Gerçek veride doğruluk |
|---|---|
| ≤ %25 | Kayda değer değişim gözlenmemiştir |
| %50 – %75 | Sınırlı düşüş |
| %100 | Belirgin domain gap |

Çalışmanın sonucu, sentetik verinin ikame değil tamamlayıcı olduğu yönündedir; sentetik
ve gerçek örneklerin öğrenilen temsil uzayında tam örtüşmediği raporlanmıştır.

**Modüle etkisi:** Sentetik veri, İP6 ve İP7 için yeterlidir; bu iş paketlerinde ağ
eğitilmemekte, geometrik yöntemin hatası ölçülmektedir ve ground truth sentetik üretimden
bedava gelmektedir. Buna karşılık İP5'te (YOLO ile gösterge tespiti) sentetik verinin tek
başına kullanılması uygun değildir; açık veri setleriyle karıştırılması gerekmektedir.
Bu bulgu, İP1'in bir tarama formalitesi değil İP5'in ön koşulu olduğunu göstermektedir.

### S2 — İbre açısı ölçüm yönteminin seçimi

Literatürde üç yöntem ailesi güncel kullanımdadır:

1. **Klasik görüntü işleme:** Hough çizgi/çember dönüşümü, ağırlık merkezi, iskeletleme
2. **Kutupsal dönüşüm ve piksel projeksiyonu:** kadranın merkez etrafında açılarak ibrenin
   sütun projeksiyonuyla belirlenmesi (3 numaralı çalışmada kullanılmıştır)
3. **Keypoint regresyonu:** merkez ve ibre ucunun doğrudan tahmini; son üç yılın baskın yaklaşımı

**Modüle etkisi:** Mevcut plan doğrulanmıştır. İP6'da klasik yöntemle başlanacak, Hough ve
kutupsal tarama sentetik veri üzerinde karşılaştırılacaktır (ground truth mevcut olduğundan
karşılaştırma maliyetsizdir). Keypoint tabanlı alternatif İP9'da ele alınacaktır. Planda
değişiklik yoktur; gerekçe literatürle desteklenmiştir.

### S3 — Perspektif bozulmasının düzeltilmesi

Düzeltme zorunludur ve bu bulgu plan değişikliği gerektirmektedir. Kamera kadrana açılı
baktığında kadran elipse dönüşmekte, kadran çizgileri ibrenin dönüş merkezine göre elips
üzerine oturmaktadır; dolayısıyla açıdan değere geçiş doğrudan bozulmaktadır. Literatürde
ayrıca elips uydurmanın tek başına yetersiz kaldığı belirtilmektedir: yöntem konturu
daireye çevirmekte, ancak kadranın iç bilgisi (çizgiler, sayılar) doğru düzeltilememektedir.
Bu nedenle perspektif dönüşümü (homografi) tercih edilmektedir.

**Modüle etkisi:** Perspektif düzeltmesinin İP14'e (zor koşullar) bırakılması uygun
değildir. Sahada kameranın kadrana tam dik bakması istisnai bir durum olduğundan, düzeltme
İP8'de gerçek fotoğrafa geçildiği anda gerekecektir.

### S4 — Saha koşullarında gereken çözünürlük (U6 numaralı uyuşmazlık)

Endüstriyel akıllı telefon çalışmasında (4 numaralı kaynak) iki aşamalı bir yapı
kullanılmaktadır: gösterge düşük çözünürlüklü karede tespit edilmekte, okuma ise
göstergenin yüksek çözünürlüklü kırpımı üzerinde, arada perspektif düzeltmesi uygulanarak
gerçekleştirilmektedir.

**Modüle etkisi:** U6 kaydındaki ikinci öneri (robotun durakta yüksek çözünürlüklü tek kare
üretmesi) literatürde standart uygulamadır. Sürekli görüntü akışının çözünürlüğünün
artırılması talep edilmeyecek; akış tespit için mevcut haliyle kullanılacak, okuma için ayrı
bir yüksek çözünürlüklü kare isteği tanımlanacaktır. Kararın sayısal dayanağı için ayrıca
sentetik kadran 60/80/120/200 piksel çapa küçültülerek İP6'nın açı hatası ölçülecektir.

---

## 3. Plana Yansıyan Kararlar

| # | Karar | Etkilenen iş paketi |
|:--:|---|---|
| K1 | İP5'te sentetik veri tek başına kullanılmayacak, açık veriyle karıştırılacaktır | İP1 → İP5 |
| K2 | Perspektif/elips düzeltmesi İP8 kapsamına alınmıştır, İP14'e bırakılmamıştır | İP8, İP14 |
| K3 | İP6'da Hough ve kutupsal tarama yöntemleri karşılaştırmalı olarak denenecektir | İP6 |
| K4 | Okuma için ayrı yüksek çözünürlüklü kare isteği tanımlanacak, sürekli akış çözünürlüğü değiştirilmeyecektir | İP10, U6 |
| K5 | İP11'de YOLO tabanlı 7-segment tespiti 640 çözünürlükte hedeflenecektir (referans mAP 0,979) | İP11 |
| K6 | Okuma sürekli değer olarak üretilecek, sınıflandırma yaklaşımı kullanılmayacaktır | İP7 |

---

## 4. Açık Kalan Hususlar

- SyncG veri setinin indirme adresi ve lisansı doğrulanamamıştır; yayın sayfası kimlik
  doğrulama istemektedir. İP1 kapsamında yeniden denenecektir. 20.000 görüntü ve keypoint
  etiketi teyit edilirse hem İP5 hem İP6 için önemli bir kaynak oluşturacaktır.
- 1 numaralı çalışmada bildirilen "4,55 ortalama hata iyileşmesi" değerinin birimi
  (derece veya değer yüzdesi) özetten anlaşılamamıştır. İP6'nın hedef eşiği belirlenmeden
  önce tam metin incelenecektir.
- Yerli/Türkçe bir gösterge veri seti aranmamıştır; fabrika göstergelerinin uluslararası
  standart kadranlar olması beklendiğinden ihtiyaç öngörülmemektedir.

---

## 5. Kaynakça

1. [Learning to Read Analog Gauges from Synthetic Data](https://arxiv.org/abs/2308.14583) — Leon-Alcazar, Alnumay, Zheng, Trigui, Patel, Ghanem (WACV 2024) · [kod](https://github.com/fuankarion/automatic-gauge-reading)
2. [A Large-Scale Synthetic Benchmark for Robust Analog Gauge Reading (SyncG)](https://www.nature.com/articles/s41597-026-07308-x) — Scientific Data, 2026
3. [Computer vision and deep transfer learning for automatic gauge reading detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC11449899/) — Scientific Reports, 2024
4. [Automatic analogue gauge reading using smartphones for industrial scenarios](https://dl.acm.org/doi/10.1145/3589883.3589925) — ICMLT 2023
5. [Detect-and-read-meters](https://github.com/shuyansy/Detect-and-read-meters) — MIT lisansı · arXiv 2302.14323
6. [TransUNet + Perspective Transformation Correction](https://doi.org/10.3390/electronics13132436) — Electronics, 2024
7. [Perspective deformation correction for circular pointer meter](https://www.sciencedirect.com/science/article/abs/pii/S0263224124003087) — Measurement, 2024
8. [Human-like reading sequence and keypoint detection](https://www.sciencedirect.com/science/article/abs/pii/S0263224125003537) — Measurement, 2025
9. Vector Detection Network: Robots Reading Analog Meters in the Wild — IEEE TAI, 2021
10. It's About Time: Analog Clock Reading in the Wild — CVPR 2022
11. [A High-Precision Automatic Pointer Meter Reading System in Low-Light Environment](https://ncbi.nlm.nih.gov/pmc/articles/PMC8309754) — Sensors, 2021
12. A pointer meter recognition method based on virtual sample generation technology — Measurement, 2020
13. [Detecting and recognizing seven segment digits using a deep learning approach](https://www.itm-conferences.org/articles/itmconf/pdf/2024/06/itmconf_amict2023_01007.pdf) — ITM Conferences, 2024
14. [Awesome-Image-based-Meter-Recognition-Reading](https://github.com/ZZZHANG-jx/Awesome-Image-based-Meter-Recognition-Reading) — derleme; tarama bu kaynaktan başlatılmıştır
15. [Analogue-Gauge-Reader](https://github.com/axn170037/Analogue-Gauge-Reader) — klasik Hough uygulaması (İP6 karşılaştırması için)
