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
| **A4** | Synthetic Data for Precision Gauge Reading | Kaggle (Endava) | DS5.0: 1.000 · DS6.0: 500 görüntü · 2,3 GB | COCO: gövde, kadran yüzü, **ibre** ve skala değerleri için bbox + **segmentasyon maskesi**; skala değerlerinde okunan değer de var. Keypoint: **ibre ucu, gösterge merkezi, min ve max skala çizgisi** | Kaggle sayfasında | Kaggle API token | İP5, İP6, İP7 |
| A5 | Pressure Gauge Reader Data | Kaggle (Aalborg atıksu pompa istasyonları) | 11,9 GB video | **ETİKETSİZ** — bkz. §2.2 | CC BY-SA 4.0 | Kaggle API token | Sınırlı |
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

## 2.1. A4 (Endava) — İndirilip İncelendi ✅

Set indirilmiş, `ReadMe.md` ve bir örnek etiket dosyası doğrudan incelenmiştir.

- **DS5.0:** 1.000 görüntü, her görüntüde bir gösterge. **Skala min/max değerleri ve yay
  uzunluğu rastgeleleştirilmiştir.** Arka planlar Stable Diffusion ile üretilmiş gerçek
  endüstriyel sahnelerdir.
- **DS6.0:** 500 görüntü, ek olarak **ana çizgi sayısı da değişkendir**.
- Üretim Houdini tabanlı özel bir hat ile yapılmıştır; modeller openmmlab çatısıyla ve
  **yalnızca sentetik veriyle sıfırdan** eğitilmiştir.
- Örnek etiket dosyasında doğrulanan alanlar: `category_name` (`dial` vb.), `bbox`,
  `segmentation` (RLE), `camera_matrix` (4×4), `camera_focal_length` ve
  **`synth_dial_value`** (göstergenin gerçek değeri).
- COCO dosyaları görev bazında ayrılmıştır: `train__inst_coco.json` / `val__inst_coco.json`
  (örnek/segmentasyon) ve `train__kps_coco.json` / `val__kps_coco.json` (keypoint).

**Modül açısından değeri:** Bu set, kendi sentetik üretecimizin ürettiği etiketin üstüne
üç şey daha koymaktadır: gerçekçi arka plan, kadran üzerinde kirlenme/aşınma ve kamera
matrisi. Kendi üretecimizin düz beyaz zemin üzerindeki temiz kadranlarıyla arasındaki fark,
İP8'e geçmeden önce yöntemin ne kadar dayanıklı olduğunu ölçmek için bilinçli bir basamak
oluşturmaktadır.

**Ayrıca dikkate değer:** Endava'nın yaklaşımı bizimkiyle aynı yönde ilerlemiştir — skala
yay uzunluğunun (180°–320°) ve ana çizgi sayısının rastgeleleştirilmesi, bizim
`gauges.yaml` üzerinden yaptığımız çeşitlendirmenin karşılığıdır. Bu, İP3'te seçilen
tasarımın bağımsız bir doğrulaması sayılabilir.

## 2.2. A5 (Aalborg) — Etiketsiz olduğu tespit edildi, öncelik düşürüldü ⚠️

İlk değerlendirmede bu set İP8 (gerçek görüntüde uçtan uca ölçüm) adayı olarak
işaretlenmişti. Endava'nın `ReadMe.md` dosyasındaki ifade bunu geçersiz kılmaktadır:

> "Since this dataset does not include labeled data, we manually annotated a small subset
> of frames from the test videos to use as ground truth."

Yani set **11,9 GB ham video** olup ground truth içermemektedir. Endava'nın yayımladığı
CSV dosyaları da etiket değil **model tahminidir** ("predicted reading"); hata ölçümünde
referans olarak kullanılamazlar.

**Karar:** 11,9 GB indirilmeyecektir. İP8'in gerçek-görüntü ground truth ihtiyacı **A2
(RealGauges)** üzerinden karşılanacaktır; orada ibre değeri ve açısı etiketlidir. A5
ileride yalnızca niteliksel gözlem veya İP14 (zor koşullar) için birkaç test videosu
düzeyinde değerlendirilebilir.

## 2.3. A1 ve A2 ERİŞİLEMİYOR — indirme bağlantısı ölü (05.08) 🔴

İP5'e geçilirken A1 (SyntheticGauges) ve A2 (RealGauges) indirilmek istenmiş, **Google
Drive klasörü HTTP 404 döndürmüştür.** İki bağımsız yöntemle denenmiştir: `gdown`
kütüphanesi (klasör listesi alınamadı) ve doğrudan HTTP isteği (404). Yayın sayfası
(`jjcvision.com`) bağlantıyı hâlâ listelemekte olup klasörün kaldırıldığı veya
erişiminin kısıtlandığı anlaşılmaktadır.

**Etkisi ikiye ayrılır:**

- **İP5 için orta.** Tespit eğitimi başka setlerle yapılabilmektedir (aşağıdaki A9).
- **İP8 için ağır.** §2.2'de A5 etiketsiz olduğu için elenmiş, İP8'in gerçek-görüntü
  ground truth ihtiyacının **A2 üzerinden karşılanacağı** kaydedilmişti. A2 de
  erişilemez olduğuna göre, gerçek görüntüde ibre değeri/açısı etiketli **elde kalan
  kaynak yoktur.** İP8'in ölçüm planı yeniden kurulmalıdır.

**İP8 için üç seçenek:**

1. **A6 (Detect-and-read-meters)** — depo iki Drive dosyası yayımlamaktadır ve ikisi de
   erişilebilir durumdadır (HTTP 200). Tanıma kümesinde ibre, kadran ve **değer** bilgisi
   bulunmaktadır. Öncelikli aday hâline gelmiştir; içeriği açılıp doğrulanmalıdır.
2. **Kendi gerçek verimizi etiketlemek.** İP13'te zaten masa üstü canlı test yapılacaktır;
   aynı düzenekte bilinen değerlerde fotoğraf çekilerek küçük ama **tam kontrollü** bir
   gerçek küme kurulabilir. Endava'nın kendi değerlendirmesini de bu yolla yaptığı §2.2'de
   kayıtlıdır.
3. Yazarlarla iletişime geçilerek A1/A2'nin yeni adresinin istenmesi (yanıt süresi belirsiz,
   plan buna bağlanamaz).

## 2.4. A9 — Roboflow-100 `gauge-u2lwv`, Hugging Face aynası ✅ **İNDİRİLDİ**

A3'ün (Roboflow) API anahtarı beklenirken, aynı ailenin bir setinin Hugging Face üzerinde
**anahtarsız** yayımlandığı tespit edilmiştir: `Francesco/gauge-u2lwv`.

- **235 gerçek endüstriyel fotoğraf** (158 eğitim · 25 doğrulama · 52 test), 640×640
- COCO kutu etiketi; kategoriler: `1 gauges` (kadran yüzü) ve `2 numbers` (kadran
  üzerindeki sayılar). Eğitim bölümünde **265 kadran** kutusu bulunmaktadır.
- Lisans `cc`, kapı yok, toplam 10,9 MB (parquet)
- Kaynak: Roboflow-100 kıyaslama derlemesi, 2022

**Değeri:** İP5'in ihtiyacı olan **gerçek** fotoğrafı sağlamaktadır ve K1 kararının
gerektirdiği karışık eğitim bu setle kurulabilmiştir. `2 numbers` kategorisi şimdilik
kullanılmamakta, İP11 için saklanmaktadır. Sınırı: yalnızca tespit etiketi vardır; ibre
açısı veya okunan değer içermez, dolayısıyla İP8'in ihtiyacını karşılamaz.

## 3. Planlanan Kullanım

| İş paketi | Kullanılacak veri | Gerekçe |
|---|---|---|
| İP5 — tespit | **A9 (gerçek) + kendi sentetik verimiz** ✅ kuruldu · sonra A4 | K1 kararı: sentetik tek başına yetersiz. A1 erişilemediği için A9 onun yerini aldı |
| İP6 — ibre açısı | Kendi sentetik verimiz ✅ *(03.08: 0,123°)*, sonra A4 (bağımsız doğrulama) | Kendi verimizde ground truth tam kontrolümüzde; A4 gerçekçi arka planla zorluk basamağı ekliyor |
| İP7 — açı→değer | Kendi sentetik verimiz ✅ *(04.08: %0,129)* + A4'ün skala min/max keypoint'leri | Kadran çapaları gerekli |
| İP8 — gerçek test | ~~A2 (RealGauges)~~ **erişilemiyor** → A6 veya kendi etiketlediğimiz küme (bkz. §2.3) | Gerçek görüntüde etiketli değer taşıyan tek kaynak elden çıktı |
| İP11 — dijital OCR | A6 | Analog ve dijital paneli birlikte ele alan tek referans |
| İP14 — zor koşullar | A4 (kirlenme/aşınma, gerçekçi aydınlatma) · gerekirse A5'ten birkaç video | A4 etiketli olduğu için hata ölçümü de yapılabilir |

---

## 4. Erişim Durumu ve İhtiyaçlar

| Erişim | Setler | Durum |
|---|---|---|
| Anahtarsız (Hugging Face) | **A9** | ✅ **İndirildi ve İP5'te kullanıldı** (05.08) — 235 gerçek fotoğraf |
| Kaggle API token | A4 | ✅ Örnek indirildi ve incelendi (31.07). Tam set için token gerekli — HF aynası da mevcut (`KhoaUIT/...`) |
| Anahtarsız (Google Drive) | A1, A2 | 🔴 **ERİŞİLEMİYOR** — klasör 404 (05.08, bkz. §2.3) |
| Anahtarsız (Google Drive) | A6 | Bağlantılar erişilebilir (HTTP 200); İP8 adayı olarak açılacak |
| Anahtarsız (GitHub) | A7 | İndirilebilir |
| Roboflow API anahtarı | A3 | Anahtar bekleniyor — **engelleyici değil**, A9 aynı aileden ve anahtarsız |
| Kaggle API token | A5 | **İndirilmeyecek** — etiketsiz, 11,9 GB (bkz. §2.2) |
| Doğrulanamadı | A8 | Yayın sayfası kimlik doğrulama istiyor |

---

## 5. Açık Kalan Hususlar

- A1/A2'nin CC BY-NC lisansı ticari kullanımı kısıtlamaktadır. Staj sonrasında proje
  ticari bir ürüne dönüşürse bu setlerle eğitilen ağırlıklar kullanılamaz; kayıt altına
  alınmıştır.
- A4'ün lisansı Kaggle sayfasında belirtilmiştir ancak dosya içinden teyit edilememiştir;
  yayında yalnızca A5'e (CC BY-SA 4.0) atıf zorunluluğu belirtilmektedir. Eğitim
  öncesinde teyit edilecektir.
- A4 yalnızca sentetik veriyle sıfırdan eğitilen bir hattın çıktısıdır; İP4'teki K1 kararı
  (sentetik tek başına yetersiz) ile aynı gerilim burada da vardır. Endava kendi
  değerlendirmesini etiketsiz gerçek videolar üzerinde el ile etiketleyerek yapmış olup
  sayısal sonuç yayımlamamıştır; bu nedenle karşılaştırma dayanağı olarak kullanılamaz.
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
