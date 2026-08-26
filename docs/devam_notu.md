# Devam Notu — Oturum Kapanışı

**Yazıldığı an:** 27.08.2026 · Gün 26/30
**Amaç:** Yeni oturum bu dosyayı okuyunca kaldığı yerden devam edebilsin.
Durum özeti bağlam dosyasında; **burada yalnızca "sırada ne var ve neye dikkat et" var.**

> 👉 **Reşit'in bakması gerekenler ayrı dosyada:** `..\..\SORULAR.md`
> (ana STAJ klasörü altında, **git'e girmiyor** — uyuşmazlık defteriyle aynı yerde.)

---

## 1. Nerede kaldık

**İP8 kapalı.** 26-27.08'de zincir ilk kez **kendi ürettiğimiz veri dışında**
sınandı ve iki yapısal boşluk ölçüldü, biri kapatıldı.

| Ölçüm | Sonuç | Dosya |
|---|---|---|
| İP8 analog — gerçek fotoğraf | **%0,473** · kapsama 10/10 | `outputs/metrics/ip8_ekran_hatasi.json` |
| İP8 lamba / vana — gerçek fotoğraf | %100 / %100 | aynı dosya |
| İP8 dijital — gerçek fotoğraf | 0/5 · sessiz hata 0 | `outputs/metrics/ip8_dijital_tani.json` |
| **BAĞIMSIZ sette okuma** (merkez etiketten) | **medyan 0,20° · p95 0,56°** · %98,2'si %5 altında | `outputs/metrics/gercek_zemin_okuma.json` |
| **BAĞIMSIZ sette okuma** (merkez kestirilmiş) | medyan 0,72° · **p95 177°** · kapsama %84,8 | aynı dosya |
| Tespit — gerçek zeminli sette (eski model) | `gauge` mAP50 **0,3925** | `outputs/metrics/ip5_gercek_zemin.json` |
| İP6 sentetik açı (değişmedi) | 0,123° · 100/100 | `outputs/metrics/ip6_aci_hatasi.json` |
| İP15 güven eşiği | 0,70 · kapsama %88,1 | `outputs/metrics/ip15_guven_esigi.json` |

**264/264 test geçiyor.**

---

## 2. 27.08'de yapılanlar

### a) Karedeki HER analog gösterge okunuyor — `pipeline.read_all_analog`

Eskiden zincir karedeki tek göstergeyi okuyordu ve bu doğruydu (kalibrasyon
göstergeye özeldir), ama diğerlerine **hiç dokunulmuyordu**. Ölçülen örnek:
`termometre.mp4`'ün ilk karesinde masada dört termometre var, tek değer
üretiliyordu.

Yeni fonksiyon her analog kutuya daire rafinesi + ibre açısı uygular. Çıktısı
`AnalogKutuOkuma` ve içinde **bilinçli olarak `value`/`unit` alanı YOKTUR** —
sebebi (b).

### b) Kimliksiz kadrana değer/birim üretilmiyor

26.08'de ölçülen sessiz hata sınıfı:

| Video | Gerçekte | Zincirin yayınladığı |
|---|---|---|
| `termometre.mp4` | 0-120 °C termometre | **2,2 bar** · ok · güven 0,724 |
| `araba.mp4` | devir saati ×1000 rpm | **0,8 bar** · ok · güven 0,839 |

Sebep koddaki bir hata değil **beyandı**: zincire `--gosterge PT-101` deniyor,
o da karedeki ilk analog kutuya PT-101'in kalibrasyonunu uyguluyordu. Demo
artık varsayılan olarak kimlik beyanı istemiyor (`--gosterge yok`); kimliksiz
kadranda yalnız açı gösteriliyor, birim yerine `?` yazıyor.

⚠ **Bu kapsama kaybıdır, çözüm değil.** Kimlik beyanla gelmeli (`waypoint`,
U11) ya da görüntüden okunmalı (S9 — Reşit'in kararı).

### c) Bağımsız veri seti indirildi ve ölçüldü

`Synanthropic/reading-analog-gauge` (Hugging Face, herkese açık, kimlik
gerektirmiyor) → `data/raw/hf_analog_gauge/` (**~2,8 GB, git dışı**):

- `corners.zip` — 8.072 görüntü, dört köşe etiketi (perspektif dörtgeni)
- `keypoint.zip` — 34.370 görüntü, **ibre ucu + merkez + skala uçları**

**Kadranlar render, zeminler gerçek endüstriyel fotoğraf.** Gerçek fotoğrafın
yerini tutmaz; kapattığı şey kadranın kendisi değil **bağlamıdır** (boru,
flanş, kablo, yansıma, derinlik).

Etiketin görüntüyle tuttuğu sınandı: 150 karede okuyucumuz etiketle **149 kez
15° içinde** uyuştu. Yani set okuma ölçümüne uygun.

**İlk kez başkasının verisinde ölçülen okuma doğruluğu** (`olc_gercek_zemin.py`):
merkez etiketten verildiğinde medyan **0,20°**, p95 **0,56°**, karelerin
%98,2'si hedef %5'in altında. Yöntem başkasının kadranında da çalışıyor.

### d) Tespit yeniden eğitildi — ama videolarda İŞE YARAMADI

Boşluk gerçekti: eski model gerçek zeminli sette `gauge` mAP50 **0,3925**
veriyordu (kendi test kümemizde 0,9632). Yeniden eğitim (`egit_gercek_zemin.py`,
2012 HF + 1353 eski kare, 60 epoch):

| Küme | Öncesi | Sonrası |
|---|---|---|
| gerçek zeminli val — `gauge` | 0,3925 | **0,9950** |
| analog eski test — `gauge` | 0,9632 | **0,9948** |
| dijital / lamba / vana | 0,995 | 0,995 |

**Ama videolarda iyileşme YOK, kısmen gerileme var** (`tani_video.py`):

| Video | eski (tespitli kare) | yeni |
|---|---|---|
| araba | %55 | **%27** |
| termometre | %100 | %91 |
| gosterge | %64 | %64 |
| fabrika (Veo) | %100 | %92 |

**Sebep — alan aşırı uyumu.** HF setinin kadranları hep aynı stilde: beyaz yüz,
ince bezel, büyük ve merkezde. Model ona özelleşti, araç göstergesi gibi
**koyu ve küçük** kadranları kaybetti. Gözle de görülüyor:
`outputs/figures/video_tani/araba_kiyas.png`.

**Dengeli karışımla ikinci eğitim de kurtarmadı** (714 HF + 1804 eski, HF payı
%60 → %28). Alan içi sayılar aynı kaldı (0,995 / analog eski test 0,9926), ama
videolarda **daha da kötü**:

| Video | eski (üretim) | HF %60 | HF %28 |
|---|---|---|---|
| araba | **%55** | %27 | %27 |
| gosterge | **%64** | %64 | %36 |
| termometre (analog/kare) | **2,45** | 1,73 | 2,09 |

Güven eşiği düşürülerek de kurtarılmadı (conf 0,25 → 0,10'da eski model yine
önde). Yani model o kadranları düşük güvenle görmüyor, **hiç görmüyor.**

### 🔴 KARAR: üretim ağırlığı `cok_sinif` OLARAK KALIYOR

`runs/detect/models/ip5/cok_sinif/weights/best.pt`. Gerçek zeminli eğitimin iki
denemesi de `runs/detect/models/ip5/gercek_zemin*/` altında duruyor —
**kullanılmıyor**, kayıt olarak saklanıyor.

**Ders:** i.i.d. bölünmüş bir val kümesindeki mAP, ALAN İÇİ başarıdır.
Genelleme ancak alan DIŞI veride görülür — burada videolarda. 0,995 gördükten
sonra videoya bakmasaydık, gerilemeyi ilan edilen bir kazanç sanacaktık.

**Setin değeri OKUMA tarafında, tespitte değil.** 34 bin ibre-ucu etiketli kare
(§2c) yöntemi bağımsız olarak doğruladı ve 180° sorununu görünür kıldı; tespit
eğitimi için ise **stil çeşitliliği yetersiz.** Gerçek fabrika fotoğrafı hâlâ
gerekli (S8/S10).

---

### e) Dijital panel gerçek fotoğrafta İLK KEZ okundu — 0/5 → 1/5

Elenen üç düzeltme (Gauss, sütun zemini, renklilik) hep **maskeyi** düzeltmeye
çalışıyordu. Dördüncü deneme başka yere baktı: hane kutuları aynı kalsın, ama
segmentin "yanık mı" kararı panelin O NOKTASINDAKİ aydınlatmasına göre
verilsin. `_panel_seviyeleri` panelin tamamı için tek referans üretiyordu ve
gerçek fotoğrafta zemin sol-sağ 1,53-1,71 kat değişiyor.

`read/digital.py::_aydinlatma_bantlari` zemini 8 banda bölüp yanık/sönük
referansını hanenin bulunduğu bandın kazancıyla **çarpıyor**. Toplamsal model de
denendi, zayıf çıktı (1/5'e karşı 0/5).

| | önce | sonra |
|---|---|---|
| **İP8 gerçek fotoğraf** | **0/5** | **1/5** · sessiz hata 0 |
| İP11 sentetik temiz | %93,3 | %93,3 |
| İP11 düşük ışık ×0,4 | %63,3 | **%68,3** |
| İP11 parlama %50 | %5,0 | %0,0 (hepsi `unreadable` — güvenli taraf) |

**Kalan 4 karede sorun hane KUTUSU.** Izgarayı tespit kutusundan kurma yolu
denendi ve **daha kötü** çıktı (#17'yi de bozdu): fotoğrafta panel eksenle hizalı
değil, eksen hizalı bir ızgara rakamların üstüne oturmuyor. Sıradaki adım
devam notunun eski maddesiyle aynı: **panelin dörtgen köşelerinden perspektif
düzeltmesi** (`detect/perspective.py` yalnız dairesel kadranı düzeltiyor).
Teşhis figürü: `outputs/figures/dijital_izgara_tani.png`.

### f) YENİ TİP: buton/tuş paneli (`keypad`) — S7'nin cevabı

Reşit'in isteği: "bir makineye bakıldığında tuş takımlarının renginden mevcutta
neyin çalıştığını anlayabilmesi lazım." Beşinci gösterge tipi olarak eklendi.

**Nasıl çalışıyor:** her buton lamba mantığıyla sınıflanır (HSV + çevre
kontrastı, `read/state.py`'dan çağrılıyor — ikinci kopya yok), çıkan bileşim
envanterdeki `machine_states` kurallarına vurulur, `value` alanına bir MAKİNE
DURUMU adı basılır. Yerleşim de kurallar da **envanterde** (`CP-701`).

| Koşul | durum | buton | kapsama | sessiz hata |
|---|---|---|---|---|
| temiz | %100 | %100 | %100 | **0** |
| düşük ışık ×0,4 | %100 | %100 | %100 | **0** |
| düşük ışık ×0,15 | %0 | %23 | %0 | **0** |
| parlama %50 | %80,6 | %90,5 | %80,6 | **0** |
| bulanık 9px · jpeg q25 · eğik 25° | %100 | %100 | %100 | **0** |

Yedi koşulun hepsinde sessiz hata sıfır. Düşük ışıkta her şey `unreadable` —
kanıt kapısı çalışıyor (aşağıda).

⚠ **Tespit sınıfı yok.** Okuyucu hazır, YOLO `keypad` tanımıyor; şimdilik
kırpılmış görüntüde okunuyor (`--tespitsiz`). Sınıf eklemek eğitim demektir ve
gerçek pano fotoğrafı gelmeden yapılırsa §2d'nin hatası tekrarlanır.

⚠ **Envanterdeki panel VARSAYIMDIR** — dört buton, yeşil/sarı/kırmızı, beş
kural. Gerçek pano fotoğrafı gelince değişecek olan YAML satırlarıdır.

### g) Gömülü hedef (Orange Pi 5 + NPU) — taşınabilirlik ölçüldü

**İyi haber: kütüphane zaten taşınabilir.** `src/gauge_vision/` içinde tek bir
`torch`/`ultralytics`/`cuda` importu yok. Tespit modeli dışarıdan veriliyor ve
arayüz küçük: `model.predict(...)[0].boxes.{xyxy,conf,cls}` + `.names`.

**ONNX aktarımı yapıldı ve doğrulandı** (`disa_aktar_onnx.py`): PyTorch ile en
büyük kutu sapması **1,38 px** (sınır 2 px). ONNX hem Hailo (`.hef`) hem RKNN
(`.rknn`) derleyicisinin girdisidir; derleme adımı kartta yapılır.

**Bir darboğaz bulundu ve kapatıldı.** `read_needle_angle` maliyeti KARENİN
boyutuyla ölçekleniyordu: aynı kadran 360 px kesitte 3,26 ms, 1080p karede
**18,13 ms**. Sebep `_polarity_mask`'in bütün kareyi eşiklemesi. ROI kırpması
eklendi → 1080p'de **3,25 ms**, kare boyutundan bağımsız. **Sonuç değişmedi**
(İP6 birebir 0,123°, İP7 %0,129, zincir %0,21); `test_sonuc_KARE_BOYUTUNDAN_bagimsiz`
kilitliyor. Zincir 95,6 → **51,1 ms**.

**Kare bütçesi izdüşümü** (`olc_gomulu.py`, tek çekirdek, 1080p, 1 gösterge):

| Senaryo | Toplam | Kare/s |
|---|---|---|
| iyimser (Hailo-8, 720p, CPU ×3) | 33 ms | **30** |
| gerçekçi (varsayılan, CPU ×4) | 46 ms | **22** |
| kötümser (RK3588 NPU, 1080p, CPU ×5) | 69 ms | **14** |
| **4 gösterge birden okunursa** | 205 ms | **5** |

⚠ Bunlar İZDÜŞÜM, ölçüm değil — NPU süresi ve yakalama maliyeti varsayım.
Script kartın üstüne kopyalanıp aynen koşturulabilir (bağımlılığı yalnız
numpy + opencv).

⚠ **30 kare/s yanlış hedef.** Devriye saniyede 1 okuma ister; gösterge saniyede
25 kez değişmiyor. Artan bütçe daha yüksek çözünürlüğe ya da **zamansal
ortalamaya** harcanmalı — ikincisi 180° sorununun (bkz. §3) çözüm yolu.

---

## 3. Neye dikkat et

### 🔴 180° ters okuma — kök sebep MERKEZ, ibre mantığı değil

Bağımsız sette ölçüldü: merkez etiketten gelirken karelerin **%1,0'ı**, merkez
`refine_dial` ile kestirilirken **%8,1'i** ibreyi 180° ters okuyor. Bu sessiz
hata sınıfıdır — sayı üretilir, güven yüksektir, değer tamamen yanlıştır.

**İki düzeltme denendi, ikisi de ÖLÇÜMLE ELENDİ. Aynı yolu üçüncü kez deneme;
gerekçeleri `read/needle.py` başındaki blokta duruyor:**

| Deneme | Sonuç |
|---|---|
| "Şerit 0,72R'yi aşıyor mu" (mutlak sonda) | İP8 ekran ölçümünü sıfırladı (kapsama 10/10 → 0/10) |
| İki yönün uzanımını birbirine göre kıyasla | Ayırmadı: doğru medyan 0,52 · ters 0,48, dağılımlar örtüşüyor |
| `refine`'in kanıt sayılarıyla kapı (artık/yayılma/destek) | En iyisi `destek/r`: terslerin %27'sini yakalıyor, doğruların %6'sını feda ediyor — konulmadı |

**Kök sebep İP6'nın kendi duyarlılık tablosunda yazıyor:** merkez kadran
**çapının %2'si** kadar kayınca ortalama hata 8,1°, **maksimum 178,8°**.
`refine_dial` bu sette ortalama %1,8 (yarıçapın) sapıyor ama p95'i %4,0 — yani
dağılımın kuyruğu tam o kritik bölgeye giriyor.

**Sıradaki denenecek yol: ZAMANSAL tutarlılık.** Zincir videoda çalışıyor ve
180°'lik bir sıçrama ardışık karelerde fiziksel olarak imkânsızdır.
`gauge_vision/temporal.py` bu iş için zaten var ve `canli_oku.py`'da kullanılıyor
ama İP8/bağımsız ölçümlerde kullanılmıyor. **Tek karede ayrılamayan şey kare
dizisinde ayrılabilir.**

### 🔴 Yeni kapı yazınca SAHTE GİRDİYLE sına — 27.08'de üçüncü kez

Buton panelinin ilk sürümünde **düz gri bir kare** dört butonu da "off" okuyup
`enerji_yok` durumunu **güven 1,00 ile** yayınlıyordu. Sebep `_lamba_durumu`'nun
parlak piksel bulamayınca "sönük" demesi ve bundan emin olması. Sahada bu,
kamera panoyu hiç görmediğinde "makinede enerji yok" diye rapor etmektir.

`refine.py` ve `roll.py`'ın ilk kapıları da rastgele gürültüyü kabul ediyordu ve
sebep aynıydı: kapı **"cevap makul mü"** diye soruyor, **"kanıt var mı"** diye
değil. `keypad.MIN_BUTON_KANITI` mercek ile pano arası bağıl kontrastı ölçüyor;
eşik iki dağılım ölçülüp aralarına kondu (gerçek buton min 0,436 · sahte girdi
maks 0,015 → 0,12). **Kusuru birim testi yakaladı**, ölçüm değil — sahte girdi
testi yazılmasaydı sessizce yayına girerdi.

### 🔴 Kimlik doğrulaması YOK ve görüntüden çıkarılamıyor

21.08'de yatıklığın ayrıklık sayısı kimlik kapısı olarak denendi ve elendi
(doğru medyan 0,011 · yanlış -0,103, dağılımlar örtüşüyor). Kimlik beyanla
gelmeli; envanterdeki `waypoint` bunun için (U11). Zincirdeki
`pipeline._tipe_uyan_kutular` **tip** filtresidir, kimlik değil — bir termometre
de `gauge` sınıfındadır.

### ⚠ Dijital panel gerçek fotoğrafta hâlâ 0/5

Teşhis bitti, tespit suçsuz (0,954 güven; kutu bir hanenin üstüne düştüğünde
rakam 1,000 güvenle DOĞRU). Çöken adım **hane kutusu bulma**; sebep
`read/digital.py::_segment_maskesi`'nin "zemin panel boyunca sabit" varsayımı —
gerçek fotoğrafta yansıma gradyanı var (sol/sağ zemin oranı 1,53-1,65).

**Üç düzeltme denendi ve elendi** (`scripts/tani_dijital.py`): Gauss ile zemin
çıkarma 0/5 · sütun bazında zemin 0/5 · renklilik kanalı 0/5 **ve #18'de yanlış
haneyi 1,000 güvenle üretti**.

**Yapılacak:** (1) panelin dörtgen köşelerinden perspektif düzeltmesi
(`detect/perspective.py` yalnız dairesel kadranı düzeltiyor), (2) hane
ızgarasını görüntüden değil TESPİT kutusundan kur (`digits.count` envanterde).
Hedef: 0/5 → en az 3/5, **sessiz hata 0 kalmak şartıyla.**

### ⚠ Eğitim bitişini `results.csv` satır sayısıyla YOKLAMA

Ultralytics son epoch satırını yazdıktan SONRA doğrulama yapıp `best.pt`'yi bir
kez daha yazar (24 MB → 6,2 MB, optimizer ayıklanır). Doğru ölçüt: **dosya
boyutu <10 MB** ve bir dakika değişmemiş olmalı. 27.08'de bu ölçütle beklendi
ve doğru çalıştı.

### ⚠ Ultralytics `box.ap50` GLOBAL sınıf kimliğiyle indekslenmez

Doğrulama kümesinde **bulunan** sınıflara göre indekslenir; eşleme
`box.ap_class_index` üzerinden yapılmalıdır. 27.08'de doğrudan `ap50[i]`
yazıldı ve hiç `gauge` örneği olmayan bir kümede "gauge 0,995" raporlandı
(aslında `digital`'in sayısıydı). `egit_gercek_zemin.py::_sinif_map` düzeltilmiş
hâli taşıyor.

### ⚠ Vana kolu renkleri — deney yapıldı, VARSAYILAN KAPALI

`synth/state.py` renk çeşitliliğini destekliyor ama üreteç onu yalnız
`--vana-renkli` ile kullanır. Eksik olan renk değil **şekil ve bağlam** —
sahadaki kol yassı bir plaka, flanşlı gövde üstünde. Okuyucu
(`read/state.py::_kol_acisi`) hâlâ en koyu bileşeni arıyor; dört alternatif
ölçüldü, hepsi geri alındı (k-means sentetikte 40/40 ama gerçek fotoğrafta
sessiz hata üretti).

---

## 4. Ekip demosu

`python demo\run_demo.py --video <yol>` — üç panel (GÖSTERGE / ALGILAMA / ANOMALİ).

**27.08'de değişti:** GÖSTERGE paneli artık karedeki **her analog kadranın**
çemberini, ibresini ve açısını çiziyor (turuncu). Varsayılan mod kimliksizdir
(`--gosterge yok`): değer ve birim üretilmez, üstte "kimlik beyanı yok" yazar.
Envanterdeki bir göstergeyi okutmak için `--gosterge PT-101` verilir; o zaman
o kutu yeşil/kırmızı kutuda değeriyle görünür, diğerleri turuncu kalır.

Renk sözleşmesi: **yeşil/kırmızı = kalibrasyonlu DEĞER beyanı** ·
**turuncu = yalnız geometri (kimlik bilinmiyor)** · **gri = yalnız tespit**.
