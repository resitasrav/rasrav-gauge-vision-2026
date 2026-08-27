# Devam Notu — Oturum Kapanışı

> ## ⏩ 27.08 AKŞAM EKİ — bu bölüm en yenidir, önce burayı oku
>
> 14 gerçek video (9573 kare) zincirden geçirildi ve **yapısal bir sessiz hata
> sınıfı ölçüldü**: karede TEK BİR kadran olmayan beş videoda 383 "gauge"
> kutusu üretildi ve **hepsi başarıyla okundu** (kapsam 1,00). Gözle
> doğrulananlar: forkliftin ön tekerleği (açı 44°), elektrikli vantilatör
> (dönen kanat her karede başka açı), beyaz ikaz lambasının düz camı, ve
> panoya **basılı** direnç sembolü (tüm kare kadran sanılıp 505 px çember).
>
> Dört iş sırayla yapıldı:
>
> | # | İş | Sonuç |
> |---|---|---|
> | 1 | İbre kanıt kapısı | sahte okuma **383 → 39** (−%89,8), gerçekte −%24,7 |
> | 2 | Zor negatif + `keypad` sınıfı | sahte kutu **64 → 4** (−%94), gerçekte −%9 |
> | 3 | Pano tipi metre (kare çerçeve, yay skala) | açı hatası **107,6° → 0,15°** |
> | 4 | Seçici anahtar (1-0 şalteri) | 8/8, sahte girdiler reddediliyor |
>
> **320 test geçiyor** (sabah 283'tü). Ayrıntılar §6'da.
>
> **Eksik bırakılan:** `read_all_analog` kapısı yeniden eğitilmiş modelle
> YENİDEN ÖLÇÜLMELİ — eşikler eski ağırlığın dağılımından seçildi.

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

---

## 6. 27.08 akşamı — dört iş, sırayla

Girdi: `demo/girdi/video/` altına 14 stok video kondu (Pexels, ~293 MB, git
dışı). Hepsi `scripts/isle_video_kumesi.py` ile işlendi; çıktı işaretli video +
video başına JSON, `demo/cikti/video/`.

**Neden bu videolar ölçüm için sınırlı:** ground truth yok. İnternetten alınmış
bir manometrenin gerçek değeri bilinmiyor, dolayısıyla **okuma doğruluğu
ölçülemez**. Ölçülebilen iki şey var ve ikisi de ground truth GEREKTİRMİYOR:
tespit oranı, ve **180° sıçrama** (ibre iki ardışık karede 180° dönemez —
fizik yasağı).

### 6.1 İbre kanıt kapısı (`pipeline.read_all_analog`)

`read_frame` (kimliği beyan edilen yol) İP15 eşiğiyle korunuyordu;
`read_all_analog` İP11 ile dün eklenmişti ve o kapı ona konmamıştı. Delik buydu.

İki DİK kapı kondu, ikisinin eşiği de ölçülerek seçildi (720 gerçek kadran
okuması / 64 yanlış pozitif):

| kapı | eşik | gerçek medyan | yanlış medyan |
|---|--:|--:|--:|
| `MIN_TESPIT_GUVENI` | 0,45 | 0,909 | 0,394 (max 0,675) |
| `MIN_IBRE_KANITI` | 0,15 | 0,596 | 0,109 |

Tek başına ibre kanıtı YETMİYOR: panoya basılı direnç sembolü gerçekten
merkezden uzanan kesintisiz koyu bir şerit, ibre güveni 0,94'e çıkıyor. Onu
tespit güveni eliyor.

Eşiğin **korunması gereken popülasyondaki bedeli** ayrıca ölçüldü (bu depodaki
ders: önceki kapı denemesi İP8'i 10/10'dan 0/10'a düşürmüştü). Sentetik v1'de
ibre güveni: temiz min 0,907 · jpeg 40 min 0,907 · çap 64 px min 0,135 ·
merkez %2 sarsıntı min 0,576 · %4 sarsıntı min 0,218. 0,15 eşiği bu
popülasyonun en kötü hâlinde %1 kayıp veriyor.

Sonuç (14 video yeniden koşuldu):

| | önce | sonra |
|---|--:|--:|
| kadran OLMAYAN videolarda okuma | 383 | **39** (-%89,8) |
| gerçek kadranlı videolarda okuma | 4787 | 3605 (-%24,7) |
| 6.mp4 (temiz tek kadran) | 239 | **239** (kayıp yok) |
| 11.mp4 (tüm kareyi kadran sanan) | 16 | **0** |

### 6.2 DENENDİ ve ELENDİ: `refine_dial` eşiklerini gevşetmek

Ölçüldü: rafine gerçek videoda **%0,6** kabul ediliyor, yani sahada fiilen
kapalı. Reddin sebebi de ölçüldü (373 gerçek kadran kutusu): artık kapısı
%74,3, kayma kapısı %19,8. `refine.py`'nin kendi uyarısı (satır 98) aynen
gerçekleşmiş — gerçek fotoğrafta artık 2,4 katına çıkıyor.

Gevşetme denendi, **elendi**. Ground truth yok ama flip fiziksel hata sinyali:

| ayar | rafine | toplam flip | oynama (4.mp4) |
|---|--:|--:|--:|
| kapalı | %0 | 10/466 | 3,24° |
| mevcut eşik | %0-1 | 10/466 | 3,25° |
| gevşek 0,15/0,30 | %31-37 | 10/466 | 4,50° |
| gevşek 0,25/0,40 | %34-44 | 10/466 | 5,00° |

Rafine üç kat sık ateşliyor, flip hiç değişmiyor, oynama artıyor. Eşikler
DEĞİŞTİRİLMEDİ; gerekçe `detect/refine.py` içine yazıldı.

### 6.3 Zor negatif + `keypad` sınıfı (İP17)

`scripts/hazirla_karistiricilar.py` modelin fiilen yanıldığı nesneleri gerçek
videolardan kırpıyor ve **etiketine göre kovalara** ayırıyor:

    negatif/ 101   teker, vantilatör kanadı, makine gövdesi -> ETİKETSİZ
    lamp/     24   ikaz lambası camı
    keypad/   10   butonlu kontrol panosu

**Bu ayrım ölçümle öğrenildi.** İlk sürümde her kırpım negatifti; sonuç: sahte
`gauge` kutusu 64->4 (istenen) **ama** 10.mp4'te `lamp` kutusu 49->1 (gerileme).
O kırpım gerçek bir ikaz lambasıydı; "gösterge değil" diye öğretilince model
onu lamba olarak da göremez oldu. Bir kırpım gerçekten bir sınıfın örneğiyse o
sınıfla etiketlenir.

Beşinci sınıf **`keypad`** eklendi (pano). Ayrı bir `button` sınıfı BİLEREK
yok: ışıklı basmalı buton ile ikaz lambası görsel olarak aynı nesnedir ve ayrı
sınıf açmak ikisini birden bozardı. `read_keypad` zaten pano kırpımı + envanter
oranlarıyla çalışıyor — eksik olan panoyu bulacak sınıftı.

Birinci koşu (etiketsiz sürüm, imgsz 416 / batch 16): sahte kutu 64->4,
gerçek 734->668. İkinci koşu (etiketli kovalar + imgsz 640 + batch 48)
sayıları `outputs/metrics/ip17_keypad.json` içinde.

**GPU notu:** imgsz 416 / batch 16 ile kart %57 kullanımda, VRAM 1,3/6,1 GB
idi — yani zamanın %43'ünü veri bekleyerek geçiriyordu. imgsz 640 / batch 48 /
workers 4 ile %100 kullanım, 5,7/6,1 GB. imgsz asıl kazanç: gerçek videolarda
kadran yarıçapı 27-70 px ölçüldü, 1080p kareyi 416'ya indirmek 27 px'lik
kadranı ~10 px'e düşürüyor.

### 6.4 Pano tipi metre — beşinci geometri (İP18)

Kare çerçeve, ~120° yay skala, ibre kenardan dönüyor (elektrik odalarındaki
ampermetre/voltmetre). Yuvarlak kadran için doğru olan üç varsayımın üçü de
yanlış. Kendi videolarımızda kanıtı var: 3.mp4'teki iki dikdörtgen VU metresi
330 karenin **hiçbirinde** tespit edilmedi.

Erişilebilir kamu veri seti arandı, **yok**: Pointer-10K (Baidu Disk,
CC BY-NC-SA), DialBench, Synanthropic, UFPR-ADMR — hepsi yuvarlak kadran.
Bu yüzden projenin diğer dört tipiyle aynı yol: `synth/panel.py`.

Envanter `face` bloğuyla geometriyi BEYAN ediyor (`shape`, `pivot`,
`sweep_radius`) — görüntüden çıkarılmıyor, çünkü yanlış bir pivot okumayı
kırmaz, sessizce KAYDIRIR. Örnek gösterge: `EM-501`.

İki bileşen de gerekli, ölçüldü (300 kare):

| yapılandırma | açı hatası (ort) | 180° ters |
|---|--:|--:|
| **envanter pivot + yay penceresi** | **0,15°** | **0** |
| envanter pivot, pencere yok | 107,6° | 45 |
| kutu merkezi + pencere | 38,0° | 0 |
| kutu merkezi, pencere yok | 126,1° | 124 |

Pencere olmadan tarama ibreyi değil **siyah çerçeveyi** buluyor (pivot alt
kenara yakın, aşağı bakan ışınlar bezele çarpıyor). `read_needle_angle` artık
`aci_penceresi` alıyor ve pano tipinde envanterdeki yaydan dolduruluyor.

**Yan bulgu:** `Scale.ccw_araligi` eklendi. `cw` kadranda açı AZALIR, dolayısıyla
CCW yayı `angle_max`'tan `angle_min`'e uzanır. Düz çıkarma EM-501'de 120°
yerine 240°'lik yayı veriyordu ve pencere açıkken bile hata 107,6°'de sabit
kalmıştı — hata buradaydı.

Değer hatası %0,262 (düz kamera), %1,397 (+/-8° yatık). İlk ölçümde %2,63
çıkmıştı; farkın tamamı `decimals: 1` yuvarlamasıydı (aralık 0-1 MW), okuma
değil.

Yatıklık kestirimi pano tipinde ATLANIYOR: yöntem kadranın çizgi halkasından
okuyor, pano metresinde öyle bir halka yok ve hiç ölçülmedi. Burada yanlış bir
yatıklığın bedeli ağır — tarama penceresini döndürüp ibreyi dışarıda bırakır.

### 6.5 Seçici anahtar — 1-0 şalteri (İP19)

Panoda iki farklı buton türü var ve FARKLI FİZİKLE okunuyorlar:

    ışıklı basmalı buton   durum merceğin RENGİNDEN
    seçici anahtar (1-0)   durum KOLUN KONUMUNDAN — ışığı yoktur

İkincisini renkle okumak "0" ile "1"i ayırt edemez; sönük bir selector her
konumda "off" görünür. Kol açısı makinesi zaten vardı (vana kolu) ve
kopyalanmadı, yeniden kullanıldı. Fark: açılar gösterge düzeyinde değil
**BUTON düzeyinde** beyan ediliyor (`kind: selector`, `lever_angles`) — bir
panoda birden çok selector olabilir ve her birinin montajı ayrı.

İki ölçülmüş ayar:

* `SELECTOR_KESIT_PAYI = 1.05` — lambanın 2,2'si burada ZARARLI. Buton bileziği
  kesite giriyor, halkanın PCA'sı yönsüz: açı 135,0°/45,0° birebir doğru
  çıkarken uzama 1,08'de kalıp kanıt kapısına takılıyordu.
* `_daire_maskele` — kesit kare, buton yuvarlak; köşelerdeki koyu bilezik
  eşiklemede kolla birleşip blob'u kareleştiriyor (uzama 1,15).

Sonuç: 8/8 doğru, düz/gürültülü/siyah girdi reddediliyor. Keypad ölçümü 7
koşulda tekrarlandı, **hepsinde sıfır sessiz hata**; karanlıkta ve 25° eğikte
kapsama sıfıra düşüyor (dürüst ret).

**Bilinen sınır:** kamera yatınca kol açısı da dönüyor ama envanterdeki beyan
sabit. Vananın da aynı sınırı var. 25° eğikte kapsama 0.

---

## 7. Sırada ne var

1. **Kapı eşiklerini yeniden ölç.** `MIN_TESPIT_GUVENI` / `MIN_IBRE_KANITI`
   eski ağırlığın dağılımından seçildi; yeni model ile
   `scripts/isle_video_kumesi.py` yeniden koşulup eşikler yeniden konmalı.
2. **Zamansal tutarlılık.** Kalan %2-10 flip'in tek yolu bu;
   `gauge_vision/temporal.py` hazır. Refine tarafı ölçümle kapandı (§6.2).
3. **Pano metresi GERÇEK fotoğrafta** — sentetik ölçüm 0,15° diyor ama bu
   "yöntem oturuyor mu" cevabı, "sahada ne olur" değil.
4. **Selector + yatıklık.** Kol açısına roll düzeltmesi.
5. **Gerçek pano fotoğrafı** (S8/S10) — keypad eşikleri, dijital perspektif ve
   birim kimliği hep buna bağlı.
