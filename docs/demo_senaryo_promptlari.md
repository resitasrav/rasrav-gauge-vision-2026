# Ekip demosu için sentetik video — üretim prompt'ları

**Amaç:** üç modülü (GÖSTERGE / ALGILAMA / ANOMALİ) TEK bir videoda aynı anda
sınayabilmek. Kaynak: `demo/girdi/video/senaryo.pdf` (Reşit'in yazdığı fabrika
devriyesi senaryosu).

---

## Özgün senaryodan neden saptım

Senaryo olduğu gibi üretilirse **GÖSTERGE paneli boş kalır.** Üç sayfada
zincirin okuyabileceği tek nesne var: pres makinesindeki yeşil→kırmızı arıza
lambası. Analog kadran, dijital panel, vana, buton panosu hiç geçmiyor.

İkinci sorun ölçülmüş: kamera 50 cm yükseklikte ve **hiç durmuyor.**

| kısıt | ölçüm | kaynak |
|---|---|---|
| kadran yarıçapı < 40 px ise polar tarama yeterli örnek alamıyor | okuma gürültüye dönüşüyor | İP6 |
| 50° eğiklikte okuma hatası | **%9,30** (maks %56,5) | İP14 |
| merkez kadran çapının %2'si kayarsa | maks **178,8°** açı hatası | İP6 |

Göstergeler operatör göz hizasında (1,4-1,6 m) durur; 50 cm'den bakınca ~50°
yukarı açı oluşur. Yani yürürken geçilen bir gösterge ne yeterince büyük ne de
yeterince dik görünür.

**Çözüm senaryoyu zorlamak değil, gerçek sisteme sadık kalmak.** Devriye
platformunda pan-tilt kamera var ve waypoint'lerde duruyor. Aşağıdaki
parçalarda robot **duruyor, kafa yukarı bakıyor, panoyu okuyor, devam ediyor**.
Bu senaryoya eklenen bir numara değil, atlanmış bir gerçek.

---

## Her prompt'un başına eklenecek sabit blok

Modelin klipler arasında üslup kaydırmasını engeller. Aynen kopyalanır.

```
Birinci şahıs robot kamerası. Lens zeminden tam 50 cm yükseklikte, yere
paralel. Hareket tamamen sarsıntısız ve sabit hızlı (yürüyen robot köpek,
stabilize kafa). Otomotiv montaj fabrikası, pürüzsüz gri epoksi zemin,
tavanda endüstriyel LED aydınlatma. Fotogerçekçi, belgesel görünümü,
renk düzeltmesi yapılmamış ham kamera hissi. Yatay 16:9, 1080p.
Metin, altyazı, arayüz öğesi YOK.
```

**Bağlantı kuralı:** her klibin SON karesini kaydet ve bir sonraki klibe
başlangıç görüntüsü olarak ver (image-to-video / extend). Bağımsız
text-to-video ile üretilirse fabrika görünümü her klipte değişir ve CapCut'ta
birleşim göze batar.

---

## Parçalar

Dokuz parça, her biri ~10 s → ~90 saniye. `[G]` GÖSTERGE, `[A]` ALGILAMA,
`[AN]` ANOMALİ hedefi demektir.

### 1 — Giriş: montaj hattı `[A]`

```
Robot koridorda ilerliyor. İki metre yukarıda, çelik konveyörde boyasız mat
gri otomobil şasileri ağır ağır geçiyor. Tavan aydınlatması metal
yüzeylerden yansıyıp loş koridora düzenli aralıklarla keskin ışık huzmeleri
düşürüyor. Sağda kalın camlı kabinlerde altı eksenli punta kaynağı robot
kolları çalışıyor; kaynak uçlarından kısa, yoğun kıvılcımlar fırlayıp cam
panele çarparak sönüyor. İleride fosforlu sarı yelekli, beyaz baretli
işçiler kapı fitili takıyor. Kamera düz ilerliyor.
```

### 2 — İlk waypoint: manometre panosu `[G]`

```
Robot yavaşlayıp sol duvardaki boru hattı panosunun önünde TAMAMEN DURUYOR.
Kamera kafası yukarı doğru dönüp panoya dikleniyor. Panoda yan yana ÜÇ ADET
yuvarlak analog basınç saati var: beyaz kadran yüzü, siyah ibre, siyah
rakamlar, paslanmaz çelik çerçeve. Kadranlar kadrajın yaklaşık üçte birini
dolduruyor ve TAM KARŞIDAN, dik açıyla görünüyor. Ortadaki saatin ibresi
kırmızı alarm bölgesine yavaşça kayıyor. Kamera bu görüntüde iki saniye
sabit kalıyor, sonra kafa aşağı inip robot ilerlemeye devam ediyor.
```

> Bu parça zincirin okuyabilmesi için TAM KARŞIDAN ve BÜYÜK olmak zorunda —
> yukarıdaki 40 px / 50° kısıtları buradan geliyor.

### 3 — Zemindeki engel + KKD ihlali `[A]` `[AN]`

```
Sarı-siyah taralı güvenlik şeridinin üzerinde ağır bir havalı somun
tabancası ve etrafa dağılmış üç parlak krom bijon somunu duruyor. Robot bu
engellerin etrafından hesaplanmış bir kavis çizerek milimetrik manevrayla
geçiyor. Tam o sırada karşı yönden hızlı adımlarla bir montaj işçisi
kadraja giriyor; 50 cm yükseklik nedeniyle çelik burunlu botlarından göğüs
hizasına kadarı net görünüyor. Lacivert tulumunun fermuarı yarıya kadar
açık, BARETİ YOK ve koruyucu gözlüğü takılı değil. İşçi yerdeki somunları
son anda fark edip hafifçe sendeleyerek sağdan teğet geçiyor.
```

### 4 — Pres makinesi: arıza lambası + buton panosu `[G]` `[AN]`

```
Yürüyüş güzergâhının ortasında, kapağı yırtılmış iki kalın karton kutudan
siyah yalıtımlı kablo demetleri zemine taşmış. Robot yanından süzülerek
geçerken soldaki pres makinesinin kontrol panosunda yeşil çalışma lambası
aniden sönüyor ve yerine hızla yanıp sönen kırmızı arıza lambası devreye
giriyor. Robot duruyor, kamera kafası panoya dönüyor: panoda üstte iki
büyük yuvarlak ikaz lambası (biri yeşil sönük, biri kırmızı yanıyor), altta
sıra sıra kare basmalı butonlar ve bir adet siyah kollu 0-1 seçici anahtar
var. Pano kadrajın yarısını dolduruyor ve tam karşıdan görünüyor. İki
saniye sabit. Makinenin güvenlik kapağı açık bırakılmış; altına eğilmiş
teknisyen ÇIPLAK ELLE keskin kenarlı sıcak sac parçasını çekiştiriyor.
```

### 5 — Batarya test alanı: duman ve alev `[AN]`

```
Zemin pürüzsüz epoksiden mat siyah antistatik kaplamaya dönüşüyor. Test
istasyonlarından birinin altından, zeminle temas eden noktadan gri, ince
ama giderek yoğunlaşan bir duman sızıyor. Duman tavan pervanelerinin
akımıyla alçak dalgalar hâlinde 50 cm yükseklikteki lense doğru geliyor.
Yaklaştıkça kaynak netleşiyor: şarja bağlı batarya modülünün alt konektör
kablosundan çıkan küçük çaplı turuncu alevler zemindeki siyah plastiği
eritiyor. Koyu siyah duman doğrudan lensin önünden geçip görüşü bir
saniyeliğine tamamen bulandırıyor, sonra açılıyor.
```

### 6 — Lojistik yolu: yağ birikintisi + AGV `[A]` `[AN]`

```
Robot dumanın etrafından geniş bir açıyla dönüp ana lojistik yoluna
çıkıyor. Zeminde, tepedeki beyaz LED'leri ayna gibi yansıtan koyu renkli
bir hidrolik yağ birikintisi var. Robot birikintiyi kadrajın merkezine alıp
etrafından pürüzsüzce dolaşıyor. Sağdan, üzerine altı siyah deri araç
koltuğu yüklenmiş bir otonom taşıma aracı (AGV), tepe lambasını sarı sarı
yakıp söndürerek geçiyor. Tekerleklerinin zeminde bıraktığı sürtünme izleri
kadrajdan çıkıyor.
```

### 7 — Dolum istasyonu: vana + dijital panel `[G]` `[AN]`

```
Zemin sıvı dökülmelerine karşı ızgaralı yapıya dönüşüyor. Yürüyüş
güzergâhının ortasında devrilmiş, kapağı açık kalmış beş litrelik beyaz
plastik kimyasal bidonu duruyor; ağzından yayılan parlak fosforlu yeşil
antifriz endüstriyel ışık altında zehirli bir yansıma yapıyor. Robot
birikintinin etrafından kavis çizerken DURUYOR ve kamera kafası sağdaki
dolum hattına dönüyor: kalın borunun üzerinde KIRMIZI KOLLU büyük bir
kelebek vana var, kolu boruya dik konumda (kapalı). Hemen yanında siyah
yüzlü bir DİJİTAL GÖSTERGE paneli parlak yeşil yedi-segment rakamlarla
"124.7" yazıyor. Vana ve panel tam karşıdan, kadrajın üçte birini
doldurarak iki saniye sabit duruyor. Arkada bakım personeli koruyucu
siperliğini yukarı kaldırmış, yüzü açık şekilde basınçlı hatta çalışıyor.
```

### 8 — Cam kırıkları + ayakkabı ihlali `[A]` `[AN]`

```
Sessiz ve aydınlık cam montaj alanı; dev vantuzlu otonom kollar geniş ön
camları havaya kaldırıp araçların üzerine yerleştiriyor. Gri epoksinin
üzerine saçılmış binlerce küçük, elmas gibi parlayan kırık temperli cam
parçacığı var. Robot cam kırıklarına basmamak için rotasını hafifçe sola
kırıyor. Tam o sırada elinde geniş endüstriyel fırçayla gelen temizlik
görevlisinin ayakları kadrajı dolduruyor: çelik burunlu iş botu yerine ince
tabanlı BEYAZ BEZ SPOR AYAKKABI giymiş. Keskin cam parçaları bez
ayakkabının hemen yanına kadar geliyor.
```

### 9 — Sevkiyat: kırık palet + forklift `[A]` `[AN]`

```
Devasa katlanır kapıların açık olduğu sevkiyat alanına çıkılıyor; zemin
betonlaşıp pürüzleşiyor, dışarıdan gün ışığı vuruyor. Yolun tam ortasında
bir forkliftin ezerek parçaladığı kırık ahşap palet duruyor; ayrılmış
tahtalardan yukarı doğru uzun, paslı, sivri çiviler çıkıyor. Robot bu
kesici engeli algılayıp etrafından dolaşırken sağdan geçen forkliftin
sürücü koltuğundaki operatör görünüyor: EMNİYET KEMERİ TAKILI DEĞİL, bir
eliyle direksiyonu tutarken diğer eliyle telsizini ayarlıyor. Robot gün
ışığı vuran geniş sevkiyat rampasına doğru ilerlemeye devam ediyor.
```

---

## Üretim notları

- **Süre sınırı (Ağustos 2026):** LTX-2.3 20 s · Kling 3.0 15-16 s (uzatmayla
  30 s+) · Vidu Q3 / Seedance 2.0 15 s · Veo 3.1 8 s · Runway Gen-4.5 5-10 s.
  Daha uzunu her modelde birleştirmedir.
- **Ses gerekmiyor** — zincir sesi kullanmıyor, üretimde kapatmak süre ve kota
  kazandırır.
- **1080p yeterli, 4K gereksiz:** `isle_video_kumesi.py` çıktıyı zaten 1920'ye
  indiriyor ve çözümleme ham kare üzerinde yapılıyor; 4K yalnızca dosya
  boyutunu büyütür.
- **En kritik parçalar 2, 4 ve 7.** Üretim kotası biterse önce bunları al —
  GÖSTERGE'nin okuyacağı tek içerik onlarda.
- Birleştirdikten sonra:

  ```powershell
  python demo\kos_ekip_demosu.py --sadece <ad>
  python demo\paylasilabilir_yap.py --hedef-mb 16
  ```

## Beklenti: hangi panel ne gösterecek

| parça | GÖSTERGE | ALGILAMA | ANOMALİ |
|:--:|---|---|---|
| 1 | — | şasi, işçi | — |
| **2** | **3 analog kadran** | — | — |
| 3 | — | işçi takibi | zemindeki alet |
| **4** | **lamba + buton panosu + selector** | teknisyen | kırmızı arıza |
| 5 | — | — | duman, alev |
| 6 | — | AGV takibi | yağ birikintisi |
| **7** | **vana + dijital panel** | personel | yeşil sıvı |
| 8 | — | görevli | cam kırıkları |
| 9 | — | forklift, operatör | çiviler |

Dokuz parçanın **üçünde** GÖSTERGE'nin okuyacağı içerik var. Özgün senaryoda
bu sayı **sıfırdı**.
