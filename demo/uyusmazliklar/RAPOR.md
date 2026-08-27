# Uyuşmazlık Taraması — Birleşik Demo İçin

**Tarih:** 13.08.2026 (staj gün 15/30) · **Tutan:** Reşit Asrav (bu tarama Claude Code ile
yapıldı, kaynak dosyalara hiçbir yazma işlemi yapılmadı — sadece okuma/statik analiz)

**Kapsam:** `demo/run_demo.py`'nin üç modülü tek pencerede birleştirmesi için gereken
"ortak sözleşme" karşılaştırması. **Bu dosya `..\..\ortak uyusmazliklar\uyusmazliklar.md`
(U1-U11) dosyasının yerine geçmez** — o dosya ekibin genel MQTT şema uyuşmazlık defteridir
ve orada zaten kayıtlı maddeler burada TEKRAR EDİLMEDİ, sadece referans verildi. Bu rapor
yalnızca demoyu kurarken ortaya çıkan, henüz kayıtlı olmayan bulguları içerir.

---

## 0. Önce söylenmesi gereken: "ortak sözleşme" öncülü gerçek projeyle tam örtüşmüyor

Görev tanımında geçen "her modül BİR KARE alır; tespit + okuma döndürür, ve
`inspect/reading` MQTT şemasıyla yayınlar" cümlesi, üç modülün **aynı görevi** (gösterge
okuma) paralel yaptığı bir senaryoyu varsayıyor. Gerçek proje böyle değil:

| Modül | Sahip | Görev | Yayınladığı topic |
|---|---|---|---|
| ALGILAMA | Bedirhan | insan/nesne tespiti + aktif takip | `vision/target_offset` |
| GÖSTERGE | Reşit (ben) | analog/dijital gösterge + lamba/vana okuma | `inspect/reading` |
| ANOMALİ | Özgür | devriye anomali tespiti + rapor | `patrol/alert` |

Üçü de aynı üst projenin (`pan_tilt_robot_projesi.md`) parçası ve hepsi `camera/frame`
girdisini paylaşıyor (bkz. §3), ama **çıktı sözleşmeleri kasıtlı olarak farklı** — her
modülün kendi iş paketi dosyasında (`DOKUMANLAR/*_is_paketleri.md`) ayrı ayrı tanımlı.
Yani "üç modülün ortak inspect/reading şemasına uyup uymadığını" karşılaştırmak anlamsız;
`inspect/reading` yalnızca GÖSTERGE'nin şeması. Aşağıdaki maddeler bunu göz önünde
bulundurarak, gerçekten karşılaştırılabilir olanı karşılaştırıyor.

---

## 1. Hiçbiri (GÖSTERGE hariç) "bir kare al → sonuç döndür" fonksiyonu sunmuyor 🔴 Yüksek etki

Demoyu kurarken beklenen: her modülden `f(kare) -> sonuç` biçiminde tek bir fonksiyon
import edip video karesi başına çağırmak.

| Modül | Beklenen | Gerçek | Kaynak |
|---|---|---|---|
| GÖSTERGE | `f(frame) -> reading` | **VAR** — `read_gauge(image, model, gauge, **kw) -> FrameResult` | `src/gauge_vision/pipeline.py:90` |
| ALGILAMA | `f(frame) -> reading` | **YOK** — tüm mantık `main()`'in `while True:` döngüsü içinde (kamera açma, model çağrısı, çizim, MQTT yayını hepsi iç içe) | `vision/live_detector.py:40-210`, kritik satır `:101` (`model.track(...)`) |
| ANOMALİ | `f(frame) -> reading` | **YOK** — `anomali_test.py` bir eğitim + toplu-tahmin scripti (`engine.fit(datamodule=...)`, `engine.predict(datamodule=...)`); tek görüntü değil, tüm MVTec-AD `bottle` test kümesi üzerinde çalışıyor | `anomali_test.py:1-49` |

**Olası etki:** Bu üç modül gerçek robotta bir araya geldiğinde de aynı sorun çıkar — kimse
diğerinin fonksiyonunu çağıramaz, yalnızca MQTT üzerinden mesajlaşabilirler (ki proje
mimarisi zaten bunu öngörüyor — "modüller birbirinin koduna değil MQTT şemalarına bağlıdır",
`CLAUDE.md` başlığı). Demo bunun aksini yapmaya çalıştığı için (tek pencere, tek process,
doğrudan fonksiyon çağrısı) bu sınırla karşılaştı.

**Bu demoda ne yapıldı (ÖNERİ değil, uygulanan çözüm):**
- ALGILAMA: `demo/run_demo.py::algilama_isle()` içinde, `live_detector.py:101`'deki AYNI
  çağrı (`model.track(source=frame, conf=conf, persist=True)`) ve aynı varsayılan model
  (`yolov8n.pt`) demo tarafında ayrı bir sarmalayıcı olarak yeniden çalıştırılıyor.
  Bedirhan'ın dosyası import edilmiyor, değiştirilmiyor, çalıştırılmıyor — sadece davranışı
  aynı kütüphane çağrısıyla tekrarlanıyor. **Bu bir kaynak değişikliği değildir**, ama
  panelde görünen "ALGILAMA modülü" gerçekte `live_detector.py` dosyası değil, onun
  demo-tarafı bir kopyasıdır. Gerçek entegrasyonda `live_detector.py` düzeltilmeden bu iki
  kod yolu ayrışabilir (biri değişip diğeri değişmezse).
- ANOMALİ: Karşılığı yok. `anomali_test.py`'yi gerçekten çağırmak MVTec-AD indirmeyi ve
  saatlerce eğitimi başlatır — bir video karesiyle hiçbir ilgisi yoktur. Bu yüzden demo bu
  modül için hiçbir çağrı denemiyor, panelde sabit `HATA: ...` gösteriyor (görevin 2.
  maddesindeki "modül hata verirse HATA etiketiyle devam etsin" kuralına göre — burada
  hata gerçek bir exception değil, baştan bilinen bir yapısal eksiklik).

**Önerilen düzeltme (ÖNERİ — uygulanmadı):**
1. `live_detector.py`'de döngü gövdesi `process_frame(frame, model, tracker_state) -> dict`
   fonksiyonuna çıkarılmalı; `main()` sadece bu fonksiyonu döngüde çağırmalı. Tek satırlık
   bir refactor, `model.track()` çağrısının kendisi değişmez.
2. Özgür'ün tarafında gerçek bir "tek görüntü → anomali skoru" fonksiyonu (`infer_frame`)
   ve eğitilmiş bir checkpoint (`.ckpt`) gerekiyor — şu an ne biri ne diğeri repoda var
   (bkz. §2).

**Muhatap:** Bedirhan (madde 1), Özgür (madde 2 ile birlikte)

---

## 2. ANOMALİ modülünde kaydedilmiş ağırlık yok 🔴 Yüksek etki

**Kaynak:** `OzgurKotbas_Akilli_Fabrika` reposunun tamamında `*.ckpt`, `*.pt`, `*.pth`
taraması **sıfır sonuç** verdi (`results/Padim/MVTecAD/bottle/` altında yalnızca görüntü
çıktıları var, model ağırlığı yok).

**Etki:** `anomali_test.py` her çalıştırıldığında sıfırdan eğitim yapıyor demektir
(`Engine(max_epochs=1)`). Canlı/demo senaryosunda kullanılabilir bir "önceden eğitilmiş
model yükle, karede tahmin yap" yolu yok. Ayrıca eğitim verisi MVTec-AD `bottle`
kategorisi — fabrika koridoru veya gösterge görüntüleriyle **konu bakımından ilgisiz**;
İP6 ("kendi verisiyle anomali") henüz tamamlanmamış (bkz. `uyusmazliklar.md` U8, 05.08
kontrolü).

**Önerilen düzeltme (ÖNERİ):** İP6/İP9 tamamlanınca eğitilmiş checkpoint repoya (veya
işaret edilen bir yola) kaydedilmeli; `infer_frame` fonksiyonu bu checkpoint'i bir kere
yükleyip tekrar tekrar tahmin yapabilmeli (her karede yeniden eğitim OLMAMALI).

**Muhatap:** Özgür

---

## 3. `camera/frame` girdi sözleşmesi zaten kayıtlı, bu demo onu kullanmıyor ⚪ Bilgi notu

Görevde "ortak sözleşme: her modül bir kare alır" ifadesi büyük olasılıkla `camera/frame`
MQTT konusuna atıfta bulunuyor — bu konudaki alan adı uyuşmazlıkları (`ts` birimi,
`frame_id` vs `seq`, `w`/`h` vs `frame_w`/`frame_h`) zaten `ortak uyusmazliklar/
uyusmazliklar.md` dosyasında **U1, U2, U3, U6** olarak kayıtlı ve orada takip ediliyor.
Bu demo MQTT'yi hiç kullanmıyor — video dosyasından doğrudan `cv2.VideoCapture` ile kare
okuyor — dolayısıyla bu risk demoyu ETKİLEMİYOR. Ancak gerçek saha entegrasyonunda (üç
modül `camera/frame`'e MQTT üzerinden abone olduğunda) hâlâ geçerli; hatırlatma amacıyla
buraya not düşüldü, madde tekrar açılmadı.

---

## 4. Girdi kare biçimi ve koordinat kuralı — GÖSTERGE ile ALGILAMA arasında UYUŞMAZLIK YOK ✅

| | GÖSTERGE | ALGILAMA | ANOMALİ |
|---|---|---|---|
| Renk uzayı | BGR (`cv2` ham kare) | BGR (`cv2` ham kare) | RGB, CHW, normalize edilmiş tensor (`prediction.image.permute(1,2,0)`, `anomali_test.py:26`) |
| Kutu formatı | `xyxy`, piksel float (`pipeline.py:172` `sonuc.boxes.xyxy`) | `xyxy`, piksel int (`live_detector.py:114`) | Kutu kavramı yok — piksel-seviye heatmap üretiyor |

GÖSTERGE ve ALGILAMA aynı temel varsayımı paylaşıyor (ikisi de ultralytics YOLO
kullanıyor, ikisi de ham BGR kare + xyxy piksel kutusu) — bu ikisi arasında **uyuşmazlık
yok**. ANOMALİ farklı bir veri temsili kullanıyor ama bu §1/§2'deki daha temel sorunun
(fonksiyon arayüzü yok) yanında ikincil; şimdilik ayrı bir madde açmaya gerek görülmedi.

---

## 5. "Okunamadı" / boş sonuç temsili — üçü de FARKLI, ve bu KASITLI olabilir ⚪ Bilgi notu

| Modül | Boş/başarısız durumu nasıl işaretliyor | Kaynak |
|---|---|---|
| GÖSTERGE | `reading.status == "unreadable"`, `value: None` — güven eşiği altına düşen HER durum için aynı sözleşme | `read/calibrate.py` (`DURUM_OK` ve kardeşleri) |
| ALGILAMA | Hedef yoksa `track_id: -1` ile mesaj YİNE yayınlanıyor (bazen yayın da kesiliyor — bkz U4) | `live_detector.py:170-181` |
| ANOMALİ | Henüz kod yok, dolayısıyla temsili de yok | — |

GÖSTERGE'nin "unreadable" kavramı ile ALGILAMA'nın "track_id: -1" kavramı birbirinin
YERİNE GEÇMEZ — biri "değeri güvenilir şekilde okuyamadım" (ölçüm hatası riski), diğeri
"aranan nesne şu an kadrajda yok" (normal, beklenen durum) anlamına geliyor. **Bunları
aynı alan adı altında birleştirmeye ÇALIŞMAK hatalı olurdu** — demo panelinde bilerek
ayrı metinler kullanıldı (GÖSTERGE: "okunamadı: ...", ALGILAMA: "hedef yok"). Bu madde bir
düzeltme önerisi değil, gelecekte biri "ikisini aynı şemaya sokalım" derse bu ayrımın
gerekçesinin kayıtlı olması için buraya yazıldı.

---

## 6. Config anahtarları — gerçek kesişim yalnızca `waypoint`, zaten kayıtlı

GÖSTERGE'nin `configs/gauges.yaml`'ı ile ALGILAMA/ANOMALİ'nin kendi ayarları arasında
doğrudan bir kesişim yok (farklı görevler, §0). Tek gerçek kesişim noktası
`gauges.yaml`'daki `waypoint: WP-04...WP-07` alanının Özgür'ün altın tur duraklarıyla
eşleşmesi gerekliliği — bu zaten `uyusmazliklar.md` **U11** olarak kayıtlı ve envanterdeki
değerlerin uydurma olduğu orada açıkça yazıyor. Burada tekrar edilmedi.

---

## Özet

| # | Konu | Durum | Muhatap |
|:--:|---|:--:|---|
| 1 | Üç modülden ikisinde kare-bazlı çağrılabilir fonksiyon yok | 🔴 açık | Bedirhan, Özgür |
| 2 | ANOMALİ'de kaydedilmiş ağırlık yok, eğitim scripti konu dışı veriyle çalışıyor | 🔴 açık | Özgür |
| 3 | `camera/frame` sözleşmesi | ⚪ zaten kayıtlı (U1/U2/U3/U6), demoyu etkilemiyor | — |
| 4 | Kare biçimi / koordinat kuralı (GÖSTERGE↔ALGILAMA) | ✅ uyuşmazlık yok | — |
| 5 | "Okunamadı" temsili üç modülde farklı | ⚪ kasıtlı farklılık, bilgi notu | — |
| 6 | Config anahtarları kesişimi | ⚪ zaten kayıtlı (U11) | Özgür |

**Sonuç:** Demoyu engelleyen tek gerçek "uyuşmazlık" madde 1 ve 2 — ikisi de ekip
arkadaşlarının tarafında, kaynak değiştirilmeden çözülemez. Demo bunları ATLAYARAK değil,
GÖRÜNÜR KILARAK (ALGILAMA: demo-tarafı sarmalayıcı ile açık künyeli; ANOMALİ: sabit HATA
paneli) tamamlandı — 3. kuralın ruhu (yanlış göstermektense göstermemek) modül
çağrılabilirliği için de uygulandı.

---

## Ek — 27.08.2026: ANOMALİ paneli artık sabit HATA göstermiyor

**Madde 1 ve 2 KAPANMADI.** Özgür'ün tarafında hâlâ tek kare alan bir fonksiyon
ve kaydedilmiş bir ağırlık yok; yukarıdaki iki madde aynen açık ve muhatabı
değişmedi. Değişen tek şey demonun bu boşlukla nasıl başa çıktığı.

Eski davranış: panel sabit bir `HATA: ...` metni gösteriyordu. Bu, "üç modül de
kendi incelemesini yapsın" istendiğinde demonun üçte birinin ölü olması demekti.

Yeni davranış: ALGILAMA'ya uygulanan çözümün aynısı ANOMALİ'ye de uygulandı —
modülün **dosyası** değil **yöntemi** demo tarafında koşturuluyor
(`demo/anomali_demo.py`). `anomali_test.py` hâlâ çalıştırılmıyor, import
edilmiyor, değiştirilmiyor.

**ALGILAMA'daki çözümden bir farkı var ve saklanmamalı.** Orada aynı KÜTÜPHANE
çağrılabiliyordu (`ultralytics.YOLO.track`, Bedirhan'ın varsayılanıyla birebir).
Burada `anomalib` bu sanal ortamda kurulu değil ve kurmak çalışan
`torch 2.13+cu126` kurulumunu riske atıyor. Bu yüzden PaDiM'in kendisi
(Defard ve ark., 2020 — Özgür'ün `anomali_test.py:17`'de seçtiği model)
torchvision ResNet18 üstünde uygulandı. Yöntem aynı üç adımdır:

1. önceden eğitilmiş CNN'in ara katmanları birleştirilir (layer1+2+3),
2. her yama konumu için normal veriden çok değişkenli Gauss çıkarılır,
3. yeni karede Mahalanobis uzaklığı = anomali haritası.

**"Normal" referansı değişti ve bu bilinçli.** MVTec-AD `bottle` kategorisi bu
videolarla konu bakımından ilgisiz (madde 2'de zaten yazılı). Referans olarak
videonun KENDİ ilk kareleri alınıyor; panel "bu videonun başına göre ne
değişti" sorusunu cevaplıyor. Devriye senaryosunda doğru soru budur — aynı
durak, aynı çerçeve, zamanla değişen sahne. **Varsayımı da açık:** sahnenin
başı zaten anormalse ölçüm yanıltır, bu yüzden referansın ne olduğu panelin
üstünde yazılı duruyor.

Eşik tahmin edilmiyor, ölçülüyor (depo kuralı — mutlak eşik bu depoda üç kez
sessiz hata üretti): uyum kümesinin kendi skor dağılımının p99'u.

**Panel başlıkları künyelendi.** Artık iki panelin başlığında
`- demo sarmalayici` yazıyor; kimse bu çıktıları ekip arkadaşlarının kodunun
çıktısı sanmasın. GÖSTERGE panelinde böyle bir ek yok, çünkü orada gerçekten
`gauge_vision` paketi çalışıyor.

| # | Konu | Durum |
|:--:|---|:--:|
| 1 | Kare-bazlı çağrılabilir fonksiyon yok | 🔴 **hâlâ açık** (Bedirhan, Özgür) |
| 2 | ANOMALİ'de kaydedilmiş ağırlık yok | 🔴 **hâlâ açık** (Özgür) |
| — | Demo bu boşlukları nasıl gösteriyor | ✅ ikisi de künyeli sarmalayıcı |
