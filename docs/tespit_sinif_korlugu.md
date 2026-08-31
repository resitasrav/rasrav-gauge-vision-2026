# Tespit Sınıf Körlüğü — 0831 Ölçümü

**Tarih:** 31.08.2026 · **Modül:** GÖSTERGE · **Tutan:** Reşit Asrav

**Ölçüm zemini:** `demo/girdi/video/yeni/0831.mp4` — 2108 kare, 70,3 sn, 1920×1080,
üretilmiş fabrika görüntüsü. **Alan dışı** bir test kümesidir: eğitim verisinin
hiçbir parçası bu videodan gelmiyor.
**Ağırlık:** `runs/detect/models/ip5/keypad5/weights/best.pt` (5 sınıf).
**Ham çıktı:** `demo/cikti/birlesik/0831_birlesik.json`

---

## Özet

Beş tespit sınıfından **yalnız `gauge` sahada çalışıyor**. `lamp`, `valve` ve
`keypad` kadrajı dolduran nesnelerde bile sıfır kutu üretiyor; `digital` ters
yönde bozuk — gösterge olmayan koyu dikdörtgenlere kutu atıyor.

Sebep eğitim kümesinin bileşimi. Gerçek endüstriyel görüntüden beslenen tek sınıf
`gauge`, sahada çalışan tek sınıf da `gauge`. Bu tesadüf değil, aynı olgunun iki
yüzü. **Algoritma sorunu değil, veri sorunu** — mimari, ağırlık dosyası ve eğitim
reçetesi aynı kalıp, gerçek görüntü verilen sınıfta 0,98 güven üretiyor.

Bu, 27.08'de araç göstergelerinde ölçülen kutupluluk körlüğüyle **aynı hata
sınıfıdır** (sentetik görünüm ≠ gerçek görünüm). Orada tek sınıfta görülmüştü;
burada dört sınıfta birden görünüyor.

---

## 1. Ölçüm — güven eşiği taraması

Eşiği 0,25'ten 0,01'e kadar indirerek, nesnenin kadrajda **büyük ve net** olduğu
karelerde:

| Nesne | Kare / sn | 0,25 | 0,10 | 0,05 | 0,02 | 0,01 |
|---|---|:--:|:--:|:--:|:--:|:--:|
| Kırmızı vana volanı (kadrajın ~%15'i) | 1650 / 55,0 | 0 | 0 | 0 | 1 → **`gauge` 0,02** | 1 |
| Kule lambası (kırmızı/yeşil yanan) | 1200 / 40,0 | 0 | 0 | 0 | **0** | **0** |
| Kule lambası | 1740 / 58,0 | 0 | 0 | 0 | **0** | **0** |
| Aydınlatmalı buton panosu | 1150 / 38,3 | 0 | 0 | 0 | 0 | 1 (çöp, `gauge` 0,01) |

Eşik meselesi değil. 0,01'de bile bulunmayan nesne için "eşiği düşürelim" bir
çözüm değildir; model o nesneyi hiç öğrenmemiştir.

Vana kutusunun **`valve` değil `gauge`** etiketiyle gelmesi ayrıca anlamlı: model
yuvarlak-ve-parlak olan her şeyi tek bir öğrenilmiş kalıba bastırıyor.

Videonun tamamındaki sınıf dağılımı, körlüğü dolaylı olarak doğruluyor:
`gauge` 811, `digital` 41, `keypad` 16, `lamp` 4, `valve` 1. Kadrajda dakikalarca
lamba ve pano dururken toplam 4 ve 16 kutu çıkması, tespitin bu sınıflarda
fiilen çalışmadığı anlamına gelir.

## 2. Kök neden — eğitim kümesinin bileşimi

`data/detect/keypad5/train/labels` etiketleri dosya adı önekine göre sayıldı
(1171 eğitim + 240 doğrulama görüntüsü):

| Kaynak | `gauge` | `digital` | `lamp` | `valve` | `keypad` | Kaynak ne? |
|---|--:|--:|--:|--:|--:|---|
| `realtr` | **265** | 0 | 0 | 0 | 0 | Gerçek endüstriyel görüntü (HuggingFace kadran seti) |
| `syn` | 53 | 0 | 0 | 0 | 0 | Kendi sentetik üretecimiz |
| `cs` | 0 | 141 | 137 | 101 | 0 | `hazirla_ip5_cok_sinif.py` — çizilmiş glif |
| `ip` | 496 | 165 | 215 | 135 | 241 | `hazirla_ip17_keypad.py` — çizilmiş glif + az sayıda gerçek kırpım |

Sentetik örneklerin neye benzediği belirleyici. `valve` örneği: düz gri gürültü
zemine çizilmiş koyu bir çubuk ve ortasında bir daire. `lamp` örneği: kusursuz
dolgulu, tek renk, kenarı keskin bir yeşil daire. Perspektif yok, krom bilezik
yok, spekülar parlama yok, arkada fabrika kalabalığı yok, derinlik yok.

Sahadaki karşılığı: vana volanı kırmızı, eğik açıdan, parlak, arkasında raflar
var. Model **glifi** öğrendi, **nesneyi** değil.

### Gerçek kırpımlar var ama tek kaynaklı

`data/detect/karistiricilar/` altında gerçek video kırpımları mevcut ve
`hazirla_ip17_keypad.py` bunları eğitim karelerine yapıştırıyor. Miktar ve
çeşitlilik yetersiz:

| Kova | Kırpım | Kaç farklı videodan |
|---|--:|---|
| `lamp/` | 24 | **1** (hepsi `10.mp4`) |
| `keypad/` | 10 | **1** (hepsi `11.mp4`) |
| `negatif/` | 101 | 4 |
| `valve/` | — | yok |
| `digital/` | — | yok |

24 kırpımın hepsi **aynı lambanın aynı ışık altındaki** görüntüsü; 10 kırpımın
hepsi **aynı panonun** görüntüsü. Bu bir sınıfı öğretmez, o tek örneği
ezberletir. `valve` ve `digital` için gerçek kırpım hiç yok.

## 3. Doğrulama kümesi de sentetik — mAP saha performansını ölçmüyor 🔴

`data/detect/keypad5/val/labels` içinde **tek bir `realtr` örneği yok**:

| Kaynak | `gauge` | `digital` | `lamp` | `valve` | `keypad` |
|---|--:|--:|--:|--:|--:|
| `cs` | 0 | 48 | 37 | 24 | 0 |
| `ip` | 130 | 42 | 56 | 31 | 72 |
| `realtr` | **0** | 0 | 0 | 0 | 0 |

Yani bu ağırlığın raporlanan mAP değeri **tamamen sentetik veri üstünde**
ölçülmüştür. Sentetik glifleri iyi bulan bir model orada yüksek mAP alır ve
sahada kör kalabilir — nitekim öyle oluyor. mAP yükselince tespitin
iyileştiğini varsaymak bu kurulumda geçersizdir.

Bu maddenin körlüğün kendisinden daha ciddi olduğu söylenebilir: körlük bir
eksiklik, bu ise **eksikliği görünmez kılan ölçüm hatasıdır.**

## 4. `digital` sınıfında ters yönlü kusur — yanlış pozitif

0831'de 37 karede `digital` tespiti var. Kare 1286 (42,9 sn) incelendi:
`digital 0,38` kutusu, zeminde duran **siyah bir makine/akü modülünün** üstünde.
Ekranda hane yok, gösterge yok.

Bu körlükten daha tehlikelidir: kutu okuma katmanına gidiyor ve depoda "sessiz
hata" diye adlandırılan sınıfa girme riski taşıyor. Kural 3 okuma aşamasında
koruyor, ama **tespit aşamasında böyle bir kapı yok**.

Aynı olgu daha önce `9s.mp4`'te de görülmüştü (DETENIDO levhaları `digital`).

## 5. Hata OLMAYAN: okunamayan %6

811 `gauge` kutusunun 50'si (%6,2) tespit edildi ama okunmadı. Kare 284
incelendi: sütun üzerinde üç manometre var, kadranlar ~45 piksel, ibre
seçilmiyor. Orada değer üretmek uydurma olurdu.

**Kural 3 tasarlandığı gibi çalışmıştır.** Bu düzeltilecek bir kusur değildir ve
"okuma oranını yükseltmek" adına gevşetilmemelidir.

Aynı şekilde videonun 20–70 sn arasındaki sessizliğinin çoğu doğrudur: 25 sn
koridor, 65 sn yükleme rampası — orada gerçekten gösterge yok. Yalnız üç noktada
(40 sn lamba, 50–58 sn vana + lamba) gerçek kaçak var, o üçü de §1'in sonucudur.

Buna karşılık kadranın gerçekten göründüğü 5–20 sn aralığında modül çalışıyor:
kare 300'de sütun üzerindeki üç manometre 0,98 / 0,82 / 0,83 güvenle bulunup
**3/3 okundu**.

---

## 6. Çözümler

Sıra önem sırasıdır. Her madde ölçülebilir bir kabul koşuluna bağlıdır; kanıt
üretilmeden tamamlandı sayılmaz.

### P0-A — Gerçek doğrulama kümesi kur (önce bu)

§3 düzeltilmeden §1'in düzeldiği **kanıtlanamaz**. Herhangi bir veri eklemeden
önce, gerçek görüntüden oluşan tutulmuş bir doğrulama bölümü gerekir.

1. `gauge` dışındaki dört sınıf için gerçek görüntü toplanır (aşağıdaki P0-B ile
   aynı çekim, ama **bölümler baştan ayrılır** — aynı sahneden hem eğitime hem
   doğrulamaya kare gitmemeli, yoksa sızıntı olur).
2. `data/detect/gercek_val/` kurulur ve `gauge5.yaml`'ın `val` alanı buraya
   bakar; sentetik doğrulama ayrı bir yaml'da saklanır, silinmez.

> **Kural 1 hatırlatması:** gerçek saha görüntüsü içeren her yeni klasör için
> `.gitignore` girdisi **klasör oluşturulmadan önce** yazılır. Depo herkese açık;
> tek koruma katmanı budur.

**Kabul ölçütü:** Mevcut ağırlık gerçek doğrulama kümesinde ölçülür ve sonuç
rapora yazılır. Sentetik mAP ile gerçek mAP arasındaki fark sayı olarak
kayıtlıdır. (Bu sayı düşük çıkacaktır — amaç iyi görünmek değil, taban çizgisini
tespit etmek.)

### P0-B — Dört sınıf için gerçek görüntü toplama

Sınıf başına hedef **≥150 gerçek örnek**, ve en az **5 farklı fiziksel nesne /
5 farklı ışık koşulu** — sayıdan çok çeşitlilik belirleyici (§2'deki 24 kırpım,
sayının tek başına yetmediğinin kanıtı).

Kaynak sırası:

1. **Saha çekimi.** Gerçek pano, gerçek lamba, gerçek vana; eğik açı, yansıma,
   kısmi örtülme, hareket bulanıklığı dahil. En değerlisi budur.
2. **Açık veri setleri.** `docs/veri_setleri_degerlendirme.md` taraması yalnız
   analog kadran için yapıldı; lamba/vana/pano/dijital panel için tekrarlanmalı.
3. **Mevcut videolardan kırpım.** `hazirla_karistiricilar.py` zaten bu yolu
   açıyor; ama kırpımlar **farklı videolardan** gelmeli, tek videodan değil.

**`docs/cekim_talimati.md` tek başına bu maddeyi kapatmaz** ve bu açıkça
yazılmalıdır: o talimat kendi sentetik göstergemizi ekranda gösterip telefonla
fotoğraflıyor. Kazandırdığı şey gerçek **optik** (mercek, ışık, sensör
gürültüsü); kazandırmadığı şey gerçek **görünüm** — ekrandaki glif hâlâ düz
vektör çizimidir. Optik boşluğu kapatır, görünüm boşluğunu kapatmaz. Sahadaki
kusur görünüm boşluğudur.

**Kabul ölçütü:** P0-A kümesinde `lamp`, `valve`, `keypad`, `digital`
sınıflarının her biri için tespit kapsamı (recall) 0'dan ölçülebilir bir değere
çıkar ve rapora yazılır. 0831.mp4'te kare 1650 (vana) ve 1200 (lamba)
`conf ≥ 0,25` ile **doğru sınıf** etiketiyle bulunur.

### P1-C — Ölçümü tekrarlanabilir hale getirme

Bu raporun tablolarının hepsi elle koşulmuş tek seferlik komutlardır.
`scripts/olc_sinif_korlugu.py` yazılmalı: girdi bir video + işaretli kare
listesi, çıktı §1'deki güven taraması tablosu. Böylece yeni ağırlık geldiğinde
ilerleme tek komutla ölçülür.

**Kabul ölçütü:** Script mevcut ağırlıkla koşturulduğunda §1 tablosunu yeniden
üretir.

### P1-D — Tespit aşamasında yanlış pozitif kapısı

§4'teki `digital` yanlış pozitifi için: `digital` kutusu, içinde hane ızgarası
bulunamadığında **tespit olarak da düşürülmeli**, yalnız okuma "unreadable"
dönmemeli. Kural 3'ün tespit aşamasındaki karşılığı yok; olması gerekiyor.

**Kabul ölçütü:** 0831.mp4'te kare 1286'daki kutu üretilmez; 9s.mp4'teki
DETENIDO levhaları `digital` sayılmaz; gerçek dijital panellerde tespit sayısı
düşmez.

---

## 7. Ne YAPILMAMALI

- **Güven eşiğini düşürmek.** §1 ölçümü bunun işe yaramadığını gösteriyor:
  0,01'de bulunmayan nesne için eşiğin anlamı yok, üstelik yanlış pozitifler
  artar.
- **Daha çok sentetik veri üretmek.** Eksik olan miktar değil, gerçekçilik.
  Aynı glifin 10.000 kopyası aynı boşluğu bırakır.
- **Sentetik mAP'e bakarak ilerleme ilan etmek.** §3 bunun neden geçersiz
  olduğunu ölçüyor. İlerleme yalnız gerçek doğrulama kümesinde ölçülür.
- **Okuma oranını yükseltmek için Kural 3'ü gevşetmek.** §5'teki %6 doğru
  davranıştır; onu "düzeltmek" sessiz hata üretir.
