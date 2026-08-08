# Reşit'e Sorular — Karar Bekleyenler

> Gece oturumunda biriken sorular. **Çalışma hiçbir noktada durdurulmadı**; her
> soru için makul bir varsayımla devam edildi ve varsayımın ne olduğu yazıldı.
> Cevap gelince ilgili yer güncellenecek.
>
> **Oturum:** 07.08.2026 gecesi → 08.08 sabahı ·
> **Yapılan:** İP14, İP15, İP11, İP12, İP10, İP13 + perspektif düzeltmesi

---

## ✅ 08.08 sabahı cevaplananlar

| | Soru | Karar |
|---|---|---|
| **S1** | İP8 ground truth kaynağı | **A — ekrandan çekim.** Sentetik kadran ekranda gösterilip fotoğraflanacak. Yarım günlük iş, elle etiketleme yok. |
| **S5** | Güven eşiği / kapsama dengesi | **0,70 kalıyor.** Kapsama %88,1, bin turda ~2 hatalı sayı kabul edildi. |
| **S4** | Eşik gösterge başına farklılaşsın mı | **Hayır** — S5'in cevabı tek genel eşiği onayladı. `conf_threshold` ezme yolu envanterde açık kalıyor ama kullanılmıyor. |

**S2 ve S3 bilinçli olarak AÇIK bırakıldı** (Reşit'in kararı, 08.08): ikisinin de
cevabı sahayı görmeden verilemez. Bunun yerine **ikisi de envanterden ayarlanır
hâle getirildi** — cevap gelince değişecek olan YAML satırı, kod değil.

| | Eskiden | Şimdi |
|---|---|---|
| **S2** vana montajı | "yatay = açık" koda gömülüydü | `states[].lever_angle` envanterde · `scripts/kalibre_vana.py` sahada ölçüp satırı yazıyor |
| **S2** tolerans | kodda `VANA_TOLERANS_DEG = 20` | `reading.tolerance_deg` envanterde (kod↔envanter ayrışması artık imkânsız) |
| **S3** eksi işareti | `allow_minus` YAML'da vardı ama **okunmuyordu** | uygulanıyor: panel eksi gösteremiyorsa eksi çözen okuma `unreadable` |

---

## 🔴 Senin kararın olmadan ilerleyemeyecek olan — TEK MADDE

### S1. İP8'in gerçek görüntü ground truth kaynağı — ✅ **A seçildi (08.08)**

**Neden bekliyor:** İP8 "gerçek gösterge fotoğraflarında uçtan uca hata tablosu"
istiyor. A1/A2 setleri erişilemez (05.08, HTTP 404), A5 etiketsiz. İbre değeri
etiketli açık kaynak kalmadı.

**Bu gece ne yapıldı:** İP8'i beklemek yerine ondan bağımsız beş iş paketi
bitirildi (İP14, İP15, İP11, İP12, İP10). Yani engel iş kaybına yol açmadı, ama
**artık gerçekten sıra ona geldi** — H4'te İP13 (canlı test) de aynı veriye
ihtiyaç duyacak.

**Üç seçenek** (`docs/veri_setleri_degerlendirme.md` §2.3'te ayrıntılı):

| | Ne | Maliyet | Ne kadar gerçek |
|---|---|---|---|
| **A** | Ekrandan çekim: sentetik kadranı ekranda gösterip fotoğraflamak | ~yarım gün | Gerçek optik yol + **tam bilinen değer** |
| B | Gerçek manometre alıp elle etiketlemek | 2-3 gün | Tam gerçek, etiket hatası riski var |
| C | Açık veriden bulup elle etiketlemek | 2+ gün | Gerçek ama az örnek |

**Önerim A.** 06.08'de telefon ekranında yapılan denemede sistem 7,775 bar'lık
kadranı 7,9 bar okumuştu (%1,25 hata) — yöntem çalışıyor. Görüntü gerçek
mercekten, gerçek ışıktan, gerçek sensörden geçiyor; değer ise birebir biliniyor
çünkü kareyi biz ürettik. **Gerçek manometrenin yerini tutmaz** (cam yansıması,
metal doku, tozlanma yok) ama sentetik ile gerçek arasında bir basamaktır ve
elle etiketleme gerektirmez.

**Senden gereken:** A/B/C tercihi. A seçilirse bir telefon ve yarım saat yeter.

---

## 🟡 Varsayımla ilerlendi — onayın gerekiyor

### S2. Vananın "açık" konumu montaja göre değişir mi? — 🟢 **açık, ama artık engel değil (08.08)**

**Varsayım (değişmedi):** kol boru hattına **paralel (yatay)** ise `open`, **dik**
ise `closed`. Envanterdeki not bunu ima ediyor ("Kol boru hattına paralel").

**Risk (değişmedi):** gerçek montajda ters olabilir ve bu **sessizce** yanlış
durum üretir — vana kapalıyken "açık" raporlanır.

**08.08'de yapılan — varsayım koddan envantere taşındı.** Durum↔açı eşleşmesi
artık `states[].lever_angle` alanından geliyor, tolerans da
`reading.tolerance_deg`'den. Montaj farkı = YAML'da iki sayıyı takas etmek;
kodda tek satır değişmiyor (2. kural).

Üç sağlama eklendi (`config.py`), üçü de sessiz hata sınıfına karşı:
- durumların **bir kısmı** açı beyan edip diğerleri etmezse reddediliyor
- iki durum toleranslarıyla **çakışıyorsa** reddediliyor (`open: 0`, `closed: 10`
  ±20° ile ayırt edilemez; kod yine cevap üretirdi ve o cevap yazı-tura olurdu)
- tolerans 0-90 dışındaysa reddediliyor

**Sahada nasıl kapatılır (5 dakika):** vananın açık ve kapalı hâlinden birkaç
fotoğraf çekilir, dosya adları `open_01.jpg` / `closed_01.jpg` yapılır ve
`python scripts\kalibre_vana.py --klasor <klasör>` koşturulur. Script kol
açılarını ölçüp YAML'a yapıştırılacak satırları basar.

*Araç bilinen ground truth'ta doğrulandı:* 12 sentetik karede `closed` 89,1°
(beyan 90 → sapma 0,9°), `open` 1,8° (beyan 0 → sapma 1,8°), yayılma < 4,7°.

**Senden hâlâ gereken:** sahaya inince o birkaç fotoğraf. Karar değil, ölçüm.

**Neden daha büyük bir öğrenici değil:** durum başına öğrenilecek **tek bir
sayı** var. Bir sınıflandırıcı bunu üç fotoğrafla daha iyi yapamaz, sadece
neden öyle karar verdiğini söyleyemez hâle gelir. Geometrik kestirim yetmezse
(ıslak/paslı kol, kısmi kapanma) doğru adım İP12'ye sınıflandırıcı eklemek
olur; o zaman bu script etiketli kümenin sağlamasını yapan araç olarak kalır.

### S3. Dijital panelde negatif değerler kabul edilebilir mi? — 🟢 **açık, ama artık engel değil (08.08)**

**08.08'de yapılan:** `allow_minus` alanı envanterde zaten vardı ama **okuyucu
onu hiç okumuyordu** — yani bayrak bir belge, kod başka bir şey yapıyordu (B6'nın
aynısı). Şimdi uygulanıyor: `allow_minus: false` iken eksi çözen bir okuma
`unreadable` dönüyor. **İşaret atılıp pozitif sayı UYDURULMUYOR** — bilinen bir
okuma hatasının üstüne makul görünen değer koymak 3. kuralın tam yasağıdır.

DP-401 şu an `true`; cevabın gelmesi tek satırlık bir değişiklik.


**Ölçülen:** eksi işaretli okumalarda güven ~0,75'e düşüyor ve DP-401'in 0,80
eşiğini geçemiyor. Sebep: eksi yalnızca orta çubuğu yakar, yüksekliği segment
kalınlığı kadardır ve hane bulma filtresine takılır.

**Şu anki davranış:** değer **doğru çözülüyor ama yayınlanmıyor** (`unreadable`).
Yani yanlış okuma yok, kapsama kaybı var.

**Varsayım:** kapsama kaybı kabul edilebilir, çünkü 3. kural yanlış okumayı
yasaklıyor. **Kalıcı çözüm** hane ızgarasını görüntüden değil İP5'in panel
kutusundan kurmak — İP13'te zincire bağlanırken yapılabilir.

**Senden gereken:** DP-401 gerçekte negatif değer gösteriyor mu? Göstermiyorsa
bu sorun hiç yok demektir ve `allow_minus: false` yapılabilir.

### S4. Güven eşiği gösterge başına farklılaşsın mı? — ✅ **Hayır (08.08)**

**Ölçülen:** eşik 0,70 seçildi; kapsama %88,1, sessiz hata %0,22. Ama eksen
kırılımında **eğiklik ekseninde kapsama %61,4** — yani eğik bakılan bir
göstergede her üç okumadan biri "okunamadı" dönüyor.

**Varsayım:** tek bir genel eşik yeterli. Envanter gösterge başına ezmeye izin
veriyor (`conf_threshold` gösterge düzeyinde tanımlanabilir) ama şu an
kullanılmıyor.

**Senden gereken:** kritik göstergelerde (PT-101 alarmlı) eşik daha mı yüksek
olmalı? Yüksek eşik = daha az yanlış ama daha çok "okunamadı".

### S5. Kapsama/risk dengesi doğru yerde mi? — ✅ **0,70 onaylandı (08.08)**

Ölçülen ödünleşme:

| Eşik | Kapsama | Bin turda kaç yanlış sayı |
|---|---|---|
| 0,55 | %89,8 | ~6 |
| **0,70** *(seçilen)* | **%88,1** | **~2** |
| 0,90 | %34,4 | 0 |

**Varsayım:** bin turda ~2 hatalı okuma kabul edilebilir; %34 kapsama
kullanılamaz. **Bu bir mühendislik kararıdır, optimizasyon değil** — sahada
kabul edilebilir riski senin/danışmanın belirlemesi gerekiyor.

---

## ⚪ Bilgi notu — sadece haberin olsun

### B1. Sentetik zorluk gerçek zorluğun yerini tutmaz

`synth/degrade.py` gerçek fotoğrafa yaklaşmak için yazıldı ama hâlâ **bizim
çizdiğimiz kadranlar**. Cam yansıması modellenmiş olsa da metal doku, tozlanma,
paslanma ve gerçek sanayi aydınlatması yok. İP14'ün tablosu "yöntem bu
bozulmalara ne kadar dayanıyor"u gösterir, "sahada ne olur"u değil.

### B2. Eğiklik tek başına baskın etken

İP14'ün en net sonucu: diğer dört bozulmanın toplamı, 30°'lik bir eğimin tek
başına ürettiği hatanın altında. **İP13'ün düzeneğini bu belirlemeli** — kamerayı
kadrana dik konumlandırmak, aydınlatmayı iyileştirmekten çok daha değerli.

### B3. Parlama okumayı değil TESPİTİ öldürüyor

%90 parlamada 25/60 karede YOLO kadranı hiç bulamıyor; bulduğu karelerde okuma
hatası bozulmasızla aynı (%0,25). Yani çözüm okuma yönteminde değil **İP5'in
eğitim kümesinde**: parlamalı örnek eklemek gerekiyor.

### B4. KT2 kod tarafında hazır

`inspect/reading` şeması dondurulmuş durumda: `schema: 1`, katı doğrulayıcı ve
20 test. Ekip onayı (U1-U3) gelmedi ama **onay geldiğinde kod değişikliği
gerekmeyecek**; farklı bir karar çıkarsa sürüm numarası artırılır.

### B5. U5 artık engel değil

Bedirhan'ın kayıt aracı `inspect/reading`'i kaydetmiyor. Yayıncı broker
bağımsız çalışacak şekilde yazıldı (dosya modu + JSONL doğrulayıcı), dolayısıyla
İP10 bu bağımlılığa takılmadan bitti. Entegrasyon sırasında yine gerekecek.

### B6. Envanterle kod arasında sessiz uyuşmazlık çıkabiliyor

Vana toleransında envanter ±20° diyordu, kod fiilen ±6° yapıyordu. İkisi de
kendi içinde tutarlı olduğu için hiçbir birim testi yakalayamazdı; ancak uçtan
uca ölçüm görünür kıldı. **Ders:** envanterdeki her sayısal beyan için, kodun o
beyanı gerçekten uyguladığını sınayan bir test olmalı. Vana için eklendi
(`test_tolerans_siniri_envanterle_tutarli`); benzerini `sweep_deg` için
İP2'de yapmıştık. Diğer beyanlar gözden geçirilmeli.
