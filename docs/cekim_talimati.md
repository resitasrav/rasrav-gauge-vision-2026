# İP8 — Ekrandan Çekim Talimatı (dört tip)

> **Toplam süre: ~40 dakika, 28 fotoğraf.** Gereken: bilgisayar + telefon.
>
> Amaç: göstergeleri ekranda gösterip telefonla fotoğraflamak. Böylece görüntü
> **gerçek mercekten, gerçek ışıktan, gerçek sensörden** geçmiş oluyor ama
> doğru cevabı biz biliyoruz, çünkü kareyi biz ürettik. Elle etiketleme yok.
>
> Artık dört tip birden çekiliyor: analog kadran (12), dijital panel (8),
> ikaz lambası (4), vana (4). Dijital/lamba/vana sayıları (%93,3 · %100 · %100)
> şimdiye kadar yalnız sentetik veride ölçüldü — gerçek optik yoldan ilk kez
> bu çekimle geçecekler.

---

## 1. Ekranda göstergeleri aç

PowerShell'de, kod reposunun içinde:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\ekran_kadran.py
```

Ekran tam ekran olur. Ortada gösterge, **altında büyük bir `#01`** görürsün.
Sıra şöyle akar (terminale de tek tek yazılır):

| Kareler | Gösterge | Ne göreceksin |
|---|---|---|
| #01–#12 | PT-101 analog kadran | ibre her karede başka değerde |
| #13–#20 | DP-401 dijital panel | 7-segment sayılar (eksili olanlar da var) |
| #21–#24 | LM-501 ikaz lambası | sönük, sönük-yeşilimsi, yeşil, kırmızı |
| #25–#28 | VL-601 vana | kol yatay, dik, hafif sapmış, 45° arada |

**Tuşlar:** `BOŞLUK` sonraki kare · `←` önceki · `q` çıkış

---

## 2. Fotoğrafları çek

Her kare için:

1. Telefonu kaldır, **göstergenin tamamı + alttaki `#NN` yazısı** kadraja girsin.
2. Odaklanmasını bekle, çek.
3. `BOŞLUK`'a bas, sonraki kare gelsin.
4. Tekrarla — 28 kere.

### Çekerken dikkat

> ⚠ **19.08'de ilk çekim yapıldı ve buradaki tarif yetersiz çıktı.** "Kadranın
> tamamı kareye girsin" denmişti; makul biçimde tüm dizüstü + oda çekildi ve
> kadran karenin küçük bir parçası kaldı, üstelik çoğu kare çok eğikti.
> Sonuç: 12 karenin yalnız 5'i okunabildi, yatıklık kestirimi 12/12 başarısız
> oldu. Aşağıdaki iki madde bu yüzden **kalın.**

| ✅ Yap | ❌ Yapma |
|---|---|
| **Gösterge kareyi DOLDURSUN** — gösterge + `#NN` şeridi karenin en az %70'i olsun | Ekranı, klavyeyi, masayı, odayı kareye alma |
| **Telefonu ekrana MÜMKÜN OLDUĞUNCA DİK tut** ve **yatay tut, eğme** | Yandan/yukarıdan bakma, telefonu döndürme |
| Ekrandan **25-40 cm** uzakta dur | Çok yaklaşma (piksel deseni çıkar) |
| Oda ışığı normal olsun | **Flaş kullanma** (ekran patlar) |
| Telefonu sabit tut, odak otursun | Dijital zoom yapma |
| `#NN` yazısı okunur kalsın | Alt şeridi kadraj dışında bırakma |

**Neden bu kadar katı:** ölçmek istediğimiz şey "gerçek mercek ve sensör zincire
ne katıyor". Kamerayı eğersen, üstüne bir de perspektif hatası biner ve iki
etki ayrılamaz hâle gelir — çıkan sayı hiçbir soruyu cevaplamaz. Eğikliğin
etkisi **zaten** İP14'te tek tek ölçüldü.

**Dijital panel karelerinde** panel yatay ve geniş; kadraja sığdırmak için
uzaklaşma, telefonu yatay çevir (panorama değil, normal fotoğraf, telefon
yan tutulmuş). Segmentlerin tek tek seçilebildiğinden emin ol.

**Moiré (ekran deseni) çıkarsa:** telefonu birkaç santim ileri/geri al. Yan
durarak çözme — eğiklik daha büyük sorun.

### İkinci tur (isteğe bağlı, sonra)

Kolay set çalıştıktan **sonra** bir de eğik set çekilebilir: aynı kareler, ama
telefon 20-30° yandan. O zaman iki tablo yan yana konur ve "gerçek fotoğrafta
eğiklik ne kadar hata katıyor" ayrı bir satır olarak raporlanır.

### En önemli kural

> **Kare atlama, iki kez çekme.** Eşleştirme çekim sırasına göre yapılıyor.
> Yanlışlıkla fazla çekersen **hemen sil**; atladığını fark edersen `←` ile geri
> dön ve o kareyi çek — ama o zaman fotoğrafların sırası bozulur, o yüzden en
> temizi baştan başlamak.
>
> Ölçüm scripti fotoğraf sayısını sayıyor; tutmuyorsa hiç ölçüm yapmadan hata
> veriyor. Yani yanlış bir tabloyla karşılaşmayacaksın — ama tekrar çekmen
> gerekir.

---

## 3. Fotoğrafları bilgisayara at

Kod reposunun içine, şu klasöre:

```
data\real\ip8_ekran\
```

Klasör yoksa oluştur. Dosya adlarına dokunma — telefon `IMG_0041`, `IMG_0042`
diye artan numara verir, sıralama ondan geliyor.

> `data/real/` **git'e girmiyor** (1. kural). Fotoğraflar bilgisayarında kalır,
> GitHub'a gitmez. (19.08'in ilk çekimi `data\real\PT-101_ekran\` altında
> duruyor; ona dokunma, kıyas için lazım.)

---

## 4. Ölçümü koştur

```powershell
python scripts\olc_ip8.py --fotograflar data\real\ip8_ekran
```

Tespit varsayılan olarak dört sınıflı YOLO ile yapılır (13.08 modeli).
Ekrana tip tip sonuç basılır ve iki dosya yazılır:

- `outputs\metrics\ip8_ekran_hatasi.json` — sayılar
- `outputs\figures\ip8_kontak_sayfasi.png` — **kontak sayfası**

---

## 5. Kontak sayfasına bak (10 saniye)

`ip8_kontak_sayfasi.png` dosyasını aç. 28 fotoğrafı yan yana görürsün; her
fotoğrafın altında sarı yazıyla ona atanan doğru cevap var (analogda değer,
dijitalde dizge, lamba/vanada durum).

**Kontrol et:** fotoğrafın *içindeki* `#03` ile altındaki sarı yazının `#03`'ü
aynı mı? Hepsinde aynıysa eşleşme doğru, tablo güvenilir.

Kaymışsa (fotoğrafta `#04` ama altta `#03` yazıyor) tablo çöptür — fazla/eksik
fotoğrafı bul, düzelt, 4. adımı tekrarla.

---

## Sonra ne olacak

Çıkan sayılar tip tip, sentetik referanslarının yanına konarak raporlanır:

| Tip | Sentetik referans | Bu çekim ölçecek |
|---|---|---|
| analog | %0,19 (zincir) | gerçek optik yolda tam skala hata |
| dijital | %93,3 dizge | dizge doğruluğu |
| lamba | %100 | durum doğruluğu |
| vana | %100 | durum doğruluğu (+ ara konumda susma) |

Gerçek sayının sentetikten **kötü çıkması beklenir ve normaldir** — aradaki
fark, "sentetikte iyi çalışıyor" ile "gerçekte çalışıyor" arasındaki mesafedir;
İP8'in bitti kriteri bu farkın ölçülmüş olmasıdır.

## Bu ölçüm neyi ölçmez

Ekranda **cam yansıması, metal doku, tozlanma ve gerçek sanayi aydınlatması
yok.** Ayrıca dijital/lamba/vana görüntüleri kendi sentetik üretecimizden
geliyor: bu çekim "optik yolun katkısını" ölçer, **"başka marka bir panele
genellemeyi" değil.** Evdeki gerçek bir fırın/klima/multimetre ekranının ve
gerçek bir ikaz lambasının fotoğrafı bu yüzden ayrıca isteniyor — o da
`../../SORULAR.md`'de (S1 · B seçeneği ile aynı başlık).
