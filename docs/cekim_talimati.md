# İP8 — Ekrandan Çekim Talimatı

> **Toplam süre: ~30 dakika.** Gereken: bilgisayar + telefon. Başka bir şey yok.
>
> Amaç: kadranı ekranda gösterip telefonla fotoğraflamak. Böylece görüntü
> **gerçek mercekten, gerçek ışıktan, gerçek sensörden** geçmiş oluyor ama
> doğru cevabı biz biliyoruz, çünkü kareyi biz ürettik. Elle etiketleme yok.

---

## 1. Ekranda kadranı aç

PowerShell'de, kod reposunun içinde:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\ekran_kadran.py --gosterge PT-101 --adet 12
```

Ekran tam ekran olur. Ortada kadran, **altında büyük bir `#01`** görürsün.
Terminale de hangi karede hangi değerin olduğu yazılır (sana lazım değil,
sadece merak edersen).

**Tuşlar:** `BOŞLUK` sonraki kare · `←` önceki · `q` çıkış

---

## 2. Fotoğrafları çek

12 kare var. Her kare için:

1. Telefonu kaldır, **kadranın tamamı + alttaki `#NN` yazısı** kadraja girsin.
2. Odaklanmasını bekle, çek.
3. `BOŞLUK`'a bas, sonraki kare gelsin.
4. Tekrarla — 12 kere.

### Çekerken dikkat

> ⚠ **19.08'de ilk çekim yapıldı ve buradaki tarif yetersiz çıktı.** "Kadranın
> tamamı kareye girsin" denmişti; makul biçimde tüm dizüstü + oda çekildi ve
> kadran karenin küçük bir parçası kaldı, üstelik çoğu kare çok eğikti.
> Zincir 12 karenin 12'sinde merkez rafinesini ve yatıklık kestirimini
> yapamadı. Aşağıdaki iki madde bu yüzden **kalın.**

| ✅ Yap | ❌ Yapma |
|---|---|
| **Kadran kareyi DOLDURSUN** — kadran + `#NN` şeridi karenin en az %70'i olsun | Ekranı, klavyeyi, masayı, odayı kareye alma |
| **Telefonu ekrana MÜMKÜN OLDUĞUNCA DİK tut** ve **yatay tut, eğme** | Yandan/yukarıdan bakma, telefonu döndürme |
| Ekrandan **25-40 cm** uzakta dur | Çok yaklaşma (piksel deseni çıkar) |
| Oda ışığı normal olsun | **Flaş kullanma** (ekran patlar) |
| Telefonu sabit tut, odak otursun | Dijital zoom yapma |
| `#NN` yazısı okunur kalsın | Alt şeridi kadraj dışında bırakma |

**Neden bu kadar katı:** ölçmek istediğimiz şey "gerçek mercek ve sensör zincire
ne katıyor". Kamerayı eğersen, üstüne bir de perspektif hatası binder ve iki
etki ayrılamaz hâle gelir — çıkan sayı hiçbir soruyu cevaplamaz. Eğikliğin
etkisi **zaten** İP14'te tek tek ölçüldü.

**Telefonu eğmemek ayrıca önemli:** yatıklık kestirimi kadranın çizgi deseninden
çalışıyor ve eğik fotoğrafta o desen bozuluyor. İlk çekimde kestirim 12/12
başarısız oldu.

**Moiré (ekran deseni) çıkarsa:** telefonu birkaç santim ileri/geri al. Yan
durarak çözme — eğiklik daha büyük sorun.

### İkinci tur (isteğe bağlı, sonra)

Kolay set çalıştıktan **sonra** bir de eğik set çekilebilir: aynı 12 kare, ama
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
data\real\PT-101_ekran\
```

Klasör yoksa oluştur. Dosya adlarına dokunma — telefon `IMG_0041`, `IMG_0042`
diye artan numara verir, sıralama ondan geliyor.

> `data/real/` **git'e girmiyor** (1. kural). Fotoğraflar bilgisayarında kalır,
> GitHub'a gitmez.

---

## 4. Ölçümü koştur

```powershell
python scripts\olc_ip8.py --fotograflar data\real\PT-101_ekran
```

Ekrana hata tablosu basılır ve iki dosya yazılır:

- `outputs\metrics\ip8_ekran_hatasi.json` — sayılar
- `outputs\figures\ip8_kontak_sayfasi.png` — **kontak sayfası**

---

## 5. Kontak sayfasına bak (10 saniye)

`ip8_kontak_sayfasi.png` dosyasını aç. 12 fotoğrafı yan yana görürsün; her
fotoğrafın altında sarı yazıyla ona atanan değer var.

**Kontrol et:** fotoğrafın *içindeki* `#03` ile altındaki sarı yazının `#03`'ü
aynı mı? Hepsinde aynıysa eşleşme doğru, tablo güvenilir.

Kaymışsa (fotoğrafta `#04` ama altta `#03` yazıyor) tablo çöptür — fazla/eksik
fotoğrafı bul, düzelt, 4. adımı tekrarla.

---

## Sonra ne olacak

Çıkan sayı, zincirin **gerçek optik yoldaki** hatasıdır ve sentetikteki %0,19
ile yan yana raporlanır. Aradaki fark, "sentetikte iyi çalışıyor" ile "gerçekte
çalışıyor" arasındaki mesafedir — İP8'in bitti kriteri budur.

İstersen aynı şeyi TI-205 ve FI-310 için de tekrarla (`--gosterge` değiştir).
FI-310 karekök ölçekli, en zorlayıcı olan o. Ama önce PT-101 yeter.

## Bu ölçüm neyi ölçmez

Ekranda **cam yansıması, metal doku, tozlanma ve gerçek sanayi aydınlatması
yok.** Bu tablo "sahada ne olur"u değil, "gerçek optik yol zincire ne kadar
hata katıyor"u söyler. Gerçek manometreyle test hâlâ ayrı bir iştir
(`../../SORULAR.md` · S1 · B seçeneği).
