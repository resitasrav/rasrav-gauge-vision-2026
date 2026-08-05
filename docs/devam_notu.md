# Devam Notu — Oturum Kapanışı

**Yazıldığı an:** 07.08.2026 · Gün 10/30 · H2'nin son günü
**Amaç:** Yeni oturum bu dosyayı okuyunca kaldığı yerden devam edebilsin. Durum özeti
CLAUDE.md'de; **burada yalnızca "sırada ne var ve neye dikkat et" var.**

---

## 1. Nerede kaldık

**İP1-İP7 bitti, zincir uçtan uca çalışıyor.** Sentetik veride (v1, eğitimde görülmemiş
100 görüntü) **%0,19 tam skala** — hedef %5, okunamayan kare 0.

| İP | Ölçüm | Dosya |
|---|---|---|
| İP6 | ibre açısı **0,123°** (kutupsal tarama) | `outputs/metrics/ip6_aci_hatasi.json` |
| İP7 | okuma hatası **%0,129 tam skala** | `outputs/metrics/ip7_okuma_hatasi.json` |
| İP5 | tespit **mAP50 0,967**, kaçırılan 1/173 | `outputs/metrics/ip5_tespit.json` |
| zincir | **%0,19** · yatıklık kestirimi 0,035° | `outputs/metrics/ip8_zincir_hatasi.json` |

149/149 test geçiyor. İki repo da temiz. Arka planda koşan bir şey yok.

**Bütçe artık okuma yönteminde tıkanıyor** — zincir kendi tabanının 1,5 katında:

| Kalem | Puan | Payı |
|---|---|---|
| Okuma yöntemi (İP6+İP7) | 0,129 | %68 |
| Tespit merkezi (rafineden sonra) | 0,051 | %27 |
| Yatıklık kestirim artığı | 0,010 | %5 |

Sentetik veride sıkılacak yer kalmadı; buradan sonra kazanç **gerçek görüntüde**
aranmalıdır. Bu sayılar zorluğu ölçmez, yalnızca yöntemin kendi tabanını gösterir.

---

## 2. Sıradaki iş — İP8: gerçek görüntüde uçtan uca test

**Neden bu:** Zincirin sentetikteki hatası artık ölçüm gürültüsü seviyesinde. Gerçek
göstergede cam yansıması, açılı bakış, sanayi aydınlatması ve tozlanma var; bunların
hiçbiri sentetikte yok. Sayının gerçek karşılığı bilinmiyor.

**🔴 Engel — çözülmeden İP8 başlayamaz: ground truth kaynağı yok.** A1/A2'nin Drive
klasörü 404 verdi, A5 etiketsiz. Gerçek görüntüde ibre değeri etiketli açık kaynak
kalmadı. Üç seçenek `docs/veri_setleri_degerlendirme.md` §2.3'te.

**Önerilen (06.08'de doğdu, henüz denenmedi): ekrandan çekim.** Sentetik kadranı ekranda
gösterip fotoğraflamak = **gerçek optik yol + tam bilinen ground truth**. Görüntü gerçek
mercekten, ışıktan, sensörden geçer; değeri birebir bilinir çünkü kareyi biz ürettik.
Gerçek manometrenin yerini tutmaz, sentetik ile gerçek arasında bir basamaktır.
Canlı demo bunun çalıştığını zaten gösterdi (telefon ekranı → %1,25 hata).

**Ölçüm hattı hazır:** `scripts/olc_zincir.py --veri <yeni_kume>` aynı ablasyon
ızgarasını yeni veri üzerinde koşturur, elle bir şey yapılmaz.

---

## 3. Sonraki üç iş (öncelik sırasıyla)

1. **İP8** — yukarıdaki engel çözülünce. Bu sırada üç eşiğin gerçek görüntüde yeniden
   ölçülmesi gerekir; hepsi sentetikte kalibre edildi ve kod içinde ⚠ ile işaretli:
   `refine.MAX_ARTIK_ORANI`, `refine.MAX_YAYILMA_ORANI`, `roll.MIN_UYUM`.
2. **KT2 — `inspect/reading` şemasını dondur.** 31 Tem'de gecikti (defterde U7 🔴).
   Alanlar belli: `gauge_id, type, value, unit, conf, status, raw_angle, img_ref`.
   `GaugeReading.as_message()` bunu zaten üretiyor. Daha fazla bekletilmemeli.
3. **İP10 — MQTT yayını.** U5 çözülmezse replay ile gösterilemez.

---

## 4. Karar bekleyenler (Reşit'ten / ekipten)

- **🔴 İP8 ground truth kaynağı** — bkz. §2. H3'e (10 Ağu) girmeden karara bağlanmalı.
- **🔴 U5** — Bedirhan'ın `recorder.py`'ı `inspect/reading`'i kaydetmiyor; İP8 ve İP10
  buna bağlı. 05.08'de hâlâ değişmemişti.
- **U11 — waypoint kimlik sözlüğü.** Gösterge kimliği hâlâ elle geliyor. Kısmi otomatik
  koruma çıktı: yanlış kimlik verilirse yatıklık kestirimi susuyor (`roll.MIN_UYUM`,
  ölçülen ayrım 0,22 vs 0,63). Sözlüğün yerini tutmaz ama sessiz hatayı görünür kılar.
- **Danışman sorusu:** "%5 ortalama hata" tam skalanın mı okunan değerin mi yüzdesi?
  Tam skala seçildi, iki tanım da ölçülüp JSON'a yazıldı.

---

## 5. Tuzaklar — zaman kaybettirenler

- **"Kapı" yazmak kapı kurmak değildir.** 07.08'de aynı hata iki modülde arka arkaya
  yapıldı: `refine.py` ve `roll.py`'ın ilk sürümlerindeki güven kapıları **rastgele
  gürültüyü kabul ediyordu** (50/50 ve 8/10). İkisinde de sebep aynıydı: kapılar
  "cevap makul mü" diye soruyordu, "kanıt var mı" diye değil. Yeni bir kapı yazınca
  **onu sahte girdiyle sınamadan bitmiş sayma** — her ikisi de ancak gürültü testiyle
  yakalandı. Eşiği tahminle değil, iki kümenin dağılımını ölçüp aralarına koy.
- **torch tekerleği.** Bu makinede RTX 4050 var ve `cu126` sürümü kurulu. `pip install
  torch` PyPI'dan **CPU** sürümünü getirir ve bu **sessizce çalışır, sadece kartı
  kullanmaz.** Bozulursa `requirements.txt` başındaki komut, sonra
  `torch.cuda.is_available()` ile doğrula. Fark: eğitim 45 dk → 5 dk.
- **Ultralytics çıktı yolu.** `project="models/ip5"` verilse bile ayarlarındaki `runs_dir`
  öne ekleniyor → gerçek yol `runs/detect/models/ip5/...`.
- **Veri sızıntısı.** Zincir ölçümü `v1` (tohum 1) üzerinde koşar; `v0`'ın 53 karesi
  karışık eğitimin içindeydi. Yeni ölçüm kümesi üretirken `--ozet` ver, yoksa İP3'ün
  kayıtlı özeti ezilir.
- **Uyuşmazlık defteri git'e girmez.** `STAJ\ortak uyusmazliklar\uyusmazliklar.md`
  yereldir, iki reponun da dışındadır. 11 madde var (U1-U11).
- **PowerShell + git commit.** Mesajda çift tırnak varsa PowerShell argümanı bölüyor.
  Uzun mesajı dosyaya yazıp `git commit -F dosya` ile ver.
- **Raporlar staj iş gününe göre tarihlenir**, dosyanın yazıldığı güne göre değil.

---

## 6. Bu oturumda yapılanlar (özet)

- **`detect/refine.py`** — kadran merkezi kutudan değil çemberden. Merkez sapması
  %1,31 → **%0,06**, maliyet 1,5 ms. Gradyan doğrularının kapalı form kesişimi;
  akümülatör yok. Fikir kaynağı Reşit'in `Cascaded-Soft-Hough` çalışması.
- **`read/roll.py`** — kamera yatıklığı kadranın çizgilerinden. Beklenen desenle
  dairesel korelasyon; hata **0,035°**, maliyet 0,8 ms.
- **`Gauge.tick_values()`** `config.py`'a taşındı — çizgi düzeni artık tek yerde,
  üreteç ile okuyucu aynı kaynaktan besleniyor.
- `olc_zincir.py` 2×3 ablasyon ızgarasına çevrildi; bütçe tablosu koşudan doğuyor.
- Zincir **%1,92 → %0,19**, okunamayan 8 → 0.
- Testler 101 → 149.
