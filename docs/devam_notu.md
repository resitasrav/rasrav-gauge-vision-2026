# Devam Notu — Oturum Kapanışı

**Yazıldığı an:** 21.08.2026 · Gün 20/30 · H4 kapandı
**Amaç:** Yeni oturum bu dosyayı okuyunca kaldığı yerden devam edebilsin.
Durum özeti bağlam dosyasında; **burada yalnızca "sırada ne var ve neye dikkat et" var.**

> 👉 **Reşit'in bakması gerekenler ayrı dosyada:** `..\..\SORULAR.md`
> (ana STAJ klasörü altında, **git'e girmiyor** — uyuşmazlık defteriyle aynı yerde.)

---

## 1. Nerede kaldık

**İP8 kapandı — hedef gerçek görüntüde sağlandı.** Kalan: İP9 (kırpılabilir), İP16.

| İP | Ölçüm | Dosya |
|---|---|---|
| **İP8 analog — GERÇEK fotoğraf** | **%0,473** · kapsama 10/10 · hedef %5 | `outputs/metrics/ip8_ekran_hatasi.json` |
| İP8 lamba — gerçek fotoğraf | **4/4 (%100)** | aynı dosya |
| İP8 vana — gerçek fotoğraf | **4/4 (%100)** · ara konum doğru `unreadable` | aynı dosya |
| İP8 dijital — gerçek fotoğraf | **0/5** · sessiz hata 0 | `outputs/metrics/ip8_dijital_tani.json` |
| Zincir (analog, sentetik) | %0,21 tam skala | `outputs/metrics/ip8_zincir_hatasi.json` |
| İP15 güven eşiği | 0,70 · kapsama %88,1 | `outputs/metrics/ip15_guven_esigi.json` |
| İP11 dijital panel (sentetik) | %93,3 dizge | `outputs/metrics/ip11_dijital.json` |
| İP12 lamba/vana (sentetik) | %100 / %100 | `outputs/metrics/ip12_lamba_vana.json` |
| İP10 MQTT | 12/12 şema uyumlu | `outputs/mqtt/*.jsonl` |
| Zincir hızı (1080p, RTX 4050) | 95,6 ms/kare — 10,5 kare/s | YOLO 28,6 + okuma 67,0 |

**247/247 test geçiyor. İki repo da temiz.**

---

## 2. Sıradaki iş — dijital panelin gerçek fotoğrafta okunması

**Neden bu:** dört tipin üçü gerçek optik yolda çalışıyor, dijital çalışmıyor.
Teşhis bitti ve tespit suçsuz: panel **0,954** güvenle tam yerinden bulunuyor,
kutu bir hanenin üstüne düştüğünde rakam **1,000 güvenle DOĞRU** çözülüyor.
Çöken adım **hane kutusu bulma** (5/5).

**Sebep:** `read/digital.py::_segment_maskesi` zeminin panel boyunca sabit
olduğunu varsayıyor (iki kademeli Otsu). Gerçek fotoğrafta ekranda yansıma
gradyanı var — sol üçte birin zemin medyanı sağın **1,53-1,65 katı** ve bu fark,
zemin ile sönük segment arasındaki farktan büyük.

**⚠ ÖNCE ŞUNU OKU: üç düzeltme denendi ve ÖLÇÜMLE ELENDİ.** Sayıları
`scripts/tani_dijital.py` içinde duruyor, aynı yolu ikinci kez deneme:

| Deneme | Sonuç |
|---|---|
| Gauss ile zemin çıkarma | 0/5 — haneler bulanığa karışıyor |
| Sütun bazında zemin kestirimi | 0/5 — gradyanı düzeltiyor, kutu sorunu sürüyor |
| Renklilik kanalı (max-min) | 0/5 **ve #18'de yanlış haneyi 1,000 güvenle üretiyor** |

**Yapılacak iki iş:**

1. **Panelin dörtgen köşelerinden perspektif düzeltmesi.**
   `detect/perspective.py` yalnız dairesel kadranı düzeltiyor; dikdörtgen panel
   için yolu yok. Fotoğraflarda panel belirgin yamuk.
2. **Hane ızgarasını görüntüden değil TESPİT kutusundan kur.** İP5 panelin
   kutusunu veriyor, hane sayısı envanterde yazılı (`digits.count`), dolayısıyla
   ızgara doğrudan kurulabilir. `read/digital.py` bunu zaten kendi yorumunda
   kalıcı çözüm diye işaret ediyor (satır ~241).

**Ölçüm:** `python scripts\olc_ip8.py --fotograflar data\real\ip8_ekran`
Hedef: dijital 0/5 → en az 3/5, **sessiz hata 0 kalmak şartıyla.**

---

## 3. Neye dikkat et

**🔴 Kimlik doğrulaması YOK ve görüntüden çıkarılamıyor.** Zincir "bu kutu
gerçekten PT-101 mi?" diye sormuyor; kendisine `--gosterge PT-101` denmiş ve
güveniyor. Bu yüzden `demo/girdi/gosterge.mp4`'teki psi manometresi
"PT-101: 5,2 bar, status ok" diye yayınlanıyor — yanlış gösterge, yanlış birim,
yüksek güven.

21.08'de yatıklığın **ayrıklık** sayısı kimlik kapısı olarak denendi ve elendi:

| Kimlik | n | min | medyan | maks |
|---|:--:|---|---|---|
| doğru | 10 | -0,031 | **0,011** | 0,111 |
| yanlış | 12 | -0,613 | **-0,103** | 0,019 |

Dağılımlar örtüşüyor. Sentetik kadranda ayırıyordu (doğru 0,112-0,147, yabancı
0,012) — **ayırt edicilik ölçüte değil sentetiğin temizliğine aitmiş.** Kimlik
beyanla gelmeli; envanterdeki `waypoint` alanı bunun için (U11).

**Zincir artık tip filtreli.** `pipeline._tipe_uyan_kutular` sınıfı `gauge.type`
ile eşleşmeyen kutuları eliyor. Bu **kimlik değil tip** kontrolüdür — bir
termometre de `gauge` sınıfındadır.

**Üretilmiş (AI) videolar okuma doğruluğunu ÖLÇEMEZ.** `data/real/` altındaki üç
Veo videosu tespit genellemesi için; kadranların ground truth'u yok, envantere
uydurma satır eklenmez. `scripts/olc_uretilmis_video.py` bunu ölçer ve raporlar.

**Vana kolu renkleri — deney yapıldı, VARSAYILAN KAPALI.** `synth/state.py`
renk/kalınlık çeşitliliğini destekliyor (`KOL_RENKLERI`, `BORU_RENKLERI`) ama
veri üreteci onu yalnız `--vana-renkli` ile kullanır. Gerekçe: üretilmiş fabrika
videosundaki altı kırmızı küresel vana kolu, renk çeşitliliğiyle eğitilen iki
modelde de bulunamadı (240 karede sıfır `valve`). **Eksik olan renk değil şekil
ve bağlam** — sahadaki kol yassı bir plaka, flanşlı gövde üstünde; üreteçteki
ince çizgi + göbek değil. Doğru yol etiketli gerçek vana fotoğrafı toplamak.

Okuyucu (`read/state.py::_kol_acisi`) hâlâ **en koyu bileşeni** arıyor. Renkli
kolda bu varsayım tutmaz ve dört alternatif ölçüldü (40 karelik renk ızgarası):
renk uzaklığı 33/40 ama **7 sessiz hata**; kenar kapısı 27/40 sıfır sessiz;
k-means renk ayrımı sentetikte **40/40 kusursuz** ama gerçek fotoğrafta sessiz
hata üretti (#26 kapalı vana → "açık", %92 güven). Hepsi geri alındı.

**⚠ EĞİTİM BİTİŞİNİ `results.csv` SATIR SAYISIYLA YOKLAMA.** Ultralytics son
epoch satırını yazdıktan SONRA doğrulama yapıp `best.pt`'yi bir kez daha yazar
(24 MB → 6,2 MB, optimizer ayıklanır). 21.08'de saatlerce yarım yazılmış
ağırlığa ölçüm yapıldı ve oynayan sayılar yanlışlıkla eğitim değişikliklerine
yazıldı. Doğru ölçüt: dosya boyutu <10 MB ve bir dakika değişmemiş olmalı —
ya da eğitim sürecinin kendi bitiş sinyali beklenmeli.

---

## 4. Ekip demosu

`python demo\run_demo.py --video <yol>` — üç panel (GÖSTERGE / ALGILAMA /
ANOMALİ). Depoya gömüldü; girdiler ve çıktılar `.gitignore`'da.

GÖSTERGE paneli artık karedeki **bütün** göstergeleri gösteriyor: okunan yeşil
kutuda değeriyle, diğerleri gri kutuda "okunmuyor" etiketiyle. Bu, "tespit
yalnız birini buluyor" yanılgısını kapatmak için eklendi — tespit hepsini
buluyor, okuma bilerek tek göstergeye bakıyor.
