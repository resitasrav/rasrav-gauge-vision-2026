# CLAUDE.md — Kod Reposu Bağlamı

> Her oturumda otomatik yüklenir. **Rapor deposu ayrıdır:**
> `../akilli-fabrika-staj-2026` — proje tanımı, 16 iş paketi ve günlük/haftalık raporlar
> orada. Buradaki iş bitince oradaki günlük rapora da işlenir.
>
> **Ekip modülleriyle uyuşmazlık bulunca** (şema ≠ kod, aracın ihtiyacı karşılamaması)
> çalışma durdurulup cevap beklenmez — bulgu önerisiyle birlikte
> `../ortak uyusmazliklar/uyusmazliklar.md` dosyasına yazılır, işe devam edilir.
> Ölçülebilir bir soruysa sentetik veride ölçülüp sayıyla gidilir. Detay: rapor
> deposundaki CLAUDE.md §5b.
>
> **Son güncelleme:** 2026-08-13

---

## 1. Ne Yapıyor

Fabrikada devriye gezen pan-tilt kameralı platformun görüntüsünden **analog gösterge,
dijital panel, ikaz lambası ve vana pozisyonunu okuyup sayıya çevirir**.

```
görüntü ─► tespit ─► merkez ─► okuma ──────────────► kalibrasyon ─► MQTT
           (İP5)     rafinesi  analog: ibre açısı     açı→değer      inspect/reading
                     + yatıklık (İP6, kutupsal)       (İP7)          (İP10)
                     kestirimi  dijital: OCR (İP11)
                               lamba/vana (İP12)
```

**Hedef metrik:** analog okuma **ortalama hata < %5** · **Sahip:** Reşit Asrav ·
**Süre:** 27.07 – 04.09.2026

---

## 2. Değişmez Kurallar

1. **Fabrika görüntüsü/videosu commit edilmez.** `data/real/` ve `data/factory/`
   `.gitignore`'da. Repo private ama kural yine de geçerli.
2. **`configs/gauges.yaml` tek doğru kaynaktır.** Gösterge bilgisi (min/max, açı, birim,
   eşik) koda gömülmez — YAML'dan okunur. Yeni gösterge = YAML'a satır, kod değişmez.
3. **Yanlış okumaktansa okumamak.** Güven eşiğinin altında değer uydurulmaz,
   `status: unreadable` basılır (İP15). Aynı ilke `config.py`'ın katı doğrulamasında da var.
4. **Ölçüm yoksa iş bitmemiştir.** Her iş paketinin sayısal bitti kriteri var; sonuç
   `outputs/metrics/` altına yazılır ve günlük rapora işlenir.
5. **Sentetik-önce.** Yöntem önce sentetik veride oturur (ground truth bedava),
   sonra gerçek fotoğrafa geçilir.
6. **🚨 TÜM COMMIT'LER REŞİT ASRAV ADINA.** Commit mesajına **asla** `Co-Authored-By`
   satırı, "Claude", "AI", "generated with" gibi hiçbir ibare eklenmez. Bu staj işi
   Reşit'in işidir ve geçmişte öyle görünmelidir. Commit atmadan önce mesajı kontrol et.
   Commit başlığı iş paketi numarasıyla başlar: `IP6: Hough ibre okuma calisiyor`
   (git geçmişinde Türkçe karakter kullanılmıyor).

---

## 3. Açı Konvansiyonu — EZBERLE

Bu projedeki en pahalı hata sınıfı yanlış açı işareti/yönüdür: kod çalışır, sayı üretir,
ama değer sessizce yanlış olur.

```
   derece · 0° = saat 3 yönü (sağ) · POZİTİF YÖN = saat yönünün TERSİ (CCW)

                90°
        135°     |     45°
             \   |   /
   180° --------+-------- 0°
             /   |   \
        225°     |    315° (= -45°)
                270° (= -90°)
```

- Tipik 270°'lik saat: `angle_min: 225 → angle_max: -45`, `direction: cw`
- **OpenCV'de y ekseni aşağı artar** → görüntüden açı:
  `atan2(-(y_uç - y_merkez), x_uç - x_merkez)` — eksi işareti bunun için
- Süpürme açısı `Scale.sweep_deg` ile hesaplanır; `config.py` 0-350° dışını reddeder
- **`sweep_deg` beyanı sağlama toplamıdır.** Yanlış `direction` geometriden anlaşılamaz:
  225 → -45 arası `ccw` yazılırsa süpürme 90° çıkar, sayı "makul" göründüğü için kod
  çalışır ve tüm okumalar sessizce yanlış olur. Envanterde kadranın süpürmesi beyan
  edilir, `config.py` hesapladığıyla karşılaştırır. **Yeni analog gösterge eklerken
  `sweep_deg` yazmayı atlama** — testi de var (`test_beyan_edilen_supurme_gerceklesiyor`)

Tam tanım: [configs/gauges.yaml](configs/gauges.yaml) dosya başlığı.

---

## 4. Klasör Düzeni — Hangi İş Nereye

```
configs/gauges.yaml     Gösterge envanteri — İP2 ✅ (zincirin temeli)
src/gauge_vision/
  config.py             Envanter + doğrulama + değer↔açı + çizgi düzeni — İP2/İP3 ✅
  synth/  dial.py       Kadran çizici (DialLook varyasyon, DialTruth etiket) — İP3 ✅
          generate.py   Tohumlu veri seti üreteci, JSONL etiket — İP3 ✅
  synth/  degrade.py    Zor koşul üreteci (perspektif/parlama/ışık/blur) — İP14 ✅
          digital.py    7-segment panel çizici — İP11 ✅
          state.py      Lamba/vana çizici — İP12 ✅
  detect/dataset.py     YOLO etiket dönüşümü + karışık eğitim kümesi — İP5 ✅
         refine.py      Kadran merkezi: kutu → çember (gradyan doğruları) — İP5 ✅
         perspective.py Eğik bakışta elips→daire düzleştirme (K2) — İP14 ✅
  read/  needle.py      İbre açısı: kutupsal tarama + Hough — İP6 ✅
         evaluate.py    Ölçüm zemini (çözünürlük/JPEG/merkez düğmeleri) — İP6 ✅
         calibrate.py   Açı→değer + durum (ok/unreadable/out_of_range/alarm) — İP7 ✅
         roll.py        Kamera yatıklığı: çizgi deseniyle korelasyon — İP8 ✅
         digital.py     7-segment okuma (segment geometrisi) — İP11 ✅
         state.py       Lamba (HSV) + vana (PCA kol açısı) — İP12 ✅
  pipeline.py           Zincir: tespit→perspektif→merkez→yatıklık→açı→değer ✅
  publish/reading.py    inspect/reading yayını + şema doğrulama — İP10 ✅
tests/                  pytest — her modülün doğruluk testi
scripts/                Tek seferlik yardımcılar (veri indirme, toplu üretim)
data/  raw/ synthetic/ real/     🔒 git'e girmez (real = fabrika görüntüsü)
models/                 🔒 ağırlıklar
outputs/ figures/ metrics/       🔒 koşu çıktıları, rapor figürleri
notebooks/              Deneme/keşif — kalıcı kod src/ altına taşınır
docs/  devam_notu.md   👈 OTURUMA BAŞLARKEN BUNU OKU — sırada ne var, neye dikkat et
       cekim_talimati.md    İP8 ekrandan çekim: adım adım ne yapılacak
                        (SORULAR.md artık burada değil → `../SORULAR.md`, git dışı)
       literatur_ozeti.md   Mini literatür + plana yansıyan K1-K6 kararları — İP4 ✅
       veri_setleri_degerlendirme.md  8 açık set, etiket türüne göre — İP1 ✅
                        Ölçüm tabloları, karar notları
```

**Kural:** notebook'ta keşfet, ama kalıcı olan her şey `src/gauge_vision/` altına
fonksiyon olarak taşınır. Rapor grafiği üreten kod da `src/` veya `scripts/` altında durur
ki tekrar üretilebilsin.

---

## 5. Komutlar

```powershell
.\.venv\Scripts\Activate.ps1          # ortamı aç (venv: Python 3.13)
python -m pytest                      # testler
python scripts\uret_sentetik.py       # 100 sentetik görüntü + etiket (İP3)
python scripts\olc_ip6.py             # açı hatası tabloları + rapor figürleri (İP6)
python scripts\olc_ip7.py             # okuma hatası % + ablasyonlar (İP7)
python scripts\hazirla_ip5_veri.py    # sentetik/gercek/karisik eğitim kümeleri (İP5)
python scripts\egit_ip5.py            # YOLO eğit + mAP ve kutu merkezi sapması (İP5)
python scripts\olc_zincir.py --veri data/synthetic/v1   # uçtan uca + 2x3 ablasyon
python scripts\olc_ip14.py --perspektif   # zor koşullar, 5 eksen (İP14)
python scripts\kalibre_ip15.py            # güven eşiği kalibrasyonu (İP15)
python scripts\olc_ip11.py --zor          # dijital panel doğruluğu (İP11)
python scripts\olc_ip12.py --zor          # lamba/vana doğruluğu (İP12)
python scripts\yayinla_ip10.py            # inspect/reading yayını (İP10)
python scripts\kalibre_vana.py --klasor data\real\VL-601   # kol açısı → YAML satırı (S2)
python scripts\canli_oku.py --kaynak 0 --gosterge PT-101  # kamera demosu
python scripts\kadran_onizle.py       # kadranı kaydırıcıyla elle dene
python -c "from gauge_vision.config import load_gauges; print(load_gauges().keys())"
```

- **venv Python 3.13** ile kuruldu. Sistemde 3.14/3.15 de var ama `ultralytics`/`torch`
  o sürümlerde tekerlek sorunu çıkarabilir — 3.13 güvenli liman.
- **GPU: RTX 4050 (6 GB), torch cu126.** Eğitim kartta koşar (3 yapılandırma 5 dk; CPU'da
  ~45 dk). `pip install torch` PyPI'dan **CPU** tekerleğini getirir ve bu **sessizce**
  çalışır, sadece kartı kullanmaz — `torch.cuda.is_available()` ile doğrula.
  Doğru kurulum `requirements.txt` başında yazılı.
- `pip install -e .` yapıldı → `import gauge_vision` her yerden çalışır, `sys.path` hilesi yok.
- Yeni bağımlılık eklerken `requirements.txt`'e de yaz (yorumlu satırlar sırasını bekliyor).

---

## 6. Kod Tarzı

- **Türkçe** docstring ve yorum (raporlar da Türkçe). Değişken/fonksiyon adları İngilizce.
- Yorum "ne yaptığını" değil **"neden öyle yaptığını"** anlatır; kodun kendisi zaten ne
  yaptığını söylüyor.
- Tip ipuçları (`type hints`) kullanılır — `dataclass` tercih edilir.
- Sihirli sayı yok: eşik, açı, boyut gibi her sabit ya `gauges.yaml`'dan gelir ya da
  modül başında adlandırılmış sabittir.
- Bir fonksiyon bir iş yapar; okuma zincirinin her adımı ayrı test edilebilir olmalı.

---

## 7. Durum

| İP | Konu | Durum |
|:--:|---|:--:|
| İP2 | `configs/gauges.yaml` envanteri + `config.py` yükleyici | ✅ 28.07 |
| İP3 | Sentetik üreteç v0 (100 görüntü + otomatik etiket) | ✅ 30.07 |
| İP4 | Mini literatür → `docs/literatur_ozeti.md` | ✅ 31.07 |
| İP1 | Veri taraması → `docs/veri_setleri_degerlendirme.md` | ✅ 31.07 |
| İP6 | Klasik ibre okuma → `read/needle.py` + `read/evaluate.py` | ✅ 03.08 · **0,123°** |
| İP7 | Açı→değer → `Scale.value_for_angle` + `read/calibrate.py` | ✅ 04.08 · **%0,129** |
| İP5 | Gösterge tespiti (YOLO) → `detect/dataset.py` + `refine.py` | ✅ 05-07.08 · **mAP50 0,967** |
| — | Zincir uçtan uca (`pipeline.py`) + yatıklık (`read/roll.py`) | ✅ 07.08 · **%0,19** |
| İP14 | Zor koşullar → `synth/degrade.py` + `detect/perspective.py` | ✅ 10-11.08 · 5 eksen |
| İP15 | Güven eşiği → `scripts/kalibre_ip15.py` | ✅ 12.08 · **0,70 kalibre** |
| İP11 | Dijital panel → `read/digital.py` | ✅ 13.08 · **%93,3 dizge** |
| İP12 | Lamba/vana → `read/state.py` | ✅ 14.08 · **%100 / %100** |
| İP10 | MQTT yayını → `publish/reading.py` | ✅ 14.08 · **12/12 şema** |
| İP8 | Gerçek görüntüde uçtan uca | 🔴 ground truth kaynağı yok |
| İP9, İP13, İP16 | bkz. rapor deposu `RESIT/Resit_is_paketleri.md` | ⬜ |

**Zincir sentetikte hedefin çok altında: %0,19 tam skala** (hedef %5), okunamayan 0/100.
Diğer tipler: dijital %93,3 dizge · lamba %100 · vana %100 · MQTT 12/12 şema uyumlu.

**İP14'ün ana bulgusu — eğiklik tek başına baskın etken:**

| Eksen | En zor seviyede hata | Not |
|---|---|---|
| **eğiklik 50°** | **%9,30** (max %56,5) | diğer dördünün toplamından fazla |
| bulanıklık 21 px | %1,32 | |
| düşük ışık ×0,15 | %0,51 | neredeyse etkisiz |
| JPEG q15 | %0,16 | **etkisi yok** |
| parlama %90 | %0,25 | ama TESPİT çöküyor (25/60 kare) |

*(Bu sayılar güven eşiği KAPALIYKEN; "ne kadar yanılabilir"i gösterir, "ne yayınlar"ı
değil. Eşik açıkken sessiz hata %0,22.)*

**Güven eşiği 0,70 artık ölçülmüş bir seçim** (İP15, 1560 kare): kapsama %88,1,
sessiz hata %0,22. Envanterdeki varsayım değer doğrulandı.

**Bu sayılar zorluğu değil, yöntemin tabanını ölçer.** Sentetikte metal doku, tozlanma
ve gerçek sanayi aydınlatması yok. Gerçek görüntü hâlâ İP8'i bekliyor.

**⚠ Üç eşik sentetikte kalibre edildi, gerçek görüntüde yeniden ölçülmeli:**
`refine.MAX_ARTIK_ORANI` · `refine.MAX_YAYILMA_ORANI` · `roll.MIN_UYUM`. Kod içinde
işaretli; ölçülen dağılımlar da yanlarında yazılı.

**H2 sırası:** plan İP5→İP6→İP7 idi, **İP6 öne alındı**. İP5 açık veri setlerinin
indirilmesini bekliyor (K1: sentetik tek başına yetersiz); İP6 elde hazır sentetik ground
truth ile hemen başlayabildi. K3 kıyası yapıldı: **kutupsal tarama** (0,123°) Hough'u
(0,328°) hem doğrulukta hem hızda geçti, İP7'nin girdisi odur.

**İP6'nın açı ölçümü merkezi doğru bildiğini varsayar** — merkez 8 px kaydığında hata
0,123° → 3,652° oluyordu. `detect/refine.py` bu varsayımı karşılamak için yazıldı:
merkez kutudan değil kadran çemberinden geliyor, sapma %1,31 → **%0,06**.

**Yeni kapı yazınca sahte girdiyle sına.** 07.08'de aynı hata iki modülde arka arkaya
yapıldı: `refine.py` ve `roll.py`'ın ilk güven kapıları **rastgele gürültüyü kabul
ediyordu** (50/50 ve 8/10). İkisinde de sebep aynı: kapı "cevap makul mü" diye
soruyordu, "kanıt var mı" diye değil. Eşik tahminle değil, iki kümenin dağılımı
ölçülüp aralarına konur.

**Mutlak eşik kullanmadan önce iki kez düşün.** 14.08'de lamba okuması `V > 90` ile
"yanıyor mu" diye soruyordu; ×0,15 ışıkta yanan lamba 35'e düşüp **60 kareyi sessizce
yanlış sınıflandırdı** (arıza lambası "off"). Doğru ölçüt mutlak parlaklık değil çevreye
göre **kontrasttı** — o oran ışık kazancıyla ölçeklenmez. Aynı hata sınıfı `digital.py`'da
da çıktı (panel çerçevesi sönük segmentten parlak). **13.08'de üçüncü kez, `needle.py`'da:**
`_dark_mask` ibrenin zeminden hep koyu olduğunu varsayıyordu; koyu kadranda beyaz ibre
(araç göstergeleri gibi, kendi envanterimizde yok ama genelleme testinde ortaya çıktı) bu
varsayımla hiç okunamıyordu. Artık iki kutbu da deneyip ibre imzasına (dar plato) hangisi
uyuyorsa onu seçiyor — düzeltildi, 230/230 test + `olc_ip6.py` sonucu değişmedi (0,123°).

**🔴 Bulundu, henüz düzeltilmedi (13.08) — `roll.py` yeni kadran stilinde sahte yatıklık
kestiriyor.** Genelleme testinde (araç hız göstergesi, gerçekte ~0° yatık) `estimate_roll`
21,3° sahte yatıklık buldu ve doğru ölçülen ibre açısını sessizce yanlış değere kaydırdı
(`roll_deg=0` zorlanınca doğru değere döndü). Aynı "sentetik kadran stiline göre kalibre
edilmiş kestirim gerçek/farklı kadranda tutmuyor" sınıfından — İP8'in kendi ölçümünde de
("yatıklık uyumu bazı karelerde negatif çıkıyor, eşik 0,4") aynı belirti var, muhtemelen
aynı kök sebep. `roll.MIN_UYUM` güven kapısı muhtemelen "cevap makul mü" değil "kanıt var
mı" testine göre yeniden kurulmalı (bkz. yukarıdaki refine.py/roll.py dersi). Henüz
dokunulmadı.

**Envanterdeki her sayısal beyan için o beyanı sınayan bir test olmalı.** Vana
toleransında envanter ±20° diyordu, kod fiilen ±6° yapıyordu; ikisi de kendi içinde
tutarlı olduğu için hiçbir birim testi yakalamadı, ancak uçtan uca ölçüm görünür kıldı.
`sweep_deg` için İP2'de yapılmıştı, vana için 14.08'de eklendi.

**Daha iyisi: beyanı kodda İKİNCİ KEZ yazma.** 18.08'de vana toleransı koddaki
`VANA_TOLERANS_DEG` sabitinden çıkarılıp `reading.tolerance_deg`'e taşındı; aynı
şekilde "yatay = açık" varsayımı `states[].lever_angle` oldu. Tek kaynak varsa
ayrışma **imkânsızdır**, testle kovalanması gerekmez. Aynı gün `allow_minus`'ın
YAML'da durup koda hiç bakmadığı da görüldü — bayrak eklemek yetmiyor, okuyan
tarafı da bağlamak gerekiyor.

**Montaj bilgisi algoritma sabiti değildir.** Hangi kol açısının "açık" demek
olduğu göstergenin montajına aittir; aynı kod ters takılmış bir vanayı da doğru
okumalı, fark YAML satırında kalmalı. Sahada ölçmek için `scripts/kalibre_vana.py`
(etiketli fotoğraf → YAML satırı). Testte varsayımın gerçekten envanterde yaşadığı
`test_montaj_varsayimi_envanterde_yasiyor` ile sınanıyor: görüntü üretecin
varsayımıyla çizilirken okuyucuya TERS envanter veriliyor, cevap ters dönmezse
varsayım hâlâ koda gömülü demektir.

**Ölçüm tablosu düz çıkıyorsa sonuç değil uyarıdır.** İP15'in ilk eşik taraması
0,00-0,70 arası tamamen düz çıktı; sebebi, girdiyi üreten ölçümün eşiği zaten uygulamış
olmasıydı (dairesel kalibrasyon). Bir tarama hiçbir şey değiştirmiyorsa önce ölçüm
düzeneğine bakılır.

**Envanterdeki değerler şu an varsayım** — gerçek gösterge listesi danışmandan gelince
`gauges.yaml` güncellenecek, kod değişmeyecek (2. kural bunun için). Sentetik veri de
bu varsayımdan beslendiği için envanter değişince `uret_sentetik.py` yeniden koşturulur.

**Mevcut sentetik veri:** `data/synthetic/v0` — 100 görüntü, tohum 0 (eğitimde kullanıldı,
53 karesi karışık kümenin içinde) · `data/synthetic/v1` — tohum 1, **sızıntısız ölçüm
kümesi**, zincir ölçümü bunda koşar. Testler: **230/230**.

---

## 8. Bu Dosyayı Güncelleme Kuralı

| Olay | Güncellenecek yer |
|---|---|
| İP bitti / başladı | §7 tablosu |
| Yeni alt paket açıldı (`synth/`, `detect/`…) | §4 klasör düzeni |
| Yeni komut / bağımlılık | §5 |
| Konvansiyon veya kural değişti | §2 / §3 |

Özet kalır: uzun anlatım rapor deposundaki günlük/haftalık rapora yazılır.
