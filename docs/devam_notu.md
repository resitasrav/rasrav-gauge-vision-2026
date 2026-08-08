# Devam Notu — Oturum Kapanışı

**Yazıldığı an:** 14.08.2026 · Gün 15/30 · H3 kapandı
**Amaç:** Yeni oturum bu dosyayı okuyunca kaldığı yerden devam edebilsin.
Durum özeti CLAUDE.md'de; **burada yalnızca "sırada ne var ve neye dikkat et" var.**

> 👉 **Reşit'in bakması gerekenler ayrı dosyada:** `..\..\SORULAR.md`
> (ana STAJ klasörü altında, **git'e girmiyor** — uyuşmazlık defteriyle aynı yerde.
> Karar defteri kişiseldir; kod deposunda durması gerekmez.)
> 1 engelleyici karar (İP8 veri kaynağı), 4 varsayım onayı, 6 bilgi notu.

---

## 1. Nerede kaldık

**On iş paketi bitti: İP1-İP7, İP10, İP11, İP12, İP14, İP15.**
Kalan: İP8 (engelli), İP9, İP13, İP16.

| İP | Ölçüm | Dosya |
|---|---|---|
| Zincir (analog) | **%0,19** tam skala | `outputs/metrics/ip8_zincir_hatasi.json` |
| İP14 zor koşullar | 5 eksen × 5 seviye | `outputs/metrics/ip14_zor_kosullar.json` |
| İP15 güven eşiği | **0,70** · kapsama %88,1 | `outputs/metrics/ip15_guven_esigi.json` |
| İP11 dijital panel | **%93,3** dizge | `outputs/metrics/ip11_dijital.json` |
| İP12 lamba/vana | **%100 / %100** | `outputs/metrics/ip12_lamba_vana.json` |
| İP10 MQTT | **12/12** şema uyumlu | `outputs/mqtt/*.jsonl` |

**221/221 test geçiyor. İki repo da temiz. Arka planda koşan bir şey yok.**

---

## 2. Sıradaki iş — İP13: canlı masa üstü test (H4)

**Neden bu:** dört gösterge tipinin dördü de ayrı ayrı çalışıyor ama **hiçbiri
zincire bağlı değil.** `pipeline.read_frame` yalnızca analog okuyor; dijital,
lamba ve vana kendi fonksiyonlarından çağrılıyor. İP13 bunları tek bir akışta
birleştirmeli.

**Ne yapacak:**

```
kare → YOLO tespiti → gauge_id (waypoint'ten, şimdilik elle)
     → tipe göre dallan:
         analog  → refine → perspektif → roll → needle → read_value
         digital → read_digital
         lamp    → read_state
         valve   → read_state
     → publish/reading.ReadingPublisher
```

**Hazır parçalar — yenisini yazma:**

- `gauge_vision.pipeline.read_frame` — analog dalı (mevcut)
- `gauge_vision.read.digital.read_digital(image, gauge)`
- `gauge_vision.read.state.read_state(image, gauge)`
- `gauge_vision.publish.reading.ReadingPublisher` — broker yoksa dosyaya yazar
- `scripts/canli_oku.py` — kamera döngüsü ve çizim zaten var

**İP11 için bir iş burada kapanıyor:** dijital hane ızgarası şu an görüntüden
kuruluyor ve eksi işaretinde tökezliyor. Zincire bağlanınca **İP5'in panel
kutusundan** kurulabilir — hane sayısı envanterde yazılı, kutu tespitten
geliyor. `read_digital`'a bir `roi` parametresi yeterli.

**Tasarım kararı (verildi):** gösterge kimliği hâlâ elle/waypoint'ten gelecek
(U11 açık). Yanlış kimliğe karşı tek otomatik işaret, yatıklık kestiriminin
susmasıdır (`roll.MIN_UYUM`, ölçülen ayrım 0,22 vs 0,63).

---

## 3. Sonraki üç iş

1. **İP8 — gerçek görüntü.** 🔴 Ground truth kaynağı kararına bağlı, bkz.
   SORULAR.md S1. Önerilen A seçeneği yarım gün sürer.
2. **İP9 — CNN alternatifi.** *(kırpılabilir)* GPU hazır, sentetik veri hazır,
   ground truth bedava. İbre açısını regresyonla kestirip kutupsal taramayla
   kıyaslamak. Kıyas tablosu K3'ün (03.08) formatını izlemeli.
3. **KT2 — ekip şema onayı.** Kod tarafı hazır (`schema: 1`, doğrulayıcı, 20
   test). U1-U3 kararı gelince değişiklik gerekmeyecek.

---

## 4. Gerçek görüntüye geçilince YENİDEN ÖLÇÜLECEK eşikler

Hepsi sentetik dağılımlara göre kalibre edildi ve kod içinde ⚠ ile işaretli:

| Sabit | Dosya | Ne için |
|---|---|---|
| `MAX_ARTIK_ORANI`, `MAX_YAYILMA_ORANI` | `detect/refine.py` | merkez rafinesi kanıt kalitesi |
| `MIN_UYUM` | `read/roll.py` | yatıklık deseni uyumu |
| `MIN_EKSEN_ORANI`, `MAX_ARTIK_ORANI` | `detect/perspective.py` | elips kabul kapıları |
| `LAMBA_PARLAKLIK_ORANI` | `read/state.py` | lamba yanık/sönük ayrımı |
| `conf_threshold: 0.70` | `configs/gauges.yaml` | **İP15 gerçek veriyle yeniden koşmalı** |

---

## 5. Tuzaklar — zaman kaybettirenler

- **"Kapı" yazmak kapı kurmak değildir.** `refine.py` ve `roll.py`'ın ilk güven
  kapıları rastgele gürültüyü kabul ediyordu (50/50 ve 8/10). Kapılar "cevap
  makul mü" diye soruyordu, "kanıt var mı" diye değil. **Yeni kapı yazınca sahte
  girdiyle sınamadan bitmiş sayma.**
- **Mutlak eşik = gizli hata.** Lamba okuması `V > 90` ile karar veriyordu;
  ×0,15 ışıkta yanan lamba 35'e düşüp **60 kareyi sessizce yanlış sınıflandırdı**.
  Doğru ölçüt çevreye göre kontrasttı. Aynı sınıf hata `digital.py`'da da çıktı.
- **Envanter ile kod sessizce ayrışabilir.** Vana toleransında envanter ±20°
  diyordu, kod ±6° yapıyordu. İkisi de kendi içinde tutarlı olduğu için hiçbir
  birim testi yakalamadı. **Her sayısal beyan için o beyanı sınayan test yaz.**
- **Düz tarama tablosu sonuç değil uyarıdır.** İP15'in ilk kalibrasyonu daireseldi
  (eşik zaten uygulanmış veriyle kalibrasyon). Bir tarama hiçbir şey
  değiştirmiyorsa önce ölçüm düzeneğine bak.
- **`_haneleri_bul` gibi filtrelerde ölçüt yanlış boyut olabilir.** Ondalık
  noktayı elemek için YÜKSEKLİK filtresi kullanmak yatay segmentleri de siler
  (hane kutusu 71 px yerine 19 px çıktı). İki boyutun büyüğüne bak.
- **torch tekerleği.** RTX 4050 + `cu126` kurulu. `pip install torch` PyPI'dan
  **CPU** sürümünü getirir ve **sessizce çalışır, sadece kartı kullanmaz.**
  `torch.cuda.is_available()` ile doğrula. Fark: eğitim 45 dk → 5 dk.
- **Ultralytics çıktı yolu.** `project="models/ip5"` verilse bile `runs_dir` öne
  ekleniyor → gerçek yol `runs/detect/models/ip5/...`.
- **Veri sızıntısı.** Zincir ölçümü `v1` (tohum 1) üzerinde koşar; `v0`'ın 53
  karesi karışık eğitimin içindeydi. Yeni ölçüm kümesi üretirken `--ozet` ver.
- **PowerShell + dosya kodlaması.** `Set-Content` ile Türkçe içeren markdown
  dosyasını yeniden yazma — mojibake üretiyor. Edit aracını kullan.
- **PowerShell + git commit.** Mesajda çift tırnak varsa argüman bölünüyor.
  Uzun mesajı dosyaya yazıp `git commit -F dosya` ile ver.
- **Uyuşmazlık defteri git'e girmez.** `STAJ\ortak uyusmazliklar\uyusmazliklar.md`
  yereldir, iki reponun da dışındadır.
- **Raporlar staj iş gününe göre tarihlenir**, dosyanın yazıldığı güne göre değil.

---

## 6. Bu oturumda yapılanlar (özet)

- **`synth/degrade.py`** — beş eksenli zor koşul üreteci; ground truth
  bozulmayla birlikte taşınıyor
- **`detect/perspective.py`** — elips→daire düzleştirme (K2); 40°'de p95
  13,44 → 4,97
- **İP14** — koşul bazlı tablo; **eğiklik tek başına baskın**, düşük ışık ve
  JPEG neredeyse etkisiz, parlama okumayı değil tespiti öldürüyor
- **İP15** — eşik 1560 karede kalibre edildi, envanterdeki 0,70 doğrulandı
- **`read/digital.py` + `synth/digital.py`** — 7-segment panel, %93,3 dizge
- **`read/state.py` + `synth/state.py`** — lamba/vana, %100/%100, sessiz yanlış
  sınıflandırma yok
- **`publish/reading.py`** — `inspect/reading` yayını + katı şema doğrulaması,
  broker bağımsız
- Testler 149 → **221**
- Günlük raporlar 10-14.08 + H3 haftalığı yazıldı
