# CLAUDE.md — Kod Reposu Bağlamı

> Her oturumda otomatik yüklenir. **Rapor deposu ayrıdır:**
> `../akilli-fabrika-staj-2026` — proje tanımı, 16 iş paketi ve günlük/haftalık raporlar
> orada. Buradaki iş bitince oradaki günlük rapora da işlenir.
>
> **Son güncelleme:** 2026-07-28

---

## 1. Ne Yapıyor

Fabrikada devriye gezen pan-tilt kameralı platformun görüntüsünden **analog gösterge,
dijital panel, ikaz lambası ve vana pozisyonunu okuyup sayıya çevirir**.

```
görüntü ──► tespit ──► kırp ──► okuma ──────────► kalibrasyon ──► MQTT
            (İP5)              analog: ibre açısı   açı→değer      inspect/reading
                               (İP6, Hough)         (İP7)          (İP10)
                               dijital: OCR (İP11)
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
  config.py             Envanter yükleyici + doğrulama — İP2 ✅
  synth/                Sentetik gösterge üreteci — İP3
  detect/               YOLO gösterge tespiti — İP5
  read/                 needle.py (İP6) · calibrate.py (İP7)
                        digital.py (İP11) · state.py lamba/vana (İP12)
  publish/              MQTT inspect/reading yayını — İP10
tests/                  pytest — her modülün doğruluk testi
scripts/                Tek seferlik yardımcılar (veri indirme, toplu üretim)
data/  raw/ synthetic/ real/     🔒 git'e girmez (real = fabrika görüntüsü)
models/                 🔒 ağırlıklar
outputs/ figures/ metrics/       🔒 koşu çıktıları, rapor figürleri
notebooks/              Deneme/keşif — kalıcı kod src/ altına taşınır
docs/                   Ölçüm tabloları, karar notları
```

**Kural:** notebook'ta keşfet, ama kalıcı olan her şey `src/gauge_vision/` altına
fonksiyon olarak taşınır. Rapor grafiği üreten kod da `src/` veya `scripts/` altında durur
ki tekrar üretilebilsin.

---

## 5. Komutlar

```powershell
.\.venv\Scripts\Activate.ps1          # ortamı aç (venv: Python 3.13)
python -m pytest                      # testler
python -c "from gauge_vision.config import load_gauges; print(load_gauges().keys())"
```

- **venv Python 3.13** ile kuruldu. Sistemde 3.14/3.15 de var ama `ultralytics`/`torch`
  o sürümlerde tekerlek sorunu çıkarabilir — İP5'e gelindiğinde 3.13 güvenli liman.
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
| İP3 | Sentetik üreteç v0 (100 görüntü + otomatik etiket) | ⬜ sıradaki |
| İP1 | Veri taraması (Roboflow/Kaggle gauge setleri) | ⬜ |
| İP4 | Mini literatür (~10 makale) | ⬜ |
| İP5-İP16 | bkz. rapor deposu `RESIT/Resit_is_paketleri.md` | ⬜ |

**Envanterdeki değerler şu an varsayım** — gerçek gösterge listesi danışmandan gelince
`gauges.yaml` güncellenecek, kod değişmeyecek (2. kural bunun için).

---

## 8. Bu Dosyayı Güncelleme Kuralı

| Olay | Güncellenecek yer |
|---|---|
| İP bitti / başladı | §7 tablosu |
| Yeni alt paket açıldı (`synth/`, `detect/`…) | §4 klasör düzeni |
| Yeni komut / bağımlılık | §5 |
| Konvansiyon veya kural değişti | §2 / §3 |

Özet kalır: uzun anlatım rapor deposundaki günlük/haftalık rapora yazılır.
