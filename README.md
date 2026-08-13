# rasrav-gauge-vision-2026

**Görsel Denetim Zekâsı — Gösterge ve Panel Okuma**
Reşit Asrav · BTÜ · Grup 03_Gama · Staj 27.07 – 04.09.2026

Fabrikada devriye gezen pan-tilt kameralı platformun görüntüsünden **analog göstergeleri,
dijital panelleri, ikaz lambalarını ve vana pozisyonlarını** otomatik okuyup sayıya çeviren
modül. Bugün bu işi bir operatör yapıyor: tur atıyor, göstergeleri okuyup not alıyor.

> 📄 Raporlar, iş paketleri ve proje tanımı ayrı repoda: **`akilli-fabrika-staj-2026`**
> 🔒 **Fabrika görüntüsü / videosu hiçbir koşulda buraya girmez.** Bu repo **public**'tir;
> kural 28.07'de "private repo" varsayımıyla konulmuştu, depo public kaldığı için tek
> koruma `.gitignore`'dur ve o yüzden kural gevşetilmek yerine sıkılaştırılmıştır:
> görüntü/video uzantıları dışlanır, `data/` ve `models/` tümüyle depo dışıdır.

---

## Boru Hattı

```
görüntü ──► tespit ──► kırp ──► okuma ──────────► kalibrasyon ──► MQTT yayını
            YOLO               analog: ibre açısı  açı → değer    inspect/reading
            (İP5)              dijital: OCR (İP11) (İP7)          (İP10)
                               lamba/vana (İP12)
```

**Hedef:** analog okuma ortalama hatası **< %5**
**Yayın şeması:** `{gauge_id, type, value, unit, conf, status, raw_angle, img_ref}`

---

## Kurulum

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest          # envanter doğrulama testleri geçmeli
```

> Python **3.13** kullanılıyor: `ultralytics`/`torch` daha yeni sürümlerde tekerlek
> sorunu çıkarabiliyor.

---

## Klasörler

| Klasör | İçerik |
|---|---|
| `configs/` | `gauges.yaml` — gösterge envanteri, **okuma zincirinin tek doğru kaynağı** |
| `src/gauge_vision/` | Modül kodu (`config.py`; ilerleyen İP'lerde `synth/`, `detect/`, `read/`, `publish/`) |
| `tests/` | pytest — envanter ve okuma doğrulaması |
| `scripts/` | Tek seferlik yardımcılar (veri indirme, toplu üretim) |
| `data/` | 🔒 `raw/` `synthetic/` `real/` — git'e girmez |
| `models/` `outputs/` | 🔒 ağırlıklar, koşu çıktıları, rapor figürleri |
| `notebooks/` `docs/` | Keşif defterleri, ölçüm tabloları ve karar notları |

---

## Gösterge Envanteri

Gösterge bilgisi koda gömülmez; hepsi [configs/gauges.yaml](configs/gauges.yaml) içinde:

```python
from gauge_vision.config import load_gauges

gauges = load_gauges()
pt101 = gauges["PT-101"]
print(pt101.unit, pt101.scale.min, pt101.scale.max, pt101.scale.sweep_deg)
# bar 0.0 10.0 270.0
```

Yeni bir gösterge test edilecekse **YAML'a satır eklenir, kod değişmez.**
Açı konvansiyonu (0° = saat 3, CCW pozitif) dosyanın başında şemasıyla tanımlı.

---

## İş Paketi Durumu

| İP | Konu | Durum |
|:--:|---|:--:|
| İP2 | Gösterge envanteri + yükleyici | ✅ |
| İP3 | Sentetik gösterge üreteci | ⬜ |
| İP5-İP7 | Tespit → ibre okuma → kalibrasyon | ⬜ |
| İP10 | MQTT `inspect/reading` yayını | ⬜ |
| İP11-İP12 | Dijital OCR · lamba/vana | ⬜ |

Tam liste ve bitti kriterleri rapor deposunda: `RESIT/Resit_is_paketleri.md`
