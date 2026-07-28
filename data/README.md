# data/ — veri klasörü (içerik git'e girmez)

Sadece bu dosya ve `.gitkeep`'ler versiyonlanır. Veri ya indirilebilir ya da
üretilebilir olmalı; repoda taşınmaz.

| Klasör | Ne konur | Nereden gelir |
|---|---|---|
| `raw/` | İndirilen açık veri setleri (Roboflow/Kaggle "gauge" setleri) | İP1 — indirme adımı `scripts/` altında belgelenir |
| `synthetic/` | Sentetik üretilen gösterge görüntüleri + etiketleri | İP3 — `python -m gauge_vision.synth` ile **yeniden üretilebilir** |
| `real/` | 🔒 Gerçek gösterge fotoğrafları / fabrika görüntüsü | **Asla commit edilmez** — `.gitignore`'da ayrıca yasaklı |

**Etiket formatı (sentetik):** her görüntünün yanında aynı adlı `.json` durur —
`{gauge_id, angle_deg, value, cx, cy, r}`. Açı ground truth'tur: ibreyi biz o açıya
koyduğumuz için ölçüm bedava gelir (sentetik-önce stratejisinin sebebi budur).
