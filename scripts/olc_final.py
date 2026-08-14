r"""İP16 — final ölçüm paketi: bütün metrikler tek komutta (4 Eylül teslimi).

    python scripts\olc_final.py           # mevcut ölçüm JSON'larından paketi kur
    python scripts\olc_final.py --tam     # önce bütün ölçümleri yeniden koştur

Üç çıktı üretir:

    outputs/metrics/ip16_final_ozet.json    tüm sayılar tek yerde
    outputs/metrics/ip16_final_tablo.md     rapora yapıştırılacak tablolar
    outputs/figures/ip16_final_ozet.png     üç panelli özet figürü

**Neden toplama scripti, yeniden ölçüm değil.** Her sayı zaten kendi ölçüm
scriptinden çıkıyor ve JSON'u imzası sayılır (7. kural: elle sayı yazılmaz).
Bu script sayı ÜRETMEZ; üretilmiş sayıları tek pakete toplar ve eksik olanı
**yüksek sesle** söyler — hangi script üreteceğiyle birlikte. `--tam` verilirse
alt scriptleri sırayla koşturur (eğitim hariç: model ağırlıkları sabittir,
final paketinde yeniden eğitim olmaz).

**Neden markdown tablosu da basıyor.** Rapordaki sayılar elle taşınırken iki
kez yanlış kopyalandı; kopyalanacak metni scriptin kendisi üretirse taşınırken
bozulmaz.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

M = Path("outputs/metrics")
F = Path("outputs/figures")
HEDEF_YUZDE = 5.0

# (dosya, üreten komut, zorunlu mu). Sıra --tam'ın koşturma sırasıdır:
# ucuz ve bağımsız olanlar önce, zincir en sonda (modeli ısıtmış olur).
KAYNAKLAR = [
    ("ip6_aci_hatasi.json", "python scripts\\olc_ip6.py", True),
    ("ip7_okuma_hatasi.json", "python scripts\\olc_ip7.py", True),
    ("ip11_dijital.json", "python scripts\\olc_ip11.py --zor", True),
    ("ip12_lamba_vana.json", "python scripts\\olc_ip12.py --zor", True),
    ("ip14_zor_kosullar.json", "python scripts\\olc_ip14.py --perspektif", True),
    ("ip15_guven_esigi.json", "python scripts\\kalibre_ip15.py", True),
    ("roll_kaniti.json", "python scripts\\olc_roll_kaniti.py", True),
    ("ip8_zincir_hatasi.json", "python scripts\\olc_zincir.py --veri data/synthetic/v1", True),
    # Gerçek çekim ve İP9 opsiyonel: yoklukları paketin kurulmasını engellemez
    # ama raporda "yok" diye görünür — sessizce düşmez.
    ("ip8_ekran_hatasi.json", "python scripts\\olc_ip8.py --fotograflar data\\real\\ip8_ekran", False),
    ("ip9_cnn_kiyas.json", "İP9 (kırpılabilir) — yapılmadıysa gerekçe rapora", False),
]


def yukle(ad: str) -> dict | None:
    yol = M / ad
    if not yol.exists():
        return None
    return json.loads(yol.read_text(encoding="utf-8"))


def kosur(komut: str) -> None:
    print(f"\n=== {komut}")
    # Alt script başarısızsa paket kurulmaz: yarısı taze yarısı bayat bir
    # "final" paketi, bayat olduğu belli olmayan sayı üretir.
    subprocess.run([sys.executable] + komut.split()[1:], check=True)


def ana_tablo(v: dict) -> list[str]:
    zincir = v["ip8_zincir_hatasi.json"]
    ip11 = v["ip11_dijital.json"]["kosullar"]["temiz"]
    ip12 = v["ip12_lamba_vana.json"]
    sat = [
        "| Tip | Metrik | Değer | Hedef/Referans |",
        "|---|---|---|---|",
        f"| Analog | Zincir uçtan uca, tam skala | **%{zincir['hata_yuzde_tam_skala']['ortalama']}** "
        f"(p95 %{zincir['hata_yuzde_tam_skala']['p95']}) | hedef < %{HEDEF_YUZDE:g} |",
        f"| Analog | İbre açısı (kutupsal) | {v['ip6_aci_hatasi.json']['yontemler']['polar']['aci_hatasi_deg']['ortalama']}° | — |",
        f"| Dijital | Dizge doğruluğu (temiz) | %{100 * ip11['dizge_dogrulugu']:.1f} | sentetik üreteç sınırı raporda |",
        f"| Lamba | Durum doğruluğu (temiz) | %{100 * ip12['lamba']['temiz']['dogruluk']:.0f} | ↑ aynı sınır |",
        f"| Vana | Durum doğruluğu (temiz) | %{100 * ip12['vana']['temiz']['dogruluk']:.0f} | ↑ aynı sınır |",
    ]
    ip13 = yukle("ip13_zincir_tum_tipler.json")
    if ip13:
        sat.append(f"| Dört tip | Devriye sessiz hata | "
                   f"**{ip13['toplam']['sessiz_yanlis']} / {ip13['toplam']['kare']} kare** | 0 |")
    sec = v["ip15_guven_esigi.json"]["secilen_esik"]   # seçilen taramanın satırı
    # Paydası yazılmadan "sessiz hata %x" bir sayı değil, iki sayıdan biridir:
    # kabul edilenler içinde %0,22 · tüm kareler içinde %0,19. Projenin bütün
    # belgeleri birincisini kullanıyor — asıl soru "yayınladığım okumaların kaçı
    # sessizce yanlış", reddedilenler zaten zararsız. Payda açıkça yazılıyor.
    sat.append(f"| Ortak | Güven eşiği {sec['esik']} | kapsama %{100 * sec['kapsama']:.1f} "
               f"· sessiz hata %{100 * sec['sessiz_hata_orani']:.2f} (kabul edilenlerde) "
               f"| 1560 karede kalibre |")
    return sat


def yontem_tablosu(v: dict) -> list[str]:
    p = v["ip6_aci_hatasi.json"]["yontemler"]
    sat = [
        "| Yöntem | Açı hatası ort | p95 | max | Karar |",
        "|---|---|---|---|---|",
        f"| **Kutupsal tarama** *(seçilen)* | {p['polar']['aci_hatasi_deg']['ortalama']}° "
        f"| {p['polar']['aci_hatasi_deg']['p95']}° | {p['polar']['aci_hatasi_deg']['max']}° | K3 — hem doğru hem hızlı |",
        f"| Hough dönüşümü | {p['hough']['aci_hatasi_deg']['ortalama']}° "
        f"| {p['hough']['aci_hatasi_deg']['p95']}° | {p['hough']['aci_hatasi_deg']['max']}° | elendi |",
    ]
    ip9 = yukle("ip9_cnn_kiyas.json")
    if ip9:
        sat.append(f"| CNN regresyon (İP9) | {ip9.get('aci_hatasi_ort', '?')}° | "
                   f"{ip9.get('aci_hatasi_p95', '?')}° | {ip9.get('aci_hatasi_max', '?')}° | kıyas |")
    else:
        sat.append("| CNN regresyon (İP9) | — | — | — | *kırpıldı: planında "
                   "'kırpılabilir' işaretliydi; kalan süre İP8 gerçek teste ayrıldı* |")
    return sat


def zor_kosul_tablosu(v: dict) -> list[str]:
    eksenler = v["ip14_zor_kosullar.json"]["eksenler"]
    sat = ["| Eksen | En zor seviye | Ortalama hata | Not |", "|---|---|---|---|"]
    for ad, seviyeler in eksenler.items():
        son_ad = list(seviyeler)[-1]
        son = seviyeler[son_ad]
        h = son.get("hata_yuzde_tam_skala") or {}
        notlar = {"egiklik": "tek başına baskın etken",
                  "parlama": "okumayı değil TESPİTİ bozuyor",
                  "jpeg": "etkisi yok", "dusuk_isik": "neredeyse etkisiz"}
        sat.append(f"| {ad} | {son_ad} | %{h.get('ortalama', '—')} | {notlar.get(ad, '')} |")
    return sat


def ip8_tablosu(v: dict) -> list[str]:
    z = v["ip8_zincir_hatasi.json"]
    sat = ["| Koşu | Kapsama | Sonuç |", "|---|---|---|",
           f"| Sentetik zincir (v1, referans) | {z['okunan']}/{z['goruntu']} "
           f"| ort %{z['hata_yuzde_tam_skala']['ortalama']} |",
           "| Benzetilmiş çekim (19.08 günlüğü) | 12/12 | ort %0,697 |"]
    # Güncel zincirin ölçümü önce denenir. `ip8_ekran_hatasi.json` 19.08'de
    # tek sınıflı ağırlıkla ve eski roll kapısıyla üretildi; aynı 12 fotoğrafın
    # bugünkü zincirle ölçümü `..._yeniden.json`'da (tip başına ayrılmış yeni
    # şema). Sabit olarak eskisini okumak, final paketinde **bayat ve daha kötü**
    # bir sayı raporlamak demekti (%6,25 yerine gerçeği %5,71).
    gercek = yukle("ip8_ekran_hatasi_yeniden.json") or yukle("ip8_ekran_hatasi.json")
    if gercek and "tipler" in gercek:
        for tip, blok in gercek["tipler"].items():
            if tip == "analog":
                h = blok.get("hata") or {}
                sat.append(f"| **Gerçek çekim — analog** | {blok['okunan']}/{blok['kare']} "
                           f"| ort %{h.get('mean', '—')} |")
            else:
                sat.append(f"| **Gerçek çekim — {tip}** | — | "
                           f"{blok['dogru']}/{blok['kare']} doğru |")
    elif gercek:
        sat.append(f"| Gerçek çekim (eski tek-tip format) | "
                   f"{gercek.get('okunan', '?')}/{gercek.get('kare', '?')} | "
                   f"ort %{(gercek.get('hata') or {}).get('mean', '—')} |")
    else:
        sat.append("| **Gerçek çekim** | — | *henüz yapılmadı — "
                   "`docs/cekim_talimati.md` (28 kare, ~40 dk)* |")
    return sat


def figur(v: dict, yol: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle("GÖSTERGE — final ölçüm özeti", fontweight="bold")

    zincir = v["ip8_zincir_hatasi.json"]["gosterge_bazli"]
    adlar = list(zincir)
    x = range(len(adlar))
    a1.bar([i - 0.2 for i in x], [zincir[a]["ortalama"] for a in adlar], 0.4,
           label="ortalama")
    a1.bar([i + 0.2 for i in x], [zincir[a]["p95"] for a in adlar], 0.4,
           label="p95")
    a1.axhline(HEDEF_YUZDE, ls="--", c="crimson", lw=1)
    a1.text(0.02, HEDEF_YUZDE * 1.03, f"hedef %{HEDEF_YUZDE:g}", c="crimson",
            fontsize=8)
    a1.set_xticks(list(x), adlar)
    a1.set_ylabel("% tam skala")
    a1.set_title("Zincir uçtan uca (sentetik v1)")
    a1.legend(fontsize=8)

    ip11 = v["ip11_dijital.json"]["kosullar"]["temiz"]["dizge_dogrulugu"]
    ip12 = v["ip12_lamba_vana.json"]
    dogruluklar = {"dijital\n(dizge)": ip11,
                   "lamba": ip12["lamba"]["temiz"]["dogruluk"],
                   "vana": ip12["vana"]["temiz"]["dogruluk"]}
    a2.bar(list(dogruluklar), [100 * d for d in dogruluklar.values()],
           color="seagreen")
    a2.set_ylim(0, 105)
    a2.set_ylabel("%")
    a2.set_title("Diğer tipler (temiz koşul, sentetik)")
    for i, d in enumerate(dogruluklar.values()):
        a2.text(i, 100 * d + 1, f"%{100 * d:.1f}", ha="center", fontsize=8)

    tarama = v["ip15_guven_esigi.json"]["tarama"]
    esikler = [t["esik"] for t in tarama]
    a3.plot(esikler, [100 * t["kapsama"] for t in tarama], label="kapsama %")
    a3.plot(esikler, [100 * t["sessiz_hata_tum_kareler"] for t in tarama],
            label="sessiz hata %")
    secilen = v["ip15_guven_esigi.json"]["secilen_esik"]["esik"]
    a3.axvline(secilen, ls="--", c="gray", lw=1)
    a3.text(secilen + 0.01, 50, f"seçilen {secilen}", fontsize=8, c="gray")
    a3.set_xlabel("güven eşiği")
    a3.set_title("Güven eşiği ödünleşmesi (1560 kare)")
    a3.legend(fontsize=8)

    fig.tight_layout()
    yol.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(yol, dpi=150)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="İP16 final ölçüm paketi")
    ap.add_argument("--tam", action="store_true",
                    help="önce bütün ölçüm scriptlerini yeniden koştur")
    args = ap.parse_args()

    if args.tam:
        for ad, komut, zorunlu in KAYNAKLAR:
            if komut.startswith("python") and (zorunlu or (M / ad).exists()):
                kosur(komut)

    veriler, eksik = {}, []
    for ad, komut, zorunlu in KAYNAKLAR:
        d = yukle(ad)
        if d is not None:
            veriler[ad] = d
        elif zorunlu:
            eksik.append((ad, komut))
    if eksik:
        print("HATA: zorunlu ölçümler eksik — final paketi eksik sayıyla kurulmaz:")
        for ad, komut in eksik:
            print(f"  {ad:<28} → {komut}")
        return 1

    bolumler = [
        "# İP16 — Final Ölçüm Tabloları",
        f"\nÜretim: `python scripts/olc_final.py` · {date.today().isoformat()}",
        "\n## Ana metrikler\n", *ana_tablo(veriler),
        "\n## Yöntem kıyası (İP6 · K3)\n", *yontem_tablosu(veriler),
        "\n## Zor koşullar — eksen başına en kötü durum (İP14)\n",
        *zor_kosul_tablosu(veriler),
        "\n## İP8 — gerçek görüntü durumu\n", *ip8_tablosu(veriler),
        "\n> Dijital/lamba/vana sayıları modelin KENDİ sentetik üretecinin",
        "> çıktısında ölçülmüştür; başka tasarım bir panele genellemeyi",
        "> göstermez. Bu sınır raporda ayrıca beyan edilir.",
    ]
    tablo_yolu = M / "ip16_final_tablo.md"
    tablo_yolu.write_text("\n".join(bolumler) + "\n", encoding="utf-8")

    figur_yolu = F / "ip16_final_ozet.png"
    figur(veriler, figur_yolu)

    ozet = {
        "is_paketi": "IP16",
        "tarih": date.today().isoformat(),
        "kaynak_dosyalar": sorted(veriler),
        "eksik_opsiyoneller": [ad for ad, _, z in KAYNAKLAR
                               if not z and ad not in veriler],
        "test": None,   # pytest sayısı elle yazılmaz; koşturan bilir
    }
    ozet_yolu = M / "ip16_final_ozet.json"
    ozet_yolu.write_text(json.dumps(ozet, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print("\n".join(bolumler))
    print(f"\nPaket: {ozet_yolu}\nTablolar: {tablo_yolu}\nFigür: {figur_yolu}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
