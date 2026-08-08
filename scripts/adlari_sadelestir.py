r"""İş paketi numaralarını dosya adlarından temizler — tek seferlik, İP16'da.

    python scripts\adlari_sadelestir.py --liste     # ne neye donusecek
    python scripts\adlari_sadelestir.py             # PROVA (hicbir sey degismez)
    python scripts\adlari_sadelestir.py --uygula    # gercekten yap

**Neden şimdi yazıldı, neden şimdi koşturulmuyor.** Dosya adlarındaki `ip6`,
`ip14` gibi önekler staj boyunca **takibi kolaylaştırmak** için kondu ve işlerini
gördüler. Ama proje bittiğinde kimse "İP14 neydi" diye tabloya bakmak zorunda
kalmamalı; ad, dosyanın ne yaptığını söylemeli.

Dönüşüm **staj sonunda, tek seferde** yapılmalı çünkü tarihli günlük raporlar
`outputs/metrics/ip14_zor_kosullar.json` gibi yollara **kanıt olarak** atıf
yapıyor. Bugün yeniden adlandırılırsa on günlük rapor olmayan dosyaları
gösterir. Rapor, o gün ne yapıldığının kaydıdır; geriye dönük düzeltilmez.
Bu dosyanın kendisi çeviri sözlüğüdür: eski rapordaki adı buradan bulursun.

**Eşleşme tek yerde duruyor** (aşağıdaki iki sözlük). Belgeye de kopyalanmadı;
`--liste` ile basılır. İki kopya tutulsaydı biri güncellenip diğeri unutulurdu —
projede bu hata sınıfı zaten iki kez çıktı.

**İP numaraları KODUN İÇİNDEN silinmiyor.** Yorumdaki "İP14'te ölçüldü" ifadesi
bir künyedir: o sayının nereden geldiğini, hangi raporda anlatıldığını söyler.
Silinirse gerekçe kaynağını kaybeder. Temizlenen yalnızca **dosya adlarıdır.**
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent

# --- 1. Yeniden adlandırılacak dosyalar (git izliyor) -------------------------
# Kural: <fiil>_<konu>.py. Fiil ne yaptığını (olc / egit / kalibre / yayinla /
# uret / hazirla), konu neyin üstünde çalıştığını söyler.
DOSYA_ADLARI: dict[str, str] = {
    "scripts/egit_ip5.py":         "scripts/egit_tespit.py",
    "scripts/hazirla_ip5_veri.py": "scripts/hazirla_tespit_verisi.py",
    "scripts/kalibre_ip15.py":     "scripts/kalibre_guven_esigi.py",
    "scripts/olc_ip6.py":          "scripts/olc_ibre_acisi.py",
    "scripts/olc_ip7.py":          "scripts/olc_aci_deger.py",
    "scripts/olc_ip8.py":          "scripts/olc_ekran_cekimi.py",
    "scripts/olc_ip11.py":         "scripts/olc_dijital_panel.py",
    "scripts/olc_ip12.py":         "scripts/olc_lamba_vana.py",
    "scripts/olc_ip13.py":         "scripts/olc_zincir_tum_tipler.py",
    "scripts/olc_ip14.py":         "scripts/olc_zor_kosullar.py",
    "scripts/yayinla_ip10.py":     "scripts/yayinla_mqtt.py",
}
# Adı zaten işini söyleyenler dokunulmadan kalıyor:
#   canli_oku · ekran_kadran · kadran_onizle · kalibre_vana · olc_zincir
#   uret_sentetik · adlari_sadelestir

# --- 2. Çıktı adları (kod içindeki sabitlerde geçiyor) ------------------------
# `outputs/` git'e girmiyor, dolayısıyla dosyaların kendisi taşınmıyor: adlar
# scriptlerdeki yol sabitlerinde değişiyor ve ölçümler yeniden koşturulunca
# yeni adla doğuyor. Ölçümü yeniden koşturmak zaten doğru olan: yeniden
# üretilemeyen bir sayı rapora girmemeli (4. kural).
CIKTI_ADLARI: dict[str, str] = {
    "ip3_sentetik_ozet":          "sentetik_ozet",
    "ip3_onizleme":               "kadran_onizleme",
    "ip3_ornek_izgara":           "sentetik_ornek_izgara",
    "ip5_veri_ozeti":             "tespit_veri_ozeti",
    "ip5_tespit":                 "tespit",
    "ip6_aci_hatasi":             "ibre_aci_hatasi",
    "ip6_hata_dagilimi":          "ibre_hata_dagilimi",
    "ip6_ornek_okuma":            "ibre_ornek_okuma",
    "ip7_okuma_hatasi":           "aci_deger_hatasi",
    "ip8_zincir_hatasi":          "zincir_hatasi",
    "ip8_ekran_manifest":         "ekran_manifest",
    "ip8_ekran_hatasi":           "ekran_cekim_hatasi",
    "ip8_kontak_sayfasi":         "ekran_kontak_sayfasi",
    "ip11_dijital":               "dijital_panel",
    "ip12_lamba_vana":            "lamba_vana",
    "ip13_zincir_tum_tipler":     "zincir_tum_tipler",
    "ip13_dijital_ornek":         "dijital_ornek",
    "ip13_lamba_ornek":           "lamba_ornek",
    "ip13_vana_ornek":            "vana_ornek",
    "canli_ip13_":                "canli_",
    "ip14_zor_kosullar":          "zor_kosullar",
    "ip14_guven_hata_ciftleri":   "guven_hata_ciftleri",
    "ip15_guven_esigi":           "guven_esigi",
}

# İçeriği taranacak yerler. `outputs/` ve `data/` yok: onlar üretilen şeyler,
# kaynak değil.
TARANAN = ("scripts", "src", "tests", "docs")
TARANAN_DOSYA = ("README.md", "CLAUDE.md")
UZANTILAR = (".py", ".md", ".yaml", ".yml", ".toml", ".txt")


def taranacak_dosyalar() -> list[Path]:
    """Taranacak kaynak dosyalar — bu dosyanın KENDİSİ hariç.

    Kendini dışlaması şart: eşleşme sözlüğü eski adları metin olarak içeriyor,
    tarama buraya da uygulansaydı sözlüğün sol sütunu da yeniden yazılır ve
    çeviri sözlüğü kendini yiyerek işe yaramaz hâle gelirdi. (Prova bunu 35
    geçişle gösterdi.)
    """
    bu_dosya = Path(__file__).resolve()
    yollar = [KOK / ad for ad in TARANAN_DOSYA if (KOK / ad).exists()]
    for dizin in TARANAN:
        d = KOK / dizin
        if d.is_dir():
            yollar += [p for p in d.rglob("*")
                       if p.suffix in UZANTILAR and "__pycache__" not in p.parts]
    return sorted(p for p in yollar if p.resolve() != bu_dosya)


def degisiklikleri_bul() -> tuple[list[tuple[Path, int]], list[tuple[str, str]]]:
    """(içeriği değişecek dosyalar, yapılamayan yeniden adlandırmalar)."""
    # Script adları da metin olarak geçiyor (komut satırı örnekleri, CLAUDE.md
    # komut tablosu), bu yüzden iki sözlük birlikte uygulanıyor.
    eslesme = {Path(e).name: Path(y).name for e, y in DOSYA_ADLARI.items()}
    eslesme.update(CIKTI_ADLARI)

    dokunulacak = []
    for p in taranacak_dosyalar():
        metin = p.read_text(encoding="utf-8")
        sayi = sum(metin.count(e) for e in eslesme)
        if sayi:
            dokunulacak.append((p, sayi))

    eksik = [(e, y) for e, y in DOSYA_ADLARI.items() if not (KOK / e).exists()]
    return dokunulacak, eksik


def icerigi_yaz(p: Path) -> int:
    eslesme = {Path(e).name: Path(y).name for e, y in DOSYA_ADLARI.items()}
    eslesme.update(CIKTI_ADLARI)
    metin = yeni = p.read_text(encoding="utf-8")
    for eski, yeni_ad in eslesme.items():
        yeni = yeni.replace(eski, yeni_ad)
    if yeni != metin:
        p.write_text(yeni, encoding="utf-8")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="İP öneklerini dosya adlarından temizler")
    ap.add_argument("--liste", action="store_true", help="eşleşmeyi bas, çık")
    ap.add_argument("--uygula", action="store_true", help="gerçekten uygula")
    args = ap.parse_args()

    if args.liste:
        print("# Dosya adlari\n")
        print("| eski | yeni |\n|---|---|")
        for e, y in DOSYA_ADLARI.items():
            print(f"| `{e}` | `{y}` |")
        print("\n# Cikti adlari (outputs/metrics, outputs/figures)\n")
        print("| eski | yeni |\n|---|---|")
        for e, y in CIKTI_ADLARI.items():
            print(f"| `{e}` | `{y}` |")
        return 0

    dokunulacak, eksik = degisiklikleri_bul()

    if eksik:
        print("UYARI — su dosyalar bulunamadi (zaten adlandirilmis olabilir):")
        for e, y in eksik:
            print(f"  {e} -> {y}")
        print()

    print(f"Yeniden adlandirilacak: {len(DOSYA_ADLARI) - len(eksik)} dosya")
    print(f"Icerigi guncellenecek : {len(dokunulacak)} dosya")
    for p, n in dokunulacak:
        print(f"  {p.relative_to(KOK).as_posix():<48} {n} gecis")

    if not args.uygula:
        print("\nPROVA — hicbir sey degismedi. Uygulamak icin: --uygula")
        return 0

    # Once icerik, sonra yeniden adlandirma: ters sirada olsaydi git mv sonrasi
    # dosya yollari degisir ve tarama listesi bayatlardi.
    for p, _ in dokunulacak:
        icerigi_yaz(p)
    for e, y in DOSYA_ADLARI.items():
        if (KOK / e).exists():
            subprocess.run(["git", "mv", e, y], cwd=KOK, check=True)

    print("\nBitti. Simdi:")
    print("  1) python -m pytest")
    print("  2) olcumleri yeniden kostur (cikti adlari degisti)")
    print("  3) commit: 'Adlandirma: is paketi onekleri dosya adlarindan kaldirildi'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
