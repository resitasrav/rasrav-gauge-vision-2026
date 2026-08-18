# Tavsiye Edilen Iyilestirmeler

Bu belge, gercek saha videosunda karar kalitesini artirmak icin oncelikli
teknik calisma listesidir. Her madde, olculebilir bir kabul kosuluna baglidir;
kanit uretilmeden "tamamlandi" sayilmaz.

## Uygulanan: Canli Akis Sabitleme

`src/gauge_vision/temporal.py`, tek karelik okuma sonucunun ardina eklenen
kareler arasi karar katmanidir. `scripts/canli_oku.py` kamera modunda bunu
varsayilan olarak kullanir.

- Ilk sayisal deger veya durum, uc ardisik kare gorulmeden gecerli sayilmaz.
- Sayisal ani sicrama, ayni yeni seviyede tekrar gorulmeden kabul edilmez.
- Lamba ve vana durum degisimi oyla onaylanir.
- Tespit en fazla iki kare kaybolursa son dogrulanmis sonuc, guveni dusurulerek
  tutulur; sonrasinda deger gizlenir.
- Tespit kutusu EMA ile yumusatilir; bu yalnizca ekrandaki ve sonraki karar
  katmanlarindaki konumu dengeler, tek kare okuyucuyu degistirmez.

Kamera kullanimi:

```powershell
python scripts/canli_oku.py --kaynak 0 --gosterge PT-101
python scripts/canli_oku.py --kaynak 0 --gosterge PT-101 --no-temporal
python scripts/canli_oku.py --kaynak 0 --gosterge PT-101 --temporal-kare 5 --kayip-toleransi 3
```

Kabul olcutu: etiketli kisa video dizilerinde, ham sonuca gore deger
titremesi, tek karelik yanlis alarm ve gecici tespit kaybi sonrasi yanlis
durum degisimi ayri ayri olculmelidir.

## P0: Dijital Panelin Gercek Goruntude Okunmasi

Mevcut teshise gore sorun panel tespiti degil, yansima gradyani altinda hane
kutularinin bulunmasidir. Onceki uclu esikleme denemeleri tekrar edilmemeli.

1. Panel cercevesinin dort koseli donusumunu bulup dikdortgen gorunume duzeltin.
2. Hane sayisini `configs/gauges.yaml` icindeki `digits.count` alanindan alin.
3. Duzeltilmis paneli esit haneli bir izgara olarak bolun; goruntuden bagimsiz
   bilesen sayma yalnizca kalite kontrolu olsun.
4. Her hane icin segment guveni ve tum sayi icin minimum guven uretilsin.

Kabul olcutu: `scripts/olc_ip8.py --fotograflar data/real/ip8_ekran` sonucunda
dijital panel en az 3/5 dogru okunmali ve sessiz yanlis deger sayisi sifir
kalmalidir.

## P0: Gosterge Kimligini Calisma Zamaninda Zorunlu Kilma

Tip filtresi bir kadranin analog oldugunu bilir, bunun PT-101 oldugunu bilmez.
Bu nedenle numerik deger, ancak kamera duragi ile `gauge_id` eslestiginde
yayina uygun sayilmalidir.

1. Her fiziksel durak icin `waypoint_gosterge_sozlugu.yaml` icinde gauge listesi
   tamamlanmali ve sahada dogrulanmalidir.
2. Canli istemci waypoint bilgisi olmadan sadece tip seviyesinde tespit
   gostermeli; kalibrasyonlu degeri yayinlamamalidir.
3. Girdi mesaji `waypoint`, `gauge_id` ve goruntu zamanini birlikte tasimalidir.

Kabul olcutu: farkli analog gostergenin kadraja girdigi senaryoda yanlis birim
veya kalibrasyonla `status: ok` mesaji uretilmemelidir.

## P1: Gercek Zaman ve Optik Dayaniklilik

- Kamera yakalama, isleme ve yayin zamanlarini ayri olcun; kare yasi ve dusen
  kare sayisini raporlayin. Isleme yakalamanin gerisinde kalirsa sinirli
  kuyrukta yalniz en guncel kareyi isleyin.
- Analog kadranda perspektif duzeltmesini gercek video ile olcun. Kabul kapisi
  olan durumlarda etkinlestirin; her karede kosulsuz acmak maliyet ve hata
  ekleyebilir.
- Tespit modeli icin yansima, hareket bulanikligi, kismi ortulme ve egik bakis
  iceren etiketli gercek kareler toplanmali. Sentetik veri varyasyonu destekler,
  saha genellemesini kanitlamaz.

Kabul olcutu: ayni kamera ve cozumurlukte p95 kare yasi, p95 isleme suresi,
tespit kapsami, mutlak deger hatasi ve yanlis alarm orani birlikte raporlanir.

## Test Stratejisi

- Birim testleri: `tests/test_temporal.py` kararsiz baslangic, ani sicrama,
  tespit kaybi, durum oylamasi ve gosterge degisimini kapsar.
- Entegrasyon testleri: etiketli video klipleri, ayni karenin art arda islenmesi
  yerine zaman sirasiyla degerlendirilmelidir.
- Regresyon esigi: yeni model veya esik ayari ancak gercek goruntude sessiz
  yanlis deger sayisini artirmiyorsa kabul edilmelidir.
