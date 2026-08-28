"""Çıktı videolarını BAŞKASINA GÖNDERİLEBİLİR hâle getirir.

    python demo/paylasilabilir_yap.py
    python demo/paylasilabilir_yap.py --hedef-mb 16      # WhatsApp sınırı

Sorunun iki yarısı vardı ve ikisi ayrı ayrı çözülüyor:

1. **Kodek.** `cv2` ile `mp4v` yazınca dosya MPEG-4 Part 2 oluyor; VLC açar ama
   tarayıcı, WhatsApp, iPhone açmaz. `demo/video_yazici.py` bunu `avc1`e
   (gerçek H.264) çevirerek çözdü.

2. **Boyut — ve bu, 1'in ÇÖZÜMÜ yüzünden KÖTÜLEŞTİ.** OpenCV Windows'ta H.264'ü
   Media Foundation ile kodluyor ve bit hızını ayarlamaya izin vermiyor;
   ölçüldü (28.08, 17 video): mp4v ile toplam 577 MB, avc1 ile **796 MB**.
   `genis_ekip` tek başına 177 → 281 MB. Yani "uyumlu ama daha da
   gönderilemez" bir çıktı oluştu.

   Çözüm gerçek bir kodlayıcı: `imageio-ffmpeg` (venv içine kurulu, sisteme
   dokunmuyor) x264'ü CRF ile kullanır. CRF sabit KALİTE hedefler, sabit bit
   hızı değil — hareketsiz sahnede küçük, hareketli sahnede büyük dosya çıkar
   ve ikisi de aynı görsel kalitede olur.

Özgün çıktılar SİLİNMEZ; küçük kopyalar `<ad>_paylas.mp4` olarak yanına yazılır.
Gözle inceleme için tam kaliteli olan, göndermek için küçük olan kullanılır.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
VARSAYILAN_KLASOR = DEMO_DIR / "cikti" / "ekip"

# 23 x264'ün varsayılanı ve "görsel olarak kayıpsıza yakın" kabul edilen yer.
# Panel çıktısı düz renkli kutular ve yazı içerdiği için burada fazlasıyla
# yeterli; 28 denendiğinde yazılar bulanıklaştı.
CRF = 23
PRESET = "veryfast"     # daha yavaş preset %10-15 kazandırıyor, 3 kat sürüyor
# yuv420p + faststart olmadan bazı oynatıcılar (Safari, WhatsApp önizleme)
# dosyayı açmıyor; kodek doğru olsa bile.
PIKSEL_BICIMI = "yuv420p"


def ffmpeg_yolu() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("imageio-ffmpeg kurulu degil: pip install imageio-ffmpeg")


def cevir(kaynak: Path, hedef: Path, ffmpeg: str, crf: int,
          en_tavan: int | None = None) -> None:
    olcek = []
    if en_tavan:
        # Tek sayı genislik/yukseklik x264'te hata verir; -2 en yakin cifte yuvarlar.
        olcek = ["-vf", f"scale='min({en_tavan},iw)':-2"]
    komut = [ffmpeg, "-y", "-loglevel", "error", "-i", str(kaynak), *olcek,
             "-c:v", "libx264", "-crf", str(crf), "-preset", PRESET,
             "-pix_fmt", PIKSEL_BICIMI, "-movflags", "+faststart",
             "-an", str(hedef)]
    subprocess.run(komut, check=True)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--klasor", type=Path, default=VARSAYILAN_KLASOR)
    p.add_argument("--crf", type=int, default=CRF,
                   help="dusuk = daha kaliteli/buyuk (18-28 arasi anlamli)")
    p.add_argument("--hedef-mb", type=float, default=None,
                   help="bu boyutu asan dosyalar icin CRF kademeli yukseltilir")
    p.add_argument("--en-tavan", type=int, default=None,
                   help="genislik tavani (or. 1280); oran korunur")
    a = p.parse_args(argv)

    ffmpeg = ffmpeg_yolu()
    kaynaklar = sorted(y for y in a.klasor.glob("*.mp4")
                       if not y.stem.endswith("_paylas"))
    if not kaynaklar:
        raise SystemExit(f"video yok: {a.klasor}")

    print(f"{len(kaynaklar)} video · CRF {a.crf} · {ffmpeg.split(chr(92))[-1]}\n")
    onceki = yeni = 0.0
    for n, k in enumerate(kaynaklar, 1):
        hedef = k.with_name(f"{k.stem}_paylas.mp4")
        eski_mb = k.stat().st_size / 1048576
        crf = a.crf
        cevir(k, hedef, ffmpeg, crf, a.en_tavan)

        # Hedef boyut verildiyse asanlar icin CRF yukseltilir. Kademeli, cunku
        # gereginden fazla sikistirmak yaziyi okunmaz yapar - ve bu panelde
        # okunacak sey tam olarak yazilar (aci, guven, alarm).
        if a.hedef_mb:
            while hedef.stat().st_size / 1048576 > a.hedef_mb and crf < 34:
                crf += 3
                cevir(k, hedef, ffmpeg, crf, a.en_tavan)

        yeni_mb = hedef.stat().st_size / 1048576
        onceki += eski_mb
        yeni += yeni_mb
        isaret = "" if not a.hedef_mb or yeni_mb <= a.hedef_mb else "  HALA BUYUK"
        print(f"[{n:2d}/{len(kaynaklar)}] {k.name:22s} {eski_mb:7.1f} -> {yeni_mb:6.1f} MB"
              f"  (CRF {crf}){isaret}")

    print(f"\ntoplam {onceki:.0f} MB -> {yeni:.0f} MB  (%{100*(1-yeni/onceki):.0f} kucuk)")
    print(f"gonderilecek dosyalar: {a.klasor}\\*_paylas.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
