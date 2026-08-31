"""Paylasilabilir MP4 yazici — H.264 (avc1), mp4v'ye geri donusle.

NEDEN VAR: `cv2.VideoWriter_fourcc(*"mp4v")` uzantisi .mp4 olan ama icinde
MPEG-4 Part 2 (eski DivX tarzi) tasiyan bir dosya uretir. Dosya bozuk degildir
ve VLC acar, ama:

  * taraycilar oynatmaz (Chrome/Edge/Firefox yalniz H.264 / VP9 / AV1),
  * WhatsApp / Telegram / Instagram reddeder veya onizleme veremez,
  * iPhone ve Windows Photos cogu zaman acmaz,
  * Google Drive / Slack onizlemesi calismaz.

Yani cikti "birine gonderilemez" hale geliyor ve bunun sebebi kodda tek bir
fourcc. `avc1` gercek H.264 uretir (dosyada `avc1` + `avcC` kutulari).

Windows'ta bu yol soyle isliyor ve konsolda KORKUTUCU gorunuyor: OpenCV once
FFmpeg'i deniyor, onun libopenh264 DLL'i bulunmadigi icin "Unable to create
encoder / Failed to initialize VideoWriter" yaziyor, SONRA Windows Media
Foundation'a dusup basariyla kodluyor. Hata satirlari gecerli, sonuc dogru;
bu yuzden acilip acilmadigi mesaja degil `isOpened()`e bakilarak sinaniyor.
"""
from __future__ import annotations

from pathlib import Path

import cv2

# Sirayla denenir; ilk acilan kullanilir. mp4v en sonda cunku uyumsuz - ama
# hicbir sey yazamamaktansa uyumsuz dosya yazmak yeglenir (uyari basilir).
KODEKLER = ("avc1", "mp4v")


def yazici_ac(yol: Path, fps: float, boyut: tuple[int, int],
              sessiz: bool = False) -> tuple["SayanYazici", str]:
    """Paylasilabilir bir VideoWriter acar; (yazici, kullanilan_kodek) doner."""
    yol.parent.mkdir(parents=True, exist_ok=True)
    for kodek in KODEKLER:
        y = cv2.VideoWriter(str(yol), cv2.VideoWriter_fourcc(*kodek), fps, boyut)
        if y.isOpened():
            if kodek != KODEKLER[0] and not sessiz:
                print(f"[UYARI] H.264 acilamadi, {kodek} kullaniliyor - bu dosya "
                      f"tarayicida/WhatsApp'ta oynamayabilir: {yol.name}")
            return SayanYazici(y, boyut, yol), kodek
        y.release()

    # HICBIR kodek acilamadiysa sebep genellikle kodek DEGILDIR. Bu mesaj
    # boyle yazildi cunku ilk hali sadece "video yazici acilamadi" diyordu ve
    # gercek sebep (dosyanin baska bir surecte acik olmasi) hic gorunmuyordu;
    # konsoldaki openh264 uyarilari da yanlis ize sokuyordu.
    kilitli = _yazilabilir_mi(yol) is False
    ipucu = ("\n  Dosya BASKA BIR SUREC tarafindan kullaniliyor gorunuyor "
             "(ayni demoyu iki kez ayni anda calistirdiniz mi?). "
             "Digerini kapatin ya da --cikti ile baska klasor verin."
             if kilitli else
             f"\n  Boyut {boyut}, fps {fps:.2f}. Sifir/tek boyut veya gecersiz "
             "fps yazici acmayi engeller.")
    raise RuntimeError(f"video yazici acilamadi: {yol}{ipucu}")


def _yazilabilir_mi(yol: Path) -> bool | None:
    """Dosya yazmaya acilabiliyor mu; belirlenemezse None."""
    if not yol.exists():
        return None
    try:
        with open(yol, "ab"):
            return True
    except OSError:
        return False


class SayanYazici:
    """VideoWriter sarmalayıcısı — boyutu uymayan kareyi SESSİZCE ATMAZ.

    `cv2.VideoWriter.write()` açılış boyutundan farklı bir kare gelince onu
    atar ve hiçbir şey söylemez: dönüş değeri yok, istisna yok, log yok.
    31.08'de bu tam olarak yaşandı — 2108 karelik video işlendi, JSON'a 2108
    kare yazıldı, mp4'e 916 kare girdi ve 70 saniyelik girdi 30 saniyelik
    çıktı verdi. Hata ancak kullanıcı süreye bakınca fark edildi.

    Bu sınıf yazılan kareyi sayar ve boyut uyuşmazlığında HEMEN patlar. Sayaç
    ayrıca çağıranın "kaç kare işledim" sayısıyla karşılaştırılabilir; ikisi
    tutmuyorsa bir şey kayboldu demektir.
    """

    def __init__(self, yazici: cv2.VideoWriter, boyut: tuple[int, int], yol: Path):
        self._y = yazici
        self.boyut = boyut          # (en, boy)
        self.yol = yol
        self.yazilan = 0

    def write(self, kare) -> None:
        boy, en = kare.shape[:2]
        if (en, boy) != self.boyut:
            raise RuntimeError(
                f"kare boyutu yazici ile uyusmuyor: kare {en}x{boy}, "
                f"yazici {self.boyut[0]}x{self.boyut[1]} ({self.yol.name}). "
                "VideoWriter bu kareyi SESSIZCE atardi; cikti girdiden kisa "
                "olurdu. Cikti tuvali sabit boyutlu olmali.")
        self._y.write(kare)
        self.yazilan += 1

    def release(self) -> None:
        self._y.release()


def kodek_oku(yol: Path) -> str:
    """Yazilmis dosyanin fourcc'si - dogrulama icin."""
    cap = cv2.VideoCapture(str(yol))
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
