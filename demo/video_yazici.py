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
              sessiz: bool = False) -> tuple[cv2.VideoWriter, str]:
    """Paylasilabilir bir VideoWriter acar; (yazici, kullanilan_kodek) doner."""
    yol.parent.mkdir(parents=True, exist_ok=True)
    for kodek in KODEKLER:
        y = cv2.VideoWriter(str(yol), cv2.VideoWriter_fourcc(*kodek), fps, boyut)
        if y.isOpened():
            if kodek != KODEKLER[0] and not sessiz:
                print(f"[UYARI] H.264 acilamadi, {kodek} kullaniliyor - bu dosya "
                      f"tarayicida/WhatsApp'ta oynamayabilir: {yol.name}")
            return y, kodek
        y.release()
    raise RuntimeError(f"video yazici acilamadi: {yol}")


def kodek_oku(yol: Path) -> str:
    """Yazilmis dosyanin fourcc'si - dogrulama icin."""
    cap = cv2.VideoCapture(str(yol))
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
