"""Tek seferlik: gosterge.mp4/araba.mp4'ten elle kutulanmış 13 zor kareyi
data/raw/video_kareler_v1'e yazar. Kutular gözle (piksel ızgara üstünden)
belirlendi — tespit eğitimi için kaba tolerans yeterli, İP6'nın ibre
hassasiyetiyle karıştırılmamalı.
"""
import os
import cv2

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEDEF = os.path.join(KOK, "data", "raw", "video_kareler_v1")
os.makedirs(os.path.join(HEDEF, "images"), exist_ok=True)
os.makedirs(os.path.join(HEDEF, "labels"), exist_ok=True)

GOSTERGE_VIDEO = r"c:/Users/resit/Desktop/STAJ/demo/girdi/gosterge.mp4"
ARABA_VIDEO = r"c:/Users/resit/Desktop/STAJ/demo/girdi/araba.mp4"

# (video, kare_no, [(x1,y1,x2,y2), ...])
KARELER = [
    (GOSTERGE_VIDEO, 0,   [(568, 383, 656, 466), (735, 595, 826, 662)]),
    (GOSTERGE_VIDEO, 45,  [(565, 385, 650, 468), (735, 595, 826, 662)]),
    (GOSTERGE_VIDEO, 90,  [(560, 383, 648, 465), (735, 595, 826, 662)]),
    (GOSTERGE_VIDEO, 135, [(555, 383, 645, 462), (735, 595, 826, 662)]),
    (GOSTERGE_VIDEO, 180, [(548, 390, 628, 470), (735, 595, 826, 662)]),
    (GOSTERGE_VIDEO, 225, [(540, 375, 625, 460), (735, 595, 826, 662)]),
    (GOSTERGE_VIDEO, 300, [(735, 595, 826, 662)]),   # ibre arkası gösterildi, elde tutulan hariç
    (GOSTERGE_VIDEO, 345, [(735, 595, 826, 662)]),   # aynı sebep
    (ARABA_VIDEO, 405, [(425, 435, 655, 600)]),
    (ARABA_VIDEO, 630, [(395, 325, 647, 486)]),
    (ARABA_VIDEO, 660, [(331, 336, 578, 493)]),
    (ARABA_VIDEO, 735, [(334, 362, 582, 505)]),
    (ARABA_VIDEO, 765, [(342, 362, 588, 496)]),
]

acik = {}
yazilan = 0
for video, kare_no, kutular in KARELER:
    if video not in acik:
        acik[video] = cv2.VideoCapture(video)
    cap = acik[video]
    cap.set(cv2.CAP_PROP_POS_FRAMES, kare_no)
    ok, frame = cap.read()
    if not ok:
        print("okunamadi", video, kare_no)
        continue
    h, w = frame.shape[:2]
    ad_govde = ("gosterge" if "gosterge" in video else "araba") + f"_{kare_no:05d}"

    satirlar = []
    for (x1, y1, x2, y2) in kutular:
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        satirlar.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    cv2.imwrite(os.path.join(HEDEF, "images", f"{ad_govde}.png"), frame)
    with open(os.path.join(HEDEF, "labels", f"{ad_govde}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(satirlar) + "\n")
    yazilan += 1
    print(ad_govde, w, h, len(kutular), "kutu")

for cap in acik.values():
    cap.release()

print(f"\ntoplam {yazilan} görüntü yazıldı: {HEDEF}")
