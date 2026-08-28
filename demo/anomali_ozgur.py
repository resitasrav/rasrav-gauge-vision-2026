"""ANOMALİ panelini ÖZGÜR'ÜN KENDİ MODÜLÜNE bağlar.

28.08'de RAPOR.md madde 1'in ANOMALİ ayağı KAPANDI: Özgür
`scripts/core/anomali_motor.py` içinde `AlgilayiciMOG2` sınıfını yayımladı ve
sınıfın `isle(frame) -> dict` diye tek kare alan bir yöntemi var. Aranan
sözleşme buydu. Artık panelde onun yönteminin bir kopyası değil, DOSYASI
koşuyor — import edilip çağrılıyor, değiştirilmiyor.

Bu yüzden `anomali_demo.py` (benim PaDiM sarmalayıcım) YEDEĞE düştü: Özgür'ün
deposu diskte bulunamazsa panel yine bir şey gösterebilsin diye duruyor, ama
tercih her zaman onun modülüdür.

DEPO NEREDE: Özgür deposunun geçmişini yeniden kurmuş — yerel klonun `main`'i
(529fd84, 14 commit) ile `origin/main` (3833b5e, 64 commit) arasında ORTAK ATA
YOK. Bu yüzden klona dokunulmadı; güncel ağaç ayrı bir worktree'ye çıkarıldı.
Aşağıdaki adaylar sırayla denenir, ilk bulunan kullanılır.

YÖNTEM FARKI, kayıt için: onunki MOG2 arka plan çıkarma (hareketli ön plan
nesnesi arar), benimki PaDiM (referans dağılımından sapma arar). İkisi farklı
soruları cevaplıyor ve devriye senaryosunda doğru soru onunkidir — "koridorda
olmaması gereken bir şey var mı". İlginç bir örtüşme: benim PaDiM ısı haritamın
yanlış parlattığı sarı hissedilebilir yüzeyi o da bağımsız olarak bulmuş ve
`build_yellow_mask` ile bastırmış.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Sırayla denenir. `_guncel` olan, geçmişi ayrışmış klonun yanına açılan
# worktree — klonun kendisi eski commit'te durduğu için modül orada YOK.
ADAY_KOKLER = (
    "ORTAK/OrtakProjeler/OzgurKotbas_guncel",
    "ORTAK/OrtakProjeler/OzgurKotbas_Akilli_Fabrika",
)
IZ_DOSYA = Path("scripts/core/anomali_motor.py")


def kok_bul(staj_dir: Path) -> Path | None:
    for aday in ADAY_KOKLER:
        yol = staj_dir / aday
        if (yol / IZ_DOSYA).is_file():
            return yol
    return None


class OzgurMotoru:
    """Özgür'ün `AlgilayiciMOG2`'sini saran ince kabuk — mantık ONUN dosyasında."""

    def __init__(self, kok: Path):
        import sys
        if str(kok) not in sys.path:
            sys.path.insert(0, str(kok))
        from scripts.core.anomali_motor import PARAMS, AlgilayiciMOG2
        self.kok = kok
        self.params = PARAMS
        self.algilayici = AlgilayiciMOG2()
        self.isindi = False

    def isle(self, kare: np.ndarray) -> tuple[np.ndarray, dict]:
        # Ö1 (Özgür'ün düzeltmesi): MOG2 ilk karede arka planı bilmez ve
        # "cold-start" yanlış alarmı üretir; ilk kare N kez beslenerek model
        # ısıtılır. Burada ÇAĞRILIYOR, yeniden yazılmıyor.
        if not self.isindi:
            self.algilayici.warmup(kare)
            self.isindi = True

        r = self.algilayici.isle(kare)
        ciz = kare.copy()

        # Ö2'nin görünür karşılığı: tavanın maskelenen kısmı ekranda da
        # gösteriliyor ki panele bakan kişi "orayı hiç bakmıyor" bilsin.
        tavan = int(ciz.shape[0] * self.params["tavan_crop_oran"])
        ciz[:tavan] = (0.45 * ciz[:tavan]).astype(np.uint8)
        cv2.line(ciz, (0, tavan), (ciz.shape[1], tavan), (90, 90, 90), 1)

        for n in r["nesneler"]:
            cv2.rectangle(ciz, (n["x"], n["y"]), (n["x"] + n["w"], n["y"] + n["h"]),
                          (0, 0, 230), 2)
            cv2.putText(ciz, f"{n['area']}px", (n["x"], max(n["y"] - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2, cv2.LINE_AA)

        if r["is_rotation"]:
            etiket, renk = "DONME - tespit askida", (0, 165, 255)
        elif r["is_alert"]:
            etiket, renk = f"ALARM  {len(r['nesneler'])} nesne", (0, 0, 230)
        else:
            etiket, renk = "normal", (0, 200, 0)
        cv2.putText(ciz, etiket, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, renk, 2,
                    cv2.LINE_AA)
        cv2.putText(ciz, f"fg {r['fg_ratio']:.4f}  akis {r['flow_mag']:.2f}",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)

        return ciz, {"alarm": bool(r["is_alert"]), "nesne": len(r["nesneler"]),
                     "fg_orani": r["fg_ratio"], "donme": bool(r["is_rotation"]),
                     "akis": r["flow_mag"]}
