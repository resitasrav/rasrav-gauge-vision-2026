"""`inspect/reading` yayını — İP10.

    from gauge_vision.publish.reading import ReadingPublisher, mesaj_dogrula

    yayinci = ReadingPublisher(host="localhost")
    yayinci.baglan()
    yayinci.yayinla(okuma, img_ref="frames/0042.jpg")

**Şema tek yerden gelir.** Mesaj gövdesini `GaugeReading.as_message()` üretir;
bu dosya yalnızca zarfı (zaman damgası, kaynak, `img_ref`) ekler ve taşır.
Alan adları burada ikinci kez tanımlanmaz — iki yerde tanımlanan bir şema
kaçınılmaz olarak ayrışır.

**Broker olmadan da çalışır.** `paho-mqtt` kurulu değilse ya da brokera
bağlanılamıyorsa yayıncı **dosya moduna** düşer: mesajlar JSONL olarak
`outputs/mqtt/` altına yazılır. Sebep pratik — U5 nedeniyle ekibin kayıt aracı
`inspect/reading`'i henüz kaydetmiyor ve broker her zaman ayakta olmuyor;
yayının doğruluğu bu bağımlılıklar çözülmeden de ölçülebilmeli. Mod
`ReadingPublisher.mod` ile bildirilir, gizlenmez.

**Doğrulama yayından ÖNCE.** `mesaj_dogrula` her mesajı sözleşmeye karşı
denetler ve uymayanı yayınlatmaz. Bozuk bir mesajı brokera bırakmak, onu
tüketen tarafta (Özgür'ün tur raporu) sessiz bir hataya dönüşür; hatayı
kaynağında durdurmak entegrasyonu günler kazandırır.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gauge_vision.read.calibrate import GaugeReading

KONU = "inspect/reading"
SEMA_SURUMU = 1

# Sözleşmedeki durum kümesi. Yeni bir durum eklemek şema değişikliğidir ve
# ekip kararı gerektirir (KT2) — bu yüzden burada sabit ve kapalı.
GECERLI_DURUMLAR = {"ok", "unreadable", "out_of_range", "alarm"}
# 27.08'de `keypad` (buton/tuş paneli) eklendi. **Mesajın ŞEKLİ değişmedi:**
# alan adları, türleri ve `schema: 1` aynı; buton paneli de lamba ve vana gibi
# `value` alanına bir DURUM ADI basıyor ("calisiyor"). Değişen tek şey `type`
# alanının alabileceği değerler kümesidir.
#
# Sürüm numarası artırılmadı çünkü alan bazında ayrıştıran hiçbir tüketici
# etkilenmiyor; yalnız `type` üzerinde KAPALI bir eşleşme yazmış bir tüketici
# tanımadığı bir değer görür. Bu, ekibe bildirilecek bir sözleşme genişlemesidir
# ve `../ortak uyusmazliklar/uyusmazliklar.md` dosyasına yazıldı (U-KP1).
# Ekip farklı karar verirse yapılacak şey `SEMA_SURUMU`'nü artırmaktır.
GECERLI_TIPLER = {"analog", "digital", "lamp", "valve", "keypad"}

# Zarf alanları — gövde `as_message()`'tan gelir, bunlar yayın anında eklenir.
ZARF_ALANLARI = ("ts", "schema", "source", "img_ref")
# `dial_angle` bilinçli olarak YOK. Yatıklık düzeltilmiş açı bir ARA
# BÜYÜKLÜKTÜR; tüketen tarafın (tur raporu) işine yaramaz ve şemaya girerse
# bir daha çıkarılamaz. `raw_angle` izlenebilirlik için duruyor: bir okuma
# tartışmalı olduğunda görüntüde ne ölçüldüğü sorulur.
GOVDE_ALANLARI = ("gauge_id", "type", "value", "unit", "conf", "status",
                  "raw_angle")

VARSAYILAN_DOSYA_DIZINI = "outputs/mqtt"


class SemaHatasi(ValueError):
    """Mesaj sözleşmeye uymuyor — yayınlanmamalı."""


def mesaj_dogrula(mesaj: dict) -> None:
    """Sözleşmeye uymayan mesajda `SemaHatasi` yükseltir.

    Denetimler bilinçli olarak KATI: eksik alan, tanınmayan durum, aralık dışı
    güven ve **`status`/`value` tutarsızlığı**. Sonuncusu en önemlisi:
    `status: unreadable` ile birlikte bir değer yayınlamak, tüketen tarafın o
    değeri kullanmasına yol açar ve tam olarak 3. kuralın engellemeye çalıştığı
    şeydir.
    """
    for alan in ZARF_ALANLARI + GOVDE_ALANLARI:
        if alan not in mesaj:
            raise SemaHatasi(f"eksik alan: {alan}")

    if mesaj["type"] not in GECERLI_TIPLER:
        raise SemaHatasi(f"tanınmayan tip: {mesaj['type']}")
    if mesaj["status"] not in GECERLI_DURUMLAR:
        raise SemaHatasi(f"tanınmayan durum: {mesaj['status']}")

    conf = mesaj["conf"]
    if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
        raise SemaHatasi(f"conf 0-1 aralığında olmalı: {conf}")

    deger = mesaj["value"]
    if deger is not None and not isinstance(deger, (int, float, str)):
        raise SemaHatasi(f"value sayı, dizge ya da null olmalı: {type(deger)}")

    # Tutarlılık: değer yoksa durum onu açıklamalı, değer varsa durum
    # "okunamadı" olmamalı.
    if deger is None and mesaj["status"] == "ok":
        raise SemaHatasi("status 'ok' ama value null")
    if deger is not None and mesaj["status"] == "unreadable":
        raise SemaHatasi("status 'unreadable' ama value dolu — 3. kural ihlali")

    if mesaj["schema"] != SEMA_SURUMU:
        raise SemaHatasi(f"şema sürümü {mesaj['schema']}, beklenen {SEMA_SURUMU}")


def mesaj_kur(okuma: GaugeReading, *, img_ref: str | None = None,
              source: str | None = None, ts: str | None = None) -> dict:
    """`GaugeReading`'i tam `inspect/reading` mesajına çevirir.

    Zaman damgası UTC ve ISO-8601: tur raporunu üreten taraf farklı saat
    diliminde çalışabilir ve yerel saat karşılaştırmayı sessizce bozar.
    """
    return {
        "ts": ts or datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "schema": SEMA_SURUMU,
        "source": source or socket.gethostname(),
        "img_ref": img_ref,
        **okuma.as_message(),
    }


@dataclass
class ReadingPublisher:
    """`inspect/reading` yayıncısı. Broker yoksa dosyaya yazar."""

    host: str = "localhost"
    port: int = 1883
    topic: str = KONU
    source: str | None = None
    dosya_dizini: str = VARSAYILAN_DOSYA_DIZINI
    zorla_dosya: bool = False

    mod: str = field(default="kapalı", init=False)     # "mqtt" | "dosya" | "kapalı"
    gonderilen: int = field(default=0, init=False)
    reddedilen: int = field(default=0, init=False)
    _istemci: object | None = field(default=None, init=False, repr=False)
    _dosya: Path | None = field(default=None, init=False, repr=False)

    def baglan(self, timeout: float = 3.0) -> str:
        """Brokera bağlanmayı dener; olmazsa dosya moduna düşer. Modu döner."""
        if not self.zorla_dosya:
            try:
                import paho.mqtt.client as mqtt

                istemci = mqtt.Client()
                istemci.connect(self.host, self.port, keepalive=int(timeout))
                istemci.loop_start()
                self._istemci = istemci
                self.mod = "mqtt"
                return self.mod
            except Exception:
                # Broker yok, kütüphane yok ya da ağ kapalı. Yayının doğruluğunu
                # ölçmek bunların hiçbirine bağlı olmamalı.
                self._istemci = None

        dizin = Path(self.dosya_dizini)
        dizin.mkdir(parents=True, exist_ok=True)
        damga = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._dosya = dizin / f"inspect_reading_{damga}.jsonl"
        self.mod = "dosya"
        return self.mod

    def yayinla(self, okuma: GaugeReading, *, img_ref: str | None = None) -> dict:
        """Okumayı doğrular ve yayınlar. Doğrulama başarısızsa `SemaHatasi`.

        Hata YUTULMAZ: bozuk bir mesajı sessizce atmak, yayının çalıştığı
        yanılsamasını yaratır. Çağıran ne olduğunu bilmeli.
        """
        mesaj = mesaj_kur(okuma, img_ref=img_ref, source=self.source)
        try:
            mesaj_dogrula(mesaj)
        except SemaHatasi:
            self.reddedilen += 1
            raise

        yuk = json.dumps(mesaj, ensure_ascii=False)
        if self.mod == "mqtt" and self._istemci is not None:
            self._istemci.publish(self.topic, yuk, qos=1)
        elif self.mod == "dosya" and self._dosya is not None:
            with self._dosya.open("a", encoding="utf-8") as f:
                f.write(yuk + "\n")
        else:
            raise RuntimeError("yayıncı bağlı değil — önce baglan() çağırın")

        self.gonderilen += 1
        return mesaj

    def kapat(self) -> None:
        if self._istemci is not None:
            try:
                self._istemci.loop_stop()
                self._istemci.disconnect()
            except Exception:
                pass
        self._istemci = None
        self.mod = "kapalı"

    def __enter__(self):
        self.baglan()
        return self

    def __exit__(self, *_):
        self.kapat()
