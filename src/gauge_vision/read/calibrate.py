"""İbre açısını gösterge değerine çevirir — okuma zincirinin son adımı (İP7).

    from gauge_vision.read.calibrate import read_value

    okuma = read_value(gauge, angle_img_deg=132.4, roll_deg=-4.2, confidence=0.93)
    okuma.value     # 4.7
    okuma.status    # "ok" | "unreadable" | "out_of_range" | "alarm"

Dönen `GaugeReading`, `inspect/reading` MQTT mesajının (İP10) birebir gövdesidir;
alan adları sözleşmeden alınmıştır, burada ikinci bir isimlendirme türetilmez.

**Ölçek kuralı burada değil `Scale`'de.** Bu dosya yalnızca okuma katmanının
kararlarını verir: yatıklık düzeltmesi, güven eşiği, kadran dışı davranışı ve
alarm. Kadranın geometrisi (süpürme yönü, karekök ölçek) tek yerde, envanteri
yükleyen sınıfta durur.

**Üç karar ve gerekçeleri:**

1. *Yatıklık düzeltmesi.* İP6 görüntüdeki açıyı ölçer; kadranın kendi
   çerçevesindeki açı `angle_img_deg - roll_deg`'dir. Sentetik veride `roll_deg`
   etiketten gelir. Gerçek görüntüde kameradan gelmez — kadranın kendi
   geometrisinden çıkarılması gerekir (İP8, K2 perspektif düzeltmesiyle birlikte).
   Bu bağımlılık gizlenmesin diye parametre zorunlu tutulmadı ama varsayılanı 0.
2. *Güven eşiği.* `gauge.conf_threshold`'un altındaki okumada değer **None**'dır.
   Düşük güvenli bir sayı yayınlamak, onu tüketen tarafın (Özgür'ün tur raporu)
   sayıyı kullanmasına yol açar; "okuyamadım" bilgisi sayının kendisinden değerlidir.
3. *Kadran dışı.* İbre dayanağa yaslanmış olabilir. Süpürmenin `SINIR_TOLERANSI`
   kadar dışındaki açı uçlara kırpılıp `ok` sayılır (ölçüm gürültüsü); daha
   ötesi `out_of_range` olur ve değer yayınlanmaz.
"""

from __future__ import annotations

from dataclasses import dataclass

from gauge_vision.config import Gauge

# Süpürmenin bu kadar dışı ölçüm gürültüsü sayılıp uca kırpılır. 0,02 × 270° ≈ 5°;
# İP6'nın p95 hatası 0,27° olduğuna göre bu pay fazlasıyla geniştir — kırpma
# yöntemin hatasını değil, ibrenin dayanağa yaslanmasını karşılamak içindir.
SINIR_TOLERANSI = 0.02

DURUM_OK = "ok"
DURUM_OKUNAMADI = "unreadable"
DURUM_KADRAN_DISI = "out_of_range"
DURUM_ALARM = "alarm"


@dataclass(frozen=True)
class GaugeReading:
    """Tek bir göstergenin okunmuş hâli — `inspect/reading` mesajının gövdesi."""

    gauge_id: str
    type: str
    value: float | None
    unit: str | None
    conf: float
    status: str
    raw_angle: float          # görüntüde ölçülen açı; izlenebilirlik için ham
    dial_angle: float | None  # yatıklık düzeltilmiş açı — kadran çerçevesinde

    def as_message(self) -> dict:
        """MQTT yükü (İP10). `img_ref` yayın anında eklenir, okuma katmanı bilmez."""
        return {
            "gauge_id": self.gauge_id,
            "type": self.type,
            "value": self.value,
            "unit": self.unit,
            "conf": round(self.conf, 3),
            "status": self.status,
            "raw_angle": round(self.raw_angle, 2),
        }


def read_value(
    gauge: Gauge,
    angle_img_deg: float,
    *,
    roll_deg: float = 0.0,
    confidence: float = 1.0,
) -> GaugeReading:
    """Ölçülen görüntü açısını değere çevirir ve durumunu belirler.

    Hiçbir koşulda hata yükseltmez: okuma zinciri saha döngüsünde çalışacaktır,
    tek bir okunamayan kare turu düşürmemelidir. Sorunlar `status` ile bildirilir.
    """
    if gauge.type != "analog":
        raise ValueError(f"{gauge.id}: read_value sadece analog göstergede çalışır "
                         f"(tip: {gauge.type})")

    bos = _bos_okuma(gauge, angle_img_deg, confidence)

    if confidence < gauge.conf_threshold:
        return bos(DURUM_OKUNAMADI)

    kadran_acisi = angle_img_deg - roll_deg
    oran = gauge.scale.fraction_for_angle(kadran_acisi)

    if not -SINIR_TOLERANSI <= oran <= 1.0 + SINIR_TOLERANSI:
        return bos(DURUM_KADRAN_DISI, dial_angle=kadran_acisi)

    deger = gauge.scale.value_for_fraction(min(1.0, max(0.0, oran)))
    deger = round(deger, gauge.decimals)

    return GaugeReading(
        gauge_id=gauge.id,
        type=gauge.type,
        value=deger,
        unit=gauge.unit,
        conf=confidence,
        status=_alarm_durumu(gauge, deger),
        raw_angle=angle_img_deg,
        dial_angle=kadran_acisi,
    )


def _alarm_durumu(gauge: Gauge, deger: float) -> str:
    """Envanterdeki alarm eşiklerini uygular.

    Eşikler koda gömülmez, `gauges.yaml`'dan gelir (2. kural): yeni bir eşik
    envantere satır eklemekle tanımlanır, burası değişmez.
    """
    alarm = gauge.alarm or {}
    for anahtar, asiliyor in (
        ("crit_above", lambda x: deger > x),
        ("crit_below", lambda x: deger < x),
        ("warn_above", lambda x: deger > x),
        ("warn_below", lambda x: deger < x),
    ):
        esik = alarm.get(anahtar)
        if esik is not None and asiliyor(float(esik)):
            return DURUM_ALARM
    return DURUM_OK


def _bos_okuma(gauge: Gauge, angle_img_deg: float, confidence: float):
    """Değersiz okuma üreten yardımcı — `value: null` tek yerden çıksın."""
    def yap(status: str, dial_angle: float | None = None) -> GaugeReading:
        return GaugeReading(
            gauge_id=gauge.id,
            type=gauge.type,
            value=None,
            unit=gauge.unit,
            conf=confidence,
            status=status,
            raw_angle=angle_img_deg,
            dial_angle=dial_angle,
        )
    return yap
