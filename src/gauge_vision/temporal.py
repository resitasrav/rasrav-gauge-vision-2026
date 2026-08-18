"""Canli video okumalarini kareler arasinda kararli hale getirir.

Tek kare okuyuculari optik olarak dogru olsalar bile tespit kutusu, yansima ve
kisa sureli ortulme nedeniyle bir sonraki karede farkli sonuc uretebilir. Bu
modul, ``pipeline.read_gauge`` sonucunu degistirmeden sonrasina eklenir:

* ilk degeri yayina almadan once art arda onay ister;
* sayisal ani sicrama ancak tekrar gorulurse kabul edilir;
* lamba/vana durum degisimini oylar;
* kisa tespit kaybinda son dogrulanmis sonucu sinirli sure tutar.

Bu bir takipci veya Kalman filtresi degildir. Tek bir, kimligi disaridan beyan
edilmis gosterge icin karar katmanidir; kimlik dogrulamasi yapmaz.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real

from gauge_vision.config import Gauge
from gauge_vision.pipeline import FrameResult
from gauge_vision.read.calibrate import DURUM_ALARM, DURUM_OK, DURUM_OKUNAMADI, GaugeReading


@dataclass(frozen=True)
class TemporalConfig:
    """Kare tabanli sabitleyici ayarlari.

    Varsayilanlar 10 FPS civarindaki canli akis icin secildi: uc karelik onay
    yaklasik 0.3 saniyedir. Daha yavas ya da daha hizli kaynaklarda CLI'dan
    ``min_confirmed_frames`` degistirilebilir.
    """

    min_confirmed_frames: int = 3
    lost_grace_frames: int = 2
    numeric_ema_alpha: float = 0.35
    confidence_ema_alpha: float = 0.40
    box_ema_alpha: float = 0.45
    max_numeric_step_fraction: float = 0.20
    sustained_change_tolerance_fraction: float = 0.05
    held_confidence_decay: float = 0.85

    def __post_init__(self) -> None:
        if self.min_confirmed_frames < 1:
            raise ValueError("min_confirmed_frames en az 1 olmali")
        if self.lost_grace_frames < 0:
            raise ValueError("lost_grace_frames negatif olamaz")
        for name in ("numeric_ema_alpha", "confidence_ema_alpha", "box_ema_alpha",
                     "max_numeric_step_fraction", "sustained_change_tolerance_fraction",
                     "held_confidence_decay"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} 0 ile 1 arasinda olmali")


class TemporalStabilizer:
    """Bir gostergeye ait ardarda ``FrameResult`` sonuclarini sabitler.

    Her kamera akisi ve her gauge icin ayri ornek kullanilmalidir. Farkli bir
    ``gauge_id`` gelirse durum otomatik sifirlanir; onceki gostergenin degeri
    yeni gostergeye tasinmaz.
    """

    def __init__(self, gauge: Gauge | None = None, config: TemporalConfig | None = None):
        self.gauge = gauge
        self.config = config or TemporalConfig()
        self.reset()

    def reset(self) -> None:
        """Gecmis kare bilgisini temizler."""
        self._gauge_id: str | None = None
        self._mode: str | None = None
        self._valid_count = 0
        self._missed_count = 0
        self._stable_reading: GaugeReading | None = None
        self._last_valid_result: FrameResult | None = None
        self._smoothed_value: float | None = None
        self._smoothed_confidence: float | None = None
        self._smoothed_detect_conf: float | None = None
        self._smoothed_box: tuple[float, float, float, float] | None = None
        self._stable_status: str | None = None
        self._status_candidate: str | None = None
        self._status_count = 0
        self._outlier_value: float | None = None
        self._outlier_count = 0
        self._state_candidate: str | None = None
        self._state_count = 0

    def update(self, result: FrameResult) -> FrameResult:
        """Yeni kare sonucunu isler ve yalnizca kararli okumayi dondurur."""
        reading = result.reading
        if not self._is_usable(reading):
            return self._missing(result)

        assert reading is not None
        mode = "numeric" if self._is_numeric(reading.value) else "state"
        if self._gauge_id not in (None, reading.gauge_id) or self._mode not in (None, mode):
            self.reset()
        self._gauge_id = reading.gauge_id
        self._mode = mode
        self._missed_count = 0
        current = self._smooth_geometry(result)

        if mode == "numeric":
            return self._numeric(current, reading)
        return self._state(current, reading)

    @staticmethod
    def _is_numeric(value: object) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    @staticmethod
    def _is_usable(reading: GaugeReading | None) -> bool:
        return (
            reading is not None
            and reading.value is not None
            and reading.status in (DURUM_OK, DURUM_ALARM)
        )

    def _numeric(self, current: FrameResult, reading: GaugeReading) -> FrameResult:
        value = float(reading.value)
        if self._is_outlier(value):
            return self._outlier(current, reading, value)

        self._clear_outlier()
        self._valid_count += 1
        self._smoothed_value = self._ema(self._smoothed_value, value, self.config.numeric_ema_alpha)
        stabilized = self._stabilized_reading(reading, value=self._rounded_value(self._smoothed_value))

        if self._valid_count < self.config.min_confirmed_frames:
            return self._unconfirmed(current, reading)
        return self._confirm(current, stabilized)

    def _outlier(self, current: FrameResult, reading: GaugeReading, value: float) -> FrameResult:
        if self._outlier_value is None or abs(value - self._outlier_value) > self._sustained_tolerance():
            self._outlier_value = value
            self._outlier_count = 1
        else:
            self._outlier_count += 1

        if self._outlier_count < self.config.min_confirmed_frames:
            return self._hold_or_unconfirmed(
                current, reading,
                f"temporal: sayisal sicrama onayi "
                f"({self._outlier_count}/{self.config.min_confirmed_frames})",
            )

        # Degisim art arda goruldu; gercek bir proses degisimi olma olasiligi
        # yuksek. Eski EMA'ya yavasca yaklastirmak yerine yeni seviyeden basla.
        self._smoothed_value = value
        self._valid_count = self.config.min_confirmed_frames
        self._clear_outlier()
        return self._confirm(
            current,
            self._stabilized_reading(reading, value=self._rounded_value(value)),
        )

    def _state(self, current: FrameResult, reading: GaugeReading) -> FrameResult:
        value = str(reading.value)
        if self._stable_reading is None:
            if value == self._state_candidate:
                self._state_count += 1
            else:
                self._state_candidate, self._state_count = value, 1
            if self._state_count < self.config.min_confirmed_frames:
                return self._unconfirmed(current, reading)
            return self._confirm(
                current, self._stabilized_reading(reading, value=value), status_confirmed=True
            )

        stable_value = str(self._stable_reading.value)
        if value == stable_value:
            self._state_candidate, self._state_count = None, 0
            return self._confirm(
                current, self._stabilized_reading(reading, value=value), status_confirmed=True
            )

        if value == self._state_candidate:
            self._state_count += 1
        else:
            self._state_candidate, self._state_count = value, 1
        if self._state_count < self.config.min_confirmed_frames:
            return self._hold_or_unconfirmed(
                current, reading,
                f"temporal: durum degisimi onayi "
                f"({self._state_count}/{self.config.min_confirmed_frames})",
            )
        self._state_candidate, self._state_count = None, 0
        return self._confirm(
            current, self._stabilized_reading(reading, value=value), status_confirmed=True
        )

    def _confirm(
        self, current: FrameResult, reading: GaugeReading, *, status_confirmed: bool = False
    ) -> FrameResult:
        if status_confirmed:
            self._stable_status = reading.status
            self._status_candidate, self._status_count = None, 0
        else:
            reading = replace(reading, status=self._stable_status_for(reading.status))
        self._stable_reading = reading
        confirmed = replace(current, reading=reading, reason="")
        self._last_valid_result = confirmed
        return confirmed

    def _stabilized_reading(self, reading: GaugeReading, *, value: float | str) -> GaugeReading:
        self._smoothed_confidence = self._ema(
            self._smoothed_confidence, reading.conf, self.config.confidence_ema_alpha
        )
        return replace(reading, value=value, conf=self._smoothed_confidence)

    def _stable_status_for(self, status: str) -> str:
        if self._stable_status is None:
            self._stable_status = status
            return status
        if status == self._stable_status:
            self._status_candidate, self._status_count = None, 0
            return status
        if status == self._status_candidate:
            self._status_count += 1
        else:
            self._status_candidate, self._status_count = status, 1
        if self._status_count >= self.config.min_confirmed_frames:
            self._stable_status = status
            self._status_candidate, self._status_count = None, 0
        return self._stable_status

    def _unconfirmed(self, current: FrameResult, reading: GaugeReading) -> FrameResult:
        return replace(
            current,
            reading=replace(reading, value=None, status=DURUM_OKUNAMADI),
            reason=(f"temporal: onay {self._valid_count}/"
                    f"{self.config.min_confirmed_frames}"),
        )

    def _hold_or_unconfirmed(
        self, current: FrameResult, reading: GaugeReading, reason: str
    ) -> FrameResult:
        if self._stable_reading is None:
            return self._unconfirmed(current, reading)
        return replace(current, reading=self._stable_reading, reason=reason)

    def _missing(self, result: FrameResult) -> FrameResult:
        self._missed_count += 1
        if self._stable_reading is None:
            self._valid_count = 0
            return result
        if self._missed_count <= self.config.lost_grace_frames and self._last_valid_result is not None:
            decay = self.config.held_confidence_decay ** self._missed_count
            held = replace(self._stable_reading, conf=self._stable_reading.conf * decay)
            return replace(
                self._last_valid_result,
                detect_conf=self._last_valid_result.detect_conf * decay,
                reading=held,
                reason=(f"temporal: son sonuc tutuluyor "
                        f"({self._missed_count}/{self.config.lost_grace_frames})"),
            )

        self.reset()
        return result

    def _smooth_geometry(self, result: FrameResult) -> FrameResult:
        box = result.box_xyxy
        if box is not None:
            if self._smoothed_box is None:
                self._smoothed_box = box
            else:
                alpha = self.config.box_ema_alpha
                self._smoothed_box = tuple(
                    alpha * now + (1.0 - alpha) * old
                    for now, old in zip(box, self._smoothed_box)
                )
            result = replace(result, box_xyxy=self._smoothed_box)

        self._smoothed_detect_conf = self._ema(
            self._smoothed_detect_conf, result.detect_conf, self.config.confidence_ema_alpha
        )
        return replace(result, detect_conf=self._smoothed_detect_conf)

    def _is_outlier(self, value: float) -> bool:
        limit = self._max_numeric_step()
        return (
            limit is not None
            and self._smoothed_value is not None
            and abs(value - self._smoothed_value) > limit
        )

    def _max_numeric_step(self) -> float | None:
        span = self._numeric_span()
        return None if span is None else span * self.config.max_numeric_step_fraction

    def _sustained_tolerance(self) -> float:
        span = self._numeric_span()
        return 0.0 if span is None else span * self.config.sustained_change_tolerance_fraction

    def _numeric_span(self) -> float | None:
        if self.gauge is None or self.gauge.id != self._gauge_id:
            return None
        if self.gauge.type == "analog" and self.gauge.scale is not None:
            return self.gauge.scale.max - self.gauge.scale.min
        limits = self.gauge.raw.get("range") or {}
        minimum, maximum = limits.get("min"), limits.get("max")
        if isinstance(minimum, Real) and isinstance(maximum, Real):
            return float(maximum) - float(minimum)
        return None

    def _rounded_value(self, value: float) -> float:
        if self.gauge is None:
            return value
        if self.gauge.type == "digital":
            decimals = (self.gauge.digits or {}).get("decimals")
        else:
            decimals = self.gauge.decimals
        return round(value, int(decimals)) if decimals is not None else value

    @staticmethod
    def _ema(previous: float | None, current: float, alpha: float) -> float:
        return current if previous is None else alpha * current + (1.0 - alpha) * previous

    def _clear_outlier(self) -> None:
        self._outlier_value, self._outlier_count = None, 0
