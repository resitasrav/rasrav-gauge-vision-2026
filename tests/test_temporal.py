"""Canli akis sonuc sabitleyicisi icin birim testleri."""

from dataclasses import replace

import pytest

from gauge_vision.config import load_gauges
from gauge_vision.pipeline import FrameResult
from gauge_vision.read.calibrate import DURUM_ALARM, DURUM_OK, GaugeReading
from gauge_vision.temporal import TemporalConfig, TemporalStabilizer


GAUGES = load_gauges()
HIZLI = TemporalConfig(
    min_confirmed_frames=3,
    lost_grace_frames=2,
    numeric_ema_alpha=0.5,
    confidence_ema_alpha=0.5,
    box_ema_alpha=0.5,
    max_numeric_step_fraction=0.20,
    sustained_change_tolerance_fraction=0.05,
)


def _reading(gauge_id: str, kind: str, value, *, status: str = DURUM_OK, conf: float = 0.90):
    return GaugeReading(
        gauge_id=gauge_id,
        type=kind,
        value=value,
        unit="bar" if kind == "analog" else None,
        conf=conf,
        status=status,
        raw_angle=90.0,
        dial_angle=90.0,
    )


def _frame(reading: GaugeReading | None, *, box=(10.0, 20.0, 110.0, 120.0)) -> FrameResult:
    return FrameResult(
        box_xyxy=box if reading is not None else None,
        detect_conf=reading.conf if reading is not None else 0.0,
        center_px=None,
        radius_px=0.0,
        needle=None,
        reading=reading,
        reason="tespit yok" if reading is None else "",
    )


def _confirm_numeric(stabilizer: TemporalStabilizer, value: float = 5.0) -> FrameResult:
    result = None
    for _ in range(3):
        result = stabilizer.update(_frame(_reading("PT-101", "analog", value)))
    assert result is not None
    return result


def test_numeric_reading_is_hidden_until_confirmed():
    stabilizer = TemporalStabilizer(GAUGES["PT-101"], HIZLI)

    first = stabilizer.update(_frame(_reading("PT-101", "analog", 5.0)))
    second = stabilizer.update(_frame(_reading("PT-101", "analog", 5.4)))
    third = stabilizer.update(_frame(_reading("PT-101", "analog", 5.2)))

    assert first.reading.value is None and first.reading.status == "unreadable"
    assert second.reading.value is None and second.reading.status == "unreadable"
    assert third.reading.value == pytest.approx(5.2)
    assert third.reading.status == DURUM_OK


def test_numeric_outlier_needs_repeated_confirmation():
    stabilizer = TemporalStabilizer(GAUGES["PT-101"], HIZLI)
    _confirm_numeric(stabilizer)

    first = stabilizer.update(_frame(_reading("PT-101", "analog", 9.5)))
    second = stabilizer.update(_frame(_reading("PT-101", "analog", 9.4)))
    third = stabilizer.update(_frame(_reading("PT-101", "analog", 9.5)))

    assert first.reading.value == pytest.approx(5.0)
    assert second.reading.value == pytest.approx(5.0)
    assert "sayisal sicrama" in first.reason
    assert third.reading.value == pytest.approx(9.5)


def test_initial_outlier_is_not_mixed_into_first_confirmed_value():
    stabilizer = TemporalStabilizer(GAUGES["PT-101"], HIZLI)

    first = stabilizer.update(_frame(_reading("PT-101", "analog", 5.0)))
    second = stabilizer.update(_frame(_reading("PT-101", "analog", 9.5)))
    third = stabilizer.update(_frame(_reading("PT-101", "analog", 9.5)))
    fourth = stabilizer.update(_frame(_reading("PT-101", "analog", 9.5)))

    assert first.reading.value is None
    assert second.reading.value is None
    assert third.reading.value is None
    assert fourth.reading.value == pytest.approx(9.5)


def test_short_detection_loss_holds_then_expires_confirmed_reading():
    stabilizer = TemporalStabilizer(GAUGES["PT-101"], HIZLI)
    _confirm_numeric(stabilizer)
    missing = _frame(None)

    first = stabilizer.update(missing)
    second = stabilizer.update(missing)
    third = stabilizer.update(missing)

    assert first.reading.value == pytest.approx(5.0)
    assert second.reading.value == pytest.approx(5.0)
    assert first.reading.conf < 0.90
    assert "son sonuc tutuluyor" in first.reason
    assert third.reading is None


def test_state_change_is_voted_before_replacing_confirmed_state():
    stabilizer = TemporalStabilizer(GAUGES["LM-501"], HIZLI)
    green = _reading("LM-501", "lamp", "green")
    red = _reading("LM-501", "lamp", "red", status=DURUM_ALARM)

    for _ in range(3):
        confirmed = stabilizer.update(_frame(green))
    first_red = stabilizer.update(_frame(red))
    second_red = stabilizer.update(_frame(red))
    third_red = stabilizer.update(_frame(red))

    assert confirmed.reading.value == "green"
    assert first_red.reading.value == "green"
    assert second_red.reading.value == "green"
    assert third_red.reading.value == "red"
    assert third_red.reading.status == DURUM_ALARM


def test_detection_box_is_smoothed_without_changing_layout_shape():
    config = replace(HIZLI, min_confirmed_frames=1)
    stabilizer = TemporalStabilizer(GAUGES["PT-101"], config)

    stabilizer.update(_frame(_reading("PT-101", "analog", 5.0), box=(0.0, 0.0, 100.0, 100.0)))
    result = stabilizer.update(_frame(_reading("PT-101", "analog", 5.0), box=(10.0, 20.0, 110.0, 120.0)))

    assert result.box_xyxy == pytest.approx((5.0, 10.0, 105.0, 110.0))


def test_gauge_change_resets_previous_value():
    stabilizer = TemporalStabilizer(config=HIZLI)
    _confirm_numeric(stabilizer)

    other = stabilizer.update(_frame(_reading("TI-205", "analog", 75.0)))

    assert other.reading.value is None
    assert other.reading.status == "unreadable"
