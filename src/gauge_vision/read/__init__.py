"""Okuma zinciri — kırpılmış gösterge görüntüsünden sayıya (İP6-İP7, İP11-İP12).

    needle.py     ibrenin GÖRÜNTÜDEKİ açısı (İP6)
    calibrate.py  açı → değer (İP7)
    digital.py    7-segment / dijital panel OCR (İP11)
    state.py      lamba / vana durumu (İP12)
"""

from gauge_vision.read.calibrate import GaugeReading, read_value
from gauge_vision.read.needle import (
    NeedleReading,
    angle_difference_deg,
    read_needle_angle,
)

__all__ = [
    "GaugeReading",
    "NeedleReading",
    "angle_difference_deg",
    "read_needle_angle",
    "read_value",
]
