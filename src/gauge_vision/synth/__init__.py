"""Sentetik gösterge üretimi (İP3).

Ground truth burada bedava: ibrenin açısını biz koyduğumuz için okuma
yönteminin hatası doğrudan ölçülebilir. İP6/İP7 önce bu veride oturur.
"""

from gauge_vision.synth.dial import DialLook, DialTruth, render_analog
from gauge_vision.synth.generate import (
    DatasetSummary,
    VariationRange,
    generate_dataset,
    load_labels,
)

__all__ = [
    "DialLook",
    "DialTruth",
    "render_analog",
    "DatasetSummary",
    "VariationRange",
    "generate_dataset",
    "load_labels",
]
