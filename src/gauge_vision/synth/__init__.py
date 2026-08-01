"""Sentetik gösterge üretimi (İP3).

Ground truth burada bedava: ibrenin açısını biz koyduğumuz için okuma
yönteminin hatası doğrudan ölçülebilir. İP6/İP7 önce bu veride oturur.
"""

from gauge_vision.synth.dial import DialTruth, render_analog

__all__ = ["DialTruth", "render_analog"]
