"""GÖSTERGE — gösterge ve panel okuma modülü.

Boru hattı:  görüntü → tespit (İP5) → kırp → okuma (İP6/İP11/İP12)
             → kalibrasyon (İP7) → MQTT yayını (İP10)

Her aşama `configs/gauges.yaml` envanterine bakar; envanter tek doğru kaynaktır.
"""

__version__ = "0.1.0"
