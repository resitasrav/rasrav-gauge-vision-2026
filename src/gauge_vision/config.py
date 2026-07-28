"""Gösterge envanterini (`configs/gauges.yaml`) yükler ve doğrular.

Zincirdeki herkes göstergeye buradan ulaşır — YAML'ı ikinci bir yerde
elle açan kod yazılmaz:

    from gauge_vision.config import load_gauges

    gauges = load_gauges()
    pt101 = gauges["PT-101"]
    print(pt101.scale.sweep_deg)   # 270.0

Doğrulama bilerek katıdır: bozuk envanter erken ve anlaşılır bir hatayla
patlar. Sessizce yanlış değer okumaktansa hiç okumamak yeğdir (aynı ilke
İP15'teki `unreadable` davranışının temeli).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Depo kökü: src/gauge_vision/config.py → üç seviye yukarı
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "gauges.yaml"

GAUGE_TYPES = ("analog", "digital", "lamp", "valve")


class ConfigError(ValueError):
    """Envanter dosyası bozuk veya eksik."""


@dataclass(frozen=True)
class Scale:
    """Analog göstergenin kadran tanımı.

    Açı konvansiyonu (gauges.yaml başlığındaki şemanın aynısı):
    derece · 0° = saat 3 yönü · pozitif yön saat yönünün TERSİ (CCW).
    """

    min: float
    max: float
    angle_min: float          # min değerdeyken ibrenin açısı
    angle_max: float          # max değerdeyken ibrenin açısı
    direction: str            # "cw" | "ccw" — min'den max'a dönüş yönü
    linear: bool = True       # False → düzgün ölçekli değil (örn. karekök debimetre)
    sweep_declared: float | None = None  # YAML'daki sweep_deg — sağlama toplamı

    @property
    def sweep_deg(self) -> float:
        """Kadranın süpürme açısı (derece). Tipik saat için 270.0."""
        if self.direction == "cw":
            return (self.angle_min - self.angle_max) % 360
        return (self.angle_max - self.angle_min) % 360


@dataclass(frozen=True)
class Gauge:
    """Envanterdeki tek bir gösterge."""

    id: str
    name: str
    type: str
    unit: str | None = None
    location: str | None = None
    waypoint: str | None = None          # Özgür'ün altın tur durağı
    conf_threshold: float = 0.70         # altında → status: unreadable
    decimals: int = 1
    scale: Scale | None = None           # analog
    digits: dict[str, Any] | None = None # digital
    states: list[dict[str, Any]] = field(default_factory=list)  # lamp / valve
    alarm: dict[str, float] = field(default_factory=dict)
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)  # ham sözlük

    @property
    def state_names(self) -> list[str]:
        return [s["name"] for s in self.states]


def load_gauges(path: str | Path | None = None) -> dict[str, Gauge]:
    """Envanteri yükler, `defaults` bloğunu uygular, doğrular.

    Dönen sözlük `gauge_id -> Gauge` şeklindedir.
    Hata durumunda `ConfigError` yükseltir.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError(f"Envanter dosyası yok: {path}")

    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict) or "gauges" not in doc:
        raise ConfigError(f"{path}: kökte 'gauges' listesi bulunamadı")

    defaults: dict[str, Any] = doc.get("defaults") or {}
    gauges: dict[str, Gauge] = {}

    for i, entry in enumerate(doc["gauges"]):
        gauge = _build_gauge(entry, defaults, where=f"{path} · gauges[{i}]")
        if gauge.id in gauges:
            raise ConfigError(f"{path}: '{gauge.id}' kimliği iki kez tanımlı")
        gauges[gauge.id] = gauge

    if not gauges:
        raise ConfigError(f"{path}: envanter boş")
    return gauges


def _build_gauge(entry: dict[str, Any], defaults: dict[str, Any], where: str) -> Gauge:
    for key in ("id", "name", "type"):
        if not entry.get(key):
            raise ConfigError(f"{where}: '{key}' alanı zorunlu")

    gid, gtype = entry["id"], entry["type"]
    if gtype not in GAUGE_TYPES:
        raise ConfigError(f"{where} ({gid}): bilinmeyen tip '{gtype}' — {GAUGE_TYPES}")

    # defaults < gösterge kendi değeri (gösterge yazmışsa onunki kazanır)
    conf = float(entry.get("reading", {}).get("conf_threshold",
                 defaults.get("conf_threshold", 0.70)))
    if not 0.0 < conf <= 1.0:
        raise ConfigError(f"{where} ({gid}): conf_threshold 0-1 aralığında olmalı, {conf} verildi")

    scale = _build_scale(entry["scale"], gid, where) if gtype == "analog" else None

    if gtype == "analog" and not entry.get("unit"):
        raise ConfigError(f"{where} ({gid}): analog göstergede 'unit' zorunlu")
    if gtype == "analog" and scale is None:
        raise ConfigError(f"{where} ({gid}): analog göstergede 'scale' zorunlu")
    if gtype == "digital" and not entry.get("digits"):
        raise ConfigError(f"{where} ({gid}): dijital göstergede 'digits' zorunlu")
    if gtype in ("lamp", "valve"):
        states = entry.get("states") or []
        if len(states) < 2:
            raise ConfigError(f"{where} ({gid}): '{gtype}' en az 2 durum ister")
        for s in states:
            if not s.get("name"):
                raise ConfigError(f"{where} ({gid}): durumlardan birinde 'name' yok")

    return Gauge(
        id=gid,
        name=entry["name"],
        type=gtype,
        unit=entry.get("unit"),
        location=entry.get("location"),
        waypoint=entry.get("waypoint"),
        conf_threshold=conf,
        decimals=int(entry.get("decimals", defaults.get("decimals", 1))),
        scale=scale,
        digits=entry.get("digits"),
        states=entry.get("states") or [],
        alarm=entry.get("alarm") or {},
        notes=entry.get("notes"),
        raw=entry,
    )


def _build_scale(raw: dict[str, Any], gid: str, where: str) -> Scale:
    missing = [k for k in ("min", "max", "angle_min", "angle_max") if k not in raw]
    if missing:
        raise ConfigError(f"{where} ({gid}): scale içinde eksik alan(lar): {missing}")

    direction = raw.get("direction", "cw")
    if direction not in ("cw", "ccw"):
        raise ConfigError(f"{where} ({gid}): direction 'cw' veya 'ccw' olmalı, '{direction}' verildi")

    scale = Scale(
        min=float(raw["min"]),
        max=float(raw["max"]),
        angle_min=float(raw["angle_min"]),
        angle_max=float(raw["angle_max"]),
        direction=direction,
        linear=bool(raw.get("linear", True)),
        sweep_declared=float(raw["sweep_deg"]) if "sweep_deg" in raw else None,
    )

    if scale.min >= scale.max:
        raise ConfigError(f"{where} ({gid}): scale.min < scale.max olmalı")
    # Süpürme 0 ise angle_min == angle_max demektir (kalibrasyon imkânsız).
    if not 0 < scale.sweep_deg <= 350:
        raise ConfigError(
            f"{where} ({gid}): süpürme açısı {scale.sweep_deg:.1f}° — "
            f"angle_min/angle_max/direction üçlüsünü kontrol et "
            f"(tipik saat: 225 → -45, cw = 270°)"
        )

    # Sağlama toplamı: yanlış 'direction' geometrik olarak yakalanamaz — ccw yazılan
    # 270°'lik bir saat sessizce 90° olur, kod çalışır, değerler yanlış çıkar.
    # Envanteri yazan insan kadranın süpürmesini bilir; beyan ederse burada tutulur.
    if scale.sweep_declared is not None and abs(scale.sweep_deg - scale.sweep_declared) > 0.5:
        raise ConfigError(
            f"{where} ({gid}): beyan edilen süpürme {scale.sweep_declared:.0f}°, "
            f"açılardan hesaplanan {scale.sweep_deg:.0f}° — "
            f"angle_min / angle_max / direction üçlüsünden biri yanlış"
        )
    return scale
