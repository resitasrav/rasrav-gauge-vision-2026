"""Waypoint-gösterge eşleme sözlüğünü yükler ve doğrular.

Bu dosya `configs/gauges.yaml` envanterinin yerine geçmez. Gösterge özellikleri
orada kalır; buradaki sözlük yalnızca Özgür'ün waypoint kimlikleri ile Reşit'in
`gauge_id` değerleri arasında köprü kurar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml

from gauge_vision.config import ConfigError, GAUGE_TYPES, Gauge, REPO_ROOT

DEFAULT_WAYPOINT_CONFIG = REPO_ROOT / "configs" / "waypoint_gosterge_sozlugu.yaml"
WAYPOINT_ID_RE = re.compile(r"^WP\d{2}$")


@dataclass(frozen=True)
class WaypointGaugeRef:
    """Bir waypoint'te görülebilen gösterge referansı."""

    gauge_id: str
    type: str | None = None
    note: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Waypoint:
    """Altın tur durağı ve o durakta beklenen gösterge referansları."""

    waypoint_id: str
    location: str
    description: str | None = None
    reference_frame: str | None = None
    gauges: list[WaypointGaugeRef] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def load_waypoints(
    path: str | Path | None = None,
    known_gauges: dict[str, Gauge] | None = None,
) -> dict[str, Waypoint]:
    """Waypoint sözlüğünü yükler.

    `known_gauges` verilirse sözlükte yazan her `gauge_id` envantere karşı
    doğrulanır. Böylece `PT-101` yerine yanlışlıkla `PT101` yazmak sessiz kalmaz.
    """
    path = Path(path) if path else DEFAULT_WAYPOINT_CONFIG
    if not path.exists():
        raise ConfigError(f"Waypoint sözlüğü yok: {path}")

    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict) or "waypoints" not in doc:
        raise ConfigError(f"{path}: kökte 'waypoints' listesi bulunamadı")

    waypointler: dict[str, Waypoint] = {}
    for i, entry in enumerate(doc["waypoints"]):
        waypoint = _build_waypoint(entry, where=f"{path} · waypoints[{i}]")
        if waypoint.waypoint_id in waypointler:
            raise ConfigError(f"{path}: '{waypoint.waypoint_id}' waypoint'i iki kez tanımlı")
        waypointler[waypoint.waypoint_id] = waypoint

    if not waypointler:
        raise ConfigError(f"{path}: waypoint sözlüğü boş")

    if known_gauges is not None:
        _validate_waypoint_gauge_refs(waypointler, known_gauges, path)
    return waypointler


def validate_gauge_waypoints(
    gauges: dict[str, Gauge],
    waypoints: dict[str, Waypoint],
) -> None:
    """`gauges.yaml` içindeki waypoint alanları sözlükle tutarlı mı?"""
    for gauge in gauges.values():
        if not gauge.waypoint:
            continue
        if not WAYPOINT_ID_RE.fullmatch(gauge.waypoint):
            raise ConfigError(
                f"{gauge.id}: waypoint '{gauge.waypoint}' WP01 biçiminde olmalı"
            )
        if gauge.waypoint not in waypoints:
            raise ConfigError(
                f"{gauge.id}: waypoint '{gauge.waypoint}' "
                f"{DEFAULT_WAYPOINT_CONFIG.name} içinde yok"
            )


def _build_waypoint(entry: dict[str, Any], where: str) -> Waypoint:
    if not isinstance(entry, dict):
        raise ConfigError(f"{where}: waypoint girdisi sözlük olmalı")

    wid = entry.get("waypoint_id")
    if not wid:
        raise ConfigError(f"{where}: 'waypoint_id' zorunlu")
    if not WAYPOINT_ID_RE.fullmatch(str(wid)):
        raise ConfigError(f"{where}: waypoint_id '{wid}' WP01 biçiminde olmalı")

    location = entry.get("konum")
    if not location:
        raise ConfigError(f"{where} ({wid}): 'konum' zorunlu")

    raw_refs = entry.get("gauges") or []
    if not isinstance(raw_refs, list):
        raise ConfigError(f"{where} ({wid}): 'gauges' liste olmalı")

    refs = [_build_gauge_ref(ref, where=f"{where} ({wid}) · gauges[{i}]")
            for i, ref in enumerate(raw_refs)]
    return Waypoint(
        waypoint_id=str(wid),
        location=str(location),
        description=entry.get("aciklama"),
        reference_frame=entry.get("referans_kare"),
        gauges=refs,
        raw=entry,
    )


def _build_gauge_ref(entry: dict[str, Any], where: str) -> WaypointGaugeRef:
    if not isinstance(entry, dict):
        raise ConfigError(f"{where}: gösterge referansı sözlük olmalı")

    gid = entry.get("gauge_id")
    if not gid:
        raise ConfigError(f"{where}: 'gauge_id' zorunlu")

    gtype = entry.get("tip")
    if gtype is not None and gtype not in GAUGE_TYPES:
        raise ConfigError(f"{where} ({gid}): bilinmeyen tip '{gtype}' — {GAUGE_TYPES}")

    return WaypointGaugeRef(
        gauge_id=str(gid),
        type=gtype,
        note=entry.get("not"),
        raw=entry,
    )


def _validate_waypoint_gauge_refs(
    waypoints: dict[str, Waypoint],
    known_gauges: dict[str, Gauge],
    path: Path,
) -> None:
    for waypoint in waypoints.values():
        for ref in waypoint.gauges:
            gauge = known_gauges.get(ref.gauge_id)
            if gauge is None:
                raise ConfigError(
                    f"{path} ({waypoint.waypoint_id}): '{ref.gauge_id}' "
                    f"gauges.yaml envanterinde yok"
                )
            if ref.type is not None and ref.type != gauge.type:
                raise ConfigError(
                    f"{path} ({waypoint.waypoint_id}): '{ref.gauge_id}' tipi "
                    f"'{ref.type}' yazılmış, envanterde '{gauge.type}'"
                )
