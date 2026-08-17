import pytest
import yaml

from gauge_vision.config import ConfigError, load_gauges
from gauge_vision.waypoints import (
    WAYPOINT_ID_RE,
    load_waypoints,
    validate_gauge_waypoints,
)


def test_waypoint_sozlugu_yukleniyor():
    waypointler = load_waypoints(known_gauges=load_gauges())
    assert set(waypointler) == {"WP01", "WP02", "WP03"}
    assert waypointler["WP03"].location == "Koridor sonu — makine yanı"


def test_waypoint_idleri_tek_formatta():
    for waypoint_id in load_waypoints():
        assert WAYPOINT_ID_RE.fullmatch(waypoint_id)
        assert "-" not in waypoint_id


def test_gauge_envanterinin_waypointleri_sozlukle_tutarli():
    validate_gauge_waypoints(load_gauges(), load_waypoints())


def test_waypointte_yazili_gauge_id_envanterde_olmali(tmp_path):
    doc = {
        "versiyon": 1,
        "waypoints": [{
            "waypoint_id": "WP01",
            "konum": "test konum",
            "gauges": [{"gauge_id": "PT-999", "tip": "analog"}],
        }],
    }
    path = tmp_path / "waypoints.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="envanterinde yok"):
        load_waypoints(path, known_gauges=load_gauges())


def test_tireli_waypoint_id_reddediliyor(tmp_path):
    doc = {
        "versiyon": 1,
        "waypoints": [{"waypoint_id": "WP-01", "konum": "test", "gauges": []}],
    }
    path = tmp_path / "waypoints.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="WP01 biçiminde"):
        load_waypoints(path)
