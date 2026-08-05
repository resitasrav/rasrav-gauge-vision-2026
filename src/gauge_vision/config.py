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

# Karekök ölçekli kadran (fark basınçlı debimetre): akış Q ∝ √ΔP, ibre ise ΔP ile
# orantılı sapar → ibrenin süpürmedeki oranı değerin KARESİ kadardır. Ölçek alt uçta
# sıkışık, üst uçta seyrektir. `linear: false` olan göstergeler bu üsse tabidir.
SQRT_SCALE_EXPONENT = 2.0


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

    def fraction_for_value(self, value: float) -> float:
        """`value` kadranın neresinde — 0.0 (min ucu) ile 1.0 (max ucu) arası oran.

        Doğrusal kadranda oran değerle aynı; karekök ölçekli kadranda değerin
        karesiyle orantılıdır (bkz. SQRT_SCALE_EXPONENT).
        """
        if not self.min <= value <= self.max:
            raise ValueError(
                f"{value} kadran aralığı dışında ({self.min}–{self.max}) — "
                f"ibre kadranın dışına çizilemez"
            )
        frac = (value - self.min) / (self.max - self.min)
        return frac if self.linear else frac ** SQRT_SCALE_EXPONENT

    def angle_for_value(self, value: float) -> float:
        """`value` değerindeyken ibrenin açısı (derece, CCW pozitif).

        Kadran geometrisi tek yerde dursun diye buraya kondu: İP3'ün sentetik
        üreteci ibreyi buna göre çizer, İP7'nin açı→değer dönüşümü bunun tersidir.
        Formül iki ayrı dosyada yazılsaydı biri düzeltilip diğeri unutulurdu.

        `cw` kadranda min'den max'a giderken açı AZALIR — pozitif yön CCW olduğu için.
        """
        offset = self.fraction_for_value(value) * self.sweep_deg
        return self.angle_min - offset if self.direction == "cw" else self.angle_min + offset

    # --- ters yön: açı → değer (İP7) ------------------------------------------
    # Aşağıdaki üçlü yukarıdaki ikilinin tersidir ve bilerek aynı sınıfta duruyor.
    # Ayrı dosyaya konsaydı ölçek kuralı (karekök kadran, süpürme yönü) iki yerde
    # yaşardı; biri düzeltilip diğeri unutulduğunda üretim doğru, okuma yanlış olurdu.

    def fraction_for_angle(self, angle_deg: float) -> float:
        """Açının süpürmedeki oranı — `fraction_for_value`'nun tersi.

        Kadranın dışına düşen açı için 0-1 dışında bir sayı döner; kırpma
        yapılmaz. Kırpma kararı okuma katmanına aittir (İP7/İP15): ibre
        dayanağa yaslanmışsa `ok`, kadranın büsbütün dışındaysa `out_of_range`.
        """
        if self.direction == "cw":
            offset = (self.angle_min - angle_deg) % 360.0
        else:
            offset = (angle_deg - self.angle_min) % 360.0

        # Kadranın GERİSİNE düşen ibreyi mod 360 devasa bir pozitif sayıya
        # çevirir: min'in 1° gerisi 359° olur ve oran 1,33 çıkar — yani "az
        # geride" olan ibre "kadranın çok ötesinde" gibi görünür. Süpürmenin
        # dışında kalan ölü bölgeyi ortadan bölüp gerisini negatife alıyoruz.
        olu_bolge = 360.0 - self.sweep_deg
        if offset > self.sweep_deg + olu_bolge / 2.0:
            offset -= 360.0
        return offset / self.sweep_deg

    def value_for_fraction(self, fraction: float) -> float:
        """Süpürmedeki oran → değer. Karekök kadranda üs tersine uygulanır."""
        if fraction < 0.0:
            raise ValueError(f"oran negatif ({fraction:.3f}) — ibre kadranın gerisinde")
        oran = fraction if self.linear else fraction ** (1.0 / SQRT_SCALE_EXPONENT)
        return self.min + oran * (self.max - self.min)

    def value_for_angle(self, angle_deg: float) -> float:
        """İbre açısı → gösterge değeri (İP7'nin çekirdeği).

        Kadran dışındaki açıda hata yükseltir; sessizce kırpılmış bir sayı
        döndürmek, ibrenin dayanağa yaslandığı durumu normal okuma gibi
        gösterirdi (3. kural).
        """
        oran = self.fraction_for_angle(angle_deg)
        if not 0.0 <= oran <= 1.0:
            raise ValueError(
                f"{angle_deg:.1f}° kadranın dışında (oran {oran:.3f}) — "
                f"süpürme {self.angle_min:.0f}° → {self.angle_max:.0f}° ({self.direction})"
            )
        return self.value_for_fraction(oran)


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
    synthetic: dict[str, Any] = field(default_factory=dict)  # İP3 çizim ayarları
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
        # Çizim ayarları da defaults < gösterge sırasıyla birleşir: TI-205 sadece
        # tick_major'ı ezip renkleri varsayılandan almaya devam edebilsin diye.
        synthetic={**(defaults.get("synthetic") or {}), **(entry.get("synthetic") or {})},
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
