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

GAUGE_TYPES = ("analog", "digital", "lamp", "valve", "keypad")

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

    @property
    def ccw_araligi(self) -> tuple[float, float]:
        """Skalanın kapladığı yay, CCW yönünde (başlangıç, bitiş) olarak.

        İbrenin fiziksel olarak bulunabileceği açı aralığı budur; okuyucunun
        tarama penceresi buradan geliyor (`read_needle_angle(aci_penceresi=…)`).

        `cw` kadranda min'den max'a giderken açı AZALIR, dolayısıyla CCW yönünde
        yay `angle_max`'tan `angle_min`'e uzanır — sıra TERSİNE çevrilmeli.
        Bu ters çevirmeyi atlamak sessizce YANLIŞ yayı seçer: EM-501'de
        (150→30, cw) düz çıkarma 120° yerine 240°'lik yayı verir ve pencerenin
        eleyeceği çerçeve tam o fazladan 120°'nin içinde kalır (27.08 ölçümü:
        pencere açık ama hata 107,6°'de sabit kaldı — hata buradaydı).
        """
        if self.direction == "cw":
            return (self.angle_max, self.angle_min)
        return (self.angle_min, self.angle_max)

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
    buttons: list[dict[str, Any]] = field(default_factory=list)  # keypad
    # Kadran yüzünün GEOMETRİSİ — yalnız analogda anlamlı. Boşsa yuvarlak kadran
    # varsayılır ve zincir bugünkü davranışını sürdürür (bkz. `face_shape`).
    face: dict[str, Any] = field(default_factory=dict)
    alarm: dict[str, float] = field(default_factory=dict)
    synthetic: dict[str, Any] = field(default_factory=dict)  # İP3 çizim ayarları
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)  # ham sözlük

    @property
    def state_names(self) -> list[str]:
        return [s["name"] for s in self.states]

    @property
    def state_angles(self) -> dict[str, float]:
        """Durum adı → o durumdaki kol açısı (derece, 180° modunda).

        Yalnızca `lever_angle` beyan eden durumlar döner; hiçbiri beyan
        etmemişse boş sözlük. Okuyucu boş sözlük görünce kendi belgelenmiş
        varsayımına düşer — ama o varsayımın nerede olduğu tek yerde durur.

        Neden envanterde: durum↔açı eşleşmesi göstergenin MONTAJ bilgisidir,
        algoritmanın değil. Aynı kod, kolu ters takılmış bir vanayı da doğru
        okumalı; fark YAML satırında kalmalı (2. kural).
        """
        return {s["name"]: float(s["lever_angle"]) % 180.0
                for s in self.states if s.get("lever_angle") is not None}

    @property
    def face_shape(self) -> str:
        """`round` (varsayılan) veya `panel` — kare çerçeveli, yay skalalı metre.

        Elektrik odalarındaki pano tipi ampermetre/voltmetrelerde çerçeve
        karedir, skala ~90°'lik bir yaydır ve ibre kutunun ORTASINDAN değil
        kenara yakın bir noktadan döner. Yuvarlak kadran için doğru olan üç
        varsayımın üçü de burada yanlış (bkz. `pivot_ratio`).
        """
        return str(self.face.get("shape", "round"))

    @property
    def pivot_ratio(self) -> tuple[float, float]:
        """İbrenin dönme noktası, TESPİT KUTUSUNA oran. Varsayılan: kutunun ortası.

        Neden envanterde: dönme noktası göstergenin MONTAJ/GEOMETRİ bilgisidir,
        algoritmanın değil — `state_angles`'taki `lever_angle` ile aynı gerekçe.
        Görüntüden kestirmek denenebilirdi ama kadranın kendisi bile ancak %0,6
        oranında çember olarak doğrulanabiliyor (bkz. detect/refine.py 27.08
        ölçümü); ondan daha zayıf bir kanıtla pivot aramak sessiz hata üretir.
        """
        p = self.face.get("pivot") or (0.5, 0.5)
        return (float(p[0]), float(p[1]))

    @property
    def sweep_radius_ratio(self) -> float | None:
        """İbre süpürme yarıçapı / kutu GENİŞLİĞİ. None → kutudan türet.

        Yuvarlak kadranda yarıçap kutunun yarısıdır; yay skalalı metrede ibre
        kutu yüksekliğinin neredeyse tamamı kadar uzanabilir, çünkü pivot
        kenardadır. Tek bir orandan türetmek ikisinden birini bozar.
        """
        r = self.face.get("sweep_radius")
        return None if r is None else float(r)

    @property
    def button_names(self) -> list[str]:
        return [b["id"] for b in self.buttons]

    @property
    def machine_states(self) -> list[dict[str, Any]]:
        """Buton kombinasyonu → makine durumu kuralları, envanterdeki sırayla.

        **Neden envanterde:** hangi buton bileşiminin "çalışıyor" demek olduğu
        MAKİNENİN bilgisidir, algoritmanın değil. Aynı kod, butonları farklı
        dizilmiş bir panoyu da doğru okumalı; fark YAML satırında kalmalı
        (2. kural — `state_angles` ile aynı gerekçe).

        Sıra anlamlıdır: ilk eşleşen kural kazanır, böylece özel bir durum
        (örn. "arıza") genel bir kuraldan önce yazılabilir.
        """
        return list(self.raw.get("machine_states") or [])

    @property
    def tolerance_deg(self) -> float:
        """Kol beyan edilen açıdan bu kadar sapabilir; dışı → `unreadable`.

        Envanterden okunuyor çünkü koda gömülü bir tolerans ile envanterdeki
        beyan sessizce ayrışabiliyor — 14.08'de tam olarak bu oldu (envanter
        ±20°, kod fiilen ±6°). Tek kaynak varsa ayrışma imkânsızdır.
        """
        return float((self.raw.get("reading") or {}).get("tolerance_deg", 20.0))

    @property
    def allow_minus(self) -> bool:
        """Dijital panel negatif değer gösterebilir mi (varsayılan: evet)."""
        return bool((self.digits or {}).get("allow_minus", True))

    def tick_values(self) -> tuple[list[float], list[float]]:
        """Ana ve ara çizgilerin DEĞERLERİ (açıları değil): (majors, minors).

        Çizgiler değer ekseninde eşit aralıklıdır; açıya çevirmeyi
        `Scale.angle_for_value` yapar. Karekök ölçekli kadranda bu, çizgilerin
        görüntüde eşit aralıklı ÇIKMAMASINI sağlar — gerçek debimetreler de
        böyledir.

        Burada duruyor çünkü çizgi düzeni kadranın kendi özelliğidir, çizim
        ayrıntısı değil: sentetik üreteç (İP3) onları ÇİZMEK için, yatıklık
        kestirimi (İP8) onları GÖRÜNTÜDE ARAMAK için kullanır. İkisi ayrı yerde
        tanımlansaydı üreteç ile okuyucu sessizce ayrışabilirdi.
        """
        if self.scale is None:
            raise ValueError(f"{self.id}: çizgi düzeni sadece analog göstergede var")

        n_major = int(self.synthetic.get("tick_major", 11))
        n_minor = int(self.synthetic.get("tick_minor", 4))

        step = (self.scale.max - self.scale.min) / (n_major - 1)
        majors = [self.scale.min + i * step for i in range(n_major)]

        minors: list[float] = []
        for i in range(n_major - 1):
            for j in range(1, n_minor + 1):
                minors.append(majors[i] + step * j / (n_minor + 1))
        return majors, minors


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


# Panolarda iki farklı buton türü var ve FARKLI FİZİKLE okunurlar: ışıklı
# basmalı buton merceğinin renginden, seçici anahtar (1-0 şalteri) kolunun
# konumundan. İkincisini renkle okumak "0" ile "1"i ayırt edemez.
BUTON_TURLERI = ("lamp", "selector")
# İki selector konumu bundan yakınsa kol açısı onları ayıramaz.
MIN_SELECTOR_AYRIMI_DEG = 25.0


def _dogrula_butonlar(entry: dict[str, Any], gid: str, where: str) -> None:
    """Buton panelinin (`keypad`) yerleşim ve kural beyanlarını sınar.

    Beş kontrol, hepsi sessiz hata sınıfına karşı — bir buton paneli yanlış
    okunduğunda çıkan şey bir sayı değil **makinenin durumudur**; "çalışıyor"
    derken duran bir makine, yanlış bir basınç değerinden tehlikelidir.

    1. **En az bir buton** ve her butonun `id`'si olmalı; `id` tekrar edemez —
       aynı ada iki buton, kural eşleşmesini yazı-turaya çevirir.
    2. **Konum oranları kare içinde kalmalı.** `center` ve `radius` tespit
       kutusuna ORANDIR (0-1); piksel yazılırsa okuma sessizce kare dışını
       örnekler ve her butonu "sönük" görür.
    3. **Butonlar ÇAKIŞMAMALI.** İki buton dairesi üst üste binerse ikisi de
       aynı pikselleri örnekler ve durumları birbirine kopyalanır.
    4. **Her butonun en az iki durumu olmalı** — tek durumlu bir buton hiçbir
       şey ayırt etmez.
    5. **Kurallar yalnız tanımlı butonlara ve durumlara atıf yapabilir.**
       Yazım hatası olan bir kural sessizce hiç eşleşmez ve panel sonsuza kadar
       `unreadable` döner; hata envanterde, belirtisi okumada çıkar.
    """
    butonlar = entry.get("buttons") or []
    if not butonlar:
        raise ConfigError(f"{where} ({gid}): 'keypad' en az bir buton ister")

    gorulen: set[str] = set()
    daireler: list[tuple[str, float, float, float]] = []
    for b in butonlar:
        bid = b.get("id")
        if not bid:
            raise ConfigError(f"{where} ({gid}): butonlardan birinde 'id' yok")
        if bid in gorulen:
            raise ConfigError(f"{where} ({gid}): buton id'si tekrar ediyor: '{bid}'")
        gorulen.add(bid)

        merkez = b.get("center")
        if not (isinstance(merkez, (list, tuple)) and len(merkez) == 2):
            raise ConfigError(f"{where} ({gid}/{bid}): 'center' [x, y] olmalı")
        try:
            cx, cy = float(merkez[0]), float(merkez[1])
            r = float(b.get("radius", 0.0))
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{where} ({gid}/{bid}): center/radius sayı olmalı — {e}") from e
        if not (0.0 < r <= 0.5):
            raise ConfigError(f"{where} ({gid}/{bid}): 'radius' 0-0,5 oranında olmalı, "
                              f"{r} verildi (piksel değil ORAN)")
        if not (r <= cx <= 1.0 - r and r <= cy <= 1.0 - r):
            raise ConfigError(f"{where} ({gid}/{bid}): buton kutunun dışına taşıyor "
                              f"(merkez {cx},{cy} · yarıçap {r})")

        durumlar = b.get("states") or []
        if len(durumlar) < 2:
            raise ConfigError(f"{where} ({gid}/{bid}): buton en az 2 durum ister")

        # Seçici anahtar (1-0 şalteri) IŞIKLA değil KOL AÇISIYLA okunur. Açısı
        # beyan edilmemiş bir selector sessizce hiçbir duruma eşleşmez ve panel
        # sonsuza kadar `unreadable` döner — hata envanterde, belirtisi okumada.
        tur = str(b.get("kind", "lamp"))
        if tur not in BUTON_TURLERI:
            raise ConfigError(f"{where} ({gid}/{bid}): 'kind' {BUTON_TURLERI} "
                              f"olmalı, '{tur}' verildi")
        if tur == "selector":
            acilar = b.get("lever_angles") or {}
            eksik = [d for d in durumlar if d not in acilar]
            if eksik:
                raise ConfigError(f"{where} ({gid}/{bid}): selector için her durumun "
                                  f"'lever_angles' değeri olmalı — eksik: {eksik}")
            try:
                sayilar = {d: float(acilar[d]) % 180.0 for d in durumlar}
            except (TypeError, ValueError) as e:
                raise ConfigError(f"{where} ({gid}/{bid}): lever_angles sayı olmalı — {e}") from e
            # İki durum birbirine çok yakınsa kol açısı onları ayıramaz; okuma
            # yazı-turaya döner. Vana tarafındaki ayrım eşiğiyle aynı mantık.
            adlar = list(sayilar)
            for i, a in enumerate(adlar):
                for c in adlar[i + 1:]:
                    fark = abs(sayilar[a] - sayilar[c]) % 180.0
                    fark = min(fark, 180.0 - fark)
                    if fark < MIN_SELECTOR_AYRIMI_DEG:
                        raise ConfigError(
                            f"{where} ({gid}/{bid}): '{a}' ve '{c}' kol açıları "
                            f"{fark:.0f}° ayrık — en az {MIN_SELECTOR_AYRIMI_DEG:.0f}° "
                            f"gerek, yoksa okuma ikisini ayıramaz")

        for ad, ox, oy, orr in daireler:
            if (cx - ox) ** 2 + (cy - oy) ** 2 < (r + orr) ** 2:
                raise ConfigError(f"{where} ({gid}): '{bid}' ve '{ad}' butonları "
                                  f"çakışıyor — ikisi aynı pikselleri örnekler")
        daireler.append((bid, cx, cy, r))

    izinli = {b["id"]: set(b.get("states") or []) for b in butonlar}
    for kural in entry.get("machine_states") or []:
        if not kural.get("name"):
            raise ConfigError(f"{where} ({gid}): machine_states kuralında 'name' yok")
        kosul = kural.get("when") or {}
        if not kosul:
            raise ConfigError(f"{where} ({gid}/{kural['name']}): 'when' boş olamaz — "
                              f"koşulsuz kural her kombinasyonu yutar")
        for bid, beklenen in kosul.items():
            if bid not in izinli:
                raise ConfigError(f"{where} ({gid}/{kural['name']}): tanımsız buton "
                                  f"'{bid}' — mevcutlar: {sorted(izinli)}")
            if beklenen not in izinli[bid]:
                raise ConfigError(f"{where} ({gid}/{kural['name']}): '{bid}' butonu "
                                  f"'{beklenen}' durumunu beyan etmiyor — "
                                  f"mevcutlar: {sorted(izinli[bid])}")


def _dogrula_kol_acilari(entry: dict[str, Any], states: list[dict[str, Any]],
                         gid: str, where: str) -> None:
    """`lever_angle` ve `tolerance_deg` beyanlarını sınar (vana).

    Üç kontrol, üçü de sessiz hata sınıfına karşı:

    1. **Ya hepsi ya hiçbiri.** Durumların bir kısmı açı beyan edip diğerleri
       etmezse, beyan etmeyenler okuyucunun varsayımına düşer ve envanter ile
       kod yarı yarıya karışır. Bu, `sweep_deg` olmadan `direction` yazmakla
       aynı hata sınıfıdır.
    2. **Tolerans anlamlı olmalı.** 0 hiçbir okumayı geçirmez, 90'dan büyük
       tolerans 180° modunda her açıyı her duruma sokar.
    3. **İki durum toleranslarıyla ÇAKIŞMAMALI.** `open: 0` ve `closed: 10`
       ±20° toleransla ayırt edilemez; kod yine de bir cevap üretirdi ve o
       cevap yazı-tura olurdu. Beyan çelişkiliyse okuma değil envanter yanlıştır.
    """
    acili = [s for s in states if s.get("lever_angle") is not None]
    if not acili:
        return
    if len(acili) != len(states):
        eksik = [s["name"] for s in states if s.get("lever_angle") is None]
        raise ConfigError(
            f"{where} ({gid}): durumların bir kısmı 'lever_angle' beyan etmiş, "
            f"{eksik} etmemiş — ya hepsi ya hiçbiri")

    try:
        acilar = {s["name"]: float(s["lever_angle"]) % 180.0 for s in acili}
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{where} ({gid}): 'lever_angle' sayı olmalı — {e}") from e

    tol = float((entry.get("reading") or {}).get("tolerance_deg", 20.0))
    if not 0.0 < tol <= 90.0:
        raise ConfigError(
            f"{where} ({gid}): tolerance_deg 0-90 aralığında olmalı, {tol} verildi")

    adlar = list(acilar)
    for i in range(len(adlar)):
        for j in range(i + 1, len(adlar)):
            a, b = acilar[adlar[i]], acilar[adlar[j]]
            fark = abs(a - b) % 180.0
            fark = min(fark, 180.0 - fark)   # kol iki uçlu: 180° modunda mesafe
            if fark < 2 * tol:
                raise ConfigError(
                    f"{where} ({gid}): '{adlar[i]}' ({a:.0f}°) ve '{adlar[j]}' "
                    f"({b:.0f}°) arasındaki {fark:.0f}°, ±{tol:.0f}° toleransla "
                    f"ayırt edilemez — açıları ayırın ya da toleransı düşürün")


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
        _dogrula_kol_acilari(entry, states, gid, where)
    if gtype == "keypad":
        _dogrula_butonlar(entry, gid, where)
    _dogrula_face(entry.get("face") or {}, gid, where)

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
        buttons=entry.get("buttons") or [],
        face=entry.get("face") or {},
        alarm=entry.get("alarm") or {},
        # Çizim ayarları da defaults < gösterge sırasıyla birleşir: TI-205 sadece
        # tick_major'ı ezip renkleri varsayılandan almaya devam edebilsin diye.
        synthetic={**(defaults.get("synthetic") or {}), **(entry.get("synthetic") or {})},
        notes=entry.get("notes"),
        raw=entry,
    )


FACE_SEKILLERI = ("round", "panel")


def _dogrula_face(face: dict[str, Any], gid: str, where: str) -> None:
    """Kadran yüzü beyanını sınar.

    Pivot sessiz hata üreten bir alandır: yanlış bir pivot okumayı kırmaz,
    KAYDIRIR. İbre açısı pivota göre ölçülüyor, dolayısıyla 0,86 yerine 0,68
    yazmak her okumayı sistematik olarak yanlış yapar ve hiçbir yerde patlamaz.
    Bu yüzden aralık kontrolü şart — 0-1 dışına çıkan bir oran zaten kutu
    dışını gösterir ve mutlaka yazım hatasıdır.
    """
    if not face:
        return
    sekil = face.get("shape", "round")
    if sekil not in FACE_SEKILLERI:
        raise ConfigError(f"{where} ({gid}): face.shape {FACE_SEKILLERI} "
                          f"olmalı, '{sekil}' verildi")
    pivot = face.get("pivot")
    if pivot is not None:
        if len(pivot) != 2:
            raise ConfigError(f"{where} ({gid}): face.pivot iki sayı ister")
        for ad, v in zip("xy", pivot):
            if not 0.0 <= float(v) <= 1.0:
                raise ConfigError(f"{where} ({gid}): face.pivot.{ad} 0-1 "
                                  f"aralığında olmalı, {v} verildi")
    r = face.get("sweep_radius")
    if r is not None and not 0.0 < float(r) <= 2.0:
        raise ConfigError(f"{where} ({gid}): face.sweep_radius 0-2 aralığında "
                          f"olmalı, {r} verildi")


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
