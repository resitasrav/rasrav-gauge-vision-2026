"""YOLO ile gösterge tespiti: üç yapılandırmayı eğitir ve kıyaslar (İP5).

    python scripts/egit_ip5.py                      # sentetik · gercek · karisik
    python scripts/egit_ip5.py --yapilandirma karisik --epoch 60

Kıyas kurgusu: üç yapılandırma **aynı** gerçek doğrulama ve test kümesini
kullanır, yalnızca eğitim verisi değişir. Böylece "sentetik tek başına yetersiz"
(K1) iddiası bizim verimizde sınanabilir hâle gelir.

**İki ayrı kabul ölçütü ölçülür ve ikincisi bu modül için birincisinden
önemlidir:**

1. `mAP50` / `mAP50-95` — standart tespit doğruluğu.
2. **Kutu merkezi hatası** — tahmin edilen kutunun merkezi, gerçek kutunun
   merkezinden kaç piksel sapıyor. İP6'nın ölçümü bunu zorunlu kıldı: 8 piksellik
   merkez kayması açı hatasını 0,12°'den 3,65°'ye çıkarıyor. Yüksek IoU veren
   ama merkezi oynak bir tespit, okuma zincirini yüksek mAP ile birlikte bozar.
   Hata kadran çapına oranlanarak da yazılır; çünkü zincire giren şey mutlak
   piksel değil, kadranın yarıçapına göre kaymadır.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from gauge_vision.detect.dataset import LABELS_DIR

VERI_KOK = "data/detect"
METRIK_YOLU = "outputs/metrics/ip5_tespit.json"
FIGUR_YOLU = "outputs/figures/ip5_tespit.png"
KOSU_KOK = "models/ip5"

YAPILANDIRMALAR = ("sentetik", "gercek", "karisik")
TEMEL_MODEL = "yolov8n.pt"


def _kutu_merkezleri(etiket_yolu: Path, g: int, y: int) -> np.ndarray:
    """YOLO etiket dosyasındaki kutuların merkezleri (piksel) ve genişlikleri."""
    if not etiket_yolu.exists():
        return np.zeros((0, 3))
    satirlar = [s.split() for s in etiket_yolu.read_text(encoding="utf-8").splitlines() if s.strip()]
    return np.array([[float(s[1]) * g, float(s[2]) * y, float(s[3]) * g] for s in satirlar])


def merkez_hatasi(model, veri_yaml: Path, conf: float = 0.25) -> dict:
    """Test kümesinde tahmin/gerçek kutu merkezleri arasındaki sapmayı ölçer.

    Eşleştirme en yakın merkeze göre yapılıyor, IoU'ya göre değil: ölçülen şey
    zaten merkez sapmasıdır ve IoU eşiği düşük örtüşmeli ama merkezi doğru
    tespitleri eleyerek sonucu iyimser gösterirdi.
    """
    import yaml

    test_dizini = Path(yaml.safe_load(veri_yaml.read_text(encoding="utf-8"))["test"])
    goruntuler = sorted(test_dizini.glob("*.png"))

    sapmalar_px: list[float] = []
    sapmalar_oran: list[float] = []
    bulunamadi = 0
    gercek_toplam = 0

    for g_yolu in goruntuler:
        sonuc = model.predict(str(g_yolu), conf=conf, verbose=False)[0]
        y_boy, x_boy = sonuc.orig_shape
        # YOLO düzeninde etiketler görüntülerin kardeş klasöründedir:
        # .../test/images/x.png → .../test/labels/x.txt
        gercekler = _kutu_merkezleri(
            test_dizini.parent / LABELS_DIR / f"{g_yolu.stem}.txt", x_boy, y_boy)
        gercek_toplam += len(gercekler)

        if len(sonuc.boxes) == 0:
            bulunamadi += len(gercekler)
            continue

        tahmin = sonuc.boxes.xywh.cpu().numpy()[:, :2]
        for gx, gy, gw in gercekler:
            uzakliklar = np.hypot(tahmin[:, 0] - gx, tahmin[:, 1] - gy)
            en_yakin = float(uzakliklar.min())
            # Kadran genişliğinin yarısından uzaktaki tahmin bu kadranın
            # eşleşmesi sayılmaz; başka bir göstergeye ait olabilir.
            if en_yakin > gw / 2:
                bulunamadi += 1
                continue
            sapmalar_px.append(en_yakin)
            sapmalar_oran.append(en_yakin / gw * 100)

    def ozet(v: list[float]) -> dict:
        if not v:
            return {"n": 0, "ortalama": 0.0, "medyan": 0.0, "p95": 0.0, "max": 0.0}
        a = np.sort(np.asarray(v))
        return {"n": int(a.size), "ortalama": round(float(a.mean()), 2),
                "medyan": round(float(np.median(a)), 2),
                "p95": round(float(a[min(a.size - 1, int(0.95 * (a.size - 1)))]), 2),
                "max": round(float(a[-1]), 2)}

    return {
        "eslesen_kadran": len(sapmalar_px),
        "eslesmeyen_kadran": bulunamadi,
        "gercek_kadran": gercek_toplam,
        "merkez_sapmasi_px": ozet(sapmalar_px),
        "merkez_sapmasi_yuzde_kadran_capi": ozet(sapmalar_oran),
    }


def egit(ad: str, *, epoch: int, imgsz: int, batch: int, seed: int) -> dict:
    from ultralytics import YOLO

    veri_yaml = Path(VERI_KOK) / ad / "gauge.yaml"
    model = YOLO(TEMEL_MODEL)
    model.train(data=str(veri_yaml), epochs=epoch, imgsz=imgsz, batch=batch,
                seed=seed, project=KOSU_KOK, name=ad, exist_ok=True,
                verbose=False, plots=False, val=True)

    olcum = model.val(data=str(veri_yaml), split="test", verbose=False)
    return {
        "mAP50": round(float(olcum.box.map50), 4),
        "mAP50_95": round(float(olcum.box.map), 4),
        "kesinlik": round(float(olcum.box.mp), 4),
        "duyarlilik": round(float(olcum.box.mr), 4),
        **merkez_hatasi(model, veri_yaml),
        "agirlik": str(Path(KOSU_KOK) / ad / "weights" / "best.pt"),
    }


def figur(sonuclar: dict, cikti: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adlar = list(sonuclar)
    fig, (sol, sag) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(adlar))
    sol.bar(x - 0.2, [sonuclar[a]["mAP50"] for a in adlar], 0.4, label="mAP50")
    sol.bar(x + 0.2, [sonuclar[a]["mAP50_95"] for a in adlar], 0.4, label="mAP50-95")
    sol.set_xticks(x, adlar)
    sol.set_ylabel("gerçek test kümesinde")
    sol.set_title("Tespit doğruluğu")
    sol.grid(alpha=0.3, axis="y")
    sol.legend()

    sag.bar(x, [sonuclar[a]["merkez_sapmasi_yuzde_kadran_capi"]["ortalama"] for a in adlar],
            0.5, color="tab:orange")
    sag.set_xticks(x, adlar)
    sag.set_ylabel("kadran çapının %'si")
    sag.set_title("Kutu merkezi sapması (İP6'nın duyarlı olduğu)")
    sag.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=130)
    plt.close(fig)
    return cikti


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="YOLO gösterge tespiti (İP5)")
    p.add_argument("--yapilandirma", nargs="*", choices=YAPILANDIRMALAR,
                   default=list(YAPILANDIRMALAR))
    p.add_argument("--epoch", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not (Path(VERI_KOK) / "gercek" / "gauge.yaml").exists():
        print("veri yok — önce scripts/hazirla_ip5_veri.py çalıştır")
        return 1

    sonuclar = {}
    for ad in args.yapilandirma:
        print(f"\n=== {ad} ===")
        sonuclar[ad] = egit(ad, epoch=args.epoch, imgsz=args.imgsz,
                            batch=args.batch, seed=args.seed)
        s = sonuclar[ad]
        print(f"  mAP50 {s['mAP50']:.3f}  mAP50-95 {s['mAP50_95']:.3f}  "
              f"merkez sapması {s['merkez_sapmasi_px']['ortalama']:.1f} px "
              f"(%{s['merkez_sapmasi_yuzde_kadran_capi']['ortalama']:.1f} kadran çapı)  "
              f"kaçırılan {s['eslesmeyen_kadran']}/{s['gercek_kadran']}")

    ozet = {
        "is_paketi": "IP5",
        "tarih": date.today().isoformat(),
        "temel_model": TEMEL_MODEL,
        "egitim": {"epoch": args.epoch, "imgsz": args.imgsz, "batch": args.batch,
                   "seed": args.seed, "cihaz": "cpu"},
        "test_kumesi": "gerçek fotoğraflar (Roboflow-100 gauge-u2lwv test bölümü)",
        "yapilandirmalar": sonuclar,
    }
    yol = Path(METRIK_YOLU)
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nölçüm: {yol}")

    if len(sonuclar) > 1:
        print(f"figür: {figur(sonuclar, Path(FIGUR_YOLU))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
