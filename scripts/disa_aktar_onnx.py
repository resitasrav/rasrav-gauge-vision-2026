r"""Tespit modelini ONNX'e aktarır ve çıktının AYNI olduğunu doğrular.

    python scripts\disa_aktar_onnx.py
    python scripts\disa_aktar_onnx.py --imgsz 416 --dogrula

**Neden bu adım var.** Sistem sahada Orange Pi 5 (RK3588) + NPU hızlandırıcılı
bir kartta koşacak. PyTorch ağırlığı (`best.pt`) o kartın NPU'sunda çalışmaz;
hem Hailo (Dataflow Compiler → `.hef`) hem Rockchip (RKNN Toolkit → `.rknn`)
zincirinin **girdisi ONNX'tir.** Bu script o girdiyi üretir ve — asıl önemlisi —
ürettiği dosyanın PyTorch ile **aynı kutuları** verdiğini sayıyla gösterir.

**Neden doğrulama şart.** Dışa aktarım sessizce bozulabilir: yanlış `imgsz`,
dinamik eksen, NMS'in modele gömülü olup olmaması... Model yine çalışır, yine
kutu üretir, kutular biraz kayar ve bu **zincirin en duyarlı olduğu büyüklüğü**
bozar (8 px merkez kayması açı hatasını 0,123° → 3,652° yapıyor). Bu yüzden
burada eşitlik iddiası ölçülüyor: aynı görüntüde iki motorun kutuları
karşılaştırılıyor.

**Bu script kartı GEREKTİRMEZ.** ONNX çıktısı bu makinede `onnxruntime` ile
koşturuluyor; kartın NPU'suna derleme adımı ayrıdır ve orada yapılır. Amaç
"taşınabilir mi" sorusunu **taşımadan önce** cevaplamaktır.

**Zincir tarafında değişiklik gerekmiyor.** `pipeline.read_frame` modeli
dışarıdan alır ve yalnız şu arayüzü ister:

    sonuc = model.predict(image, conf=..., verbose=False)[0]
    sonuc.boxes.xyxy / .conf / .cls / len()
    sonuc.names

`OnnxTespit` bu arayüzü uygular; kartta `.hef`/`.rknn` çalıştıran bir sınıf da
aynı arayüzü uygularsa zincir kodu tek satır değişmez.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

VARSAYILAN_AGIRLIK = "runs/detect/models/ip5/cok_sinif/weights/best.pt"
METRIK_YOLU = Path("outputs/metrics/onnx_aktarim.json")

# Kutuların "aynı" sayılması için izin verilen sapma (piksel). Zincirin merkez
# duyarlılığı bunu belirliyor: İP6 ölçümüne göre 8 px kayma açı hatasını 30
# katına çıkarıyor, 2 px'lik bir fark ise ölçüm gürültüsünün içinde kalıyor.
MAX_KUTU_SAPMASI_PX = 2.0
MAX_GUVEN_SAPMASI = 0.02


class _Kutular:
    """Ultralytics `Boxes` nesnesinin zincirin kullandığı kadarı."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _Sonuc:
    def __init__(self, kutular: _Kutular, adlar: dict[int, str]):
        self.boxes = kutular
        self.names = adlar


class OnnxTespit:
    """ONNX modelini Ultralytics arayüzüyle sarmalar.

    Kartta çalışacak sınıf da tam olarak bunu yapacaktır; tek farkı
    `_ileri()` içinde `onnxruntime` yerine Hailo/RKNN çalışma zamanını
    çağırmasıdır. Ön işleme (letterbox) ve son işleme (NMS + ölçek geri alma)
    aynen kalır — orası motordan bağımsızdır.
    """

    def __init__(self, onnx_yolu: str | Path, adlar: dict[int, str], imgsz: int = 416):
        import onnxruntime as ort

        self.oturum = ort.InferenceSession(str(onnx_yolu),
                                           providers=["CPUExecutionProvider"])
        self.girdi_adi = self.oturum.get_inputs()[0].name
        self.names = adlar
        self.imgsz = imgsz

    def _on_isle(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Letterbox: en-boy oranı KORUNARAK `imgsz` kareye oturtur.

        Oran korunmazsa kadran elipse dönüşür ve tespit kutusu kayar — 19.08'de
        `ekran_kadran.py`'da tam bu hata yaşandı ve o günün bütün sayıları
        geçersiz oldu. Aynı hata burada tekrarlanmasın diye letterbox açıkça
        yazılıyor.
        """
        h, w = image.shape[:2]
        olcek = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * olcek)), int(round(h * olcek))
        kucuk = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        tuval = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        dx, dy = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        tuval[dy:dy + nh, dx:dx + nw] = kucuk

        x = cv2.cvtColor(tuval, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return x.transpose(2, 0, 1)[None], olcek, (dx, dy)

    def predict(self, image: np.ndarray, *, conf: float = 0.25,
                verbose: bool = False, **_):
        girdi, olcek, (dx, dy) = self._on_isle(image)
        ham = self.oturum.run(None, {self.girdi_adi: girdi})[0]

        # YOLOv8 çıktısı (1, 4+nc, N): ilk dört satır cx,cy,w,h; kalanı sınıf
        # skorları. Sigmoid/softmax yok — skorlar doğrudan olasılık.
        p = ham[0]
        kutular_xywh = p[:4].T
        skorlar = p[4:].T
        sinif = skorlar.argmax(axis=1)
        guven = skorlar.max(axis=1)

        tut = guven >= conf
        kutular_xywh, sinif, guven = kutular_xywh[tut], sinif[tut], guven[tut]
        if kutular_xywh.shape[0] == 0:
            return [_Sonuc(_Kutular(np.zeros((0, 4), np.float32),
                                    np.zeros((0,), np.float32),
                                    np.zeros((0,), np.float32)), self.names)]

        cx, cy, bw, bh = kutular_xywh.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)

        # NMS modele gömülü değil — burada yapılıyor. Kartta da burada kalacak:
        # NPU'lar NMS'i çoğu zaman desteklemez, CPU'da yapmak standart yoldur.
        idx = cv2.dnn.NMSBoxes(
            [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xyxy],
            guven.astype(np.float32).tolist(), conf, 0.45)
        idx = np.array(idx).reshape(-1) if len(idx) else np.zeros((0,), int)
        xyxy, sinif, guven = xyxy[idx], sinif[idx], guven[idx]

        # Letterbox'ı geri al: önce kaydırma, sonra ölçek.
        xyxy[:, [0, 2]] -= dx
        xyxy[:, [1, 3]] -= dy
        xyxy /= olcek

        sira = np.argsort(-guven)
        return [_Sonuc(_Kutular(xyxy[sira].astype(np.float32),
                                guven[sira].astype(np.float32),
                                sinif[sira].astype(np.float32)), self.names)]


def _kutulari_kiyasla(a, b) -> dict:
    """İki motorun kutularını eşleştirip en büyük sapmayı döner."""
    if len(a.boxes) == 0 and len(b.boxes) == 0:
        return {"kutu": 0, "sapma_px": 0.0, "guven_sapmasi": 0.0, "esit": True}
    if len(a.boxes) != len(b.boxes):
        return {"kutu": f"{len(a.boxes)} ≠ {len(b.boxes)}", "esit": False}

    # PyTorch tarafı GPU tensörü döndürebilir; `np.asarray` orada patlar.
    def _dizi(v):
        return np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=float)

    ax, bx = _dizi(a.boxes.xyxy), _dizi(b.boxes.xyxy)
    ac, bc = _dizi(a.boxes.conf), _dizi(b.boxes.conf)
    sapma = float(np.abs(ax - bx).max())
    g_sapma = float(np.abs(ac - bc).max())
    return {"kutu": len(a.boxes), "sapma_px": round(sapma, 3),
            "guven_sapmasi": round(g_sapma, 4),
            "esit": sapma <= MAX_KUTU_SAPMASI_PX and g_sapma <= MAX_GUVEN_SAPMASI}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Tespit modelini ONNX'e aktar ve doğrula")
    p.add_argument("--agirlik", default=VARSAYILAN_AGIRLIK)
    p.add_argument("--imgsz", type=int, default=416, help="eğitimdeki değerle aynı olmalı")
    p.add_argument("--kare", type=int, default=8, help="doğrulamada kaç sentetik kare")
    args = p.parse_args(argv)

    agirlik = Path(args.agirlik)
    if not agirlik.exists():
        print(f"ağırlık yok: {agirlik}")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(agirlik))
    adlar = {int(k): v for k, v in model.names.items()}
    print(f"ağırlık: {agirlik}\nsınıflar: {adlar}\nimgsz: {args.imgsz}")

    print("\nONNX'e aktarılıyor…")
    onnx_yolu = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=False,
                                  dynamic=False, verbose=False))
    print(f"yazıldı: {onnx_yolu} ({onnx_yolu.stat().st_size / 1e6:.1f} MB)")

    onnx_model = OnnxTespit(onnx_yolu, adlar, args.imgsz)

    # Doğrulama kareleri: envanterdeki dört tip, farklı konum ve ölçeklerde.
    from gauge_vision.config import load_gauges
    from gauge_vision.synth.dial import render_analog

    gauges = load_gauges()
    analog = next(g for g in gauges.values() if g.type == "analog")
    rng = np.random.default_rng(0)

    print(f"\n{'kare':>6s} {'pytorch':>9s} {'onnx':>9s} {'kutu':>8s} "
          f"{'sapma px':>10s} {'güven Δ':>9s}")
    kayitlar = []
    for i in range(args.kare):
        boy = int(rng.integers(180, 420))
        kadran, _ = render_analog(analog, float(rng.uniform(analog.scale.min,
                                                           analog.scale.max)),
                                  size=boy)
        kare = np.full((720, 1280, 3), 95, np.uint8)
        x = int(rng.integers(0, 1280 - boy))
        y = int(rng.integers(0, 720 - boy))
        kare[y:y + boy, x:x + boy] = kadran

        t0 = time.perf_counter()
        pt = model.predict(kare, conf=0.25, verbose=False)[0]
        pt_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        ox = onnx_model.predict(kare, conf=0.25)[0]
        ox_ms = (time.perf_counter() - t0) * 1000

        k = _kutulari_kiyasla(pt, ox)
        kayitlar.append({**k, "pytorch_ms": round(pt_ms, 1), "onnx_ms": round(ox_ms, 1)})
        print(f"{i:>6d} {pt_ms:>8.1f}ms {ox_ms:>8.1f}ms {str(k['kutu']):>8s} "
              f"{k.get('sapma_px', '—'):>10} {k.get('guven_sapmasi', '—'):>9}")

    esit = all(k["esit"] for k in kayitlar)
    en_kotu = max((k.get("sapma_px", 0.0) for k in kayitlar), default=0.0)
    print(f"\nEN BÜYÜK KUTU SAPMASI: {en_kotu:.3f} px "
          f"(sınır {MAX_KUTU_SAPMASI_PX} px)")
    print("SONUÇ: ONNX çıktısı PyTorch ile AYNI ✅" if esit
          else "SONUÇ: ⚠ ÇIKTILAR AYRIŞIYOR — kart üstünde kullanmadan önce bakılmalı")

    rapor = {"tarih": date.today().isoformat(), "agirlik": str(agirlik),
             "onnx": str(onnx_yolu), "imgsz": args.imgsz, "siniflar": adlar,
             "esit": esit, "en_buyuk_kutu_sapmasi_px": round(en_kotu, 3),
             "kareler": kayitlar,
             "not": "ONNX, Hailo (.hef) ve RKNN (.rknn) derleyicilerinin girdisidir; "
                    "derleme adımı kartın kendi araç zincirinde yapılır"}
    METRIK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    METRIK_YOLU.write_text(json.dumps(rapor, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"ölçüm: {METRIK_YOLU}")
    return 0 if esit else 1


if __name__ == "__main__":
    raise SystemExit(main())
