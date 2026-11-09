# -*- coding: utf-8 -*-
"""Single-image inference latency, round-robin protocol to cancel thermal drift.
Each model is measured for ROUNDS interleaved rounds; per-round median is kept,
and the median-of-medians is reported. Baselines are built from yaml with nc=9
(latency is weight-independent); ablation models use trained checkpoints.
Output: paper/analysis/latency.json
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.environ.get("ULTRALYTICS_ROOT", "ultralytics"))

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load, DetectionModel, RTDETRDetectionModel

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(ROOT, "results", "latency.json")
W   = os.environ.get("RUNS_ROOT", "runs/detect/runs/wafer")
IMGSZ, NC, WARMUP, REPS, ROUNDS = 416, 9, 30, 60, 5

CFG = os.path.join(ROOT, "configs")
# All models are built from yaml (nc=9): latency is architecture-determined and
# weight-independent (verified: ckpt weights transplanted into a yaml-built
# model give identical latency; YOLO(ckpt) wrapper adds per-call overhead).
BASE = [("YOLOv8n","yolov8n.yaml","det"),("YOLOv9t","yolov9t.yaml","det"),
        ("YOLOv10n","yolov10n.yaml","det"),("YOLOv11n","yolo11n.yaml","det"),
        ("YOLOv12n","yolo12n.yaml","det"),("RT-DETR-L","rtdetr-l.yaml","rtdetr"),
        ("+PRAttn+GeoFuse", os.path.join(CFG, "yolo11n-a7-prattn-geofuse.yaml"), "det"),
        ("+CBAM", os.path.join(CFG, "yolo11n-cbam.yaml"), "det"),
        ("+EMA", os.path.join(CFG, "yolo11n-ema.yaml"), "det"),
        ("YOLOv26n", "yolo26n.yaml", "det")]

def build():
    models = []
    for name, cfg, kind in BASE:
        d = yaml_model_load(cfg); d["nc"] = NC
        cls = RTDETRDetectionModel if kind == "rtdetr" else DetectionModel
        m = cls(d, ch=3, nc=NC, verbose=False); m.fuse()
        models.append((name, m))
    return models

def bench_once(model, device):
    x = torch.rand(1, 3, IMGSZ, IMGSZ, device=device)
    with torch.no_grad():
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]

res = {}
for dev_name in ["cuda", "cpu"]:
    dev = torch.device(dev_name)
    models = [(n, m.to(dev).eval().float()) for n, m in build()]
    meds = {n: [] for n, _ in models}
    for n, m in models:  # shared warmup brings the device to thermal steady state
        x = torch.rand(1, 3, IMGSZ, IMGSZ, device=dev)
        with torch.no_grad():
            for _ in range(WARMUP): m(x)
        if dev.type == "cuda": torch.cuda.synchronize()
    for r in range(ROUNDS):
        for n, m in models:  # round-robin: drift is spread evenly over models
            meds[n].append(bench_once(m, dev))
        print(f"{dev_name} round {r+1}/{ROUNDS} done", flush=True)
    for n, _ in models:
        v = sorted(meds[n]); med = v[len(v) // 2]
        res.setdefault(n, {})[dev_name] = {"median_ms": round(med, 2),
                                           "fps": round(1000.0 / med, 1),
                                           "rounds_ms": [round(t, 2) for t in v]}
        print(f"{n:22s} {dev_name:4s} {med:7.2f} ms  ({res[n][dev_name]['fps']:6.1f} FPS)"
              f"  rounds={res[n][dev_name]['rounds_ms']}", flush=True)
    del models
    if dev.type == "cuda": torch.cuda.empty_cache()

json.dump({"device_gpu": torch.cuda.get_device_name(0),
           "cpu": os.environ.get("PROCESSOR_IDENTIFIER", ""),
           "imgsz": IMGSZ, "reps": REPS, "rounds": ROUNDS,
           "protocol": "round-robin, median of per-round medians",
           "results": res}, open(OUT, "w"), indent=1)
print("saved", OUT)
