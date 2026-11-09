# -*- coding: utf-8 -*-
"""Params / GFLOPs (@416) / size, all models built with nc=9 to match the task."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("ULTRALYTICS_ROOT", "ultralytics"))

import yaml, torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
from ultralytics.utils.torch_utils import get_flops

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(ROOT, "results", "complexity.json")
W = os.environ.get("RUNS_ROOT", "runs/detect/runs/wafer")
IMGSZ, NC = 416, 9

BASE = [
    ("YOLOv8n",   "yolov8n.yaml",  "det"),
    ("YOLOv9t",   "yolov9t.yaml",  "det"),
    ("YOLOv10n",  "yolov10n.yaml", "det"),
    ("YOLOv11n",  "yolo11n.yaml",  "det"),
    ("YOLOv12n",  "yolo12n.yaml",  "det"),
    ("RT-DETR-L", "rtdetr-l.yaml", "rtdetr"),
]
TRAINED = [
    ("Ours (v11n+aug+CB)", "w11_base_s0"),
    ("+PRAttn+GeoFuse",    "w11_a7_s0"),
    ("+CBAM",              "w_cbam_s0"),
    ("+EMA",               "w_ema_s0"),
    ("YOLOv26n",           "w_base_s1"),
]

def measure(model):
    n_p = sum(p.numel() for p in model.parameters())
    return round(n_p / 1e6, 2), round(get_flops(model, IMGSZ), 2)

res = {}
for name, cfg, kind in BASE:
    try:
        from ultralytics.nn.tasks import yaml_model_load
        d = yaml_model_load(cfg)
        d["nc"] = NC
        cls = RTDETRDetectionModel if kind == "rtdetr" else DetectionModel
        m = cls(d, ch=3, nc=NC, verbose=False)
        p, f = measure(m)
        res[name] = {"params_m": p, "gflops": f, "size_mb": None}
        print("%-22s params=%6.2fM  GFLOPs=%6.2f" % (name, p, f))
    except Exception as e:
        res[name] = {"error": str(e)[:140]}
        print("%-22s FAILED %s" % (name, str(e)[:110]))

for name, run in TRAINED:
    pt = os.path.join(W, run, "weights", "best.pt")
    try:
        mm = YOLO(pt)
        p, f = measure(mm.model)
        res[name] = {"params_m": p, "gflops": f,
                     "size_mb": round(os.path.getsize(pt) / 1024 / 1024, 1)}
        print("%-22s params=%6.2fM  GFLOPs=%6.2f  size=%.1fMB" % (name, p, f, res[name]["size_mb"]))
    except Exception as e:
        res[name] = {"error": str(e)[:140]}
        print("%-22s FAILED %s" % (name, str(e)[:110]))

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print("saved", OUT)
