# -*- coding: utf-8 -*-
"""Robustness to imaging perturbations + confusion matrices.
Evaluates w11_base_s0 (ours) and w11_a7_s0 (+PRAttn+GeoFuse) on the wafer test
set under 7 photometric conditions. All boxes are preserved (photometric only).
Outputs: paper/analysis/robustness.json  (mAP50, mAP50-95, per-class AP50)
         paper/analysis/confusion_base.json / confusion_a7.json
"""
import os
import tempfile, sys, json, shutil, warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.environ.get("ULTRALYTICS_ROOT", "ultralytics"))

import cv2, numpy as np, yaml, torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

ROOT   = os.environ.get("WAFER_AUG_ROOT", "datasets/wafer_aug")
SRC_IM = os.path.join(ROOT, "test", "images")
SRC_LB = os.path.join(ROOT, "test", "labels")
TMP    = os.path.join(tempfile.gettempdir(), "wafer_robust")  # TEMP env has stray spaces
W      = os.environ.get("RUNS_ROOT", "runs/detect/runs/wafer")
OUT    = 'results'

NAMES  = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','None','Random','Scratch']

def perturb(img, kind):
    if kind == "clean":     return img
    if kind == "blur_s1":   return cv2.GaussianBlur(img, (0, 0), 1.0)
    if kind == "blur_s2":   return cv2.GaussianBlur(img, (0, 0), 2.0)
    if kind == "noise_s10":
        n = np.random.default_rng(0).normal(0, 10, img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if kind == "noise_s25":
        n = np.random.default_rng(0).normal(0, 25, img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if kind == "down50":
        h, w = img.shape[:2]
        return cv2.resize(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "down25":
        h, w = img.shape[:2]
        return cv2.resize(cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(kind)

CONDS = ["clean", "blur_s1", "blur_s2", "noise_s10", "noise_s25", "down50", "down25"]

def build_cond(cond):
    imdir = os.path.join(TMP, cond, "images")
    lbdir = os.path.join(TMP, cond, "labels")
    if os.path.isdir(imdir) and len(os.listdir(imdir)) == 1300:
        return imdir
    os.makedirs(imdir, exist_ok=True); os.makedirs(lbdir, exist_ok=True)
    for f in os.listdir(SRC_LB):
        if f.endswith(".txt"):
            dst = os.path.join(lbdir, f)
            if not os.path.exists(dst): os.link(os.path.join(SRC_LB, f), dst)
    files = [f for f in os.listdir(SRC_IM) if f.endswith(".jpg")]
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(SRC_IM, f))
        cv2.imwrite(os.path.join(imdir, f), perturb(img, cond))
        if i % 400 == 0: print(f"  {cond}: {i}/1300", flush=True)
    return imdir

def make_yaml(imdir):
    d = {"path": ROOT, "train": "train/images", "val": "valid/images",
         "test": imdir, "nc": 9, "names": NAMES}
    y = os.path.join(TMP, "data.yaml"); yaml.safe_dump(d, open(y, "w"))
    return y

def run_val(weights, data_yaml):
    m = YOLO(weights)
    r = m.val(data=data_yaml, split="test", imgsz=416, batch=16,
              workers=0, verbose=False, plots=False)
    return {"map50": round(float(r.box.map50), 4),
            "map":   round(float(r.box.map), 4),
            "ap50_per_class": {NAMES[i]: round(float(v), 4) for i, v in enumerate(r.box.ap50)}}

def run_conf(weights, data_yaml, tag):
    # plots=True is required: ultralytics only updates confusion_matrix then
    args = dict(model=weights, data=data_yaml, split="test", imgsz=416,
                batch=16, workers=0, verbose=False, plots=True,
                project=os.path.join(TMP, "conf_plots"), name=tag)
    v = DetectionValidator(args=args)
    v()
    cm = v.confusion_matrix.matrix  # (nc+1) x (nc+1), last = background
    json.dump({"names": NAMES + ["background"], "matrix": cm.tolist()},
              open(os.path.join(OUT, f"confusion_{tag}.json"), "w"), indent=1)
    print(f"confusion {tag} saved")

MODELS = {"base": os.path.join(W, "w11_base_s0", "weights", "best.pt"),
          "a7":   os.path.join(W, "w11_a7_s0", "weights", "best.pt")}

def conf_only():
    y_clean = make_yaml(build_cond("clean"))
    for tag, w in MODELS.items():
        run_conf(w, y_clean, tag)

def main():
    results = {}
    models = MODELS
    for cond in CONDS:
        print(f"=== {cond} ===", flush=True)
        imdir = build_cond(cond)
        y = make_yaml(imdir)
        results[cond] = {}
        for tag, w in models.items():
            results[cond][tag] = run_val(w, y)
            print(f"  {tag}: mAP50={results[cond][tag]['map50']}  mAP={results[cond][tag]['map']}", flush=True)
        json.dump(results, open(os.path.join(OUT, "robustness.json"), "w"), indent=1)
    y_clean = make_yaml(build_cond("clean"))
    for tag, w in models.items():
        run_conf(w, y_clean, tag)
    print("saved", os.path.join(OUT, "robustness.json"))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "conf":
        conf_only()
    else:
        main()
