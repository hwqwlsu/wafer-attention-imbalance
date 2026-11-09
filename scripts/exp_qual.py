# -*- coding: utf-8 -*-
"""Qualitative figure set:
  A) fig10_qualgrid : 3x3 grid, one clean detection per class (base model)
  B) fig11_failures : 1x4 typical failure cases (GT dashed green, TP cyan,
                      missed GT dashed red, FP solid red)
  C) fig12_noisecollapse : 2x2 same image under clean/noise_s25 x base/a7
All predictions are real: base = w11_base_s0/best.pt, a7 = w11_a7_s0/best.pt.
"""
import os
import tempfile, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.environ.get("ULTRALYTICS_ROOT", "ultralytics"))

import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from ultralytics import YOLO

REPO = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("WAFER_AUG_ROOT", "datasets/wafer_aug")
IMG, LBL = os.path.join(ROOT, "test", "images"), os.path.join(ROOT, "test", "labels")
ROB = os.path.join(tempfile.gettempdir(), "wafer_robust")
W = os.environ.get("RUNS_ROOT", "runs/detect/runs/wafer")
FIG = os.path.join(REPO, "figures")
ANA = os.path.join(REPO, "results", "qual_preds.json")
NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
         "Near-full", "None", "Random", "Scratch"]
IMGSZ, CONF = 416, 0.25

rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                 "font.size": 7, "figure.dpi": 600, "savefig.dpi": 600,
                 "savefig.bbox": "tight", "pdf.fonttype": 42})
C_GT, C_TP, C_FN, C_FP = "#2CA02C", "#17BECF", "#D62728", "#D62728"

# ---------- inference ----------
def infer(model, img_dir, names):
    out = {}
    for n in names:
        p = os.path.join(img_dir, n)
        r = model.predict(p, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        out[n] = {"xyxy": r.boxes.xyxy.cpu().numpy().tolist(),
                  "conf": r.boxes.conf.cpu().numpy().tolist(),
                  "cls":  r.boxes.cls.cpu().numpy().astype(int).tolist()}
    return out

def load_gt(name):
    p = os.path.join(LBL, os.path.splitext(name)[0] + ".txt")
    gt = []
    if os.path.exists(p):
        for ln in open(p):
            v = ln.split()
            if len(v) >= 5:
                gt.append([int(v[0])] + [float(x) for x in v[1:5]])
    return gt  # [cls, cx, cy, w, h] normalised

def gt_xyxy(g, W=416, H=416):
    c, cx, cy, w, h = g
    return [c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    i = iw * ih
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

def match(pred, gts):
    """pred: dict xyxy/conf/cls; gts: list of [cls,cx,cy,w,h].
    Return tp[list of (pi,gi,conf)], fn[list of gi], fp[list of pi]."""
    P = [(pred["xyxy"][i], pred["cls"][i], pred["conf"][i]) for i in range(len(pred["cls"]))]
    G = [gt_xyxy(g) for g in gts]
    used_p, used_g, tp = set(), set(), []
    pairs = sorted([(iou(p[0], g[1:]), pi, gi, p[2]) for pi, p in enumerate(P)
                    for gi, g in enumerate(G) if p[1] == g[0]], reverse=True)
    for iouv, pi, gi, cf in pairs:
        if iouv < 0.5 or pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); tp.append((pi, gi, cf))
    return tp, [gi for gi in range(len(G)) if gi not in used_g], \
           [pi for pi in range(len(P)) if pi not in used_p]

def draw(ax, img, pred=None, gts=None, tp=(), fn=(), fp=(), title="",
         draw_pred=True, gt_label=False):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    WHT = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.6)
    if gts is not None:
        for gi, g in enumerate(gts):
            c, x1, y1, x2, y2 = gt_xyxy(g)
            missed = gi in fn
            col = C_FN if missed else C_GT
            ax.add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                         ec=col, lw=1.3, ls="--" if missed else (0, (4, 2))))
            if missed:
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, "missed", color=C_FN,
                        fontsize=6, ha="center", va="center", weight="bold", bbox=WHT)
            elif gt_label:
                ax.text(x1 + 2, y1 + 2, NAMES[c], color=C_GT, fontsize=5.5,
                        va="top", bbox=WHT)
    if draw_pred and pred is not None:
        tp_p = {p for p, _, _ in tp}
        for i in range(len(pred["cls"])):
            x1, y1, x2, y2 = pred["xyxy"][i]
            good = i in tp_p
            if (not good) and i not in fp:
                continue
            col = C_TP if good else C_FP
            ax.add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                         ec=col, lw=1.3, ls="-" if good else ":"))
            ax.text(x1 + 2, y1 + 2, f"{NAMES[pred['cls'][i]]} {pred['conf'][i]:.2f}",
                    color=col, fontsize=5.5, va="top", bbox=WHT)
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.axis("off")

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig); print("saved", name)

# ---------- run ----------
if os.path.exists(ANA):
    PRED = json.load(open(ANA))
else:
    files = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".png", ".jpg")))
    PRED = {"base": infer(YOLO(os.path.join(W, "w11_base_s0", "weights", "best.pt")), IMG, files)}
    PRED["a7"] = infer(YOLO(os.path.join(W, "w11_a7_s0", "weights", "best.pt")), IMG, files)
    json.dump(PRED, open(ANA, "w"))
    print("preds cached:", ANA)

files = sorted(PRED["base"].keys())

# ---------- A: 3x3 clean grid, one class each ----------
picked = {}
for c, cname in enumerate(NAMES):
    best, bs = None, -1
    for f in files:
        gts = load_gt(f)
        if len(gts) != 1 or gts[0][0] != c:
            continue
        tp, fn, fp = match(PRED["base"][f], gts)
        if tp and not fn and not fp and tp[0][2] > bs:
            best, bs = f, tp[0][2]
    picked[c] = (best, bs)
    print(f"grid {cname:10s} -> {best} conf={bs:.3f}")

fig, axes = plt.subplots(3, 3, figsize=(7.0, 7.0))
for c, ax in enumerate(axes.flat):
    f, cf = picked[c]
    img = cv2.imread(os.path.join(IMG, f))
    tp, fn, fp = match(PRED["base"][f], load_gt(f))
    draw(ax, img, PRED["base"][f], load_gt(f), tp, fn, fp,
         title=f"{NAMES[c]}  (conf {cf:.2f})")
leg = [mpatches.Patch(fill=False, ec=C_GT, ls=(0, (4, 2)), lw=1.3, label="ground truth"),
       mpatches.Patch(fill=False, ec=C_TP, ls="-", lw=1.3, label="prediction (ours)")]
fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, fontsize=7)
fig.tight_layout(pad=0.4, rect=(0, 0.02, 1, 1))
save(fig, "fig10_qualgrid")

# ---------- B: 1x4 failure panel (base model), single-instance images only ----------
cases = []
for f in files:
    gts = load_gt(f)
    if len(gts) != 1:
        continue
    pr = PRED["base"][f]
    tp, fn, fp = match(pr, gts)
    n_pred = len(pr["cls"])
    if fn and n_pred <= 2:
        cases.append(("FN", f, fn[0], NAMES[gts[fn[0]][0]], n_pred))  # clean-ish miss
    elif tp and fp:
        big = [pi for pi in fp if pr["conf"][pi] > 0.4]
        if big:
            cases.append(("FP", f, big[0], NAMES[pr["cls"][big[0]]],
                          pr["conf"][big[0]]))                     # extra false alarm
cases.sort(key=lambda x: x[4])  # fewest predictions first -> cleanest panel
fn_el = next(x for x in cases if x[0] == "FN" and x[3] == "Edge-Loc")
fn_sc = next(x for x in cases if x[0] == "FN" and x[3] == "Scratch")
fn_lc = next((x for x in cases if x[0] == "FN" and x[3] == "Loc"), None)
fp_hi = max((x for x in cases if x[0] == "FP"), key=lambda x: x[4])
sel = [s for s in [fn_el, fn_sc, fn_lc, fp_hi] if s]
print("failure cases:", [(s[0], s[3], s[1]) for s in sel])

fig, axes = plt.subplots(1, len(sel), figsize=(7.0, 1.95))
if len(sel) == 1:
    axes = [axes]
for ax, s in zip(axes, sel):
    kind, f, idx, cname = s[0], s[1], s[2], s[3]
    img = cv2.imread(os.path.join(IMG, f))
    gts = load_gt(f)
    tp, fn, fp = match(PRED["base"][f], gts)
    if kind == "FN":
        draw(ax, img, PRED["base"][f], gts, tp, [idx], fp, title=f"missed {cname}")
    else:
        draw(ax, img, PRED["base"][f], gts, tp, [], [idx], title=f"false alarm {cname}")
leg = [mpatches.Patch(fill=False, ec=C_GT, ls=(0, (4, 2)), lw=1.3, label="ground truth"),
       mpatches.Patch(fill=False, ec=C_FN, ls="--", lw=1.3, label="missed detection"),
       mpatches.Patch(fill=False, ec=C_TP, ls="-", lw=1.3, label="correct prediction"),
       mpatches.Patch(fill=False, ec=C_FP, ls=":", lw=1.3, label="false alarm")]
fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.4, rect=(0, 0.035, 1, 1))
save(fig, "fig11_failures")

# ---------- C: 2x2 noise collapse on the same image ----------
clean_dir = os.path.join(ROB, "clean", "images")
noisy_dir = os.path.join(ROB, "noise_s25", "images")
mB = YOLO(os.path.join(W, "w11_base_s0", "weights", "best.pt"))
mA = YOLO(os.path.join(W, "w11_a7_s0", "weights", "best.pt"))
CHOICE = os.path.join(os.path.dirname(ANA), "qual_c_choice.json")
if os.path.exists(CHOICE):
    best_f = json.load(open(CHOICE))["file"]
else:
    common = sorted(set(os.listdir(clean_dir)) & set(os.listdir(noisy_dir)))
    best_f, best_gap = None, -1
    for f in common:
        gts = load_gt(f)
        if len(gts) != 1:
            continue
        pb = mB.predict(os.path.join(noisy_dir, f), imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        pa = mA.predict(os.path.join(noisy_dir, f), imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        cb = pb.boxes.conf.max().item() if len(pb.boxes) else 0
        ca = pa.boxes.conf.max().item() if len(pa.boxes) else 0
        if cb > 0.5 and cb - ca > best_gap:
            best_f, best_gap = f, cb - ca
        if best_gap > 0.8:
            break  # good enough; keep the scan fast
    json.dump({"file": best_f, "gap": best_gap}, open(CHOICE, "w"))
print("noise collapse image:", best_f)

cells = [("clean + base", clean_dir, mB), ("clean + PRAttn+GeoFuse", clean_dir, mA),
         ("noise $\\sigma$=25 + base", noisy_dir, mB), ("noise $\\sigma$=25 + PRAttn+GeoFuse", noisy_dir, mA)]
fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.2))
for ax, (ttl, d, mdl) in zip(axes.flat, cells):
    p = os.path.join(d, best_f)
    r = mdl.predict(p, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    pr = {"xyxy": r.boxes.xyxy.cpu().numpy().tolist(),
          "conf": r.boxes.conf.cpu().numpy().tolist(),
          "cls": r.boxes.cls.cpu().numpy().astype(int).tolist()}
    gts = load_gt(best_f)
    tp, fn, fp = match(pr, gts)
    draw(ax, cv2.imread(p), pr, gts, tp, fn, fp, title=ttl)
fig.tight_layout(pad=0.4)
save(fig, "fig12_noisecollapse")
print("done")
