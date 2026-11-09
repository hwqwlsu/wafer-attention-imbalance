import os
# -*- coding: utf-8 -*-
"""Rebuild Fig. 6: ground truth | rebalanced baseline | +PRAttn+GeoFuse.

Requires the seed-0 weights of the paper models (w11_base_s0, w11_a7_s0).
Usage:
    python make_fig6_clean.py --base <best_base.pt> --plug <best_a7.pt>
Everything else (sample selection, boxes) is computed, nothing is hand-edited.
"""
import argparse, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle

REPO = os.environ.get("ULTRALYTICS_ROOT", "ultralytics")
sys.path.insert(0, REPO)

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(ROOT, "manuscript")
FIG = os.path.join(BASE, "figs")
DATA = os.environ.get("WAFER_AUG_ROOT", "datasets/wafer_aug")
SPLIT = os.path.join(DATA, "test")
NAMES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full',
         'None', 'Random', 'Scratch']
TARGETS = ["Loc", "Scratch", "Random"]

rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
})
C_GT, C_PR = "#1F5FA8", "#C1272D"


def read_labels(txt):
    rows = []
    with open(txt) as f:
        for line in f:
            v = line.split()
            if len(v) >= 5:
                rows.append((int(float(v[0])), *[float(x) for x in v[1:5]]))
    return rows


def pick_samples():
    """One image per target class, preferably with a single annotated box."""
    labs = {os.path.basename(p)[:-4]: read_labels(p)
            for p in sorted(glob.glob(os.path.join(SPLIT, "labels", "*.txt")))}
    picks = {}
    for want in TARGETS:
        cid = NAMES.index(want)
        strict = [k for k, v in labs.items() if len(v) == 1 and v[0][0] == cid]
        loose = [k for k, v in labs.items() if any(r[0] == cid for r in v) and len(v) <= 3]
        cand = strict or loose
        cand.sort(key=lambda k: (len(labs[k]), k))
        picks[want] = (cand[0], labs[cand[0]])
    return picks


def draw(ax, img, boxes, color, show_name=True, fs=6.2):
    ax.imshow(img)
    h, w = img.shape[:2]
    for cls, cx, cy, bw, bh in boxes:
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        x1, y1 = max(3, x1), max(3, y1)
        x2, y2 = min(w - 3, x2), min(h - 3, y2)   # 3px margin so full-image boxes stay visible
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               lw=1.1, edgecolor=color, zorder=3))
        if show_name:
            above = y1 > 14
            ax.text(x1 + 2, y1 - 3 if above else y1 + 3, NAMES[cls], color=color,
                    fontsize=fs, ha="left", va="bottom" if above else "top", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.75))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("#4D4D4D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--plug", required=True)
    ap.add_argument("--out", default="fig6_qualitative")
    a = ap.parse_args()

    from ultralytics import YOLO
    m_base, m_plug = YOLO(a.base), YOLO(a.plug)
    picks = pick_samples()

    nrow, ncol = len(TARGETS), 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.32 * nrow + 0.32))
    heads = ["Ground truth", "Ours (YOLOv11n + aug + CB)", "+PRAttn + GeoFuse"]
    for j, h in enumerate(heads):
        axes[0, j].set_title(h, fontsize=8.2, pad=6)

    for i, cls_name in enumerate(TARGETS):
        key, gts = picks[cls_name]
        img_path = os.path.join(SPLIT, "images", key + ".jpg")
        img = plt.imread(img_path)
        if img.ndim == 3:
            img = img[:, :, :3]
        draw(axes[i, 0], img, gts, C_GT)
        for j, m in enumerate([m_base, m_plug], start=1):
            r = m.predict(img_path, imgsz=416, conf=0.25, verbose=False)[0]
            b = r.boxes
            boxes = []
            for xyxy, c in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = xyxy; w, h = img.shape[1], img.shape[0]
                boxes.append((c, ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                              (x2 - x1) / w, (y2 - y1) / h))
            draw(axes[i, j], img, boxes, C_PR)
        axes[i, 0].set_ylabel(cls_name, fontsize=8.2, labelpad=4)

    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    fig.savefig(os.path.join(FIG, a.out + ".png"))
    fig.savefig(os.path.join(FIG, a.out + ".pdf"))
    plt.close(fig)
    print("saved", a.out, "| samples:", {k: v[0] for k, v in picks.items()})


if __name__ == "__main__":
    main()
