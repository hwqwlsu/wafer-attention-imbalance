# -*- coding: utf-8 -*-
"""Fig 1: system pipeline diagram (matplotlib, journal style).
Fig 6: GT vs prediction qualitative comparison (PIL compose)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "manuscript")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE, LBLUE, RED, GREY, GREEN = "#2166AC", "#D1E5F0", "#B2182B", "#7F7F7F", "#1A9850"

# ================= Fig 1: pipeline =================
fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

def box(x, y, w, h, text, fc="white", ec="black", fs=7.2, lw=0.9, style="round,pad=0.25", tc="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, mutation_scale=1.0)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.25)

def arrow(x1, y1, x2, y2, color="black", lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=9, lw=lw, color=color, shrinkA=1, shrinkB=1))

# --- stage 1: data ---
box(1, 22, 16, 12, "WM-811K wafer\ndefect maps\n(9 classes)", fc=LBLUE)
box(1, 6, 16, 10, "Class imbalance\n27.3 : 1", fc="white", ec=RED, tc=RED, fs=6.8)
arrow(9, 22, 9, 16.5, color=RED, lw=0.9)

# --- stage 2: offline aug ---
box(21, 22, 17, 12, "Offline minority\noversampling\n(+ copy-paste aug)", fc=LBLUE)
box(21, 6, 17, 10, "Rebalanced train set\n11,544 images (4.1 : 1)", fc="white", ec=GREEN, tc=GREEN, fs=6.8)
arrow(29.5, 22, 29.5, 16.5, color=GREEN, lw=0.9)
arrow(17.5, 28, 20.5, 28)

# --- stage 3: model ---
box(42.5, 14, 22, 20, "YOLOv11n detector\n\nBackbone + C2PSA\nPAN neck\nDecoupled head", fc=LBLUE)
arrow(38.5, 28, 42, 28)
# CB loss feeding training
box(42.5, 1.5, 22, 8, "Class-Balanced (CB) loss\neffective number re-weighting", fc="#FDDBC7", ec=RED, fs=6.8)
arrow(53.5, 9.8, 53.5, 13.5, color=RED, lw=1.0)

# --- ablated modules (dashed, marked as negative) ---
box(68.5, 30, 27, 9, "Attention plug-ins (ablated):\nPRAttn / CBAM / EMA / GeoFuse", fc="white", ec=GREY, fs=6.6, style="round,pad=0.25")
for ln in ax.lines: pass
ax.patches[-1].set_linestyle((0, (4, 2)))
arrow(65, 34, 68, 34, color=GREY, lw=0.9, style="-|>")
ax.text(82, 27.5, "significant accuracy drop (p < 0.05)\nredundant with built-in C2PSA", fontsize=6.4,
        color=RED, ha="center", va="top", style="italic")
arrow(82, 29.7, 82, 28.2, color=RED, lw=0.8)

# --- stage 4: eval ---
box(70, 8, 25, 12, "Evaluation\n3-seed mean ± std\npaired t-test / per-class AP", fc=LBLUE, fs=6.9)
arrow(65, 22, 74, 20.2)

# stage labels
for x, t in [(9, "Data"), (29.5, "Rebalance"), (53.5, "Model + Loss"), (82, "Analysis")]:
    ax.text(x, 39.5, t, ha="center", fontsize=8, fontweight="bold", color="#333333")

fig.savefig(os.path.join(FIG, "fig1_pipeline.png"))
fig.savefig(os.path.join(FIG, "fig1_pipeline.pdf"))
plt.close(fig)
print("saved fig1_pipeline")

# ================= Fig 6: GT vs pred =================
from PIL import Image, ImageDraw, ImageFont

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.environ.get("RUNS_ROOT", "runs/detect/runs/wafer/w_v11n_s0_test")
rows = []
for i in [0, 1, 2]:
    gt = Image.open(os.path.join(SRC, f"val_batch{i}_labels.jpg")).convert("RGB")
    pr = Image.open(os.path.join(SRC, f"val_batch{i}_pred.jpg")).convert("RGB")
    rows.append((gt, pr))

w, h = rows[0][0].size
pad = 10; lab_h = 34
canvas = Image.new("RGB", (2*w + 3*pad, 3*h + 4*pad + lab_h), "white")
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except Exception:
    font = ImageFont.load_default()

# column headers
d.text((pad + w//2, pad + lab_h//2), "Ground truth", anchor="mm", fill="black", font=font)
d.text((2*pad + w + w//2, pad + lab_h//2), "Ours (YOLOv11n + aug + CB)", anchor="mm", fill="black", font=font)
for r, (gt, pr) in enumerate(rows):
    y = pad + lab_h + r*(h + pad)
    canvas.paste(gt, (pad, y))
    canvas.paste(pr, (2*pad + w, y))
out = os.path.join(FIG, "fig6_qualitative.png")
canvas.save(out, dpi=(600, 600))
print("saved fig6_qualitative", canvas.size)
