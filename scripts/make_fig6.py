# -*- coding: utf-8 -*-
"""Fig 6: qualitative GT vs prediction, zoomed 2x2 wafer crops."""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(ROOT, "manuscript")
FIG = os.path.join(BASE, "figs")
SRC = os.environ.get("RUNS_ROOT", "runs/detect/runs/wafer/w_v11n_s0_test")

CROP = 896  # 2x2 cells of 448 px
pairs = []
for b in [1, 2]:
    gt = Image.open(os.path.join(SRC, f"val_batch{b}_labels.jpg")).convert("RGB").crop((0, 0, CROP, CROP))
    pr = Image.open(os.path.join(SRC, f"val_batch{b}_pred.jpg")).convert("RGB").crop((0, 0, CROP, CROP))
    pairs.append((gt, pr))

pad, lab_h = 16, 80
W = 2 * CROP + 3 * pad
H = 2 * CROP + 3 * pad + lab_h
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arialbd.ttf", 44)
except Exception:
    font = ImageFont.truetype("arial.ttf", 44)

d.text((pad + CROP // 2, pad + lab_h // 2), "Ground truth", anchor="mm", fill="black", font=font)
d.text((2 * pad + CROP + CROP // 2, pad + lab_h // 2), "Prediction (Ours)", anchor="mm", fill="black", font=font)
for r, (gt, pr) in enumerate(pairs):
    y = pad + lab_h + r * (CROP + pad)
    canvas.paste(gt, (pad, y))
    canvas.paste(pr, (2 * pad + CROP, y))
    # thin separator between the two columns
    d.rectangle([pad + CROP + pad // 2, y, pad + CROP + pad // 2 + 2, y + CROP], fill=(200, 200, 200))

out = os.path.join(FIG, "fig6_qualitative.png")
canvas.save(out, dpi=(600, 600))
print("saved", out, canvas.size)
