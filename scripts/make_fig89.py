import os
# -*- coding: utf-8 -*-
"""Fig 8 (robustness curves) and Fig 9 (confusion matrices).
Data: analysis/robustness.json, analysis/confusion_{base,a7}.json (real runs)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(ROOT, "manuscript")
R = json.load(open(os.path.join(BASE, "analysis", "robustness.json"), encoding="utf-8"))
CB = json.load(open(os.path.join(BASE, "analysis", "confusion_base.json"), encoding="utf-8"))
CA = json.load(open(os.path.join(BASE, "analysis", "confusion_a7.json"), encoding="utf-8"))
FIG = os.path.join(BASE, "figs"); os.makedirs(FIG, exist_ok=True)

rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 900, "savefig.dpi": 900, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_BASE, C_A7 = "#2166AC", "#B2182B"

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("saved", name)

# ---------- Fig 8: robustness to photometric perturbations ----------
CONDS = ["clean", "blur_s1", "blur_s2", "noise_s10", "noise_s25", "down50", "down25"]
XLAB = ["clean", "blur\n$\\sigma$=1", "blur\n$\\sigma$=2", "noise\n$\\sigma$=10",
        "noise\n$\\sigma$=25", "down\n50%", "down\n25%"]
b50 = [R[c]["base"]["map50"] * 100 for c in CONDS]
a50 = [R[c]["a7"]["map50"] * 100 for c in CONDS]
b95 = [R[c]["base"]["map"] * 100 for c in CONDS]
a95 = [R[c]["a7"]["map"] * 100 for c in CONDS]
x = np.arange(len(CONDS))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, b, a, ttl in [(axes[0], b50, a50, "(a) mAP$_{50}$"),
                      (axes[1], b95, a95, "(b) mAP$_{50:95}$")]:
    ax.plot(x, b, "-o", color=C_BASE, ms=4, lw=1.2, label="Ours (baseline)")
    ax.plot(x, a, "-s", color=C_A7, ms=4, lw=1.2, label="+PRAttn+GeoFuse")
    for xi, (bv, av) in enumerate(zip(b, a)):
        gap = bv - av
        lab = f"{gap:+.1f}"
        if abs(gap) >= 4.0:
            # large gap: place the label in the clear space between the curves
            y_txt = (bv + av) / 2.0
        else:
            # small gap: place it above the higher curve with a white backing
            y_txt = max(bv, av) + 0.35
        ax.annotate(lab, (xi, y_txt), ha="center", va="center",
                    fontsize=6.3, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.9))
    ax.set_xticks(x); ax.set_xticklabels(XLAB)
    ax.set_title(ttl, fontsize=8.5)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    lo = min(b + a); ax.set_ylim(lo - 1.5, max(b + a) + 1.5)
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend(frameon=False, loc="lower left")
save(fig, "fig8_robustness")

# ---------- Fig 9: confusion matrices (row-normalised) ----------
def norm_cm(cm):
    m = np.array(cm, dtype=float)
    rs = m.sum(1, keepdims=True)
    return np.divide(m, rs, out=np.zeros_like(m), where=rs > 0)

names = CB["names"]
nb, na = norm_cm(CB["matrix"]), norm_cm(CA["matrix"])
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
for ax, m, ttl in [(axes[0], nb, "(a) Ours (w11_base)"),
                   (axes[1], na, "(b) +PRAttn+GeoFuse (w11_a7)")]:
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_title(ttl, fontsize=8.5)
    ax.set_xlabel("Predicted"); 
    for i in range(len(names)):
        for j in range(len(names)):
            v = m[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                        color="white" if v > 0.55 else "black")
axes[0].set_ylabel("True")
fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
save(fig, "fig9_confusion")
