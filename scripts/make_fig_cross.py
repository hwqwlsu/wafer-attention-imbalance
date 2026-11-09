# -*- coding: utf-8 -*-
"""Cross-dataset figure: metallic vs PCB, six detectors + per-class PCB detail.
All numbers are read from runs/wafer/results.jsonl (no estimates)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

LEDGER = os.path.join(os.environ.get("RUNS_ROOT", "runs/wafer"), "results.jsonl")
BASE = os.environ.get("FIG_ROOT", "figures")
FIG = os.path.join(BASE, "figs"); os.makedirs(FIG, exist_ok=True)

rows = [json.loads(l) for l in open(LEDGER, encoding="utf-8") if l.strip()]
R = {r["name"]: r for r in rows}

rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_METAL, C_PCB, C_BASE, C_PLUG = "#92C5DE", "#2166AC", "#2166AC", "#B2182B"

# (label, metal run, pcb run)
MODELS = [
    ("YOLOv8n", "m_v8n_s0", "p_v8n_s0"),
    ("YOLOv11n", "m_v11n_s0", "p_v11n_s0"),
    ("YOLOv12n", "m_v12n_s0", "p_v12n_s0"),
    ("RT-DETR-L", "m_rtdetr_s0", "p_rtdetr_s0"),
    ("YOLOv26n", "m_base_s0", "p_base_s0"),
    ("YOLOv26n\n+PRAttn+GeoFuse", "m_a7_s0", "p_a7_s0"),
]
metal = np.array([R[m]["map50"] * 100 for _, m, _ in MODELS])
pcb = np.array([R[p]["map50"] * 100 for _, _, p in MODELS])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6),
                              gridspec_kw={"width_ratios": [1.0, 1.15]})

# ---------------- (a) six detectors on both datasets ----------------
y = np.arange(len(MODELS)); h = 0.34
ax1.barh(y - h / 2, metal, h, color=C_METAL, edgecolor="none", label="Metallic", zorder=3)
ax1.barh(y + h / 2, pcb, h, color=C_PCB, edgecolor="none", label="PCB", zorder=3)
for yi, v in zip(y - h / 2, metal):
    ax1.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=6.3)
for yi, v in zip(y + h / 2, pcb):
    ax1.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=6.3)
ax1.set_yticks(y); ax1.set_yticklabels([n for n, _, _ in MODELS], fontsize=7.3)
ax1.invert_yaxis()
ax1.set_xlim(42, 99); ax1.set_xlabel(r"AP$_{50}$ (%)")
ax1.set_title("(a) Six detectors, seed 0", fontsize=8.5)
ax1.legend(frameon=False, loc="lower right", handlelength=1.0, borderpad=0.1,
           labelspacing=0.3)
ax1.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)

# ---------------- (b) PCB per-class, baseline vs plug-in ----------------
PCB_CLS = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]
b = np.array(R["p_base_s0"]["per_class_ap50"]) * 100
a = np.array(R["p_a7_s0"]["per_class_ap50"]) * 100
yy = np.arange(len(PCB_CLS))
for i in yy:
    ax2.plot([b[i], a[i]], [i, i], "-", color="#BFBFBF", lw=2.0, zorder=2)
ax2.scatter(b, yy, s=22, color=C_BASE, zorder=3, label="YOLOv26n baseline", linewidths=0)
ax2.scatter(a, yy, s=22, color=C_PLUG, zorder=4, label="+PRAttn+GeoFuse", linewidths=0)
for i in yy:
    d = a[i] - b[i]
    ax2.text(max(a[i], b[i]) + 1.3, i, f"{d:+.1f}", va="center", fontsize=6.3,
             color=C_PLUG if d < 0 else "#4D4D4D")
ax2.set_yticks(yy); ax2.set_yticklabels(PCB_CLS, fontsize=7.3)
ax2.invert_yaxis()
ax2.set_xlim(63, 102); ax2.set_xlabel(r"AP$_{50}$ (%)")
ax2.set_title(r"(b) PCB per class, $\Delta$ aggregate $=+0.05$ pt", fontsize=8.5)
ax2.legend(frameon=False, loc="upper left", handletextpad=0.3, borderpad=0.1,
           labelspacing=0.3)
ax2.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)

fig.tight_layout(w_pad=1.6)
fig.savefig(os.path.join(FIG, "fig_cross_dataset.png"))
fig.savefig(os.path.join(FIG, "fig_cross_dataset.pdf"))
plt.close(fig)
print("saved fig_cross_dataset")
print("metal  ", np.round(metal, 2))
print("pcb    ", np.round(pcb, 2))
