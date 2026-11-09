import os
# -*- coding: utf-8 -*-
"""Generate publication-quality figures (SCI Q2 standard).
Outputs PNG (600 dpi) + PDF to paper/figs/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(ROOT, "manuscript")
A = json.load(open(os.path.join(BASE, "analysis", "stats.json"), encoding="utf-8"))
G, T, PC, DIST, CLS = A["groups"], A["paired_tests"], A["perclass"], A["class_dist"], A["classes"]
FIG = os.path.join(BASE, "figs"); os.makedirs(FIG, exist_ok=True)

# ---------- global style ----------
rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C = {"base": "#2166AC", "neg": "#B2182B", "grey": "#7F7F7F",
     "palette": ["#2166AC", "#67A9CF", "#D1E5F0", "#FDDBC7", "#EF8A62", "#B2182B"]}

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("saved", name)

# ---------- Fig 2: class distribution original vs augmented ----------
orig = np.array(DIST["train_orig"]); aug = np.array(DIST["train_aug"])
x = np.arange(len(CLS)); w = 0.38
fig, ax = plt.subplots(figsize=(7.0, 2.6))
b1 = ax.bar(x - w/2, orig, w, label="Original train set (10,400 images)", color=C["base"], edgecolor="none")
b2 = ax.bar(x + w/2, aug, w, label="After offline minority oversampling (11,544 images)", color="#92C5DE", edgecolor="none")
for xi, v in zip(x - w/2, orig):
    ax.text(xi, v + 40, str(v), ha="center", va="bottom", fontsize=6.5)
for xi, v in zip(x + w/2, aug):
    if v != orig[np.where(x + w/2 == xi)[0][0]]:
        ax.text(xi, v + 40, str(v), ha="center", va="bottom", fontsize=6.5, color=C["neg"])
ax.set_xticks(x); ax.set_xticklabels(CLS, rotation=20, ha="right")
ax.set_ylabel("Number of defect instances")
ax.set_ylim(0, 3900)
ax.legend(frameon=False, loc="upper right")
ax.annotate("max imbalance 27.3 : 1\n(Edge-Loc 3,251 vs. Near-full 119)", xy=(5, 240), xytext=(5.6, 2200),
            fontsize=7.5, color=C["neg"], arrowprops=dict(arrowstyle="->", color=C["neg"], lw=0.8))
save(fig, "fig2_class_distribution")

# ---------- Fig 3: SOTA comparison ----------
sota = [("YOLOv8n", "w_v8n"), ("YOLOv9t", "w_v9t"), ("YOLOv10n", "w_v10n"),
        ("YOLOv11n", "w_v11n"), ("YOLOv12n", "w_v12n"), ("RT-DETR-L", "w_rtdetr"),
        ("Ours", "w11_base")]
items = [(n, G[g]["map50"]["mean"] * 100, G[g]["map50"]["std"] * 100 if G[g]["n"] > 1 else 0,
          G[g]["map"]["mean"] * 100) for n, g in sota]
items.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(3.6, 2.8))
y = np.arange(len(items))
cols = [C["neg"] if n == "Ours" else C["base"] for n, *_ in items]
ax.barh(y, [it[1] for it in items], height=0.62, color=cols, edgecolor="none", zorder=3)
ax.errorbar([it[1] for it in items], y, xerr=[it[2] for it in items], fmt="none",
            ecolor="black", elinewidth=0.9, capsize=2, zorder=4)
for yi, it in zip(y, items):
    ax.text(it[1] + 0.09, yi, f"{it[1]:.2f}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([it[0] for it in items])
ax.set_xlabel("mAP$_{50}$ on wafer test set (%)")
ax.set_xlim(91.5, 97)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
save(fig, "fig3_sota")

# ---------- Fig 4: main ablation error bars + significance ----------
order = [("w11_base", "Ours\n(YOLOv11n+aug+CB)"), ("w11_a7", "+PRAttn\n+GeoFuse"),
         ("w_a8", "+GeoFuse"), ("w_cbam", "+CBAM"), ("w_ema", "+EMA")]
means = [G[g]["map50"]["mean"] * 100 for g, _ in order]
stds = [G[g]["map50"]["std"] * 100 for g, _ in order]
td = {t["g2"]: t for t in T}
fig, ax = plt.subplots(figsize=(5.0, 3.0))
x = np.arange(len(order))
cols = [C["base"]] + [C["neg"]] * 4
bars = ax.bar(x, means, width=0.56, color=cols, edgecolor="none", zorder=3,
              yerr=stds, error_kw=dict(ecolor="black", elinewidth=0.9, capsize=2.5))
# individual seed points
for i, (g, _) in enumerate(order):
    vals = np.array(G[g]["map50"]["values"]) * 100
    ax.scatter(np.full(len(vals), i) + np.linspace(-0.13, 0.13, len(vals)), vals,
               s=7, color="black", alpha=0.65, zorder=5, linewidths=0)
# significance brackets vs baseline (staggered heights)
def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
for i, (g, _) in enumerate(order):
    if i == 0:
        continue
    p = td[g]["p"]
    yb = 96.28 + (4 - i) * 0.17
    ax.plot([0, 0, i, i], [yb, yb + 0.06, yb + 0.06, yb], lw=0.8, color="black", clip_on=False)
    ax.text(i / 2, yb + 0.07, star(p), ha="center", va="bottom", fontsize=8)
labels = ["Ours\n(v11n+aug+CB)", "+PRAttn\n+GeoFuse", "+GeoFuse", "+CBAM", "+EMA"]
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("mAP$_{50}$ (%)")
ax.set_ylim(94.0, 97.15)
ax.set_yticks(np.arange(94.0, 97.0, 0.5))
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
save(fig, "fig4_ablation")

# ---------- Fig 5: per-class AP50 heatmap ----------
mg = [("w11_base", "Ours (base)"), ("w11_a7", "+PRAttn+GeoFuse"), ("w_a8", "+GeoFuse"),
      ("w_cbam", "+CBAM"), ("w_ema", "+EMA")]
M = np.array([[PC[g]["mean"][i] * 100 for i in range(9)] for g, _ in mg])
fig, ax = plt.subplots(figsize=(7.0, 2.5))
im = ax.imshow(M, cmap="RdYlGn", vmin=86, vmax=100, aspect="auto")
ax.set_xticks(range(9)); ax.set_xticklabels(CLS, rotation=20, ha="right")
ax.set_yticks(range(5)); ax.set_yticklabels([n for _, n in mg], fontsize=7.5)
for i in range(5):
    for j in range(9):
        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                color="black")
cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
cb.set_label("AP$_{50}$ (%)", fontsize=7.5); cb.ax.tick_params(labelsize=7)
ax.set_xticks(np.arange(-0.5, 9, 1), minor=True); ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.grid(which="minor", color="white", lw=1.2)
ax.tick_params(which="both", length=0)
save(fig, "fig5_perclass_heatmap")

# ---------- Fig 7: negative-result analysis (2 panels) ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.35]})
# (a) paired seed slope chart base -> a7
b = np.array(G["w11_base"]["map50"]["values"]) * 100
a = np.array(G["w11_a7"]["map50"]["values"]) * 100
for i in range(3):
    ax1.plot([0, 1], [b[i], a[i]], "o-", color=C["grey"], lw=1.0, ms=3.5,
             markerfacecolor="white", markeredgewidth=0.9, alpha=0.8)
ax1.plot([0, 1], [b.mean(), a.mean()], "o-", color=C["neg"], lw=2.2, ms=5.5, zorder=5)
ax1.text(-0.05, b.mean(), f"mean {b.mean():.2f}", ha="right", va="center", fontsize=7, color=C["neg"])
ax1.text(1.05, a.mean(), f"mean {a.mean():.2f}", ha="left", va="center", fontsize=7, color=C["neg"])
ax1.set_xlim(-0.5, 1.5); ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Ours (base)", "+PRAttn+GeoFuse"])
ax1.set_ylabel("mAP$_{50}$ (%)")
ax1.set_ylim(94.6, 96.5)
ax1.set_title("(a) Paired per-seed comparison\n$\\Delta=-0.78$ pt, $t(2)=-25.2$, $p=0.0016$", fontsize=8)
# (b) per-class delta bars
d = (np.array(PC["w11_a7"]["mean"]) - np.array(PC["w11_base"]["mean"])) * 100
idx = np.argsort(d)
ax2.barh(np.arange(9), d[idx], height=0.6,
         color=[C["neg"] if v < -1 else "#EF8A62" if v < -0.5 else C["grey"] for v in d[idx]],
         edgecolor="none", zorder=3)
ax2.set_yticks(np.arange(9)); ax2.set_yticklabels([CLS[i] for i in idx], fontsize=7.5)
for i, v in enumerate(d[idx]):
    ax2.text(v - 0.06, i, f"{v:+.2f}", ha="right", va="center", fontsize=6.5)
ax2.axvline(0, color="black", lw=0.8)
ax2.set_xlabel("$\\Delta$AP$_{50}$ vs. baseline (pt)")
ax2.set_xlim(-3.2, 0.3)
ax2.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
ax2.set_title("(b) Per-class accuracy change", fontsize=8)
fig.tight_layout(w_pad=2.2)
save(fig, "fig7_negative_analysis")

print("ALL DONE")
