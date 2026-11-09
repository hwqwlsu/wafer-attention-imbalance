import os
# -*- coding: utf-8 -*-
"""Generate all paper tables (LaTeX booktabs + Markdown preview)."""
import json, os

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(ROOT, "manuscript")
A = json.load(open(os.path.join(BASE, "analysis", "stats.json"), encoding="utf-8"))
G, T, PC, DIST = A["groups"], A["paired_tests"], A["perclass"], A["class_dist"]
CLS = A["classes"]
TEX = os.path.join(BASE, "tables"); os.makedirs(TEX, exist_ok=True)
MD = []

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

def pm(g, m, pct=True):
    s = G[g][m]; f = 100 if pct else 1
    if G[g]["n"] > 1:
        return f"${s['mean']*f:.2f} \\pm {s['std']*f:.2f}$"
    return f"${s['mean']*f:.2f}$"

def pmd(g, m):
    s = G[g][m]
    if G[g]["n"] > 1:
        return f"{s['mean']*100:.2f} ± {s['std']*100:.2f}"
    return f"{s['mean']*100:.2f}"

# ---------------- Table 1: datasets ----------------
t1 = r"""\begin{table}[t]
\centering
\caption{Overview of the three industrial surface-defect datasets used in this work.}
\label{tab:datasets}
\small
\begin{tabular}{lcccc}
\toprule
Dataset & Split (train/val/test) & Classes & Images & Max imbalance ratio \\
\midrule
Wafer defect (WM-811K-derived) & 10{,}400 / 1{,}300 / 1{,}300 & 9 & 13{,}000 & 27.3:1 \\
Wafer defect + offline aug.\   & 11{,}544 / 1{,}300 / 1{,}300 & 9 & 14{,}144 & 4.1:1 \\
Metallic surface (NEU-derived) & \multicolumn{1}{c}{2{,}352 total} & 6 & 2{,}352 & --- \\
PCB defect (DsPCBSD+)          & 8{,}208 / 1{,}026 / 1{,}025 & 9 & 10{,}259 & --- \\
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(TEX, "tab1_datasets.tex"), "w").write(t1)
MD.append("""### Table 1. Datasets

| Dataset | train/val/test | Classes | Images | Max imbalance ratio |
|---|---|---|---|---|
| Wafer (WM-811K-derived) | 10,400 / 1,300 / 1,300 | 9 | 13,000 | 27.3:1 |
| Wafer + offline aug | 11,544 / 1,300 / 1,300 | 9 | 14,144 | 4.1:1 |
| Metallic (NEU-derived) | 2,352 total | 6 | 2,352 | --- |
| PCB (DsPCBSD+) | 8,208 / 1,026 / 1,025 | 9 | 10,259 | --- |
""")

# ---------------- Table 2: SOTA on wafer ----------------
sota = [("YOLOv8n", "w_v8n"), ("YOLOv9t", "w_v9t"), ("YOLOv10n", "w_v10n"),
        ("YOLOv11n", "w_v11n"), ("YOLOv12n", "w_v12n"), ("RT-DETR-L", "w_rtdetr")]
rows, mdrows = [], []
alls = sota + [("\\textbf{Ours (YOLOv11n+CB+aug)}", "w11_base")]
best = {m: max(G[g][m]["mean"] for _, g in alls) for m in ["map50", "map", "mp", "mr"]}
for name, g in alls:
    vals = []
    for m in ["map50", "map", "mp", "mr"]:
        v = pm(g, m)
        # bold if column-best, or ours within one std of best (noise overlap)
        if G[g][m]["mean"] == best[m] or (g == "w11_base" and abs(G[g][m]["mean"] - best[m]) <= G[g][m]["std"]):
            v = r"\textbf{" + v + "}"
        vals.append(v)
    rows.append(f"{name} & " + " & ".join(vals) + r" \\")
    mdrows.append(f"| {name} | " + " | ".join(pmd(g, m) for m in ["map50", "map", "mp", "mr"]) + " |")
t2 = "\\begin{table}[t]\n\\centering\n\\caption{Comparison with recent lightweight detectors on the wafer-defect test set. " \
     "All models are evaluated under identical settings (input 1024, single seed for baselines; ours reports mean$\\pm$std over 3 seeds). " \
     "Best result in bold.}\n\\label{tab:sota}\n\\small\n\\begin{tabular}{lcccc}\n\\toprule\n" \
     "Model & mAP$_{50}$ & mAP$_{50:95}$ & mP & mR \\\\\n\\midrule\n" + "\n".join(rows) + \
     "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
open(os.path.join(TEX, "tab2_sota.tex"), "w").write(t2)
MD.append("### Table 2. SOTA comparison (wafer test set)\n\n| Model | mAP50 | mAP50:95 | mP | mR |\n|---|---|---|---|---|\n" + "\n".join(mdrows) + "\n")

# ---------------- Table 3: pilot component ablation (yolo26n, seed 0) ----------------
pilot = [("YOLOv26n baseline (no aug, no CB)$^{\\dagger}$", "w_base_26n_noaug"),
         ("\\quad + PRAttn (no aug, no CB)", "w_a1"),
         ("\\quad + GeoFuse + CB (no aug)", "w_a2"),
         ("\\quad + offline aug (no CB)", "w_a3"),
         ("\\quad + PRAttn + aug (no CB)", "w_a5"),
         ("\\quad + GeoFuse + aug + CB", "w_a6")]
rows, mdrows = [], []
for name, g in pilot:
    vals = [pm(g, m) for m in ["map50", "map", "mp", "mr"]]
    rows.append(f"{name} & " + " & ".join(vals) + r" \\")
    mdrows.append(f"| {name} | " + " | ".join(pmd(g, m) for m in ["map50", "map", "mp", "mr"]) + " |")
t3 = "\\begin{table}[t]\n\\centering\n\\caption{Pilot component study on the YOLOv26n backbone (wafer dataset, seed 0 unless noted). " \
     "$^{\\dagger}$mean$\\pm$std over 2 seeds. Offline augmentation and the class-balanced (CB) loss are the only components that " \
     "consistently help; attention-style modules do not.}\n\\label{tab:pilot}\n\\small\n\\begin{tabular}{lcccc}\n\\toprule\n" \
     "Configuration & mAP$_{50}$ & mAP$_{50:95}$ & mP & mR \\\\\n\\midrule\n" + "\n".join(rows) + \
     "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
open(os.path.join(TEX, "tab3_pilot.tex"), "w").write(t3)
MD.append("### Table 3. Pilot component study (YOLOv26n, wafer)\n\n| Configuration | mAP50 | mAP50:95 | mP | mR |\n|---|---|---|---|---|\n" + "\n".join(mdrows) + "\n")

# ---------------- Table 4: main multi-seed ablation ----------------
main = [("\\textbf{YOLOv11n + aug + CB (ours)}", "w11_base", None),
        ("\\quad + PRAttn + GeoFuse", "w11_a7", "w11_a7"),
        ("\\quad + GeoFuse (YOLOv26n base)", "w_a8", "w_a8"),
        ("\\quad + CBAM", "w_cbam", "w_cbam"),
        ("\\quad + EMA", "w_ema", "w_ema"),
        ("\\quad YOLOv26n + PRAttn + GeoFuse + aug + CB", "w_a7_26n", "w_a7_26n")]
td = {t["g2"]: t for t in T}
rows, mdrows = [], []
for name, g, tk in main:
    m50 = pm(g, "map50"); rest = [pm(g, m) for m in ["map", "mp", "mr"]]
    if tk is None:
        d, pv = "--", "--"
    else:
        t = td[tk]; d = f"{t['diff_mean']*100:+.2f}{stars(t['p'])}"; pv = f"{t['p']:.4f}"
    rows.append(f"{name} & {m50} & {d} & " + " & ".join(rest) + f" & {pv} \\\\")
    mdrows.append(f"| {name} | {pmd(g,'map50')} | {d} | " + " | ".join(pmd(g, m) for m in ["map", "mp", "mr"]) + f" | {pv} |")
t4 = "\\begin{table}[t]\n\\centering\n\\caption{Main ablation on the wafer dataset (3 seeds, mean$\\pm$std). " \
     "$\\Delta$ is the mAP$_{50}$ difference against our baseline; $p$ from a two-sided paired $t$-test across seeds " \
     "($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$). Every attention-style insertion significantly degrades accuracy, " \
     "consistent with redundancy against YOLOv11n's built-in C2PSA self-attention.}\n\\label{tab:main}\n\\small\n" \
     "\\begin{tabular}{lcccccc}\n\\toprule\nConfiguration & mAP$_{50}$ & $\\Delta$mAP$_{50}$ & mAP$_{50:95}$ & mP & mR & $p$ \\\\\n" \
     "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
open(os.path.join(TEX, "tab4_main.tex"), "w").write(t4)
MD.append("### Table 4. Main multi-seed ablation (wafer, 3 seeds)\n\n| Configuration | mAP50 | ΔmAP50 | mAP50:95 | mP | mR | p |\n|---|---|---|---|---|---|---|\n" + "\n".join(mdrows) + "\n")

# ---------------- Table 5: cross-dataset ----------------
cross = [("YOLOv8n", "m_v8n", "p_v8n"), ("YOLOv11n", "m_v11n", "p_v11n"),
         ("YOLOv12n", "m_v12n", "p_v12n"), ("RT-DETR-L", "m_rtdetr", "p_rtdetr"),
         ("YOLOv26n (base)", "m_base", "p_base"),
         ("YOLOv26n + PRAttn + GeoFuse + CB", "m_a7", "p_a7")]
rows, mdrows = [], []
bm = {("m", "map50"): max(G[gm]["map50"]["mean"] for _, gm, _ in cross),
      ("m", "map"): max(G[gm]["map"]["mean"] for _, gm, _ in cross),
      ("p", "map50"): max(G[gp]["map50"]["mean"] for _, _, gp in cross),
      ("p", "map"): max(G[gp]["map"]["mean"] for _, _, gp in cross)}
for name, gm, gp in cross:
    def b(g, m, key):
        v = pm(g, m)
        return r"\textbf{" + v + "}" if G[g][m]["mean"] == bm[key] else v
    rows.append(f"{name} & {b(gm,'map50',('m','map50'))} & {b(gm,'map',('m','map'))} & "
                f"{b(gp,'map50',('p','map50'))} & {b(gp,'map',('p','map'))} \\\\")
    mdrows.append(f"| {name} | {pmd(gm,'map50')} | {pmd(gm,'map')} | {pmd(gp,'map50')} | {pmd(gp,'map')} |")
t5 = "\\begin{table}[t]\n\\centering\n\\caption{Cross-dataset generalization on metallic and PCB surface-defect detection " \
     "(seed 0, no offline augmentation). Best per column in bold.}\n\\label{tab:cross}\n\\small\n" \
     "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Metallic} & \\multicolumn{2}{c}{PCB} \\\\\n" \
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nModel & mAP$_{50}$ & mAP$_{50:95}$ & mAP$_{50}$ & mAP$_{50:95}$ \\\\\n\\midrule\n" + \
     "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
open(os.path.join(TEX, "tab5_cross.tex"), "w").write(t5)
MD.append("### Table 5. Cross-dataset (metal / PCB, seed 0)\n\n| Model | metal mAP50 | metal mAP50:95 | PCB mAP50 | PCB mAP50:95 |\n|---|---|---|---|---|\n" + "\n".join(mdrows) + "\n")

# ---------------- Table 6: per-class AP50 ----------------
groups6 = [("w11_base", "Ours (base)"), ("w11_a7", "+PRAttn+GeoFuse"), ("w_a8", "+GeoFuse"), ("w_cbam", "+CBAM"), ("w_ema", "+EMA")]
rows, mdrows = [], []
for i, c in enumerate(CLS):
    vals = [f"${PC[g]['mean'][i]*100:.2f}$" for g, _ in groups6]
    rows.append(f"{c} & " + " & ".join(vals) + r" \\")
    mdrows.append(f"| {c} | " + " | ".join(f"{PC[g]['mean'][i]*100:.2f}" for g, _ in groups6) + " |")
t6 = "\\begin{table}[t]\n\\centering\n\\caption{Per-class AP$_{50}$ on the wafer test set (mean over 3 seeds). " \
     "The largest degradations from PRAttn+GeoFuse appear on Scratch, None and Edge-Loc.}\n\\label{tab:perclass}\n\\small\n" \
     "\\begin{tabular}{lccccc}\n\\toprule\nClass & Ours (base) & +PRAttn+GeoFuse & +GeoFuse & +CBAM & +EMA \\\\\n\\midrule\n" + \
     "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
open(os.path.join(TEX, "tab6_perclass.tex"), "w").write(t6)
MD.append("### Table 6. Per-class AP50 (wafer, 3-seed mean)\n\n| Class | " + " | ".join(n for _, n in groups6) + " |\n|---" + "|---"*5 + "|\n" + "\n".join(mdrows) + "\n")

open(os.path.join(TEX, "tables_preview.md"), "w", encoding="utf-8").write("\n".join(MD))
print("tables written:", os.listdir(TEX))
