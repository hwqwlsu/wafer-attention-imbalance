# -*- coding: utf-8 -*-
"""Statistical analysis for wafer defect detection paper.
Reads the 44-run ledger (+ w_a7_s0 from .bak), computes group stats,
paired significance tests, per-class AP analysis, dataset class distribution.
Outputs JSON + CSV to paper/analysis/.
"""
import json, os, math, csv
import numpy as np
from scipy import stats

ROOT = os.environ.get("RUNS_ROOT", "runs/wafer")
OUT = 'results'
os.makedirs(OUT, exist_ok=True)

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'None', 'Random', 'Scratch']

# ---------- load ledger ----------
runs = {}
with open(os.path.join(ROOT, "results.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # fix the known corrupted line 44 (map50w_ema_s0 splice) -> take the w_ema_s0 part
        if '"map50w_ema_s0"' in line:
            line = line.replace('0.9486487301339436, "map50w_ema_s0"', '0.9486487301339436, "map50"')
            # reconstruct: original line had w_ema_s0 fields after splice; parse carefully
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            # handle corrupted w_cbam_s2/w_ema_s0 line: extract w_ema_s0 json manually
            continue
        runs[r["name"]] = r

# w_ema_s0 corrupted line fallback: hard-parse from raw text if missing
if "w_ema_s0" not in runs:
    with open(os.path.join(ROOT, "results.jsonl"), encoding="utf-8") as f:
        raw = f.read()
    i = raw.find('"map50w_ema_s0"')
    seg = raw[i:raw.find("\n", i)]
    seg = seg.replace('"map50w_ema_s0"', '"name": "w_ema_s0", "map50"')
    seg = '{"' + seg.split('{"', 1)[-1] if not seg.startswith('{') else seg
    runs["w_ema_s0"] = json.loads(seg)

# add w_a7_s0 from .bak
with open(os.path.join(ROOT, "results.jsonl.bak"), encoding="utf-8") as f:
    for line in f:
        if '"name": "w_a7_s0"' in line:
            runs["w_a7_s0"] = json.loads(line)

print(f"loaded {len(runs)} runs")

# ---------- groups ----------
GROUPS = {
    # main multi-seed (wafer aug+CB)
    "w11_base": ["w11_base_s0", "w11_base_s1", "w11_base_s2"],
    "w11_a7":   ["w11_a7_s0", "w11_a7_s1", "w11_a7_s2"],
    "w_a8":     ["w_a8_s0", "w_a8_s1", "w_a8_s2"],
    "w_cbam":   ["w_cbam_s0", "w_cbam_s1", "w_cbam_s2"],
    "w_ema":    ["w_ema_s0", "w_ema_s1", "w_ema_s2"],
    # yolo26n pilot multi-seed
    "w_a7_26n": ["w_a7_s0", "w_a7_s1", "w_a7_s2"],
    "w_base_26n_noaug": ["w_base_s1", "w_base_s2"],
    # SOTA wafer single-seed
    "w_v8n": ["w_v8n_s0"], "w_v9t": ["w_v9t_s0"], "w_v10n": ["w_v10n_s0"],
    "w_v11n": ["w_v11n_s0"], "w_v12n": ["w_v12n_s0"], "w_rtdetr": ["w_rtdetr_s0"],
    # yolo26n pilot single-seed exploration
    "w_a1": ["w_a1_s0"], "w_a2": ["w_a2_s0"], "w_a3": ["w_a3_s0"],
    "w_a5": ["w_a5_s0"], "w_a6": ["w_a6_s0"],
    # cross-dataset
    "m_v8n": ["m_v8n_s0"], "m_v11n": ["m_v11n_s0"], "m_v12n": ["m_v12n_s0"],
    "m_base": ["m_base_s0"], "m_a7": ["m_a7_s0"], "m_rtdetr": ["m_rtdetr_s0"],
    "p_v8n": ["p_v8n_s0"], "p_v11n": ["p_v11n_s0"], "p_v12n": ["p_v12n_s0"],
    "p_base": ["p_base_s0"], "p_a7": ["p_a7_s0"], "p_rtdetr": ["p_rtdetr_s0"],
}

def group_stats(names):
    vals = {m: np.array([runs[n][m] for n in names]) for m in ["map50", "map", "mp", "mr"]}
    out = {"n": len(names), "runs": names}
    for m, v in vals.items():
        out[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "values": [float(x) for x in v]}
        if len(v) > 1:
            se = v.std(ddof=1) / math.sqrt(len(v))
            tcrit = stats.t.ppf(0.975, len(v) - 1)
            out[m]["ci95"] = float(tcrit * se)
    return out

G = {g: group_stats(ns) for g, ns in GROUPS.items()}

# ---------- paired tests: w11_base vs each attention module (per-seed pairing) ----------
def paired(g1, g2):
    a = np.array([runs[n]["map50"] for n in GROUPS[g1]])
    b = np.array([runs[n]["map50"] for n in GROUPS[g2]])
    d = b - a
    t, p = stats.ttest_rel(b, a)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("inf")
    se = d.std(ddof=1) / math.sqrt(len(d))
    tcrit = stats.t.ppf(0.975, len(d) - 1)
    return {"g1": g1, "g2": g2, "diff_mean": float(d.mean()), "diff_ci95": float(tcrit * se),
            "t": float(t), "df": len(d) - 1, "p": float(p), "cohens_dz": float(dz),
            "diffs": [float(x) for x in d]}

TESTS = [paired("w11_base", g) for g in ["w11_a7", "w_a8", "w_cbam", "w_ema"]]
TESTS.append(paired("w11_base", "w_a7_26n"))

# ---------- per-class AP50: w11_base vs w11_a7 (mean over seeds) ----------
perclass = {}
for g in ["w11_base", "w11_a7", "w_a8", "w_cbam", "w_ema"]:
    arr = np.array([runs[n]["per_class_ap50"] for n in GROUPS[g]])
    perclass[g] = {"mean": [float(x) for x in arr.mean(axis=0)],
                   "std": [float(x) for x in arr.std(axis=0, ddof=1)]}

# ---------- dataset class distribution (original train labels) ----------
def count_classes(label_dir):
    cnt = np.zeros(9, dtype=int)
    for fn in os.listdir(label_dir):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, fn)) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    cnt[int(ln.split()[0])] += 1
    return cnt.tolist()

WAFER = os.environ.get("WAFER_ROOT", "datasets/wafer")
WAFER_AUG = os.environ.get("WAFER_AUG_ROOT", "datasets/wafer_aug")

dist = {
    "train_orig": count_classes(os.path.join(WAFER, "train", "labels")),
    "valid": count_classes(os.path.join(WAFER, "valid", "labels")),
    "test": count_classes(os.path.join(WAFER, "test", "labels")),
    "train_aug": count_classes(os.path.join(WAFER_AUG, "train", "labels")),
}

# ---------- save ----------
out = {"classes": CLASSES, "groups": G, "paired_tests": TESTS, "perclass": perclass, "class_dist": dist}
with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

# CSV: multi-seed summary
with open(os.path.join(OUT, "multiseed_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["group", "n", "mAP50_mean", "mAP50_std", "mAP50_ci95", "mAP_mean", "mAP_std",
                "mP_mean", "mR_mean"])
    for g in ["w11_base", "w11_a7", "w_a8", "w_cbam", "w_ema", "w_a7_26n", "w_base_26n_noaug"]:
        s = G[g]
        w.writerow([g, s["n"], f"{s['map50']['mean']:.6f}", f"{s['map50']['std']:.6f}",
                    f"{s['map50'].get('ci95', 0):.6f}", f"{s['map']['mean']:.6f}", f"{s['map']['std']:.6f}",
                    f"{s['mp']['mean']:.6f}", f"{s['mr']['mean']:.6f}"])

with open(os.path.join(OUT, "paired_tests.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["baseline", "variant", "diff_mean", "diff_ci95", "t", "df", "p", "cohens_dz"])
    for t in TESTS:
        w.writerow([t["g1"], t["g2"], f"{t['diff_mean']:.6f}", f"{t['diff_ci95']:.6f}",
                    f"{t['t']:.4f}", t["df"], f"{t['p']:.6f}", f"{t['cohens_dz']:.4f}"])

# ---------- print summary ----------
print("\n=== multi-seed groups (mAP50) ===")
for g in ["w11_base", "w11_a7", "w_a8", "w_cbam", "w_ema", "w_a7_26n", "w_base_26n_noaug"]:
    s = G[g]["map50"]
    print(f"{g:22s} n={G[g]['n']}  {s['mean']*100:.2f} ± {s['std']*100:.2f}")

print("\n=== paired t-tests vs w11_base (mAP50) ===")
for t in TESTS:
    print(f"{t['g2']:12s} diff={t['diff_mean']*100:+.2f}pt  t({t['df']})={t['t']:+.3f}  p={t['p']:.4f}  dz={t['cohens_dz']:+.2f}")

print("\n=== class distribution ===")
for k, v in dist.items():
    print(k, dict(zip(CLASSES, v)), "total:", sum(v))

print("\n=== per-class AP50 (w11_base mean) ===")
for i, c in enumerate(CLASSES):
    print(f"{c:10s} {perclass['w11_base']['mean'][i]*100:.2f}  (a7 delta {(perclass['w11_a7']['mean'][i]-perclass['w11_base']['mean'][i])*100:+.2f})")
