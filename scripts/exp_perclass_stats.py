import os
# -*- coding: utf-8 -*-
"""Per-class paired t-tests across 3 seeds, from the results.jsonl ledger."""
import json
import numpy as np
from scipy import stats

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEDGER = os.path.join(os.environ.get("RUNS_ROOT", "runs/wafer"), "results.jsonl")
OUT    = os.path.join(ROOT, "results", "perclass_tests.json")

GROUPS = {
    "w11_base": ["w11_base_s0", "w11_base_s1", "w11_base_s2"],
    "w11_a7":   ["w11_a7_s0", "w11_a7_s1", "w11_a7_s2"],
    "w_a8":     ["w_a8_s0", "w_a8_s1", "w_a8_s2"],
    "w_cbam":   ["w_cbam_s0", "w_cbam_s1", "w_cbam_s2"],
    "w_ema":    ["w_ema_s0", "w_ema_s1", "w_ema_s2"],
}
CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','None','Random','Scratch']

recs = {}
for line in open(LEDGER, encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        recs[r["name"]] = [100.0 * float(x) for x in r["per_class_ap50"]]

out = {}
for variant in ["w11_a7", "w_a8", "w_cbam", "w_ema"]:
    out[variant] = {}
    for i, cls in enumerate(CLASSES):
        a = np.array([recs[r][i] for r in GROUPS["w11_base"]])
        b = np.array([recs[r][i] for r in GROUPS[variant]])
        t, p = stats.ttest_rel(b, a)
        d = b - a
        dz = float(d.mean() / (d.std(ddof=1) + 1e-12))
        out[variant][cls] = {"base": [round(x, 2) for x in a], "var": [round(x, 2) for x in b],
                             "delta": round(float(d.mean()), 2), "t": round(float(t), 2),
                             "p": round(float(p), 4), "dz": round(dz, 2)}

json.dump(out, open(OUT, "w"), indent=1)
for v in out:
    sig = [(c, out[v][c]["delta"], out[v][c]["p"]) for c in CLASSES if out[v][c]["p"] < 0.05]
    print(v, "->", sig)
print("saved", OUT)
