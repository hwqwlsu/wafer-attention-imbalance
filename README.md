# Is Class Imbalance the Bottleneck? — code and per-seed results

Code, configurations and per-seed results for the manuscript

> **Is Class Imbalance the Bottleneck? A Controlled Re-evaluation of Attention,
> Rebalancing, and Robustness for Wafer Map Defect Detection**
> Wenquan Huang, Teng Li, Qing Cheng — School of Artificial Intelligence, Anhui University

The study is a *controlled re-evaluation*, not a new module: six real-time
detectors are trained under one protocol, four attention plug-ins are tested
over three seeds with paired tests, and per-class AP is used to test the
imbalance hypothesis. Everything needed to regenerate the numbers in the paper
(training runner, model configs, evaluation scripts, per-seed ledger and
statistics) is in this repository.

## Repository layout

```
run_exp.py            single-run trainer/evaluator (adds class-balanced loss)
run_queue.py          sequential job runner; skips runs already in the ledger
configs/              YOLO model YAMLs for the four attention variants
data/                 dataset YAMLs (paths are relative; see below)
scripts/              experiments, statistics, figures, table generation
results/              per-seed and aggregated results used in the paper
```

`results/` is the artefact behind the "per-seed results released" claim:

| file | content |
|---|---|
| `multiseed_summary.csv` | mean/std/95% CI of mAP50, mAP50:95, mP, mR per variant (n=3) |
| `paired_tests.csv` | paired t-test baseline vs each variant (t, df, p, Cohen's dz) |
| `perclass_tests.json` | per-class AP50 and per-class paired tests |
| `stats.json` | class distributions, group means, statistics bundle |
| `complexity.json` | parameters, FLOPs, latency per model |
| `latency.json` | round-robin latency benchmark |
| `robustness.json` | 7 photometric conditions x 2 models |
| `confusion_base.json`, `confusion_a7.json` | confusion matrices |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`ultralytics` must be the checkout that carries the class-balanced BCE term
described in [docs/cb_loss_patch.md](docs/cb_loss_patch.md). With the stock
package the `--cb` flag is silently ignored and results will differ.

## Data

The three datasets are **not** redistributed here; download them from their
original sources and point the YAML files at your copy.

| dataset | source | classes | used for |
|---|---|---|---|
| wafer maps (WM-811K-derived) | WM-811K, Wu et al. 2015; YOLO conversion by us | 9 | main benchmark |
| metallic surface (NEU-derived) | NEU-DET, YOLO export | 6 | cross-dataset |
| PCB (DsPCBSD+) | figshare DOI 10.6084/m9.figshare.24970329 | 9 | cross-dataset |

Expected layout (see `data/wafer.yaml`):

```
datasets/wafer/{train,valid,test}/{images,labels}
datasets/wafer_aug/{train,valid,test}/{images,labels}
```

Path overrides are read from the environment, so nothing needs editing:

| variable | meaning | default |
|---|---|---|
| `WAFER_ROOT` | original wafer split | `datasets/wafer` |
| `WAFER_AUG_ROOT` | split after offline minority synthesis | `datasets/wafer_aug` |
| `RUNS_ROOT` | training output tree / results ledger | `runs/wafer` |
| `ULTRALYTICS_ROOT` | path added to `sys.path` for the patched checkout | `ultralytics` |
| `WAFER_REPO` | repository root used by figure scripts | auto |

## Reproducing the main table

```bash
# baseline and the four plug-ins, three seeds each
python run_exp.py --model yolo11n.pt --data data/wafer_aug.yaml --cb \
                  --name w11_base --seed 0
python run_exp.py --model configs/yolo11n-a7-prattn-geofuse.yaml \
                  --pretrained yolo11n.pt --data data/wafer_aug.yaml --cb \
                  --name w11_a7 --seed 0
# ... repeat with --seed 1 and --seed 2, then aggregate
python scripts/analyze.py            # -> results/multiseed_summary.csv, paired_tests.csv
```

`run_queue.py jobfile.json` runs a list of such jobs and skips names already
present in `runs/wafer/results.jsonl`, so a queue can be restarted safely.

One JSON line per run is appended to the ledger:

```json
{"name": "w11_base", "seed": 0, "cb": true,
 "map50": 0.9609, "map": 0.8231, "mp": 0.9217, "mr": 0.9208,
 "per_class_ap50": [...]}
```

## Further experiments

```bash
python scripts/exp_robustness.py      # 7 perturbations, 2 models, confusion matrices
python scripts/exp_perclass_stats.py  # per-class paired tests
python scripts/exp_latency.py         # round-robin latency
python scripts/compute_complexity.py  # parameters / FLOPs
python scripts/make_fig89.py          # Fig. 8 and Fig. 9
python scripts/make_fig_cross.py      # cross-dataset figure
```

## Protocol notes

* Training: 100 epochs, 416x416, batch 16, cosine LR, identical hyper-parameters
  for every model including the baselines.
* Statistics: three seeds, paired t-test against the tuned baseline, Cohen's dz,
  plus per-class AP50. With df=2 the p-values are coarse; this is stated in the
  paper's Limitations.
* The cross-dataset tables are ranking experiments, not application benchmarks —
  the metallic release annotates five of six declared classes.

## Citation

```bibtex
@article{huang2026imbalance,
  title  = {Is Class Imbalance the Bottleneck? A Controlled Re-evaluation of
            Attention, Rebalancing, and Robustness for Wafer Map Defect Detection},
  author = {Huang, Wenquan and Li, Teng and Cheng, Qing},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Contact

Wenquan Huang — hwqwlsu@163.com
