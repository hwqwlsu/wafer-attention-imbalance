# Class-balanced (CB) loss patch

`run_exp.py --cb` does not change the model; it computes effective-number class
weights from the training labels and exports them as a comma-separated list in
the `WAFER_CB_WEIGHTS` environment variable:

```python
beta = 0.999
eff = (1 - beta) / (1 - np.power(beta, np.maximum(counts, 1)))
w = eff * (nc / eff.sum())          # normalised so that mean(w) = 1
os.environ["WAFER_CB_WEIGHTS"] = ",".join(str(float(x)) for x in w)
```

An environment variable is used on purpose: it survives Windows
`multiprocessing.spawn`, which re-imports the module and would drop module-level
globals or monkeypatched classes.

The consumption side lives in the Ultralytics checkout used for the paper
(`ultralytics/utils/loss.py`, class `v8DetectionLoss`). The change weights the
binary cross-entropy classification term per class:

```python
# __init__, after self.bce = nn.BCEWithLogitsLoss(reduction="none")
_w = os.environ.get("WAFER_CB_WEIGHTS", "")
self.cb_w = (torch.tensor([float(x) for x in _w.split(",")], device=device)
             if _w else None)

# in the loss computation, where the BCE term is reduced
loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum()
if self.cb_w is not None:
    # target_scores is (batch, anchors, nc); weight each class channel
    wts = self.cb_w.view(1, 1, -1)
    loss_cls = (self.bce(pred_scores, target_scores.to(dtype)).sum(-1) * wts).sum()
```

Notes:

* Weighting the classification (BCE) term only, not the box regression or DFL
  terms, matches the "class-balanced loss" description in the paper.
* The weight vector is normalised to mean 1, so the effective learning rate on
  the classification head is unchanged; only the per-class distribution of the
  gradient changes.
* With the stock PyPI `ultralytics` package these lines are absent and
  `--cb` has no effect. Any Ultralytics 8.x release works after applying the
  patch; the exact version used for the reported runs is recorded in the
  paper's implementation section.
* The same weights are used at every epoch, i.e. they are static, computed once
  from the training split before training starts.
