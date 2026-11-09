# -*- coding: utf-8 -*-
"""Unified experiment runner for the wafer-defect study.

Supports: arbitrary model cfg/pt, dataset yaml, seed, class-balanced (CB)
loss flag (innovation 2 loss part), pretrained weight transfer, auto test-set
evaluation after training, and appends one JSON line per run to
runs/wafer/results.jsonl for later statistics.

Examples:
  python run_exp.py --model yolo26n.pt --data wafer.yaml --name base_s0 --seed 0
  python run_exp.py --model configs/yolo26n-a2-geofuse.yaml --pretrained yolo26n.pt --cb --data wafer.yaml --name a2_s0
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import ultralytics.utils.loss as U

# CB weights are passed to the trainer through the WAFER_CB_WEIGHTS env var,
# which v8DetectionLoss reads when building its BCE term. Env vars survive
# Windows multiprocessing.spawn, unlike module globals / monkeypatched classes.


def class_weights(data_yaml, nc):
    import numpy as np
    import yaml
    d = yaml.safe_load(open(data_yaml, encoding='utf-8'))
    root = d['path']
    counts = np.zeros(nc)
    lbl_sub = d['train'].replace('images', 'labels')
    for f in glob.glob(os.path.join(root, lbl_sub, '*.txt')):
        for line in open(f, encoding='utf-8'):
            p = line.split()
            if p:
                counts[int(p[0])] += 1
    beta = 0.999
    eff = (1 - beta) / (1 - np.power(beta, np.maximum(counts, 1)))
    w = eff * (nc / eff.sum())
    return w, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='')
    ap.add_argument('--data', default='')
    ap.add_argument('--name', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--pretrained', default='')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=416)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cb', action='store_true')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--cache', default='')
    ap.add_argument('--patience', type=int, default=100)
    ap.add_argument('--resfile', default='')
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    import yaml as _yaml

    if args.resume:
        cands = glob.glob(os.path.join('runs', '**', 'runs', 'wafer', args.name, 'weights', 'last.pt'), recursive=True)
        cands += glob.glob(os.path.join('runs', 'wafer', args.name, 'weights', 'last.pt'))
        if not cands:
            raise SystemExit('[resume] last.pt not found for ' + args.name)
        args.model = cands[0]
        print('[resume] from', args.model)

    cbw = None
    if args.cb:
        import numpy as np
        nc = _yaml.safe_load(open(args.data, encoding='utf-8'))['nc']
        w, counts = class_weights(args.data, nc)
        os.environ['WAFER_CB_WEIGHTS'] = ','.join(str(float(x)) for x in w)
        print('[CB] class counts:', counts.tolist())
        print('[CB] weights:', [round(float(x), 3) for x in w])

    model = YOLO(args.model)
    if args.resume:
        model.train(resume=True)
    else:
        if args.pretrained and os.path.exists(args.pretrained):
            try:
                model.load(args.pretrained)
                print('[init] pretrained weights transferred from', args.pretrained)
            except Exception as e:
                print('[init] pretrained transfer skipped:', e)

        model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            seed=args.seed, cos_lr=True, workers=args.workers, patience=args.patience,
            project='runs/wafer', name=args.name, exist_ok=True, verbose=False,
            cache=args.cache if args.cache else False,
        )

    # test-set evaluation with the best weights
    # trainer save_dir is authoritative (project dir gains a task/ level, e.g. runs/detect/runs/wafer/<name>)
    save_dir = str(getattr(getattr(model, 'trainer', None), 'save_dir', '') or '')
    best = os.path.join(save_dir, 'weights', 'best.pt') if save_dir else ''
    if not os.path.exists(best):
        # fallback: search both possible roots
        cands = glob.glob(os.path.join('runs', '**', 'runs', 'wafer', args.name, 'weights', 'best.pt'), recursive=True)
        cands += glob.glob(os.path.join('runs', 'wafer', args.name, 'weights', 'best.pt'))
        best = cands[0] if cands else best
    res = None
    if os.path.exists(best):
        m2 = YOLO(best)
        res = m2.val(data=args.data, split='test', imgsz=args.imgsz,
                     project='runs/wafer', name=args.name + '_test', exist_ok=True, verbose=False)

    out = {
        'name': args.name, 'model': args.model, 'data': args.data,
        'seed': args.seed, 'cb': args.cb,
    }
    if res is not None:
        out.update({
            'map50': float(res.box.map50), 'map': float(res.box.map),
            'mp': float(res.box.mp), 'mr': float(res.box.mr),
            'per_class_ap50': [float(x) for x in res.box.ap50],
        })
    os.makedirs('runs/wafer', exist_ok=True)
    outfile = args.resfile if args.resfile else 'runs/wafer/results.jsonl'
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    with open(outfile, 'a', encoding='utf-8') as f:
        f.write(json.dumps(out, ensure_ascii=False) + '\n')
    print('[done]', json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
