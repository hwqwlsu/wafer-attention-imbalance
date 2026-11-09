# -*- coding: utf-8 -*-
# Dataset overview figure: 3 rows (wafer / metallic / PCB) x 4 columns, GT boxes drawn.
# Selection rules:
#   wafer    -> one canonical sample per requested class (largest single box, cleanest pattern)
#   metallic -> one sample per requested class, favouring the most prominent (largest) annotation
#   PCB      -> 2 sparsest images + 2 densest images
# Output: paper/submission/figs/fig0_datasets.{pdf,png}  (PNG exported at 800 dpi)
import os, glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WA = os.environ.get("WAFER_ROOT", "datasets/wafer")
ME = os.environ.get("METAL_ROOT", "datasets/metal")
PCB = os.environ.get("PCB_ROOT", "datasets/pcb")

DATASETS = [
    dict(
        row="Wafer (WM-811K-derived)",
        img_dir=os.path.join(WA, "train", "images"),
        lab_dir=os.path.join(WA, "train", "labels"),
        names=['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'None', 'Random', 'Scratch'],
        mode="per_class",
        want=['Center', 'Edge-Ring', 'Scratch', 'Donut'],
        # hand-picked: the clearest examples of each pattern
        pin=['wafer_image_index_255159_png', 'wafer_image_index_100574_png',
             'wafer_image_index_136626_png', 'wafer_image_index_105569_png'],
    ),
    dict(
        row="Metallic (NEU-derived)",
        img_dir=os.path.join(ME, "train", "images"),
        lab_dir=os.path.join(ME, "train", "labels"),
        names=['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches'],
        mode="per_class",
        want=['crazing', 'inclusion', 'pitted_surface', 'scratches'],
    ),
    dict(
        row="PCB (DsPCBSD+)",
        img_dir=os.path.join(PCB, "images", "train"),
        lab_dir=os.path.join(PCB, "labels", "train"),
        names=['SH', 'SP', 'SC', 'OP', 'MB', 'HB', 'CS', 'CFO', 'BMFO'],
        mode="sparse_dense",
    ),
]

PALETTE = [(255, 56, 56), (255, 157, 51), (255, 225, 52), (87, 219, 99),
           (52, 199, 235), (62, 90, 235), (196, 84, 235), (235, 84, 170),
           (140, 140, 140), (255, 255, 255)]


def read_yolo(lab_path):
    boxes = []
    if not os.path.isfile(lab_path):
        return boxes
    with open(lab_path) as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 5:
                boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def draw(img, boxes, names):
    h, w = img.shape[:2]
    out = img.copy()
    fs = max(0.55, 0.0016 * w)      # label height ~3.5-4% of the image width
    thk = 2 if w < 1200 else 3
    bt = max(2, int(w / 400.0))
    for c, xc, yc, bw, bh in boxes:
        x0 = int((xc - bw / 2) * w); y0 = int((yc - bh / 2) * h)
        x1 = int((xc + bw / 2) * w); y1 = int((yc + bh / 2) * h)
        col = PALETTE[c % len(PALETTE)]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, bt)
        name = names[c] if c < len(names) else str(c)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fs, thk)
        pad = max(2, th // 5)
        if y0 - th - 2 * pad - 4 >= 0:  # label above the box
            ty = y0 - pad
            cv2.rectangle(out, (x0, ty - th - pad), (x0 + tw + 2 * pad, ty + pad), col, -1)
            cv2.putText(out, name, (x0 + pad, ty - pad), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thk, cv2.LINE_AA)
        else:                            # box touches the top edge -> label inside
            ty = y0 + th + 2 * pad
            cv2.rectangle(out, (x0 + bt, y0 + bt), (x0 + tw + 3 * pad, y0 + th + 3 * pad), col, -1)
            cv2.putText(out, name, (x0 + 2 * pad, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thk, cv2.LINE_AA)
    return out


def index_dataset(ds):
    """[(img_path, boxes, {class: max_area})] for every labelled image."""
    labs = sorted(glob.glob(os.path.join(ds["lab_dir"], "*.txt")))
    out = []
    for lp in labs:
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = None
        for ext in (".jpg", ".png", ".jpeg", ".bmp"):
            cand = os.path.join(ds["img_dir"], stem + ext)
            if os.path.isfile(cand):
                ip = cand; break
        if ip is None:
            continue
        boxes = read_yolo(lp)
        if not boxes:
            continue
        area = {}
        for c, _, _, bw, bh in boxes:
            area[c] = max(area.get(c, 0.0), bw * bh)
        out.append((ip, boxes, area))
    return out


def pick_per_class(ds, recs):
    chosen, used = [], set()
    for want in ds["want"]:
        cid = ds["names"].index(want)
        cands = [r for r in recs if cid in r[2] and r[0] not in used]
        if not cands:
            continue
        # the annotation should be a visible sub-region (not the whole frame),
        # prominent enough to be recognisable, with few other boxes around it
        good = [r for r in cands if 0.02 <= r[2][cid] <= 0.55]
        pool = good if good else cands
        pool.sort(key=lambda r: (-r[2][cid], len(r[1])))
        chosen.append((pool[0][0], pool[0][1])); used.add(pool[0][0])
    # top up if a class had no sample
    for ip, boxes, _ in sorted(recs, key=lambda r: len(r[1])):
        if len(chosen) >= 4:
            break
        if ip not in used:
            chosen.append((ip, boxes)); used.add(ip)
    return chosen[:4]


def pick_sparse_dense(recs):
    pool = [r for r in recs if len(r[1]) >= 1]
    pool.sort(key=lambda r: len(r[1]))
    sparse = pool[:2]
    # medium density keeps the picture readable while still showing crowded parts
    mid = [r for r in pool if 6 <= len(r[1]) <= 16]
    dense = sorted(mid, key=lambda r: -len(r[1]))[:2] if mid else pool[-2:]
    out, used = [], set()
    for ip, boxes, _ in sparse + dense:
        if ip in used:
            continue
        out.append((ip, boxes)); used.add(ip)
    return out[:4]


fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.6))
for r, ds in enumerate(DATASETS):
    recs = index_dataset(ds)
    if ds.get("pin"):
        samples = []
        for stem in ds["pin"]:
            for ip, boxes, _ in recs:
                if os.path.basename(ip).startswith(stem):
                    samples.append((ip, boxes)); break
    else:
        samples = pick_per_class(ds, recs) if ds["mode"] == "per_class" else pick_sparse_dense(recs)
    print(ds["row"], "->", [(os.path.basename(p), len(b)) for p, b in samples])
    assert len(samples) == 4, (ds["row"], len(samples))
    for c, (ip, boxes) in enumerate(samples):
        img = cv2.imdecode(np.fromfile(ip, dtype=np.uint8), cv2.IMREAD_COLOR)
        # upscale small sources so the drawn labels stay crisp at print resolution
        h0, w0 = img.shape[:2]
        if min(h0, w0) < 800:
            s = 800.0 / min(h0, w0)
            interp = cv2.INTER_NEAREST if ds["mode"] == "per_class" and "Wafer" in ds["row"] else cv2.INTER_CUBIC
            img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=interp)
        img = draw(img, boxes, ds["names"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = axes[r, c]
        ax.imshow(img, interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if c == 0:
            ax.set_ylabel(ds["row"], fontsize=13, rotation=90, labelpad=12)
fig.tight_layout(rect=[0, 0.005, 1, 1])
out = os.path.join(ROOT, "figures", "fig0_datasets")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=800)
# small preview for quick on-screen inspection
from PIL import Image
im = Image.open(out + ".png")
im.resize((1400, int(1400 * im.size[1] / im.size[0])), Image.LANCZOS).save(
    os.path.join(os.path.dirname(out), "fig0_datasets_preview.png"))
print("saved", out + ".pdf", im.size)
