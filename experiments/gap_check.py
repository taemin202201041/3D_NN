"""Train vs validation vs test accuracy at the best epoch (is the annotation input limited by over-fitting?). read-only.
Usage: python gap_check.py <experiment> <variant> [<experiment> <variant> ...]"""
import glob
import json
import sys

import numpy as np

a = sys.argv[1:]
print(f"{'run':36s} {'n':>2s} {'best_ep':>7s} {'train':>6s} {'val':>6s} {'test':>6s}")
for e, v in zip(a[::2], a[1::2]):
    rows = []
    for f in glob.glob(f"results/runs/{e}__{v}__s*.json"):
        r = json.load(open(f, encoding="utf-8"))
        if "metrics" not in r:
            continue
        h, be = r["history"], int(r["metrics"]["best_epoch"])
        tr = h["accuracy"][be - 1] if "accuracy" in h else np.nan
        va = h["val_accuracy"][be - 1] if "val_accuracy" in h else np.nan
        rows.append((be, tr, va, r["metrics"]["test"]["acc_all"]))
    if rows:
        x = np.array(rows, float).mean(0)
        print(f"{(e[:4] + ' ' + v)[:36]:36s} {len(rows):2d} {x[0]:7.1f} {x[1]:6.3f} {x[2]:6.3f} {x[3]:6.3f}")
