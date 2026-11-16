"""Per-run mean movement of finished dynamic runs (read-only). Usage: python check_move_seeds.py <experiment prefix> [name filter]"""
import glob
import json
import sys

import numpy as np

import visualize as V

exp = sys.argv[1]
flt = sys.argv[2] if len(sys.argv) > 2 else ""
for f in sorted(glob.glob(f"results/runs/{exp}__*{flt}*.json")):
    pre = f.replace("\\", "/").split("/")[-1][:-5]
    meta = json.load(open(f, encoding="utf-8"))
    m = meta["config"]["layout"]["dynamic"]["method"]
    if m == "fixed":
        continue
    try:
        r, c1, w = V.load_run(pre)
    except SystemExit:
        continue
    c0 = V.grid_coords(r["config"])
    d = np.linalg.norm(c1 - c0, axis=1)
    print(f"{pre[:46]:46s} {m:8s} best_epoch {meta['metrics']['best_epoch']:2d} open {meta['config']['layout']['dynamic']['open_epoch']:2d} mean move {d.mean():.2f} moved>0.05: {int((d > 0.05).sum()):3d}")
