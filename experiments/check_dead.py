"""Are there neurons with (almost) no connections / no movement?  And modularity Q / small-world sigma of finished runs. read-only."""
import collections
import glob
import json

import numpy as np

import visualize as V

for pre in ("e02_requested__position_full__s1", "e02_requested__swap_full__s1", "e01_example_dynamic__relayout_d3__s1"):
    r, c1, w = V.load_run(pre)
    c0 = V.grid_coords(r["config"])
    a = np.abs(w)
    np.fill_diagonal(a, 0)
    s = a.sum(0) + a.sum(1)
    d = np.linalg.norm(c1 - c0, axis=1)
    print(f"{pre[:36]:36s} weak neurons (strength<0.2): {int((s < 0.2).sum())} | barely moved (<0.05): {int((d < 0.05).sum())}/100 | nonzero weights {float((a > 1e-3).mean()):.2f}")

g = collections.defaultdict(list)
for f in glob.glob("results/runs/e02_requested__*.json"):
    r = json.load(open(f, encoding="utf-8"))
    st = r.get("structure") or {}
    g[f.replace("\\", "/").split("__")[1]].append((st.get("Q"), st.get("n_modules"), st.get("sigma"), st.get("edges_active_offdiag"), r["metrics"]["test"]["acc_all"]))
print("\nname               Q(modularity) modules  sigma(small-world)  active edges  test acc")
for k, v in sorted(g.items()):
    if not any(x[0] is not None for x in v):
        continue
    f_ = lambda i: np.nanmean([x[i] for x in v if x[i] is not None])
    print(f"{k:18s} {f_(0):8.3f} {f_(1):10.1f} {f_(2):14.2f} {f_(3):14.0f} {f_(4):10.3f}")
