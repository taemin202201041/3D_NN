"""Test metrics per variant for one experiment (mean over seeds, n = number of finished runs). Usage: python tab.py e25_scale"""
import collections
import glob
import json
import sys

import numpy as np

exp = sys.argv[1]
g = collections.defaultdict(list)
for f in glob.glob(f"results/runs/{exp}__*.json"):
    r = json.load(open(f, encoding="utf-8"))
    if "metrics" not in r:
        continue
    g[f.replace("\\", "/").split("__")[1]].append(r["metrics"]["test"])
keys = ["acc_all", "descriptive_per_option", "explanatory_per_option", "explanatory_per_question", "predictive_per_question",
        "counterfactual_per_option", "counterfactual_per_question"]
print("variant n", *[k.replace("_per_", "/")[:14] for k in keys])
for v, L in sorted(g.items()):
    print(v, len(L), *[f"{np.mean([m.get(k, np.nan) for m in L]):.3f}" for k in keys])
