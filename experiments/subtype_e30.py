"""Where does accuracy stay low?  Descriptive sub-type accuracy and per-question multiple-choice accuracy of finished runs (mean over seeds).
Usage: python subtype_e30.py <experiment> <variant> [<experiment> <variant> ...]   read-only"""
import glob
import json
import sys

import numpy as np

args = sys.argv[1:]
keys = ["desc_count", "desc_exist", "desc_query_color", "desc_query_shape", "desc_query_material", "descriptive_per_option",
        "explanatory_per_option", "explanatory_per_question", "predictive_per_option", "predictive_per_question",
        "counterfactual_per_option", "counterfactual_per_question"]
print(f"{'run':34s}" + " ".join(k.replace('descriptive', 'desc').replace('explanatory', 'expl').replace('predictive', 'pred').replace('counterfactual', 'cf').replace('_per_', '/')[:11].rjust(11) for k in keys))
for e, v in zip(args[::2], args[1::2]):
    rs = [json.load(open(f, encoding="utf-8")) for f in glob.glob(f"results/runs/{e}__{v}__s*.json")]
    rs = [r for r in rs if "metrics" in r]
    row = []
    for k in keys:
        x = [r["metrics"]["test"].get(k) for r in rs]
        x = [a for a in x if a is not None]
        row.append(f"{np.mean(x):11.3f}" if x else f"{'-':>11s}")
    print(f"{(e[:3] + ' ' + v)[:30]:30s} n={len(rs)} " + " ".join(row))
