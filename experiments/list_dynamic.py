"""List finished runs that move neurons (layout.dynamic.method != fixed) with their data settings. read-only."""
import collections
import glob
import json

rows = collections.defaultdict(list)
for f in sorted(glob.glob("results/runs/*.json")):
    r = json.load(open(f, encoding="utf-8"))
    if "metrics" not in r:
        continue
    c = r["config"]
    d, l = c["data"], c["layout"]["dynamic"]
    name = f.replace("\\", "/").split("__")[1]
    rows[(r["experiment"], name)].append(
        (r["run_id"], d["perception"], d["question_encoding"], "/".join(t[:3] for t in d["question_types"]), tuple(d["train_scenes"]),
         l["method"], l["open_epoch"], round(r["metrics"]["test"]["acc_all"], 3), r["model"]["layout"].get("n_swaps", 0)))
for (e, n), v in rows.items():
    x = v[0]
    if x[5] == "fixed" and "fixed" not in n:
        continue
    if x[5] != "fixed" or n.startswith("fixed"):
        print(f"{e[:24]:24s} {n[:18]:18s} n={len(v)} {x[1][:15]:15s} {x[2][:12]:12s} {x[3]:16s} train{x[4]} {x[5]:8s} o{x[6]:<3} acc {[y[7] for y in v]} swaps {x[8]}")
