"""Why are multiple-choice 'whole question correct' scores below chance?  Read-only analysis of finished runs.
Per MC type (test split): true share of 'correct' options, share the model predicts as 'correct', distribution of the number of
correct options per question, and the whole-question score of simple strategies (all-correct, all-wrong, always-prior) vs the models.
"""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import run as R                                    # noqa: E402
from dataset import load                           # noqa: E402
from layout import Geometry                        # noqa: E402
from model import SeRNN                            # noqa: E402
from questions import CLASSES, QTYPES              # noqa: E402

CORR, WRONG = CLASSES.index("correct"), CLASSES.index("wrong")
runs = {}
for f in glob.glob(str(HERE / "results/runs/e09_*.json")) + glob.glob(str(HERE / "results/runs/e08_grid_o1__fixed_full*.json")):
    r = json.load(open(f, encoding="utf-8"))
    if r.get("status") == "ok":
        runs[r["run_id"]] = r
cfg0 = next(iter(runs.values()))["config"]
data = load(cfg0, HERE)
s = data["splits"]
test = s["test"]
vt = torch.from_numpy(data["video"]).to(R.dev)
y, qt, qk = test["y"], test["qtype"], test["qkey"]

print("[1] true structure of the multiple-choice questions (test)")
for i in (1, 2, 3):
    sel = qt == i
    keys, inv = np.unique(qk[sel], return_inverse=True)
    n_opt = np.bincount(inv)
    n_cor = np.bincount(inv, weights=(y[sel] == CORR).astype(float)).astype(int)
    print(f"  {QTYPES[i]:15s} questions {len(keys):5d}  options/q {Counter(n_opt.tolist())}  share correct {np.mean(y[sel] == CORR):.3f}")
    print(f"  {'':15s} #correct per question: {dict(sorted(Counter(n_cor.tolist()).items()))}")
    allc = np.mean(n_cor == n_opt); allw = np.mean(n_cor == 0)
    # best constant answer per option-count (prior): predict 'correct' if P(correct | n_opt) > 0.5
    pri = []
    for k in np.unique(n_opt):
        m = n_opt == k
        p = n_cor[m].sum() / n_opt[m].sum()
        pri.append((k, p))
    print(f"  {'':15s} whole-question score: all-correct {allc:.3f}  all-wrong {allw:.3f}  | P(correct) by #options: " + ", ".join(f"{k}: {p:.2f}" for k, p in pri))

print("\n[2] what do the models predict on those options (mean of seeds)")
agg = defaultdict(lambda: defaultdict(list))
for rid, r in sorted(runs.items()):
    name = rid.split("__")[1]
    cfg = r["config"]
    w = HERE / "results/weights" / f"{rid}.weights.pt"
    if not w.exists():
        continue
    ck = torch.load(w, map_location=R.dev, weights_only=False)
    geom = Geometry(cfg, R.dev)
    model = SeRNN(cfg, geom, vt.shape[2], s["train"]["q"].shape[1], int(cfg["seed"])).to(R.dev)
    model.load_state_dict(ck["model"]); geom.set_coords(ck["coords"])
    p, lg, _, _ = R.evaluate(model, vt, test)
    for i in (1, 2, 3):
        sel = qt == i
        pc = np.mean(p[sel] == CORR)
        keys, inv = np.unique(qk[sel], return_inverse=True)
        ok = (p[sel] == y[sel])
        wrong = np.bincount(inv, weights=(~ok).astype(float))
        agg[name][f"{QTYPES[i][:4]}_pred_correct_share"].append(pc)
        agg[name][f"{QTYPES[i][:4]}_per_option"].append(ok.mean())
        agg[name][f"{QTYPES[i][:4]}_whole_q"].append(np.mean(wrong == 0))
        # margin: does the score of 'correct' vs 'wrong' vary between options at all?
        d = (lg[sel][:, CORR] - lg[sel][:, WRONG])
        agg[name][f"{QTYPES[i][:4]}_logit_diff_std"].append(d.std())
cols = [f"{t}_{k}" for t in ("expl", "pred", "coun") for k in ("pred_correct_share", "per_option", "whole_q", "logit_diff_std")]
print("model".ljust(16), *[c.replace("_pred_correct_share", "_predC").replace("_logit_diff_std", "_sd")[:14].ljust(14) for c in cols])
for n, d in sorted(agg.items()):
    print(n.ljust(16), *[f"{np.mean(d[c]):.3f}".ljust(14) for c in cols])
