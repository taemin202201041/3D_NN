"""MC follow-up: do the options of one question get different answers? (read-only)"""
import glob, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
import run as R
from dataset import load
from layout import Geometry
from model import SeRNN
from questions import CLASSES, QTYPES
CORR, WRONG = CLASSES.index("correct"), CLASSES.index("wrong")
runs = {}
for f in glob.glob(str(HERE / "results/runs/e09_*.json")) + glob.glob(str(HERE / "results/runs/e08_grid_o1__fixed_full*.json")):
    r = json.load(open(f, encoding="utf-8"))
    if r.get("status") == "ok": runs[r["run_id"]] = r
cfg0 = next(iter(runs.values()))["config"]; data = load(cfg0, HERE); s = data["splits"]; test = s["test"]
vt = torch.from_numpy(data["video"]).to(R.dev)
y, qt, qk, q = test["y"], test["qtype"], test["qkey"], test["q"]
# how different are the INPUT vectors of the options of one question?
for i in (1, 2, 3):
    sel = np.where(qt == i)[0]; keys, inv = np.unique(qk[sel], return_inverse=True)
    qs = q[sel].astype(np.float32); cen = np.zeros((len(keys), qs.shape[1])); np.add.at(cen, inv, qs); cen /= np.bincount(inv)[:, None]
    within = np.linalg.norm(qs - cen[inv], axis=1).mean(); between = np.linalg.norm(cen - cen.mean(0), axis=1).mean()
    print(f"{QTYPES[i]:15s} question-vector spread: within one question's options {within:.3f}  vs between questions {between:.3f}")
res = {}
for rid, r in sorted(runs.items()):
    name = rid.split("__")[1]; cfg = r["config"]; w = HERE / "results/weights" / f"{rid}.weights.pt"
    ck = torch.load(w, map_location=R.dev, weights_only=False); geom = Geometry(cfg, R.dev)
    model = SeRNN(cfg, geom, vt.shape[2], s["train"]["q"].shape[1], int(cfg["seed"])).to(R.dev)
    model.load_state_dict(ck["model"]); geom.set_coords(ck["coords"])
    p, lg, _, _ = R.evaluate(model, vt, test)
    d = lg[:, CORR] - lg[:, WRONG]
    for i in (1, 2, 3):
        sel = np.where(qt == i)[0]; keys, inv = np.unique(qk[sel], return_inverse=True)
        pc = (p[sel] == CORR).astype(float); n = np.bincount(inv)
        same = np.bincount(inv, weights=pc) ; allsame = np.mean((same == 0) | (same == n))     # every option got the same label
        ds = d[sel]; mu = np.bincount(inv, weights=ds) / n; wsd = np.sqrt(np.bincount(inv, weights=(ds - mu[inv]) ** 2) / n).mean()
        out = {"allsame": allsame, "within_sd": wsd}
        if i == 2:   # exactly one correct of two: pick the option with the higher score
            top = np.array([sel[inv == k][np.argmax(d[sel][inv == k])] for k in range(len(keys))])
            out["pick_best"] = np.mean(y[top] == CORR)
        res.setdefault(name, {}).setdefault(QTYPES[i][:4], []).append(out)
print("\nmodel  | per type: share of questions where ALL options got the same label / within-question logit sd / predictive 'pick the higher-scored option' accuracy")
for n, dd in sorted(res.items()):
    print(n.ljust(14), " | ".join(f"{t}: same {np.mean([o['allsame'] for o in v]):.2f} sd {np.mean([o['within_sd'] for o in v]):.3f}" + (f" pick {np.mean([o['pick_best'] for o in v]):.3f}" if 'pick_best' in v[0] else "") for t, v in dd.items()))
