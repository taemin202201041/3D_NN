"""Read-only analyses on finished runs (changes nothing in results/):
 A  can the question vector alone tell the descriptive subtype (count/exist/color/shape/material)?
 B  per-subtype descriptive accuracy: question-only model vs video models vs per-subtype majority
 C  shuffle the videos among test scenes and re-evaluate trained models: accuracy drop = how much the model uses the video
Usage: python probe_analysis.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import run as R                                   # noqa: E402  (reuses evaluate / device)
from dataset import load                          # noqa: E402
from layout import Geometry                       # noqa: E402
from metrics import scores                        # noqa: E402
from model import SeRNN                           # noqa: E402
from questions import SUBTYPES                    # noqa: E402

runs = {}
for f in glob.glob(str(HERE / "results/runs/e0[89]_*.json")):
    r = json.load(open(f, encoding="utf-8"))
    if r.get("status") == "failed" or not r.get("metrics"):
        continue
    runs[r["run_id"]] = r
PICK = ("e09_baselines_same__question_only", "e09_baselines_same__lstm", "e09_baselines_same__linear_rnn",
        "e08_grid_o1__fixed_full", "e08_grid_o1__position_full", "e08_grid_o1__swap_B1_full", "e08_grid_o1__position_B2_full")
sel = {k: r for k, r in runs.items() if k.rsplit("__", 2)[0] in PICK}
cfg0 = next(iter(sel.values()))["config"]
data = load(cfg0, HERE)
s = data["splits"]

# ---------------------------------------------------------------- A: subtype from the question vector
d = {k: s[k] for k in ("train", "val", "test")}
def desc(sp):
    m = sp["qtype"] == 0
    return torch.tensor(sp["q"][m]).float(), torch.tensor(sp["subtype"][m]).long()
xtr, ytr = desc(d["train"]); xte, yte = desc(d["test"])
mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
lin = torch.nn.Linear(xtr.shape[1], len(SUBTYPES))
opt = torch.optim.Adam(lin.parameters(), 1e-2)
for _ in range(300):
    opt.zero_grad(); torch.nn.functional.cross_entropy(lin((xtr - mu) / sd), ytr).backward(); opt.step()
acc = float((lin((xte - mu) / sd).argmax(-1) == yte).float().mean())
maj = float((yte == torch.bincount(ytr).argmax()).float().mean())
print(f"[A] subtype classifier from question vector: test acc {acc:.3f}  (majority {maj:.3f}, subtypes {list(SUBTYPES)})")

# ---------------------------------------------------------------- B: per-subtype accuracy
by = {}
for rid, r in sel.items():
    name = rid.split("__")[1] if rid.startswith("e09") else rid.split("__")[1]
    by.setdefault(name, []).append(r["metrics"]["test"])
keys = sorted({k for v in by.values() for t in v for k in t if k.startswith("desc_")})
print("\n[B] test accuracy per descriptive subtype (mean of seeds)")
print("model".ljust(22), *[k[5:].ljust(14) for k in keys])
for n, v in sorted(by.items()):
    print(n.ljust(22), *[f"{np.mean([t.get(k, np.nan) for t in v]):.3f}".ljust(14) for k in keys])
b = next(iter(sel.values()))["metrics"].get("baselines", {})
if b:
    bt = b.get("test", {})
    print("majority/chance entries:", {k: v for k, v in bt.items() if isinstance(v, dict)} if any(isinstance(v, dict) for v in bt.values()) else list(bt)[:6])

# ---------------------------------------------------------------- C: shuffled-video evaluation
vt = torch.from_numpy(data["video"]).to(R.dev)
lo, hi = cfg0["data"]["test_scenes"]
rng = np.random.default_rng(0)
perm = np.arange(vt.shape[0]); perm[lo:hi] = rng.permutation(np.arange(lo, hi))
test = s["test"]; shuf = dict(test); shuf["scene"] = perm[test["scene"]]
print("\n[C] test: real video vs video shuffled among test scenes")
rows = {}
for rid, r in sorted(sel.items()):
    cfg = r["config"]
    w = HERE / "results/weights" / f"{rid}.weights.pt"
    if not w.exists():
        continue
    ck = torch.load(w, map_location=R.dev, weights_only=False)
    geom = Geometry(cfg, R.dev)
    model = SeRNN(cfg, geom, vt.shape[2], s["train"]["q"].shape[1], int(cfg["seed"])).to(R.dev)
    model.load_state_dict(ck["model"]); geom.set_coords(ck["coords"])
    out = []
    for sp in (test, shuf):
        p = R.evaluate(model, vt, sp)[0]
        sc = scores(p, test)
        out.append((sc["descriptive_per_option"], sc["explanatory_per_question"], sc["predictive_per_question"], sc["counterfactual_per_question"]))
    rows.setdefault(rid.rsplit("__", 2)[0].split("__", 1)[1], []).append(out)
print("model".ljust(18), "desc real/shuf   expl real/shuf   pred real/shuf   cf real/shuf")
for n, v in rows.items():
    a = np.mean(v, axis=0)           # [2, 4]
    print(n.ljust(18), "   ".join(f"{a[0][i]:.3f}/{a[1][i]:.3f}" for i in range(4)))
