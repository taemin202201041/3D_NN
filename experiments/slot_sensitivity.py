"""Why did slot shuffling / the shared object encoder change the accuracy?  (annotation input, descriptive)
For each finished run: test accuracy with the original slot order and with the 6 object slots randomly re-ordered at TEST time (partner columns re-ordered
with them).  A model that memorised slot positions loses accuracy; an order-free model does not.  Also train / val / test accuracy at the best epoch (over-fitting
gap) and the number of trainable parameters.
Usage: python slot_sensitivity.py <experiment> <variant> [<experiment> <variant> ...]   (mean over the finished seeds)  read-only."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

import run as RUN
import visualize as V  # noqa: F401
from dataset import batches, load
from layout import Geometry
from metrics import scores
from model import SeRNN, permute_slots

HERE = Path(__file__).resolve().parent
_cache = {}


@torch.no_grad()
def test_acc(model, vt, split, shuffle, seed=0):
    model.eval()
    g = torch.Generator(device=RUN.dev).manual_seed(seed)
    preds = []
    for x, y in batches(vt, split, 1024, False, device=RUN.dev):
        v = x[0]
        if shuffle:
            order = torch.argsort(torch.rand(v.shape[0], 6, device=RUN.dev, generator=g), dim=1)
            v = permute_slots(v, order)
        lg = model(v, x[1], x[2])
        preds.append(lg.argmax(-1).cpu().numpy())
    return scores(np.concatenate(preds), split)["acc_all"]


def one(path):
    r = json.load(open(path, encoding="utf-8"))
    cfg = r["config"]
    key = (cfg["data"]["perception"], cfg["data"]["question_encoding"])
    if key not in _cache:
        data = load(cfg, HERE)
        _cache[key] = (data, torch.from_numpy(data["video"]).to(RUN.dev))
    data, vt = _cache[key]
    s = data["splits"]
    geom = Geometry(cfg, RUN.dev)
    model = SeRNN(cfg, geom, data["video"].shape[2], s["train"]["q"].shape[1], int(cfg["seed"]), emb=data["meta"].get("emb")).to(RUN.dev)
    ck = torch.load(HERE / "results/weights" / f"{r['run_id']}.weights.pt", map_location=RUN.dev, weights_only=False)
    model.load_state_dict(ck["model"])
    geom.set_coords(ck["coords"])
    a0 = test_acc(model, vt, s["test"], False)
    a1 = np.mean([test_acc(model, vt, s["test"], True, seed=k) for k in range(3)])
    be = int(r["metrics"]["best_epoch"])
    h = r["history"]
    return a0, a1, h["accuracy"][be - 1], h["val_accuracy"][be - 1], r["model"]["trainable_params"]


if __name__ == "__main__":
    a = sys.argv[1:]
    print(f"{'run':34s} {'n':>2s} {'train':>6s} {'val':>6s} {'test':>6s} {'test(slots shuffled)':>21s} {'drop':>6s} {'params':>8s}")
    for e, v in zip(a[::2], a[1::2]):
        rows = []
        for f in sorted(glob.glob(f"results/runs/{e}__{v}__s*.json")):
            if "metrics" in json.load(open(f, encoding="utf-8")):
                rows.append(one(f))
        if rows:
            x = np.array(rows, float).mean(0)
            print(f"{(e[:3] + ' ' + v)[:34]:34s} {len(rows):2d} {x[2]:6.3f} {x[3]:6.3f} {x[0]:6.3f} {x[1]:21.3f} {x[0] - x[1]:6.3f} {int(x[4]):8d}")
