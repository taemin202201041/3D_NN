"""What does each module (cluster) of a trained modular network do?  (annotation input, descriptive sub-types)

Three views, all from a finished run (results/weights + the same data split):
  1. LESION   : remove one module's neurons (input, recurrent and readout weights set to 0) and re-score the test split per sub-type
                (count / exist / colour / shape / material).  A random set of neurons of the same size is the control.
  2. INPUT    : mean |input weight| of a module's neurons per input feature group (position, velocity, collision, colour, shape, material, ..., question).
  3. READOUT  : mean |readout weight| of a module's neurons per answer-class group (yes/no, counts, shapes, colours, materials).
Usage: python analysis_modules.py <run id prefix> [--random 8]  ->  stdout + results/figs/module_roles_<run_id>.png
Modules = the 5 slabs of the START grid (first coordinate), as in gateway_structure.py.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

import perception as P            # noqa: E402
import questions as Q             # noqa: E402
import run as RUN                 # noqa: E402
import visualize as V             # noqa: E402
from dataset import load         # noqa: E402
from layout import Geometry       # noqa: E402
from metrics import scores        # noqa: E402
from model import SeRNN           # noqa: E402

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
HERE = Path(__file__).resolve().parent
SUB = [("count", "개수"), ("exist", "존재"), ("query_color", "색"), ("query_shape", "모양"), ("query_material", "재질")]
IN_GROUPS = {"위치": ["x", "y", "z"], "속도": ["vx", "vy", "vz"], "방향·회전": ["ox", "oy", "oz", "wx", "wy", "wz"], "화면 안": ["visible"],
             "충돌": ["collide"] + [f"partner{i}" for i in range(P.SLOTS)] + ["cx", "cy", "cz"], "색": [f"color_{c}" for c in P.COLORS],
             "재질": [f"material_{m}" for m in P.MATERIALS], "모양": [f"shape_{s}" for s in P.SHAPES]}
OUT_GROUPS = {"예/아니오": ["yes", "no"], "개수 0-5": ["0", "1", "2", "3", "4", "5"], "모양": Q.SHAPES, "색": Q.COLORS, "재질": Q.MATERIALS}


@torch.no_grad()
def mean_sub(model, vt, split):
    """sub-type accuracies with the answer restricted to the classes that fit the question group (so a lesion cannot 'win' or 'lose' by pushing
    the answer into another group, e.g. answering a colour to a shape question)."""
    from dataset import batches
    model.eval()
    preds = []
    for x, y in batches(vt, split, 1024, False, device=RUN.dev):
        lg = model(*x)
        lg = lg + (model.allowed[x[2]] - 1.0) * 1e9
        preds.append(lg.argmax(-1).cpu().numpy())
    sc = scores(np.concatenate(preds), split)
    return np.array([sc.get(f"desc_{k}", np.nan) for k, _ in SUB]), sc.get("descriptive_per_option", np.nan)


def lesion(model, idx):
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.core.kernel[:, idx] = 0
        m.core.recurrent_kernel[idx, :] = 0
        m.core.recurrent_kernel[:, idx] = 0
        m.core.bias[idx] = 0
        m.readout.weight[:, idx] = 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--random", type=int, default=20)
    a = ap.parse_args()
    hits = sorted(glob.glob(str(HERE / "results/runs" / f"{a.run}*.json")))
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} runs match")
    r = json.load(open(hits[0], encoding="utf-8"))
    cfg, rid = r["config"], r["run_id"]
    if cfg["data"]["perception"] != "annotation":
        raise SystemExit("input-feature view needs the annotation input")
    data = load(cfg, HERE)
    vt = torch.from_numpy(data["video"]).to(RUN.dev)
    s = data["splits"]
    geom = Geometry(cfg, RUN.dev)
    model = SeRNN(cfg, geom, data["video"].shape[2], s["train"]["q"].shape[1], int(cfg["seed"]), emb=data["meta"].get("emb")).to(RUN.dev)
    ck = torch.load(HERE / "results/weights" / f"{rid}.weights.pt", map_location=RUN.dev, weights_only=False)
    model.load_state_dict(ck["model"])
    geom.set_coords(ck["coords"])
    c0 = V.grid_coords(cfg)
    mod = np.clip(np.rint(c0[:, 0] - c0[:, 0].min()).astype(int), 0, 4)
    n_mod = int(mod.max()) + 1

    base, base_all = mean_sub(model, vt, s["test"])
    print(f"{rid}\ntest descriptive {base_all:.3f}; per sub-type " + ", ".join(f"{n} {v:.3f}" for (_, n), v in zip(SUB, base)))

    # 1. lesion
    drops = np.zeros((n_mod, len(SUB)))
    for m in range(n_mod):
        drops[m] = base - mean_sub(lesion(model, np.flatnonzero(mod == m)), vt, s["test"])[0]
    rng = np.random.default_rng(0)
    size = int(np.bincount(mod).mean())
    rnd = np.array([base - mean_sub(lesion(model, rng.choice(len(mod), size, replace=False)), vt, s["test"])[0] for _ in range(a.random)])
    print(f"\n[1] LESION - accuracy drop (points x100) when the module is removed; control = {size} random neurons (mean ± sd of {a.random})")
    print("module  " + "".join(f"{n:>8s}" for _, n in SUB))
    for m in range(n_mod):
        print(f"{m:6d}  " + "".join(f"{100 * d:8.1f}" for d in drops[m]))
    print("random  " + "".join(f"{100 * x:5.1f}±{100 * y:<2.1f}" for x, y in zip(rnd.mean(0), rnd.std(0))))

    # 2. input profile
    kern = model.core.kernel.detach().cpu().numpy()                      # [d_in, units]
    d_vid = data["video"].shape[2]
    feat = np.abs(kern[:d_vid].reshape(P.SLOTS, P.F, -1)).mean(0)        # [feature, units], mean over the 6 slots
    prof_in = np.zeros((n_mod, len(IN_GROUPS) + 1))
    for g, (name, fs) in enumerate(IN_GROUPS.items()):
        rows = [P.FEATURES.index(f) for f in fs]
        for m in range(n_mod):
            prof_in[m, g] = feat[rows][:, mod == m].mean()
    for m in range(n_mod):
        prof_in[m, -1] = np.abs(kern[d_vid:][:, mod == m]).mean()
    z_in = (prof_in - prof_in.mean(0)) / (prof_in.std(0) + 1e-9)

    # 3. readout profile
    ro = np.abs(model.readout.weight.detach().cpu().numpy())             # [classes, units]
    prof_out = np.zeros((n_mod, len(OUT_GROUPS)))
    for g, (name, cs) in enumerate(OUT_GROUPS.items()):
        rows = [Q.CLASSES.index(c) for c in cs]
        for m in range(n_mod):
            prof_out[m, g] = ro[rows][:, mod == m].mean()
    z_out = (prof_out - prof_out.mean(0)) / (prof_out.std(0) + 1e-9)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    lim = max(0.02, np.abs(drops).max())
    im = ax[0].imshow(drops * 100, cmap="RdBu_r", vmin=-lim * 100, vmax=lim * 100)
    ax[0].set_xticks(range(len(SUB))); ax[0].set_xticklabels([n for _, n in SUB]); ax[0].set_yticks(range(n_mod)); ax[0].set_yticklabels([f"모듈 {m}" for m in range(n_mod)])
    for m in range(n_mod):
        for j in range(len(SUB)):
            ax[0].text(j, m, f"{100 * drops[m, j]:.1f}", ha="center", va="center", fontsize=10)
    ax[0].set_title(f"1. 모듈을 끄면 정확도가 떨어지는 정도 (점)\n무작위 {size}개 대조 평균 {100 * rnd.mean():.1f}", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    for k, (z, names, title) in enumerate([(z_in, list(IN_GROUPS) + ["질문"], "2. 모듈이 받는 입력 (열별 표준화: 진할수록 그 입력을 더 세게 받음)"),
                                           (z_out, list(OUT_GROUPS), "3. 모듈이 영향을 주는 답 종류 (열별 표준화)")]):
        im = ax[k + 1].imshow(z, cmap="RdBu_r", vmin=-2, vmax=2)
        ax[k + 1].set_xticks(range(len(names))); ax[k + 1].set_xticklabels(names, rotation=30, ha="right"); ax[k + 1].set_yticks(range(n_mod)); ax[k + 1].set_yticklabels([f"모듈 {m}" for m in range(n_mod)])
        ax[k + 1].set_title(title, fontsize=10, loc="left")
        fig.colorbar(im, ax=ax[k + 1], fraction=0.046)
    fig.suptitle(rid, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "results/figs" / f"module_roles_{rid}.png"
    fig.savefig(out, dpi=110)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
