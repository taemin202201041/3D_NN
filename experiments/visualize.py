"""Before/after picture of one finished run: neuron positions (grid start vs final), movement, and the recurrent weights.

Usage:  python visualize.py <run_id or unique prefix> [--top 300] [--out results/figs]
Reads results/runs/<id>.json, results/coords/<id>.npy (final coordinates), results/weights/<id>.weights.pt (recurrent kernel x mask).
Panels: (1) grid start + strongest connections, (2) final positions + the same connections, (3) movement arrows,
        (4) |W| matrix, (5) |W| against distance before / after, (6) distances between all neuron pairs (minimum-distance check).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection   # noqa: E402

HERE = Path(__file__).resolve().parent


def load_run(prefix: str):
    hits = sorted(glob.glob(str(HERE / "results/runs" / f"{prefix}*.json")))
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} runs match '{prefix}': {[Path(h).stem for h in hits][:6]}")
    r = json.load(open(hits[0], encoding="utf-8"))
    rid = r["run_id"]
    coords = np.load(HERE / "results/coords" / f"{rid}.npy")
    ck = torch.load(HERE / "results/weights" / f"{rid}.weights.pt", map_location="cpu", weights_only=False)
    sd = ck["model"]
    if "core.recurrent_kernel" not in sd:
        raise SystemExit("this run has no plain recurrent kernel (LSTM?) - nothing to draw")
    w = (sd["core.recurrent_kernel"] * sd["core.rec_mask"]).numpy()
    return r, coords, w


def grid_coords(cfg: dict) -> np.ndarray:
    from layout import Geometry
    return Geometry(cfg, "cpu").coords0


def draw_net(ax, xyz, edges, w, color_by, title, lim):
    ax.scatter(*xyz.T, c=color_by, cmap="viridis", s=26, depthshade=True, edgecolors="k", linewidths=0.3)
    segs = [[xyz[i], xyz[j]] for i, j in edges]
    mag = np.array([abs(w[i, j]) for i, j in edges])
    lc = Line3DCollection(segs, colors=plt.cm.Reds(0.25 + 0.75 * mag / mag.max()), linewidths=0.4 + 1.6 * mag / mag.max(), alpha=0.55)
    ax.add_collection3d(lc)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    r, c1, w = load_run(a.run)
    cfg = r["config"]
    c0 = grid_coords(cfg)
    n = len(c0)
    dim3 = c0.shape[1] == 3
    if not dim3:                                       # 2D layouts: lift to z = 0 so the same code draws them
        c0 = np.c_[c0, np.zeros(n)]; c1 = np.c_[c1, np.zeros(n)]
    off = ~np.eye(n, dtype=bool)
    mag = np.abs(w) * off
    top = np.argsort(mag.ravel())[::-1][:a.top]
    edges = [divmod(int(k), n) for k in top]
    d0 = np.linalg.norm(c0[:, None] - c0[None], axis=-1)
    d1 = np.linalg.norm(c1[:, None] - c1[None], axis=-1)
    disp = np.linalg.norm(c1 - c0, axis=1)
    lay = r["model"]["layout"]
    allp = np.vstack([c0, c1])
    lim = [(allp[:, k].min() - 0.3, allp[:, k].max() + 0.3) for k in range(3)]
    color = c0[:, 2] if dim3 else c0[:, 1]             # colour = layer of the START grid, so moved neurons can be followed

    fig = plt.figure(figsize=(17, 9.5))
    ax = fig.add_subplot(2, 3, 1, projection="3d"); draw_net(ax, c0, edges, w, color, f"1. start: grid, strongest {a.top} connections", lim)
    ax = fig.add_subplot(2, 3, 2, projection="3d"); draw_net(ax, c1, edges, w, color, f"2. after learning ({lay['dynamic_method']}), same connections", lim)
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.scatter(*c0.T, c="lightgray", s=14); ax.scatter(*c1.T, c=color, cmap="viridis", s=24, edgecolors="k", linewidths=0.3)
    ax.quiver(*c0.T, *(c1 - c0).T, color="tab:blue", alpha=0.6, arrow_length_ratio=0.0, linewidth=0.8)
    ax.set_title(f"3. movement (gray = start)  mean {disp.mean():.2f}, max {disp.max():.2f}", fontsize=10)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2]); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    order = np.lexsort((c1[:, 0], c1[:, 1], c1[:, 2]))        # neurons sorted by final position
    ax = fig.add_subplot(2, 3, 4)
    mo = (np.abs(w) * off)[np.ix_(order, order)]
    im = ax.imshow(mo, cmap="Reds", vmax=np.percentile(mo[mo > 0], 99)); ax.set_title("4. |W| without self-connections (sorted by final position)", fontsize=10)
    ax.set_xlabel("to"); ax.set_ylabel("from"); fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(2, 3, 5)
    m = off & (np.abs(w) > 0)
    ax.scatter(d0[m], np.abs(w)[m], s=3, alpha=0.25, label="start grid", color="gray")
    ax.scatter(d1[m], np.abs(w)[m], s=3, alpha=0.25, label="after", color="tab:red")
    ax.set_xlabel("distance between neurons"); ax.set_ylabel("|W|"); ax.set_yscale("log"); ax.legend(markerscale=4, loc="lower left"); ax.set_title("5. weight vs distance", fontsize=10)
    wc0, wc1 = float((np.abs(w) * d0 * off).sum()), float((np.abs(w) * d1 * off).sum())
    ax.text(0.98, 0.97, f"sum |W|*d: {wc0:.1f} -> {wc1:.1f}", transform=ax.transAxes, fontsize=9, ha="right", va="top")

    ax = fig.add_subplot(2, 3, 6)
    ax.hist(d0[off], bins=60, alpha=0.5, label="start grid", color="gray"); ax.hist(d1[off], bins=60, alpha=0.5, label="after", color="tab:red")
    md = float(cfg["layout"]["dynamic"]["min_dist"])
    ax.axvline(md, color="k", ls="--", label=f"min_dist {md}")
    ax.set_title(f"6. pair distances  (smallest: {d0[off].min():.2f} -> {d1[off].min():.3f})", fontsize=10); ax.set_xlabel("distance"); ax.legend()

    t = r["metrics"]["test"]
    fig.suptitle(f"{r['run_id']}\ntest all {t['acc_all']:.3f}  descriptive {t.get('descriptive_per_option', float('nan')):.3f}  best epoch {r['metrics']['best_epoch']}  "
                 f"swaps {lay.get('n_swaps', 0)}  mean move {disp.mean():.2f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{r['run_id']}.png"
    fig.savefig(f, dpi=110)
    print("saved", f)


if __name__ == "__main__":
    main()
