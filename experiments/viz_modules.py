"""Modules and small-world structure after moving: neurons coloured by module (5 slabs of the start grid), intra-module edges thin grey,
inter-module (gateway) edges coloured; below: |W| matrix sorted by module (block-diagonal = modular).
Usage: python viz_modules.py <prefix1> <prefix2> ... [--names a b c] [--out name]   -> results/figs/modules_<out>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                     # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

import visualize as V                                  # noqa: E402

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
HERE = Path(__file__).resolve().parent
INK, MUTED = "#1f2933", "#6b7785"
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]          # colour-blind safe, fixed order = module id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="+")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--out", default="structures")
    ap.add_argument("--max_edges", type=int, default=700)
    ap.add_argument("--compact", action="store_true")
    a = ap.parse_args()
    runs = [V.load_run(p) for p in a.run]
    names = a.names or a.run
    n = len(runs)
    fig = plt.figure(figsize=(8.4, 4.6) if a.compact else (4.8 * n, 10.2))
    for k, ((r, c1, w), nm) in enumerate(zip(runs, names)):
        cfg = r["config"]
        c0 = V.grid_coords(cfg)
        mod = np.rint(c0[:, 0] - c0[:, 0].min()).astype(int)            # slab index along the first grid axis
        mod = np.clip(mod, 0, 4)
        aw = np.abs(w); np.fill_diagonal(aw, 0)
        thr = float(cfg["eval"].get("edge_threshold", 0.01))
        idx = np.argwhere(aw > thr)
        if len(idx) > a.max_edges:
            idx = idx[np.argsort(aw[idx[:, 0], idx[:, 1]])[::-1][:a.max_edges]]
        inter = mod[idx[:, 0]] != mod[idx[:, 1]]
        st = r.get("structure") or {}
        ax = fig.add_subplot(2, n, k + 1, projection="3d")
        for sel, col, lw, al in ((~inter, (0.55, 0.6, 0.65, 0.18), 0.4, None), (inter, (0.12, 0.14, 0.17, 0.75), 1.0, None)):
            segs = [[c1[i], c1[j]] for i, j in idx[sel]]
            if segs:
                ax.add_collection3d(Line3DCollection(segs, colors=[col] * len(segs), linewidths=lw))
        ax.scatter(*c1.T, c=[OKABE[m] for m in mod], s=34, edgecolors=INK, linewidths=0.3, depthshade=True)
        allp = np.vstack([c0, c1]); ax.set_xlim(allp[:, 0].min() - .3, allp[:, 0].max() + .3); ax.set_ylim(allp[:, 1].min() - .3, allp[:, 1].max() + .3); ax.set_zlim(allp[:, 2].min() - .3, allp[:, 2].max() + .3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        q_, s_, e_ = st.get("Q", float("nan")), st.get("sigma", float("nan")), st.get("edges_active_offdiag", "?")
        ttl = f"{nm}\nQ {q_:.2f}  σ {s_:.1f}  연결 {e_}" if a.compact else f"{nm}\n모듈성 Q {q_:.2f} | small-world σ {s_:.1f} | 활성 연결 {e_}"
        ax.set_title(ttl, fontsize=9 if a.compact else 10, color=INK, loc="left")
        ax2 = fig.add_subplot(2, n, n + k + 1)
        order = np.lexsort((np.arange(len(mod)), mod))
        m = aw[np.ix_(order, order)]
        ax2.imshow(m, cmap="Blues", vmax=np.percentile(m[m > 0], 99) if (m > 0).any() else 1)
        edges_ = np.cumsum(np.bincount(mod, minlength=5))[:-1] - 0.5
        for e in edges_:
            ax2.axhline(e, color=INK, lw=0.6); ax2.axvline(e, color=INK, lw=0.6)
        ax2.set_xticks([]); ax2.set_yticks([])
        if not a.compact:
            ax2.set_title("연결 세기 |W| (모듈 순서로 정렬, 대각 블록 = 모듈)", fontsize=9, color=MUTED, loc="left")
    if not a.compact:
        fig.suptitle("동적 이동 후 모듈 구조: 색 = 모듈, 진한 선 = 모듈 사이 연결", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 if a.compact else 0.95))
    f = HERE / "results/figs" / (f"modules_{a.out}_compact.png" if a.compact else f"modules_{a.out}.png")
    fig.savefig(f, dpi=105); print("saved", f)


if __name__ == "__main__":
    main()
