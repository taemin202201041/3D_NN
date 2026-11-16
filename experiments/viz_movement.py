"""How far did the neurons move, and what shape did they end up in?  (for the meeting)

Single run : python viz_movement.py <run_id prefix>
   -> results/figs/move_<run_id>.png   (2 x 4: start grid, final shape (3D, top view, side view), arrows, how far (histogram),
                                         movement vs connection strength, numbers)
Comparison : python viz_movement.py --compare <prefix1> <prefix2> ... [--names fixed position swap ...] [--out name]
   -> results/figs/compare_<out>.png   (one column per run: final shape 3D + top view, with mean / max movement)
Colour: one hue, light -> dark = how far the neuron moved (grey = did not move).  Reads results/runs, coords, weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib.colors import Normalize                # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

import visualize as V                                  # noqa: E402

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
HERE = Path(__file__).resolve().parent
INK, MUTED, BLUE, GRID = "#1f2933", "#6b7785", "#1f5fa8", "#d9dee4"
CMAP = plt.cm.Blues


def lift(c):
    return c if c.shape[1] == 3 else np.c_[c, np.zeros(len(c))]


def stats(r, c0, c1, w):
    n = len(c0)
    off = ~np.eye(n, dtype=bool)
    d0 = np.linalg.norm(c0[:, None] - c0[None], axis=-1)
    d1 = np.linalg.norm(c1[:, None] - c1[None], axis=-1)
    disp = np.linalg.norm(c1 - c0, axis=1)
    a = np.abs(w) * off
    return dict(n=n, disp=disp, d0=d0, d1=d1, off=off, a=a, wire0=float((a * d0).sum()), wire1=float((a * d1).sum()),
                min0=float(d0[off].min()), min1=float(d1[off].min()), strength=a.sum(0) + a.sum(1))


def cube(ax, lim):
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.set_pane_color((1, 1, 1, 0)); p._axinfo["grid"]["color"] = GRID


def edges_of(a, top):
    idx = np.argsort(a.ravel())[::-1][:top]
    return [divmod(int(k), a.shape[0]) for k in idx]


def draw3d(ax, xyz, color, size, title, edges=None, a=None, lim=None):
    if edges is not None:
        segs = [[xyz[i], xyz[j]] for i, j in edges]
        mag = np.array([a[i, j] for i, j in edges]); mag = mag / mag.max()
        ax.add_collection3d(Line3DCollection(segs, colors=[(0.12, 0.37, 0.66, 0.10 + 0.35 * m) for m in mag], linewidths=0.35 + 1.0 * mag))
    ax.scatter(*xyz.T, c=color, s=size, edgecolors=INK, linewidths=0.25, depthshade=True)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    if lim is not None:
        cube(ax, lim)


def one(prefix: str, top: int, out: str):
    r, c1, w = V.load_run(prefix)
    cfg = r["config"]
    c0 = V.grid_coords(cfg)
    c0, c1 = lift(c0), lift(c1)
    s = stats(r, c0, c1, w)
    disp = s["disp"]
    lay = r["model"]["layout"]
    dyn = cfg["layout"]["dynamic"]
    allp = np.vstack([c0, c1])
    lim = [(allp[:, k].min() - 0.3, allp[:, k].max() + 0.3) for k in range(3)]
    norm = Normalize(0, max(disp.max(), 1e-6))
    col = CMAP(0.15 + 0.85 * norm(disp))
    edges = edges_of(s["a"], top)

    fig = plt.figure(figsize=(19, 9.6))
    ax = fig.add_subplot(2, 4, 1, projection="3d"); draw3d(ax, c0, "#b8c0c9", 30, "1. 시작: 격자", edges, s["a"], lim)
    ax = fig.add_subplot(2, 4, 2, projection="3d"); draw3d(ax, c1, col, 36, f"2. 학습 후 모양 (진할수록 많이 이동)", edges, s["a"], lim)

    for k, (ix, iy, name) in enumerate([(0, 1, "3. 위에서 본 모양 (회색 = 시작, 파랑 = 학습 후)"), (0, 2, "4. 옆에서 본 모양 (x-z)")]):
        ax = fig.add_subplot(2, 4, 3 + k)
        for i, j in edges:
            ax.plot([c1[i, ix], c1[j, ix]], [c1[i, iy], c1[j, iy]], color=BLUE, alpha=0.10 + 0.25 * s["a"][i, j] / s["a"].max(), lw=0.5)
        ax.scatter(c0[:, ix], c0[:, iy], s=16, color="#c9d0d7", zorder=2, label="시작")
        ax.scatter(c1[:, ix], c1[:, iy], s=30, c=col, edgecolors=INK, linewidths=0.25, zorder=3, label="학습 후")
        ax.set_title(name, fontsize=10.5, color=INK, loc="left"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(2, 4, 5, projection="3d")
    ax.scatter(*c0.T, c="#c9d0d7", s=14)
    ax.quiver(*c0.T, *(c1 - c0).T, colors=[tuple(x) for x in col], arrow_length_ratio=0.0, linewidth=1.3)
    ax.scatter(*c1.T, c=col, s=26, edgecolors=INK, linewidths=0.25)
    ax.set_title("5. 이동 방향 (회색 = 시작점)", fontsize=10.5, color=INK, loc="left"); cube(ax, lim)

    ax = fig.add_subplot(2, 4, 6)
    ax.hist(disp, bins=24, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axvline(disp.mean(), color=INK, lw=1.2); ax.text(disp.mean(), ax.get_ylim()[1] * 0.95, f" 평균 {disp.mean():.2f}", fontsize=9, color=INK, va="top")
    ax.axvline(disp.max(), color=MUTED, lw=1.0, ls="--"); ax.text(disp.max(), ax.get_ylim()[1] * 0.80, f"최대 {disp.max():.2f} ", fontsize=9, color=MUTED, ha="right", va="top")
    ax.set_xlabel("뉴런이 움직인 거리 (격자 간격 1 기준)"); ax.set_ylabel("뉴런 수"); ax.set_title("6. 얼마나 움직였나", fontsize=10.5, color=INK, loc="left")

    ax = fig.add_subplot(2, 4, 7)
    ax.scatter(s["strength"], disp, s=22, color=BLUE, alpha=0.75, edgecolors="white", linewidths=0.4)
    cc = float(np.corrcoef(s["strength"], disp)[0, 1]) if disp.std() > 0 else float("nan")
    ax.set_xlabel("뉴런의 연결 세기 합 (들어오는 + 나가는 |W|)"); ax.set_ylabel("이동 거리")
    ax.set_title(f"7. 연결이 센 뉴런이 더 움직였나 (상관 {cc:.2f})", fontsize=10.5, color=INK, loc="left")

    ax = fig.add_subplot(2, 4, 8); ax.axis("off")
    t = r["metrics"]["test"]
    lines = [("이동 방식", f"{lay['dynamic_method']}  (열리는 에포크 {dyn['open_epoch']})"),
             ("좌표 학습률 / 반발 / 최소거리", f"{dyn['coord_lr']} / {dyn['repulsion']} / {dyn['min_dist']}"),
             ("교환 횟수", f"{lay.get('n_swaps', 0)}"),
             ("평균 / 최대 이동", f"{disp.mean():.2f} / {disp.max():.2f}"),
             ("움직인 뉴런 (이동 > 0.05)", f"{int((disp > 0.05).sum())} / {s['n']}"),
             ("가장 가까운 두 뉴런 (전 → 후)", f"{s['min0']:.2f} → {s['min1']:.3f}"),
             ("배선 비용 합 |W|·거리 (전 → 후)", f"{s['wire0']:.1f} → {s['wire1']:.1f}"),
             ("테스트 전체 정확도", f"{t['acc_all']:.3f}")]
    ax.set_title("8. 숫자 요약", fontsize=10.5, color=INK, loc="left")
    for k, (a_, b_) in enumerate(lines):
        y = 0.92 - k * 0.115
        ax.text(0.0, y, a_, fontsize=9.5, color=MUTED, transform=ax.transAxes); ax.text(1.0, y, b_, fontsize=10.5, color=INK, ha="right", transform=ax.transAxes)
    fig.suptitle(f"{r['run_id']}   (선 = 가장 센 연결 {top}개)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    f = HERE / out / f"move_{r['run_id']}.png"
    f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=110); print("saved", f)


def compare(prefixes, names, top, out, out_dir, compact=False):
    runs = [V.load_run(p) for p in prefixes]
    n = len(runs)
    c0s = [lift(V.grid_coords(r["config"])) for r, _, _ in runs]
    allp = np.vstack(c0s + [lift(c) for _, c, _ in runs])
    lim = [(allp[:, k].min() - 0.3, allp[:, k].max() + 0.3) for k in range(3)]
    mx = max(float(np.linalg.norm(lift(c) - c0, axis=1).max()) for (_, c, _), c0 in zip(runs, c0s))
    norm = Normalize(0, max(mx, 1e-6))
    fig = plt.figure(figsize=(12.2, 5.1) if compact else (4.6 * n, 10.4))
    for k, ((r, c1, w), c0, nm) in enumerate(zip(runs, c0s, names)):
        c1 = lift(c1); s = stats(r, c0, c1, w); disp = s["disp"]
        col = CMAP(0.15 + 0.85 * norm(disp)); edges = edges_of(s["a"], top)
        t = r["metrics"]["test"]
        ax = fig.add_subplot(3, n, k + 1, projection="3d")
        draw3d(ax, c1, col, 34, f"{nm}", edges, s["a"], lim)
        ax2 = fig.add_subplot(3, n, n + k + 1)
        for i, j in edges:
            ax2.plot([c1[i, 0], c1[j, 0]], [c1[i, 1], c1[j, 1]], color=BLUE, alpha=0.08 + 0.25 * s["a"][i, j] / s["a"].max(), lw=0.5)
        ax2.scatter(c0[:, 0], c0[:, 1], s=14, color="#c9d0d7", zorder=2)
        ax2.scatter(c1[:, 0], c1[:, 1], s=28, c=col, edgecolors=INK, linewidths=0.25, zorder=3)
        ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title("위에서 본 모양", fontsize=9.5, color=MUTED, loc="left")
        ax3 = fig.add_subplot(3, n, 2 * n + k + 1); ax3.axis("off")
        items = [("평균 이동", f"{disp.mean():.2f}"), ("배선 비용", f"{s['wire0']:.0f} → {s['wire1']:.0f}"), ("가장 가까운 쌍", f"{s['min1']:.2f}"), ("테스트 정확도", f"{t['acc_all']:.3f}")] if compact else                 [("평균 이동", f"{disp.mean():.2f}"), ("최대 이동", f"{disp.max():.2f}"), ("교환 횟수", f"{r['model']['layout'].get('n_swaps', 0)}"),
                 ("배선 비용 전→후", f"{s['wire0']:.0f} → {s['wire1']:.0f}"), ("가장 가까운 쌍", f"{s['min1']:.2f}"), ("테스트 전체", f"{t['acc_all']:.3f}")]
        step = 0.25 if compact else 0.16
        for q, (a_, b_) in enumerate(items):
            y = 0.95 - q * step
            ax3.text(0.02, y, a_, fontsize=9 if compact else 10, color=MUTED, transform=ax3.transAxes); ax3.text(0.98, y, b_, fontsize=10 if compact else 11, color=INK, ha="right", transform=ax3.transAxes)
    if not compact:
        fig.suptitle("학습 후 뉴런 배치 비교 (진할수록 많이 이동, 같은 색 눈금)", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 if compact else 0.96))
    f = HERE / out_dir / (f"compare_{out}_compact.png" if compact else f"compare_{out}.png")
    f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=105); print("saved", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="*")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--out", default="results/figs")
    ap.add_argument("--name", default="dynamic")
    ap.add_argument("--compact", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare(a.run, a.names or a.run, a.top, a.name, a.out, a.compact)
    else:
        one(a.run[0], a.top, a.out)
