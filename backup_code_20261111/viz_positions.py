"""3D visualisation of neuron positions: fixed grid vs slot-swap vs continuous (box + repulsion).
Colour = Louvain community of the trained weight matrix (top-10% edges, symmetrised); lines = strongest 3% of connections."""
import json, os, glob, sys
import numpy as np, networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results', 'seRNN_pos')
n = np.arange(5); x, y, z = np.meshgrid(n, n, np.arange(4)); GRID = np.stack([x.ravel(), y.ravel(), z.ravel()], 1).astype(float)


def communities(W):
    A = np.abs(W); B = (A > np.quantile(A, 0.9)).astype(float); Bs = ((B + B.T) > 0).astype(float); np.fill_diagonal(Bs, 0)
    lab = np.zeros(100, int)
    for i, c in enumerate(nx.community.louvain_communities(nx.from_numpy_array(Bs), seed=0)): lab[list(c)] = i
    return lab


def load(kind, strength, seed):
    base = os.path.join(R, f'pos_grid_{kind}_s{strength}_seed{seed}')
    W = np.load(base + '_W.npy'); P = np.load(base + '_P.npy'); j = json.load(open(base + '.json')); return W, P, j


def draw(ax, W, P, title, acc):
    lab = communities(W); A = np.abs(W); A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    thr = np.quantile(A[np.triu_indices(100, 1)], 0.97)
    for i, k in zip(*np.where(np.triu(A) > thr)):
        ax.plot(*zip(P[i], P[k]), color='#888', lw=0.5, alpha=0.5)
    ax.scatter(*P.T, c=lab, cmap='tab10', s=38, edgecolor='k', linewidth=0.3, depthshade=False)
    ax.set_title(f'{title}\n(test 정확도 {acc:.2f}, 커뮤니티 {lab.max() + 1}개)', fontsize=9)
    ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_zlim(0, 3); ax.set_xticks([0, 2, 4]); ax.set_yticks([0, 2, 4]); ax.set_zticks([0, 1.5, 3]); ax.tick_params(labelsize=6)
    ax.view_init(22, -58)


if __name__ == '__main__':
    strength = 0.03
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    kinds = [('fixed', '격자 고정'), ('swap', '칸 교환 (격자 칸 고정, 배정만 변경)'), ('cont', '연속 이동 (부피 제한 + 반발)')]
    # keep only seeds where all three conditions learned the task (acc > 0.9) so the structure is meaningful
    ok = []
    for s in range(1, 41):
        try:
            if all(load(k, strength, s)[2]['test_acc'] > 0.9 for k, _ in kinds): ok.append(s)
        except FileNotFoundError: pass
    use = [s for s in seeds if s in ok] or ok[:3]
    print('seeds used', use, '(all three learned:', ok, ')')
    fig = plt.figure(figsize=(15, 4.6 * len(use)))
    for r, s in enumerate(use):
        for c, (k, name) in enumerate(kinds):
            W, P, j = load(k, strength, s)
            draw(fig.add_subplot(len(use), 3, r * 3 + c + 1, projection='3d'), W, P, f'seed {s} · {name}', j['test_acc'])
    fig.suptitle('뉴런 위치(점)와 가장 강한 연결(선). 색 = 학습된 가중치에서 찾은 커뮤니티', fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'results', 'fig_positions_3d.png'), dpi=120); plt.close()
    # movement of each neuron (swap / cont) relative to its starting grid slot
    s = use[0]; fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (k, name) in zip(axs, kinds[1:]):
        W, P, j = load(k, strength, s); d = np.linalg.norm(P - GRID, axis=1)
        ax.hist(d, bins=20, color='#1f77b4'); ax.set_title(f'{name}\nseed {s}: 뉴런별 이동 거리 (평균 {d.mean():.2f})', fontsize=9); ax.set_xlabel('시작 칸으로부터의 거리')
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'results', 'fig_positions_move.png'), dpi=120)
    print('saved')
