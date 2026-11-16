"""The three hand-designed connectivity structures A, B1, B2 as implemented in seRNN_structure.py (top: 3D, bottom: connection matrix)."""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import seRNN_structure as S
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
N = 100; G = S.GRID; MOD = S.MOD; cols = plt.cm.tab10(np.arange(5))
info = {'A': ('A: 군집 안은 완전연결 + 다리 뉴런 2개씩', 2), 'B1': ('B1: 군집마다 허브(뉴런1) 1개, 별 모양', 1), 'B2': ('B2: 군집마다 허브(뉴런1, 2) 2개', 2)}
fig = plt.figure(figsize=(15, 9.5))
for c, k in enumerate(('A', 'B1', 'B2')):
    M = S.build_mask(k).copy(); np.fill_diagonal(M, False); title, ns = info[k]
    special = np.concatenate([S.special(g, ns) for g in S.GROUPS]); is_sp = np.zeros(N, bool); is_sp[special] = True
    ax = fig.add_subplot(2, 3, c + 1, projection='3d')
    for i, j in zip(*np.where(np.triu(M | M.T, 1))):
        cross = MOD[i] != MOD[j]
        ax.plot(*zip(G[i], G[j]), color='#c0392b' if cross else '#9aa0a6', lw=1.3 if cross else 0.35, alpha=0.9 if cross else 0.3)
    ax.scatter(*G[~is_sp].T, c=cols[MOD[~is_sp]], s=26, edgecolor='k', linewidth=0.3, depthshade=False)
    ax.scatter(*G[is_sp].T, c=cols[MOD[is_sp]], s=150, marker='*', edgecolor='k', linewidth=0.8, depthshade=False)
    ax.set_title(title + '\n별(★) = 다리/허브 뉴런, 빨간 선 = 군집 사이 연결', fontsize=10); ax.view_init(20, -60)
    ax.set_xlabel('x (군집 방향)', fontsize=8); ax.set_ylabel('y', fontsize=8); ax.set_zlabel('z', fontsize=8); ax.tick_params(labelsize=6)
    order = np.concatenate([np.concatenate([S.special(g, ns), np.setdiff1d(g, S.special(g, ns))]) for g in S.GROUPS])
    ax2 = fig.add_subplot(2, 3, c + 4); Mo = (M | M.T)[np.ix_(order, order)]
    ax2.imshow(Mo, cmap='Greys', interpolation='nearest')
    for b in range(1, 5): ax2.axhline(b * 20 - 0.5, color='#c0392b', lw=0.6); ax2.axvline(b * 20 - 0.5, color='#c0392b', lw=0.6)
    ax2.set_title('연결 행렬 (검정 = 연결 있음, 뉴런을 군집 순서로 정렬)\n연결 %d개(양방향 쌍 %d) · 밀도 %.1f%%' % (M.sum(), (np.triu(M | M.T, 1)).sum(), 100 * (M.sum() + N) / N**2), fontsize=9)
    ax2.set_xlabel('받는 뉴런 (각 군집의 앞쪽이 허브/다리)', fontsize=8); ax2.set_ylabel('보내는 뉴런', fontsize=8); ax2.tick_params(labelsize=6)
fig.suptitle('설계한 연결 구조 A, B1, B2 (뉴런 100개, 5개 군집 x 20개). 색 = 군집', fontsize=12)
plt.tight_layout(); plt.savefig('results/fig_structures_ab1b2.png', dpi=115)
