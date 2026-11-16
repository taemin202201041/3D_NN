"""Success rate (test acc > 0.9, out of 100 runs) with 95% Clopper-Pearson interval for every condition."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results', 'seRNN_struct')
groups = [('대조군', [('none', '규제 없는 RNN (none)'), ('comm', 'seRNN 규제 (comm)')], '#6b7280'),
          ('처음부터 구조 강제', [('A', 'A 군집+다리'), ('B1', 'B1 허브1'), ('B2', 'B2 허브2')], '#2a6fb0'),
          ('같은 연결 수, 무작위', [('rA', 'rA'), ('rB1', 'rB1'), ('rB2', 'rB2')], '#7aa6d6'),
          ('격자에서 구조로 (빠르게, 10에포크)', [('aA', 'aA'), ('aB1', 'aB1'), ('aB2', 'aB2')], '#c0603a'),
          ('격자에서 구조로 (느리게, 30에포크)', [('sA', 'sA'), ('sB1', 'sB1'), ('sB2', 'sB2')], '#e0a070')]
fig, ax = plt.subplots(figsize=(9, 7)); y = 0; yt = []; yl = []
for gname, items, col in groups:
    ax.text(1, y - 0.2, gname, fontsize=9, color=col, fontweight='bold', va='bottom'); y += 0.9
    for c, lab in items:
        acc = np.array([json.load(open(f))['test_acc'] for f in sorted(glob.glob(os.path.join(R, f'st_{c}_seed*.json')))]); n = len(acc); k = int((acc > .9).sum())
        lo = stats.beta.ppf(.025, k, n - k + 1) if k else 0; hi = stats.beta.ppf(.975, k + 1, n - k) if k < n else 1
        ax.plot([lo * 100, hi * 100], [y, y], color=col, lw=2.2); ax.plot(k / n * 100, y, 'o', color=col, ms=7, mec='white')
        ax.text(103, y, f'{k}/{n}', va='center', fontsize=8); yt.append(y); yl.append(lab); y += 1
    y += 0.5
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9); ax.invert_yaxis(); ax.set_xlim(0, 110); ax.set_xticks(range(0, 101, 20))
ax.set_xlabel('성공률 (%) = 100번 중 테스트 정확도 0.9 초과 비율'); ax.axvline(76, color='#6b7280', ls=':', lw=1); ax.text(76, -0.9, 'none 76%', fontsize=8, color='#6b7280', ha='center')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_title('조건별 성공률과 95% 신뢰구간 (점 = 성공률, 선 = 신뢰구간)', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'results', 'fig_structure_success.png'), dpi=120)
for c in ('none','comm','A','B1','B2','rA','rB1','rB2','aA','aB1','aB2','sA','sB1','sB2'):
    s = [json.load(open(f))['seconds'] for f in glob.glob(os.path.join(R, f'st_{c}_seed*.json'))]; print(c, 'median sec %.1f  p90 %.1f max %.0f' % (np.median(s), np.quantile(s, .9), max(s)))
