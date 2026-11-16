"""Histogram of final test accuracy over 40 independent runs per method (strength 0.03)."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results', 'seRNN_pos')
kinds = [('fixed', '격자 고정'), ('swap', '칸 교환'), ('cont', '연속 이동')]
bins = np.arange(0.55, 1.0501, 0.05)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (k, name) in zip(axs, kinds):
    a = np.array([json.load(open(f))['test_acc'] for f in sorted(glob.glob(os.path.join(R, f'pos_grid_{k}_s0.03_seed*.json')))])
    ax.hist(a, bins=bins, color='#3b6ea5', edgecolor='white')
    ax.axvline(0.9, color='#c0392b', ls='--', lw=1.2); ax.axvline(0.625, color='#888', ls=':', lw=1.2)
    lo, mid, hi = (a < 0.7).sum(), ((a >= 0.7) & (a < 0.9)).sum(), (a >= 0.9).sum()
    ax.set_title(f'{name}\n0.7 미만 {lo}개 · 0.7–0.9 {mid}개 · 0.9 이상 {hi}개', fontsize=10)
    ax.set_xlabel('테스트 정확도'); print(k, lo, mid, hi, 'in[0.6,0.65):', ((a >= .6) & (a < .65)).sum(), 'in[0.95,1.0]:', (a >= .95).sum())
axs[0].set_ylabel('모델 수 (40개 중)')
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'results', 'fig_accuracy_hist.png'), dpi=120)
