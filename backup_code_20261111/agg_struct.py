import json, glob, os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results', 'seRNN_struct')
CONDS = ['none', 'comm', 'A', 'B1', 'B2', 'rA', 'rB1', 'rB2', 'aA', 'aB1', 'aB2', 'sA', 'sB1', 'sB2']
D = {c: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f'st_{c}_seed*.json')))] for c in CONDS}
def cp(k, n, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0; hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0; return lo, hi
rows = []
for c in CONDS:
    acc = np.array([r['test_acc'] for r in D[c]]); n = len(acc); k = int((acc > 0.9).sum()); lo, hi = cp(k, n)
    t = np.mean([r['seconds'] for r in D[c]])
    rows.append(dict(cond=c, n=n, mean_acc=acc.mean(), sd=acc.std(), median=np.median(acc), k=k, rate=k / n, ci_lo=lo, ci_hi=hi,
                     n_low=int((acc < 0.7).sum()), n_mid=int(((acc >= .7) & (acc <= .9)).sum()), sec=t, sec_per_success=t / (k / n) if k else np.nan,
                     edges=int(np.median([r['nonzero_total'] for r in D[c]]))))
T = pd.DataFrame(rows).set_index('cond'); pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(T.to_string()); T.to_csv(os.path.join(HERE, 'results', 'seRNN_struct_summary.csv'))
pairs = [('none', x) for x in ('A', 'B1', 'B2')] + [(a, 'r' + a) for a in ('A', 'B1', 'B2')] + [('comm', x) for x in ('aA', 'aB1', 'aB2', 'sA', 'sB1', 'sB2')] + [('a' + x, 's' + x) for x in ('A', 'B1', 'B2')] + [('none', 'comm')]
out = []
for a, b in pairs:
    xa = np.array([r['test_acc'] for r in D[a]]); xb = np.array([r['test_acc'] for r in D[b]])
    ka, kb = int((xa > .9).sum()), int((xb > .9).sum())
    pf = stats.fisher_exact([[ka, len(xa) - ka], [kb, len(xb) - kb]])[1]; pm = stats.mannwhitneyu(xa, xb)[1]
    out.append(dict(a=a, b=b, rate_a=ka / len(xa), rate_b=kb / len(xb), diff=kb / len(xb) - ka / len(xa), p_fisher=pf, p_mwu=pm))
P = pd.DataFrame(out)
def holm(p):
    p = np.array(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj
P['holm_fisher'] = holm(P.p_fisher); P['holm_mwu'] = holm(P.p_mwu)
print(P.to_string()); P.to_csv(os.path.join(HERE, 'results', 'seRNN_struct_tests.csv'), index=False)
