"""Did the 'modular + small-world + few connector neurons' form emerge without being forced?  (no figure; numbers only)"""
import glob, json, numpy as np, networkx as nx
import seRNN_structure as S
N = 100
def graph_stats(Bs, seed=0):
    g = nx.from_numpy_array(Bs); comms = nx.community.louvain_communities(g, seed=seed); lab = np.zeros(N, int)
    for i, c in enumerate(comms): lab[list(c)] = i
    Q = nx.community.modularity(g, comms)
    same = (lab[:, None] == lab[None]) & ~np.eye(N, dtype=bool); diff = lab[:, None] != lab[None]
    big = [c for c in comms if len(c) >= 5]
    win = np.mean([Bs[np.ix_(list(c), list(c))].sum() / (len(c) * (len(c) - 1)) for c in big]) if big else np.nan   # density inside communities
    btw = Bs[diff].sum() / diff.sum()                                                                               # density between communities
    cross = (Bs * diff).sum(1); conn = (cross > 0).mean()                                                           # fraction of neurons that have any cross-community link
    top = np.sort(cross)[::-1]; conc = top[:max(1, N // 10)].sum() / max(cross.sum(), 1)                            # share of cross links owned by the top 10 % neurons
    return dict(Q=Q, ncomm=len(comms), within=win, between=btw, connector_frac=conn, cross_top10=conc)
def from_W(W):
    A = np.abs(W); B = (A > np.quantile(A, .9)).astype(float); Bs = ((B + B.T) > 0).astype(float); np.fill_diagonal(Bs, 0); return Bs
rows = {}
def add(tag, W, acc):
    r = graph_stats(from_W(W)); r['smallworld'] = S.structure_metrics(W, S.GRID_D if hasattr(S, 'GRID_D') else np.zeros((N, N)))['smallworld']; r['acc'] = acc; rows.setdefault(tag, []).append(r)
for k, nm in (('fixed', 'pos fixed grid'), ('swap', 'pos slot swap'), ('cont', 'pos continuous')):
    for f in sorted(glob.glob(f'results/seRNN_pos/pos_grid_{k}_s0.03_seed*_W.npy')):
        add(nm, np.load(f), json.load(open(f.replace('_W.npy', '.json')))['test_acc'])
for f in sorted(glob.glob('results/seRNN_struct/st_comm_seed*_W.npy')):
    add('comm (round 1)', np.load(f), json.load(open(f.replace('_W.npy', '.json')))['test_acc'])
print('%-18s %4s %6s %6s %7s %7s %9s %9s %8s' % ('group', 'n', 'Q', 'ncomm', 'within', 'between', 'connector', 'top10share', 'smallw'))
for tag, L in rows.items():
    m = lambda k: np.nanmean([r[k] for r in L])
    print('%-18s %4d %6.3f %6.1f %7.3f %7.3f %9.3f %9.3f %8.2f' % (tag, len(L), m('Q'), m('ncomm'), m('within'), m('between'), m('connector_frac'), m('cross_top10'), m('smallworld')))
    ok = [r for r in L if r['acc'] > .9]
    if ok: print('   (learned only, n=%d)        %6.3f %6.1f %7.3f %7.3f %9.3f %9.3f %8.2f' % (len(ok), *[np.nanmean([r[k] for r in ok]) for k in ('Q', 'ncomm', 'within', 'between', 'connector_frac', 'cross_top10', 'smallworld')]))
# calibration: the hand-built ideal forms themselves (binary graph, no threshold)
print('--- calibration: the designed forms')
for k in ('A', 'B1', 'B2'):
    M = S.build_mask(k).astype(float); np.fill_diagonal(M, 0); r = graph_stats(((M + M.T) > 0).astype(float)); print(k, {a: round(float(b), 3) for a, b in r.items()})
