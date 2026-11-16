"""Module + small-world connectivity masks on the 3D 5x5x4 grid (100 neurons). 3D only.

Modules: 5 slabs of 20 neurons (first grid coordinate), dense inside (all intra-module connections allowed).
Gateways: the k neurons of each module closest to the module centre -- only gateways connect to other modules.
  A  : k = 4  (several gateways, still a minority of the 20)
  B2 : k = 2
  B1 : k = 1
Inter-module wiring between gateways of different modules:
  full : every gateway <-> every gateway of other modules (always 1 module-hop)
  rand : each possible gateway pair is kept with probability p (default 0.5); resampled until all 5 modules are connected
Mask names: 'A_full', 'A_rand', 'B2_full', ... Masks are symmetric, diagonal always allowed.
"""
import numpy as np

N = 100
GRID = np.array([[x, y, z] for x in range(5) for y in range(5) for z in range(4)], dtype='float32')   # (100, 3)
MOD = GRID[:, 0].astype(int)
GROUPS = [np.where(MOD == c)[0] for c in range(5)]
K = {'A': 4, 'B2': 2, 'B1': 1}


def gateways(k):
    out = []
    for g in GROUPS:
        d = np.linalg.norm(GRID[g] - GRID[g].mean(0), axis=1)
        out.append(g[np.argsort(d, kind='stable')[:k]])
    return out


def build_mask(name, p=0.5, seed=0):
    base, wiring = name.split('_')
    gw = gateways(K[base]); rng = np.random.default_rng(seed)
    M = np.zeros((N, N), bool)
    for g in GROUPS: M[np.ix_(g, g)] = True
    pairs = [(i, j) for a in range(5) for b in range(a + 1, 5) for i in gw[a] for j in gw[b]]
    while True:
        keep = pairs if wiring == 'full' else [q for q in pairs if rng.random() < p]
        T = np.zeros((N, N), bool)
        for i, j in keep: T[i, j] = T[j, i] = True
        if wiring == 'full' or modules_connected(T):
            break
    M |= T
    np.fill_diagonal(M, True)
    return M


def modules_connected(T):
    link = np.zeros((5, 5), bool)
    for i, j in zip(*np.where(T)): link[MOD[i], MOD[j]] = True
    seen, stack = {0}, [0]
    while stack:
        a = stack.pop()
        for b in np.where(link[a])[0]:
            if b not in seen: seen.add(b); stack.append(b)
    return len(seen) == 5


if __name__ == '__main__':
    for n in ('A_full', 'A_rand', 'B2_full', 'B2_rand', 'B1_full', 'B1_rand'):
        M = build_mask(n); off = M & ~np.eye(N, dtype=bool)
        inter = off & (MOD[:, None] != MOD[None])
        print(f'{n:8s} directed edges {off.sum():5d}  inter-module {inter.sum():4d}  density {off.mean():.3f}  symmetric {bool((M == M.T).all())}')
