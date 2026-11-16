"""Accuracy breakdown (every metric is reported) and structure metrics of W ⊙ M."""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

from questions import ALLOWED, QTYPES, SUBTYPES


def scores(pred: np.ndarray, s: dict) -> dict:
    """pred: predicted class per sample. Multiple-choice 'per_question' = all choices right."""
    ok = pred == s["y"]
    out = {"acc_all": float(ok.mean()) if len(ok) else float("nan"), "n_samples": int(len(ok))}
    for i, name in enumerate(QTYPES):
        sel = s["qtype"] == i
        if not sel.any():
            continue
        out[f"{name}_per_option"] = float(ok[sel].mean())
        if i > 0:
            _, inv = np.unique(s["qkey"][sel], return_inverse=True)
            wrong = np.bincount(inv, weights=(~ok[sel]).astype(float))
            out[f"{name}_per_question"] = float((wrong == 0).mean())
    for j, name in enumerate(SUBTYPES):
        sel = s["subtype"] == j
        if sel.any():
            out[f"desc_{name}"] = float(ok[sel].mean())
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def baselines(train: dict, split: dict, n_classes: int) -> dict:
    """Majority answer (from train, per answer group) and expected chance (uniform over allowed answers)."""
    maj = np.zeros(len(ALLOWED), int)
    for g in range(len(ALLOWED)):
        y = train["y"][train["agroup"] == g]
        maj[g] = np.bincount(y, minlength=n_classes).argmax() if len(y) else 0
    glob_major = np.bincount(train["y"], minlength=n_classes).argmax()
    chance_mask = float((1.0 / ALLOWED.sum(1))[split["agroup"]].mean())
    return {"majority_per_group": scores(maj[split["agroup"]], split),
            "majority_global": scores(np.full(len(split["y"]), glob_major), split),
            "chance_all23": round(1.0 / n_classes, 4),
            "chance_masked": round(chance_mask, 4)}


# ---------------------------------------------------------------------- structure
def _modularity(a):
    k = a.sum(1)
    two_m = k.sum()
    if two_m == 0:
        return 0.0, 1
    b = a - np.outer(k, k) / two_m
    groups, final = [np.arange(len(a))], []
    while groups:
        g = groups.pop()
        bg = b[np.ix_(g, g)].copy()
        bg -= np.diag(bg.sum(1))
        vals, vecs = np.linalg.eigh(bg)
        s = np.where(vecs[:, -1] >= 0, 1.0, -1.0)
        if vals[-1] <= 1e-8 or s @ bg @ s <= 1e-8 or abs(s.sum()) == len(s):
            final.append(g)
            continue
        groups += [g[s > 0], g[s < 0]]
    labels = np.empty(len(a), int)
    for i, g in enumerate(final):
        labels[g] = i
    return float((b * (labels[:, None] == labels[None, :])).sum() / two_m), len(final)


def _clust_path(a):
    deg = a.sum(1)
    tri = np.diag(a @ a @ a)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl = np.where(deg > 1, tri / (deg * (deg - 1)), 0.0)
    sp = shortest_path(a, unweighted=True, directed=False)
    off = ~np.eye(len(a), dtype=bool) & np.isfinite(sp)
    return float(cl.mean()), float(sp[off].mean()) if off.any() else float("nan")


def _sigma(a, rng, n_rand=20):
    c, l = _clust_path(a)
    n, m = len(a), int(a.sum() / 2)
    iu = np.triu_indices(n, 1)
    cr, lr = [], []
    for _ in range(n_rand):
        r = np.zeros((n, n))
        pick = rng.choice(len(iu[0]), m, replace=False)
        r[iu[0][pick], iu[1][pick]] = 1
        r = r + r.T
        c1, l1 = _clust_path(r)
        cr.append(c1); lr.append(l1)
    return float((c / np.mean(cr)) / (l / np.mean(lr))) if np.mean(cr) > 0 else float("nan")


def structure(w_eff: np.ndarray, dist: np.ndarray, threshold: float, comm_fn) -> dict:
    """Directed W (i -> j). Edge counts include self-connections; possible = N*N."""
    aw = np.abs(w_eff).astype(np.float64)
    n = len(aw)
    off = ~np.eye(n, dtype=bool)
    out = {"edges_possible": n * n,
           "edges_nonzero": int((aw > 0).sum()),
           "edges_active": int((aw > threshold).sum()),
           "edges_active_offdiag": int((aw[off] > threshold).sum()),
           "edge_threshold": threshold,
           "total_abs": float(aw.sum()),
           "wiring_cost": float((aw * dist).sum())}
    out["mean_len"] = out["wiring_cost"] / out["total_abs"] if out["total_abs"] > 0 else float("nan")
    nz = off & (aw > 0)
    out["rho_w_d"] = float(spearmanr(aw[nz], dist[nz]).statistic) if nz.sum() > 2 else float("nan")
    c = comm_fn(aw)
    out["comm_mean"] = float(c[off].mean())
    out["comm_len"] = float((c * dist).sum() / c.sum()) if c.sum() > 0 else float("nan")
    vals = aw[off]
    thr = np.quantile(vals, 0.9) if (vals > 0).any() else np.inf
    a = ((aw >= thr) & off & (aw > 0)).astype(float)
    a = np.maximum(a, a.T)
    out["Q"], out["n_modules"] = _modularity(a)
    out["sigma"] = _sigma(a, np.random.default_rng(0)) if a.sum() > 0 else float("nan")
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}
