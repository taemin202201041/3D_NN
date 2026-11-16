"""Fixed connection topologies (which directed edges may exist), applied as the recurrent mask M.

model.topology
    A_full A_rand B2_* B1_*  gateway_structure.py (5 modules x 20 neurons, 4/2/1 gateways per module)
    full       every directed edge allowed (default, 10,000 incl. self-connections)
    gateway    neurons are split into spatial clusters. Inside a cluster every edge is allowed.
               Each cluster has `gateways` gateway neurons; gateways of different clusters are fully
               connected to each other. Non-gateway neurons have NO edge to other clusters, so all
               traffic between clusters must pass through the gateways (modular + short paths).
    bridges    inside a cluster every edge is allowed. In addition every neuron gets `bridges`
               outgoing edges to random neurons of other clusters (no gateways). A few long-range
               shortcuts on top of modules, like a small-world network.

Clusters are spatial: k-means on the STARTING coordinates (same neuron identity for the dynamic model).
"""

from __future__ import annotations

import numpy as np


def spatial_clusters(coords: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = coords[rng.choice(len(coords), k, replace=False)].astype(float)
    lab = np.zeros(len(coords), int)
    for _ in range(iters):
        d = ((coords[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all() and _ > 0:
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                c[j] = coords[lab == j].mean(0)
    return lab


def topology_mask(kind: str, coords: np.ndarray, cfg_topo: dict, seed: int) -> tuple[np.ndarray, dict]:
    n = len(coords)
    if kind == "full":
        return np.ones((n, n), np.float32), {"kind": "full"}
    if kind in ("A_full", "A_rand", "B2_full", "B2_rand", "B1_full", "B1_rand"):    # our 5-module / k-gateway structures
        from gateway_structure import build_mask
        m = build_mask(kind, seed=seed).astype(np.float32)
        return m, {"kind": kind, "edges_allowed": int(m.sum())}
    k = int(cfg_topo.get("clusters", 4))
    lab = spatial_clusters(coords, k, seed=0)
    same = lab[:, None] == lab[None, :]
    mask = same.astype(np.float32)
    info = {"kind": kind, "clusters": k, "cluster_sizes": np.bincount(lab, minlength=k).tolist()}
    rng = np.random.default_rng(seed)
    if kind == "gateway":
        g = int(cfg_topo.get("gateways", 2))
        gates = []
        center = np.array([coords[lab == j].mean(0) for j in range(k)])
        for j in range(k):
            members = np.flatnonzero(lab == j)
            others = np.delete(center, j, axis=0).mean(0)  # gateways face the other clusters
            order = members[np.argsort(((coords[members] - others) ** 2).sum(1))]
            gates += order[:g].tolist()
        gates = np.array(sorted(gates))
        mask[np.ix_(gates, gates)] = 1.0
        info.update(gateways_per_cluster=g, gateway_nodes=gates.tolist())
    elif kind == "bridges":
        b = int(cfg_topo.get("bridges", 1))
        for i in range(n):
            outside = np.flatnonzero(lab != lab[i])
            mask[i, rng.choice(outside, size=min(b, len(outside)), replace=False)] = 1.0
        info.update(bridges_per_node=b)
    else:
        raise ValueError(f"unknown model.topology {kind}")
    info["edges_allowed"] = int(mask.sum())
    return mask, info
