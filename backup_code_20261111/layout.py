"""Neuron coordinates + distance matrix for the spatial penalty -- DYNAMIC version (PyTorch).

Neurons start on a grid (neighbour spacing 1) and move during training.
layout.dynamic.method
    position  coordinates are trainable (own Adam, lr coord_lr), clipped to the box, + repulsion
              repulsion * sum_{i!=j} relu(min_dist - d_ij)^2 against collapse; frozen until `open_epoch`.
    swap      BIMT-style: after each epoch (from open_epoch) swap positions of neuron pairs when that lowers sum |W|*D.
    both      position + swap.
    relayout  train `open_epoch` epochs, rearrange ONCE (greedy swaps until none helps), then keep positions fixed.
Used by model.py / run.py:  distance(), extra_loss(), coord_params(), step(), on_epoch_end(epoch), set_coords(c),
coords_numpy(), io_nodes(), summary().
"""
from __future__ import annotations

import numpy as np
import torch


def make_coords(kind: str, shape: list[int], spacing: float, seed: int, n_clusters: int = 4) -> np.ndarray:
    dim, n = len(shape), int(np.prod(shape))
    hi = (np.asarray(shape) - 1) * spacing
    if kind == "grid":
        g = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, dim)
        return g.astype(np.float64) * spacing
    rng = np.random.default_rng(seed)
    if kind == "random":
        return rng.uniform(0, hi, size=(n, dim))
    if kind == "cluster":
        centers = rng.uniform(0, hi, size=(n_clusters, dim))
        return np.clip(centers[np.arange(n) % n_clusters] + rng.normal(0, 0.5 * spacing, size=(n, dim)), 0, hi)
    raise ValueError(f"unknown layout.kind {kind}")


def pairwise(c: np.ndarray) -> np.ndarray:
    return np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))


class Geometry:
    mode = "dynamic"

    def __init__(self, cfg: dict, device="cpu"):
        lay, dyn = cfg["layout"], cfg["layout"]["dynamic"]
        self.dim = int(lay["dim"])
        self.shape = list(lay["shape"][str(self.dim)])
        n = int(np.prod(self.shape))
        if n != cfg["model"]["units"]:
            raise ValueError(f"layout {self.shape} has {n} nodes, model.units={cfg['model']['units']}")
        self.kind, self.spacing, self.norm = lay["kind"], float(lay["spacing"]), lay["distance_norm"]
        self.method, self.open_epoch = dyn["method"], int(dyn["open_epoch"])
        self.lr, self.rep, self.min_dist = float(dyn["coord_lr"]), float(dyn["repulsion"]), float(dyn["min_dist"])
        self.swaps_per_epoch = int(dyn.get("swaps_per_epoch", 10))
        self.coords0 = make_coords(self.kind, self.shape, self.spacing, lay.get("seed", 0)).astype(np.float32)
        self.hi = torch.tensor((np.asarray(self.shape) - 1) * self.spacing, dtype=torch.float32, device=device)
        self.trainable = self.method in ("position", "both")
        self.coords = torch.nn.Parameter(torch.tensor(self.coords0, device=device), requires_grad=self.trainable)
        self.opt = torch.optim.Adam([self.coords], lr=self.lr) if self.trainable else None
        self.offdiag = 1.0 - torch.eye(n, device=device)
        self.is_open = False
        self.model, self.n_swaps, self.relayout_passes = None, 0, 0
        self.rng = np.random.default_rng(int(cfg["seed"]))

    def attach(self, model):
        self.model = model

    def coord_params(self):
        return [self.coords] if self.trainable else []

    def _raw(self):
        d = self.coords[:, None, :] - self.coords[None, :, :]
        return torch.sqrt((d * d).sum(-1) + 1e-12) * self.offdiag

    def distance(self):
        d = self._raw()
        return d / (d.sum() / self.offdiag.sum()) if self.norm == "mean1" else d

    def extra_loss(self):
        if not self.trainable or self.rep == 0:
            return torch.zeros((), device=self.coords.device)
        return self.rep * (torch.relu(self.min_dist - self._raw()) ** 2 * self.offdiag).sum()

    @torch.no_grad()
    def step(self):
        """call after backward(): move the coordinates (only once open)"""
        if self.trainable and self.is_open and self.coords.grad is not None:
            self.opt.step()
            self.coords.copy_(torch.max(torch.min(self.coords, self.hi), torch.zeros_like(self.hi)))
        self.coords.grad = None

    def on_epoch_end(self, epoch: int):
        if self.method == "relayout":
            if epoch + 1 == self.open_epoch and self.model is not None:
                self._relayout()
            return
        if epoch + 1 >= self.open_epoch:
            self.is_open = True
            if self.method in ("swap", "both") and self.model is not None:
                self._swap()

    def _relayout(self, max_passes: int = 30):
        for p in range(max_passes):
            before, keep = self.n_swaps, self.swaps_per_epoch
            self.swaps_per_epoch = len(self.coords0)
            self._swap()
            self.swaps_per_epoch = keep
            if self.n_swaps == before:
                break
        self.relayout_passes = p + 1

    def _swap(self):
        """Greedy BIMT swap: delta(i<->j) = 2[(AD)_ij + (AD)_ji - (AD)_ii - (AD)_jj + 2 a_ij d_ij], A = |W|+|W|^T."""
        w = np.abs(self.model.core.effective_recurrent().detach().cpu().numpy()).astype(np.float64)
        a = w + w.T
        np.fill_diagonal(a, 0.0)
        c = self.coords_numpy()
        for i in self.rng.permutation(len(c))[: self.swaps_per_epoch]:
            d = pairwise(c.astype(np.float64))
            ad = a @ d
            delta = 2 * (ad[i, :] + ad[:, i] - ad[i, i] - np.diag(ad) + 2 * a[i, :] * d[i, :])
            delta[i] = 0.0
            j = int(np.argmin(delta))
            if delta[j] < -1e-9:
                c[[i, j]] = c[[j, i]]
                self.n_swaps += 1
        self.set_coords(c)

    @torch.no_grad()
    def set_coords(self, c):
        self.coords.copy_(torch.as_tensor(np.asarray(c, np.float32)))

    def coords_numpy(self) -> np.ndarray:
        return self.coords.detach().cpu().numpy().copy()

    def distance_numpy(self) -> np.ndarray:
        with torch.no_grad():
            return self.distance().cpu().numpy()

    def io_nodes(self, n: int = 25):
        """Input/output neurons are fixed by their STARTING position (neuron identity)."""
        order = np.argsort(self.coords0[:, self.dim - 1 if self.dim == 3 else 0], kind="stable")
        return np.sort(order[:n]), np.sort(order[-n:])

    def summary(self) -> dict:
        raw = pairwise(self.coords_numpy().astype(np.float64))
        off = ~np.eye(len(raw), dtype=bool)
        disp = np.linalg.norm(self.coords_numpy() - self.coords0, axis=1)
        return {"mode": self.mode, "kind": self.kind, "dim": self.dim, "shape": self.shape, "spacing": self.spacing,
                "distance_norm": self.norm, "n_nodes": len(raw), "dynamic_method": self.method, "coord_lr": self.lr,
                "open_epoch": self.open_epoch, "repulsion": self.rep, "min_dist": self.min_dist,
                "swaps_per_epoch": self.swaps_per_epoch, "n_swaps": self.n_swaps,
                "relayout_passes": self.relayout_passes,
                "mean_displacement": round(float(disp.mean()), 4), "max_displacement": round(float(disp.max()), 4),
                "mean_dist_raw": round(float(raw[off].mean()), 4), "max_dist_raw": round(float(raw.max()), 4),
                "min_dist_raw": round(float(raw[off].min()), 4)}
