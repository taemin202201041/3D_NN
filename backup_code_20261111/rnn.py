"""THE RNN PART -- the file you change.  Everything before (perception, questions, dataset) and after (readout, answer
mask, loss, scoring, JSON) lives elsewhere and does not care what is in here.

Contract (model.py relies on exactly this):
    core = RecurrentCore(cfg, d_in, seed, in_mask)        in_mask: None or 0/1 vector over neurons that receive input
    core(x)                      x: (B, T, d_in)  ->  h_T: (B, units)   state after the last step
    core.effective_recurrent()   (units, units) matrix whose entries are the directed edges i -> j  (mask applied)
    core.rec_mask                buffer (units, units) 0/1, set by run.py from model.topology / pruning
    penalty(core.effective_recurrent(), geom.distance(), form, gamma, eps)   spatial penalty, added to the task loss
model.cell = "rnn" (default) is the team's seRNN cell; model.activation="linear" makes it a linear RNN.
model.cell = "lstm" is a plain nn.LSTM baseline (no spatial structure: use reg.form=none, topology=full, io=all).
Default cell:   pre = x K + (sum_k a_k h_{t-k}) (W * M) + b ;  h = (1-a) h_prev + a act(pre)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ACTS = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "softplus": nn.functional.softplus,
        "gelu": nn.functional.gelu, "linear": lambda z: z}


class LSTMCore(nn.Module):
    """baseline: nn.LSTM with `units` hidden units, state after the last step. The 100x100 'recurrent matrix' reported for
    structure metrics is only the input-gate block of weight_hh -- not a neuron-to-neuron graph, do not interpret it."""

    def __init__(self, cfg: dict, d_in: int, seed: int, in_mask=None):
        super().__init__()
        m = cfg["model"]
        if cfg["reg"]["form"] != "none" or m["topology"] != "full" or m["io"] != "all" or m["autapse_only"] or cfg["prune"]["method"] != "none":
            raise ValueError("model.cell=lstm supports only reg.form=none, topology=full, io=all, no pruning")
        torch.manual_seed(seed)
        self.units = m["units"]
        self.lstm = nn.LSTM(d_in, self.units, batch_first=True)
        self.register_buffer("rec_mask", torch.ones(self.units, self.units))

    def effective_recurrent(self):
        return self.lstm.weight_hh_l0[: self.units]

    def forward(self, x):
        return self.lstm(x)[0][:, -1]


class SeRNNCore(nn.Module):
    def __init__(self, cfg: dict, d_in: int, seed: int, in_mask=None):
        super().__init__()
        m = cfg["model"]
        n = m["units"]
        g = torch.Generator().manual_seed(seed)
        self.units, self.lags, self.leak, self.act = n, int(m["lags"]), m["leak_alpha"], ACTS[m["activation"]]
        lim = (6.0 / (d_in + n)) ** 0.5
        self.kernel = nn.Parameter((torch.rand(d_in, n, generator=g) * 2 - 1) * lim)             # Glorot uniform
        q, r = torch.linalg.qr(torch.randn(n, n, generator=g))
        self.recurrent_kernel = nn.Parameter(q * torch.sign(torch.diagonal(r)))                  # orthogonal
        self.bias = nn.Parameter(torch.zeros(n))
        if self.lags > 1:
            a0 = torch.zeros(self.lags)
            a0[0] = 1.0
            self.lag_w = nn.Parameter(a0)
        self.register_buffer("rec_mask", torch.ones(n, n))
        self.register_buffer("in_mask", torch.ones(n) if in_mask is None else torch.as_tensor(in_mask, dtype=torch.float32))

    def effective_recurrent(self):
        return self.recurrent_kernel * self.rec_mask

    def forward(self, x):
        B, T, _ = x.shape
        hist = [x.new_zeros(B, self.units) for _ in range(self.lags)]                            # hist[0] = h_{t-1}
        u = x @ (self.kernel * self.in_mask[None, :]) + self.bias
        W = self.effective_recurrent()
        for t in range(T):
            mix = hist[0] if self.lags == 1 else sum(self.lag_w[k] * hist[k] for k in range(self.lags))
            new = self.act(u[:, t] + mix @ W)
            h = new if self.leak is None else (1.0 - self.leak) * hist[0] + self.leak * new
            hist = [h] + hist[:-1]
        return hist[0]


def RecurrentCore(cfg: dict, d_in: int, seed: int, in_mask=None):
    return (LSTMCore if cfg["model"].get("cell", "rnn") == "lstm" else SeRNNCore)(cfg, d_in, seed, in_mask)


def penalty(w_eff, dist, form: str, gamma: float, eps: float):
    """spatial penalty on the effective recurrent matrix: dc = g sum|W| D C, c = g sum|W| C, d = g sum|W| D, l1, none"""
    a = w_eff.abs()
    if form == "none" or gamma == 0:
        return a.new_zeros(())
    if form == "l1":
        return gamma * a.sum()
    if form == "d":
        return gamma * (a * dist).sum()
    if form in ("dc", "c"):
        inv = torch.diag((a.sum(1) + eps) ** -0.5)
        c = torch.linalg.matrix_exp(inv @ a @ inv) * (1 - torch.eye(len(a), device=a.device))   # communicability, diag 0
        return gamma * (a * c * (dist if form == "dc" else 1.0)).sum()
    raise ValueError(f"unknown reg.form {form}")


def communicability_np(abs_w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    from scipy.linalg import expm
    inv = np.diag((abs_w.sum(1) + eps) ** -0.5)
    c = expm(inv @ abs_w @ inv)
    np.fill_diagonal(c, 0.0)
    return c
