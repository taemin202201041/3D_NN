"""Front + back around the swappable core (rnn.py).

    x_t   = [video_t (SLOTS*F) ; question (2V+4)]  (question repeated at every step) + Gaussian input noise (train only)
    h_T   = core(x)                                 <- rnn.py
    logit = h_T W_out + b_out   (+ -1e9 on answers impossible for the question type if model.answer_mask)
    loss  = cross-entropy + spatial penalty + coordinate loss (repulsion) of the dynamic layout
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from questions import ALLOWED, N_CLASSES
from rnn import RecurrentCore, penalty


class SeRNN(nn.Module):
    def __init__(self, cfg: dict, geom, d_video: int, d_q: int, seed: int):
        super().__init__()
        m, r = cfg["model"], cfg["reg"]
        self.cfg = cfg
        self.geom = geom
        geom.attach(self)
        self.form, self.gamma, self.eps = r["form"], float(r["gamma"]), float(r["eps"])
        self.answer_mask, self.noise = bool(m["answer_mask"]), float(m["input_noise"])
        in_mask = out_mask = None
        if m["io"] == "restricted":
            ins, outs = geom.io_nodes(m.get("io_nodes", 25))
            in_mask = np.zeros(m["units"], np.float32)
            in_mask[ins] = 1
            out_mask = np.zeros(m["units"], np.float32)
            out_mask[outs] = 1
        elif m["io"] != "all":
            raise ValueError(f"unknown model.io {m['io']}")
        self.core = RecurrentCore(cfg, d_video + d_q, seed, in_mask)
        self.register_buffer("out_mask", torch.ones(m["units"]) if out_mask is None else torch.as_tensor(out_mask))
        self.readout = nn.Linear(m["units"], N_CLASSES)
        self.register_buffer("allowed", torch.as_tensor(ALLOWED))

    def forward(self, video, q, grp):
        if self.cfg["model"].get("video_off", False):      # control: question-only model (video features zeroed)
            video = torch.zeros_like(video)
        x =torch.cat([video, q[:, None, :].expand(-1, video.shape[1], -1)], -1)
        if self.training and self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        logits = self.readout(self.core(x) * self.out_mask)
        return logits + (self.allowed[grp] - 1.0) * 1e9 if self.answer_mask else logits

    def spatial_penalty(self):
        return penalty(self.core.effective_recurrent(), self.geom.distance(), self.form, self.gamma, self.eps)
