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


SLOTS, FEAT, PARTNER0, PRESENT = 6, 37, 14, 23       # annotation input: 6 object slots x 37 features per step (perception.py)


def permute_slots(video, order=None):
    """video [B, T, 6*37] -> the 6 object slots in a random order per sample (the same order at every time step).
    partner0..5 are columns that NAME slots, so they are re-ordered with the same permutation.  new slot i = old slot order[i]."""
    B, T, _ = video.shape
    x = video.reshape(B, T, SLOTS, FEAT)
    if order is None:
        order = torch.argsort(torch.rand(B, SLOTS, device=video.device), dim=1)
    x = torch.gather(x, 2, order[:, None, :, None].expand(B, T, SLOTS, FEAT))
    pc = x[..., PARTNER0:PARTNER0 + SLOTS]
    pc = torch.gather(pc, 3, order[:, None, None, :].expand(B, T, SLOTS, SLOTS))
    x = torch.cat([x[..., :PARTNER0], pc, x[..., PARTNER0 + SLOTS:]], -1)
    return x.reshape(B, T, SLOTS * FEAT)


class SeRNN(nn.Module):
    def __init__(self, cfg: dict, geom, d_video: int, d_q: int, seed: int, emb=None):
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
        self.slot_shuffle = bool(m.get("slot_shuffle", False))   # augmentation (training only): random order of the 6 object slots
        self._force_shuffle = False                              # analysis switch: shuffle also at evaluation time
        od = int(m.get("obj_dim", 0))                            # shared per-object encoder (annotation input only): 37 -> od per slot, same weights for all 6 slots
        self.objmode = m.get("obj_mode", "concat")               # "concat" (6*od, keeps the slot positions) | "pool" (sum and max over slots, order-free)
        self.objenc = None
        d_video_eff = d_video
        if (od or self.slot_shuffle) and d_video != SLOTS * FEAT:
            raise ValueError("model.slot_shuffle / model.obj_dim need the annotation input (6 slots x 37 features)")
        if od:
            self.objenc = nn.Sequential(nn.Linear(FEAT, od), nn.ReLU())
            d_video_eff = SLOTS * od if self.objmode == "concat" else 2 * od
        vd = int(m.get("video_dim", 0))                        # optional learned linear reduction of the per-frame video feature (0 = off)
        self.vproj = nn.Linear(d_video_eff, vd) if vd else None
        self.qenc = m.get("q_encoder", "mean")               # "mean" (default, question = one fixed vector) | "positional" | "seq_core" (word-order encodings, no LSTM/GRU)
        d_in = (vd or d_video_eff) + d_q
        if self.qenc == "mean_embed":                            # control: word embeddings learned from THIS dataset's questions (answers only), averaged
            if emb is None:
                raise ValueError("model.q_encoder needs data.question_encoding = word_seq")
            qd = int(m.get("q_dim", 300))
            self.qemb = nn.Embedding(len(emb), qd, padding_idx=0)
            d_in = (vd or d_video_eff) + qd
        elif self.qenc != "mean":
            if emb is None:
                raise ValueError("model.q_encoder needs data.question_encoding = word_seq")
            qd = int(m.get("q_dim", 32))
            self.qemb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float32), freeze=True, padding_idx=0)
            self.qproj = nn.Linear(self.qemb.embedding_dim, qd)
            d_in = (vd or d_video_eff) + (d_q * qd if self.qenc == "positional" else qd)
        self.core = RecurrentCore(cfg, d_in, seed, in_mask)
        self.register_buffer("out_mask", torch.ones(m["units"]) if out_mask is None else torch.as_tensor(out_mask))
        self.readout = nn.Linear(m["units"], N_CLASSES)
        self.register_buffer("allowed", torch.as_tensor(ALLOWED))

    def forward(self, video, q, grp):
        if self.cfg["model"].get("video_off", False):      # control: question-only model (video features zeroed)
            video = torch.zeros_like(video)
        p_drop = float(self.cfg["model"].get("video_dropout", 0.0))     # regularisation: dropout on the video features (training only)
        if self.training and p_drop > 0:
            video = torch.nn.functional.dropout(video, p_drop, True)
        if self.slot_shuffle and (self.training or self._force_shuffle):
            video = permute_slots(video)
        if self.objenc is not None:
            B, T, _ = video.shape
            x5 = video.reshape(B, T, SLOTS, FEAT)
            h = self.objenc(x5) * x5[..., PRESENT:PRESENT + 1]            # empty slots contribute nothing
            video = h.reshape(B, T, -1) if self.objmode == "concat" else torch.cat([h.sum(2), h.max(2).values], -1)
        if self.vproj is not None:
            video = self.vproj(video)
        if self.qenc == "mean":
            x = torch.cat([video, q[:, None, :].expand(-1, video.shape[1], -1)], -1)
        elif self.qenc == "mean_embed":
            ids = q.long()
            msk = (ids > 0).unsqueeze(-1).float()
            qv = (self.qemb(ids) * msk).sum(1) / msk.sum(1).clamp(min=1.0)
            x = torch.cat([video, qv[:, None, :].expand(-1, video.shape[1], -1)], -1)
        else:
            ids = q.long()
            e = self.qproj(self.qemb(ids)) * (ids > 0).unsqueeze(-1)            # [B, L, qd], padding = 0
            if self.qenc == "positional":                                       # words keep their position, no recurrence
                x = torch.cat([video, e.flatten(1)[:, None, :].expand(-1, video.shape[1], -1)], -1)
            else:                                                               # the core itself reads the words one by one, then the frames
                B, L, T = video.shape[0], e.shape[1], video.shape[1]
                xw = torch.cat([video.new_zeros(B, L, video.shape[2]), e], -1)
                xv = torch.cat([video, e.new_zeros(B, T, e.shape[2])], -1)
                x = torch.cat([xw, xv], 1)
        if self.training and self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        logits = self.readout(self.core(x) * self.out_mask)
        return logits + (self.allowed[grp] - 1.0) * 1e9 if self.answer_mask else logits

    def spatial_penalty(self):
        return penalty(self.core.effective_recurrent(), self.geom.distance(), self.form, self.gamma, self.eps)
