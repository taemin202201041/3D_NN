"""Cache building and loading (numpy; batches() gives torch tensors).

data_large_v1 layout (paths.clevrer_root):  manifest.json  features/<scene:05d>.pt  questions/train.json  questions/validation.json
manifest lists the scenes of each split; rows of the feature table are  train (2000) + dev (500) + local_test (500),  so
    data.train_scenes [0, 2000]   data.val_scenes [2000, 2500]   data.test_scenes [2500, 3000]    index ROWS of that table.

Cache (built once):
    video_raw.npy   float16 [3000, T, D]     perception.perceive() of every scene
    samples.npz     question samples (see questions.encode_videos)
    meta.json       vocab, classes, train statistics for standardisation, scene ids
Loading standardises the features with train statistics and selects splits / subsamples without rebuilding the cache.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import perception as P
import questions as Q

MANIFEST_SPLITS = ("train", "dev", "local_test")


def cache_dir(cfg: dict, here: Path) -> Path:
    d = cfg["data"]
    return (here / cfg["paths"]["cache_dir"]).resolve() / f"{d['perception']}_{d['question_encoding']}_t{d['steps']}"


def build_cache(cfg: dict, here: Path) -> Path:
    out = cache_dir(cfg, here)
    if (out / "meta.json").exists():
        return out
    if cfg["data"]["perception"] == "annotation":
        return build_annotation_cache(cfg, here, out)
    root = (here / cfg["paths"]["clevrer_root"]).resolve()
    qdir = root / cfg["paths"].get("questions_dir", "questions")
    need = [root / "manifest.json", qdir / "train.json", qdir / "validation.json", root / cfg["paths"].get("features_dir", "features")]
    missing = [str(x) for x in need if not x.exists()]
    if missing:
        raise FileNotFoundError("data files not found. Set paths.clevrer_root in config/base.json. Missing: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scenes = [s for k in MANIFEST_SPLITS for s in manifest[k]]
    row_of = {s: i for i, s in enumerate(scenes)}
    paper = (here / cfg["paths"].get("paper_dir", "data/clevrer_paper")).resolve()
    if cfg["data"]["perception"] == "resnet50_npy":                      # CLEVRER-paper features (make_features_r50.py)
        perc = P.NpyPerception(paper, cfg["data"]["steps"])
    elif cfg["data"]["perception"] == "resnet50_motion":                 # ResNet-50 appearance + optical-flow motion (make_motion_features.py)
        perc = P.NpyConcatPerception(paper, ["features_r50", "motion_g4"], cfg["data"]["steps"], name="resnet50_motion")
    elif cfg["data"]["perception"] == "motion_g4":                       # motion only (control)
        perc = P.NpyConcatPerception(paper, ["motion_g4"], cfg["data"]["steps"], name="motion_g4")
    elif cfg["data"]["perception"] == "resnet50_grid2":                  # spatial 2x2 grid of the last conv map (make_features_r50_grid.py)
        perc = P.NpyPerception(paper, cfg["data"]["steps"], stem="features_r50_g2", name="resnet50_grid2")
    else:
        perc = P.FeaturePerception(root, cfg["paths"].get("features_dir", "features"), cfg["data"]["steps"])
    w2v = None
    if cfg["data"]["question_encoding"] == "word2vec_mean":              # CLEVRER-paper question encoding (make_w2v.py)
        ws = json.loads((paper / "w2v_words.json").read_text(encoding="utf-8"))
        vec = np.load(paper / "w2v.npy")
        w2v = {w: vec[i] for i, w in enumerate(ws) if vec[i].any()}
    t0 = time.time()
    x0 = perc.perceive(scenes[0])
    video = np.lib.format.open_memmap(out / "video_raw.npy", mode="w+", dtype=np.float16, shape=(len(scenes), *x0.shape))
    for i, s in enumerate(scenes):
        video[i] = perc.perceive(s)
    video.flush()
    enc_program = cfg["data"]["question_encoding"] == "program_token_counts"   # diagnostic: video path + the team's question encoding
    if enc_program:
        oq = (here / cfg["paths"].get("official_questions_dir", "../data_large_v1/data_large_v1/cache")).resolve()
        train_q = json.loads((oq / "official_questions_train.json").read_text(encoding="utf-8"))
        val_q = json.loads((oq / "official_questions_validation.json").read_text(encoding="utf-8"))
    else:
        train_q = json.loads((qdir / "train.json").read_text(encoding="utf-8"))
        val_q = json.loads((qdir / "validation.json").read_text(encoding="utf-8"))
    keep = lambda qs: [v for v in qs if v["scene_index"] in row_of]
    train_q, val_q = keep(train_q), keep(val_q)
    vocab = Q.build_vocab([train_q], program=enc_program)                # vocabulary from the training scenes only
    samples = Q.encode_videos(train_q + val_q, vocab, row_of, w2v, program=enc_program)
    np.savez(out / "samples.npz", **samples)
    tr = np.asarray(video[: len(manifest["train"])], np.float32).reshape(-1, x0.shape[-1])
    meta = {"vocab": vocab, "classes": Q.CLASSES, "groups": Q.GROUPS, "qtypes": Q.QTYPES, "subtypes": Q.SUBTYPES,
            "scenes": scenes, "split_sizes": {k: len(manifest[k]) for k in MANIFEST_SPLITS},
            "feat_mean": tr.mean(0).tolist(), "feat_std": (tr.std(0) + 1e-6).tolist(), "steps": int(x0.shape[0]), "feat_dim": int(x0.shape[1]),
            "perception": perc.name, "encoder_settings": perc.settings, "n_samples": int(len(samples["y"])),
            "built_seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[cache] done: {out} ({time.time() - t0:.0f}s)", flush=True)
    return out


N_ANN_SCENES = 15000                       # CLEVRER videos 0-9999 (train) + 10000-14999 (validation); row = video id


def build_annotation_cache(cfg: dict, here: Path, out: Path) -> Path:
    """Team input: CLEVRER annotation as the perception output [25, 6 slots * 37], question = program-token counts (official questions
    with `program`). Rows are video ids 0..14999, so data.*_scenes are video-id ranges exactly like the team's code."""
    paths = cfg["paths"]
    adir = (here / paths.get("annotations_dir", "data/clevrer/annotations")).resolve()
    qdir = (here / paths.get("official_questions_dir", "../data_large_v1/data_large_v1/cache")).resolve()
    qf = [qdir / "official_questions_train.json", qdir / "official_questions_validation.json"]
    missing = [str(x) for x in qf + [adir / "annotation_00000-01000", adir / "annotation_14000-15000"] if not x.exists()]
    if missing:
        raise FileNotFoundError("annotation input needs: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    perc = P.AnnotationPerception(adir, cfg["data"].get("frames", 128), cfg["data"]["steps"])
    t0 = time.time()
    video = np.lib.format.open_memmap(out / "video_raw.npy", mode="w+", dtype=np.float16, shape=(N_ANN_SCENES, perc.steps, P.SLOTS * P.F))
    for s in range(N_ANN_SCENES):
        video[s] = perc.perceive(s)
        if s % 3000 == 2999:
            print(f"[cache] perceived {s + 1}/{N_ANN_SCENES} scenes ({time.time() - t0:.0f}s)", flush=True)
    video.flush()
    train_q = json.loads(qf[0].read_text(encoding="utf-8"))
    val_q = json.loads(qf[1].read_text(encoding="utf-8"))
    enc = cfg["data"]["question_encoding"]
    program = enc == "program_token_counts"
    w2v = None
    if enc == "word2vec_mean":                                           # diagnostic: annotation input + the original word2vec question vector
        paper = (here / paths.get("paper_dir", "data/clevrer_paper")).resolve()
        ws = json.loads((paper / "w2v_words.json").read_text(encoding="utf-8"))
        vec = np.load(paper / "w2v.npy")
        w2v = {w: vec[i] for i, w in enumerate(ws) if vec[i].any()}
    vocab = Q.build_vocab([train_q], program=program)
    unseen = set(Q.build_vocab([val_q], program=program)) - set(vocab)
    if unseen and program:
        raise ValueError(f"program tokens in validation not in train vocab: {sorted(unseen)}")
    row_of = {i: i for i in range(N_ANN_SCENES)}
    samples = Q.encode_videos(train_q + val_q, vocab, row_of, w2v, program=program)
    np.savez(out / "samples.npz", **samples)
    tr = np.asarray(video[:9000], np.float32).reshape(-1, P.F)                  # statistics of present objects, scenes 0-8999 (as the team)
    tr = tr[tr[:, P.I_PRESENT] > 0]
    meta = {"annotation": True, "vocab": vocab, "classes": Q.CLASSES, "groups": Q.GROUPS, "qtypes": Q.QTYPES, "subtypes": Q.SUBTYPES,
            "scenes": list(range(N_ANN_SCENES)), "features": P.FEATURES, "continuous": P.CONTINUOUS,
            "cont_mean": tr[:, P.CONTINUOUS].mean(0).tolist(), "cont_std": (tr[:, P.CONTINUOUS].std(0) + 1e-6).tolist(),
            "steps": int(perc.steps), "feat_dim": int(P.SLOTS * P.F), "perception": perc.name, "encoder_settings": perc.settings,
            "n_samples": int(len(samples["y"])), "built_seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[cache] done: {out} ({time.time() - t0:.0f}s)", flush=True)
    return out


def _annotation_video(raw, meta: dict, mask_offscreen: bool) -> np.ndarray:
    """standardise the continuous features with train statistics, zero the slots of absent (and optionally off-screen) objects"""
    mean, std = np.asarray(meta["cont_mean"], np.float32), np.asarray(meta["cont_std"], np.float32)
    out = np.empty(raw.shape, np.float16)
    for lo in range(0, raw.shape[0], 1000):
        x = np.asarray(raw[lo:lo + 1000], np.float32).reshape(-1, raw.shape[1], P.SLOTS, P.F)
        x[..., P.CONTINUOUS] = (x[..., P.CONTINUOUS] - mean) / std
        hide = x[..., P.I_PRESENT] == 0
        if mask_offscreen:
            hide |= x[..., P.I_VISIBLE] == 0
        x[hide] = 0.0
        out[lo:lo + 1000] = x.reshape(x.shape[0], x.shape[1], -1)
    return out


_CACHE: dict = {}
_PCA: dict = {}


def load(cfg: dict, here: Path) -> dict:
    """Returns video [n_scenes, T, D] float16 (standardised) and per-split sample dicts."""
    d = cfg["data"]
    path = build_cache(cfg, here)
    key = str(path) + (f"|offscreen_masked={bool(d.get('mask_offscreen', True))}" if d["perception"] == "annotation" else "")
    if key not in _CACHE:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        raw = np.load(path / "video_raw.npy", mmap_mode="r")
        if meta.get("annotation"):
            video = _annotation_video(raw, meta, bool(d.get("mask_offscreen", True)))
        else:
            mean, std = np.asarray(meta["feat_mean"], np.float32), np.asarray(meta["feat_std"], np.float32)
            video = ((np.asarray(raw, np.float32) - mean) / std).astype(np.float16)
        _CACHE[key] = (meta, video, dict(np.load(path / "samples.npz")))
    meta, video, s = _CACHE[key]
    k = int(d.get("pca_dim", 0))
    if k:                                    # fixed (not learned) reduction: top-k principal components of the TRAIN frames, each unit variance
        lo_t, hi_t = d["train_scenes"]
        pk = (key, k, lo_t, hi_t)
        if pk not in _PCA:
            x = video[lo_t:hi_t].reshape(-1, video.shape[-1]).astype(np.float32)
            mu = x.mean(0)
            ev, evec = np.linalg.eigh(np.cov((x - mu).T))
            top = np.argsort(ev)[::-1][:k]
            comp, sd = evec[:, top].astype(np.float32), np.sqrt(ev[top]).astype(np.float32) + 1e-6
            flat = ((video.reshape(-1, video.shape[-1]).astype(np.float32) - mu) @ comp) / sd
            _PCA[pk] = flat.reshape(video.shape[0], video.shape[1], k).astype(np.float16)
            print(f"[pca] {video.shape[-1]} -> {k} dims, explained variance {float(ev[top].sum() / ev.sum()):.3f}", flush=True)
        video = _PCA[pk]
    allowed_types = [Q.QTYPES.index(t) for t in d["question_types"]]
    rng = np.random.default_rng(12345)
    splits = {}
    for name in ["train", "val", "test"]:
        lo, hi = d[f"{name}_scenes"]
        rows = np.arange(lo, hi)
        if d.get("subsample", 1.0) < 1.0:
            rows = np.sort(rng.choice(rows, max(1, int(len(rows) * d["subsample"])), replace=False))
        sel = np.isin(s["scene"], rows) & np.isin(s["qtype"], allowed_types)
        splits[name] = {k: v[sel] for k, v in s.items()}
        splits[name]["n_scenes"] = int(len(rows))
    return {"meta": meta, "video": video, "splits": splits}


def batches(video, split: dict, batch: int, shuffle: bool, seed: int = 0, device="cpu"):
    """`video` = ONE torch tensor [n_scenes, T, D] (float16) on `device`. Yields ((video, q, agroup), y) float32/long."""
    import torch

    n = len(split["y"])
    t = {k: torch.as_tensor(split[k]).to(device) for k in ("scene", "q", "agroup", "y")}
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, batch):
        b = order[i:i + batch]
        yield (video[t["scene"][b].long()].float(), t["q"][b].float(), t["agroup"][b].long()), t["y"][b].long()
