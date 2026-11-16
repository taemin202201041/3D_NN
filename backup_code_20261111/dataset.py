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
    train_q = json.loads((qdir / "train.json").read_text(encoding="utf-8"))
    val_q = json.loads((qdir / "validation.json").read_text(encoding="utf-8"))
    keep = lambda qs: [v for v in qs if v["scene_index"] in row_of]
    train_q, val_q = keep(train_q), keep(val_q)
    vocab = Q.build_vocab([train_q])                                     # vocabulary from the training scenes only
    samples = Q.encode_videos(train_q + val_q, vocab, row_of, w2v)
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


_CACHE: dict = {}


def load(cfg: dict, here: Path) -> dict:
    """Returns video [n_scenes, T, D] float16 (standardised) and per-split sample dicts."""
    d = cfg["data"]
    path = build_cache(cfg, here)
    key = str(path)
    if key not in _CACHE:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        raw = np.load(path / "video_raw.npy", mmap_mode="r")
        mean, std = np.asarray(meta["feat_mean"], np.float32), np.asarray(meta["feat_std"], np.float32)
        video = ((np.asarray(raw, np.float32) - mean) / std).astype(np.float16)
        _CACHE[key] = (meta, video, dict(np.load(path / "samples.npz")))
    meta, video, s = _CACHE[key]
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
