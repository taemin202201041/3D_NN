"""Is the information the questions need actually inside the frozen ResNet-50 features?  (read-only, results go to stdout + results/probe_features.json)

A linear probe (logistic regression on standardised features) tries to read ground-truth facts from the features:
  frame level : number of visible objects, which colours / shapes / materials are visible        (one frame -> its own annotation)
  video level : total number of objects, which attributes exist, number of collisions            (mean and std of the 25 frames)
Features compared: global pool5 (2048 / frame)  vs  2x2 spatial grid (8192 / frame).
Train rows 0-1999, test rows 2500-2999 of the 3000-scene table (same rows as the experiments). Chance = predict the majority.
"""
import json
from pathlib import Path

import numpy as np
import torch

import perception as P

HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
paper = HERE / "data/clevrer_paper"
scenes = json.loads((paper / "features_r50.json").read_text(encoding="utf-8"))["scenes"]
frame_idx = np.round(np.linspace(0, 127, 25)).astype(int)
ann = P.AnnotationPerception(HERE / "data/clevrer/annotations")

# ---- labels from the annotation (same json the team uses)
n_obj, collisions = [], []
attr_video = []                         # [scene, 13]  colours(8) materials(2) shapes(3) existing in the video
n_vis, attr_frame = [], []              # [scene, 25], [scene, 25, 13]
CM = P.COLORS + P.MATERIALS + P.SHAPES
for s in scenes:
    a = json.loads(ann.path(s).read_text(encoding="utf-8"))
    props = {p["object_id"]: p for p in a["object_property"]}
    n_obj.append(len(props)); collisions.append(len(a["collision"]))
    def vec(ids):
        v = np.zeros(13, np.float32)
        for k in ids:
            p = props[k]; v[CM.index(p["color"])] = 1; v[CM.index(p["material"])] = 1; v[CM.index(p["shape"])] = 1
        return v
    attr_video.append(vec(props))
    vis_n, vis_a = [], []
    for f in frame_idx:
        ids = [o["object_id"] for o in a["motion_trajectory"][f]["objects"] if o["inside_camera_view"]]
        vis_n.append(len(ids)); vis_a.append(vec(ids))
    n_vis.append(vis_n); attr_frame.append(vis_a)
n_obj, collisions = np.array(n_obj), np.minimum(np.array(collisions), 4)
attr_video, n_vis, attr_frame = np.array(attr_video), np.array(n_vis), np.array(attr_frame)
print("objects per video", np.bincount(n_obj), " collisions(0-4+)", np.bincount(collisions), " visible per frame", np.bincount(n_vis.ravel()))

TR, TE = slice(0, 2000), slice(2500, 3000)


def fit_eval(xtr, ytr, xte, yte, kind):
    """kind 'multi' -> y int classes; 'bin' -> y [n, L] 0/1.  returns (probe accuracy, majority accuracy)"""
    xtr, xte = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (xtr, xte))
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    if kind == "multi":
        k = int(max(ytr.max(), yte.max())) + 1
        yt, ye = torch.tensor(ytr, device=dev).long(), torch.tensor(yte, device=dev).long()
        lin = torch.nn.Linear(xtr.shape[1], k).to(dev)
        opt = torch.optim.Adam(lin.parameters(), 3e-3, weight_decay=1e-3)
        for _ in range(300):
            opt.zero_grad(); torch.nn.functional.cross_entropy(lin(xtr), yt).backward(); opt.step()
        acc = float((lin(xte).argmax(-1) == ye).float().mean())
        maj = float((ye == torch.bincount(yt).argmax()).float().mean())
        return acc, maj
    yt, ye = torch.tensor(ytr, device=dev), torch.tensor(yte, device=dev)
    lin = torch.nn.Linear(xtr.shape[1], yt.shape[1]).to(dev)
    opt = torch.optim.Adam(lin.parameters(), 3e-3, weight_decay=1e-3)
    for _ in range(300):
        opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(lin(xtr), yt).backward(); opt.step()
    pred = (lin(xte) > 0).float()
    acc = float((pred == ye).float().mean())
    maj = float(((yt.mean(0) > 0.5).float()[None] == ye).float().mean())
    return acc, maj


out = {}
for name, stem in [("global 2048", "features_r50"), ("grid2x2 8192", "features_r50_g2")]:
    x = np.load(paper / f"{stem}.npy", mmap_mode="r")
    x = np.asarray(x, np.float32)                                              # [3000, 25, D]
    # frame level: every frame is a sample
    xf_tr, xf_te = x[TR].reshape(-1, x.shape[-1]), x[TE].reshape(-1, x.shape[-1])
    r = {}
    r["frame: #visible objects (0-6)"] = fit_eval(xf_tr, n_vis[TR].ravel(), xf_te, n_vis[TE].ravel(), "multi")
    r["frame: visible colour/material/shape (13 yes/no)"] = fit_eval(xf_tr, attr_frame[TR].reshape(-1, 13), xf_te, attr_frame[TE].reshape(-1, 13), "bin")
    # video level: mean and std over the 25 frames
    xv = np.concatenate([x.mean(1), x.std(1)], -1)
    r["video: total objects"] = fit_eval(xv[TR], n_obj[TR], xv[TE], n_obj[TE], "multi")
    r["video: which attributes exist (13 yes/no)"] = fit_eval(xv[TR], attr_video[TR], xv[TE], attr_video[TE], "bin")
    r["video: #collisions (0-4+)"] = fit_eval(xv[TR], collisions[TR], xv[TE], collisions[TE], "multi")
    out[name] = r
    del x, xf_tr, xf_te, xv
print(f"\n{'task':52s} {'majority':>9s}  " + "  ".join(f"{n:>14s}" for n in out))
for t in out["global 2048"]:
    print(f"{t:52s} {out['global 2048'][t][1]:9.3f}  " + "  ".join(f"{out[n][t][0]:14.3f}" for n in out))
(HERE / "results/probe_features.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
