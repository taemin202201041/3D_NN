"""Does motion information (feature change between frames) make collisions readable?  read-only.
 window level: label = a collision happens between sampled frame t and t+1 (the team's 'collide' flag); probe inputs compared:
   f_t only | f_t and f_{t+1} | f_t and (f_{t+1}-f_t)
 video level: number of collisions from [mean, std] (appearance only) vs + [mean |diff|, max |diff|] of consecutive frames.
"""
import json
from pathlib import Path
import numpy as np, torch
import perception as P
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
paper = HERE / "data/clevrer_paper"
scenes = json.loads((paper / "features_r50.json").read_text(encoding="utf-8"))["scenes"]
idx = np.round(np.linspace(0, 127, 25)).astype(int); edges = list(idx[1:]) + [128]
ann = P.AnnotationPerception(HERE / "data/clevrer/annotations")
win, ncol = [], []
for s in scenes:
    a = json.loads(ann.path(s).read_text(encoding="utf-8"))
    fr = np.array([c["frame_id"] for c in a["collision"]], int)
    win.append([int(((fr >= idx[t]) & (fr < edges[t])).any()) for t in range(25)]); ncol.append(min(len(fr), 4))
win, ncol = np.array(win), np.array(ncol)
print("share of windows with a collision:", round(float(win.mean()), 3))
x = np.asarray(np.load(paper / "features_r50.npy", mmap_mode="r"), np.float32)
TR, TE = slice(0, 2000), slice(2500, 3000)
def probe(xtr, ytr, xte, yte, multi=False):
    xtr, xte = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (xtr, xte))
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6; xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    yt, ye = torch.tensor(ytr, device=dev).long(), torch.tensor(yte, device=dev).long()
    k = int(max(ytr.max(), yte.max())) + 1; lin = torch.nn.Linear(xtr.shape[1], k).to(dev); opt = torch.optim.Adam(lin.parameters(), 3e-3, weight_decay=1e-3)
    for _ in range(300): opt.zero_grad(); torch.nn.functional.cross_entropy(lin(xtr), yt).backward(); opt.step()
    p = lin(xte).argmax(-1); acc = float((p == ye).float().mean()); maj = float((ye == torch.bincount(yt).argmax()).float().mean())
    if k == 2:                       # balanced accuracy (windows with a collision are a minority)
        rec = [float((p[ye == c] == c).float().mean()) for c in (0, 1)]; return acc, maj, sum(rec) / 2
    return acc, maj, None
f0, f1 = x[:, :-1], x[:, 1:]                                   # windows t=0..23 (the last window has no next frame)
lab = win[:, :-1]
print(f"\n{'window-level collision probe':40s} acc    majority  balanced-acc")
for name, feats in [("f_t only", f0), ("f_t and f_t+1", np.concatenate([f0, f1], -1)), ("f_t and (f_t+1 - f_t)", np.concatenate([f0, f1 - f0], -1))]:
    a, m, b = probe(feats[TR].reshape(-1, feats.shape[-1]), lab[TR].ravel(), feats[TE].reshape(-1, feats.shape[-1]), lab[TE].ravel())
    print(f"{name:40s} {a:.3f}  {m:.3f}     {b:.3f}")
d = np.abs(np.diff(x, axis=1))
app = np.concatenate([x.mean(1), x.std(1)], -1); mot = np.concatenate([app, d.mean(1), d.max(1)], -1)
print(f"\n{'video-level #collisions (0-4+)':40s} acc    majority")
for name, feats in [("appearance (mean,std)", app), ("appearance + motion (|diff|)", mot)]:
    a, m, _ = probe(feats[TR], ncol[TR], feats[TE], ncol[TE]); print(f"{name:40s} {a:.3f}  {m:.3f}")
