"""Do the optical-flow features make collisions readable?  Same probes as probe_motion.py, now with motion_g4 (64 per step).  read-only."""
import json
from pathlib import Path
import numpy as np, torch
import perception as P
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
paper = HERE / "data/clevrer_paper"
scenes = json.loads((paper / "motion_g4.json").read_text(encoding="utf-8"))["scenes"]
idx = np.round(np.linspace(0, 127, 25)).astype(int); edges = list(idx[1:]) + [128]
ann = P.AnnotationPerception(HERE / "data/clevrer/annotations")
win, ncol = [], []
for s in scenes:
    a = json.loads(ann.path(s).read_text(encoding="utf-8")); fr = np.array([c["frame_id"] for c in a["collision"]], int)
    win.append([int(((fr >= idx[t]) & (fr < edges[t])).any()) for t in range(25)]); ncol.append(min(len(fr), 4))
win, ncol = np.array(win), np.array(ncol)
m = np.asarray(np.load(paper / "motion_g4.npy"), np.float32)                         # [3000, 25, 64]
r50 = np.asarray(np.load(paper / "features_r50.npy", mmap_mode="r"), np.float32)     # [3000, 25, 2048]
TR, TE = slice(0, 2000), slice(2500, 3000)
def probe(xtr, ytr, xte, yte, steps=300, hidden=0):
    xtr, xte = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (xtr, xte))
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6; xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    yt, ye = torch.tensor(ytr, device=dev).long(), torch.tensor(yte, device=dev).long()
    k = int(max(ytr.max(), yte.max())) + 1
    net = (torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, k)) if hidden else torch.nn.Linear(xtr.shape[1], k)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-3)
    for _ in range(steps): opt.zero_grad(); torch.nn.functional.cross_entropy(net(xtr), yt).backward(); opt.step()
    p = net(xte).argmax(-1); acc = float((p == ye).float().mean()); maj = float((ye == torch.bincount(yt).argmax()).float().mean())
    bal = sum(float((p[ye == c] == c).float().mean()) for c in range(2)) / 2 if k == 2 else None
    return acc, maj, bal
def ctx(a, w):                                   # concat features of steps t-w..t+w (edge-padded)
    pad = np.pad(a, ((0, 0), (w, w), (0, 0)), mode="edge"); return np.concatenate([pad[:, i:i + 25] for i in range(2 * w + 1)], -1)
print(f"share of windows with a collision: {win.mean():.3f}\n{'window-level collision probe':44s} acc    majority  balanced-acc")
for name, feats, h in [("ResNet f_t (before)", r50, 0), ("motion m_t", m, 0), ("motion m_t, MLP", m, 64), ("motion m_t-1..m_t+1", ctx(m, 1), 0),
                       ("motion m_t-1..m_t+1, MLP", ctx(m, 1), 64), ("ResNet f_t + motion m_t-1..m_t+1", np.concatenate([r50, ctx(m, 1)], -1), 0)]:
    a, mj, b = probe(feats[TR].reshape(-1, feats.shape[-1]), win[TR].ravel(), feats[TE].reshape(-1, feats.shape[-1]), win[TE].ravel(), hidden=h)
    print(f"{name:44s} {a:.3f}  {mj:.3f}     {b:.3f}")
vid = np.concatenate([m.mean(1), m.std(1), m.max(1)], -1)
print(f"\n{'video-level #collisions (0-4+)':44s} acc    majority")
for name, feats in [("motion mean/std/max", vid), ("ResNet mean/std + motion", np.concatenate([r50.mean(1), r50.std(1), vid], -1))]:
    a, mj, _ = probe(feats[TR], ncol[TR], feats[TE], ncol[TE]); print(f"{name:44s} {a:.3f}  {mj:.3f}")
