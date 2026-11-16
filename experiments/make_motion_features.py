"""Motion features straight from the pixels: dense optical flow (Farneback) between consecutive frames of the 128-frame video, pooled on a
4x4 grid and averaged over the window of each of the 25 sampled steps.  Per step and grid cell: mean dx, mean dy, mean |flow|, max |flow| -> 4*4*4 = 64 numbers.
ResNet (ImageNet) features say WHAT is in a frame; these say WHERE things MOVE.  Output experiments/data/clevrer_paper/motion_g4.npy  float16 [3000, 25, 64] (+json scene order).
Usage: python make_motion_features.py [N]   (N = only first N scenes, for timing)
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.argv, _argv = [sys.argv[0]], sys.argv
import make_features_r50 as M          # noqa: E402  scenes, video_path (local mp4 or ranged download), IDX, OUT
sys.argv = _argv
N = int(sys.argv[1]) if len(sys.argv) > 1 else len(M.scenes)
G = 4
W, H = 120, 80
IDX = np.round(np.linspace(0, 127, 25)).astype(int)
ENDS = list(IDX[1:]) + [128]


def motion(scene: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(M.video_path(scene)))
    grays = []
    while len(grays) < 128:
        ok, bgr = cap.read()
        if not ok:
            break
        grays.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    assert len(grays) == 128, f"scene {scene}: {len(grays)} frames"
    cells = np.zeros((127, G, G, 4), np.float32)                      # per frame pair: mean dx, dy, mean |f|, max |f|
    for i in range(127):
        f = cv2.calcOpticalFlowFarneback(grays[i], grays[i + 1], None, 0.5, 2, 9, 2, 5, 1.1, 0)
        mag = np.linalg.norm(f, axis=-1)
        for a in range(G):
            for b in range(G):
                sl = (slice(a * H // G, (a + 1) * H // G), slice(b * W // G, (b + 1) * W // G))
                cells[i, a, b] = (f[sl][..., 0].mean(), f[sl][..., 1].mean(), mag[sl].mean(), mag[sl].max())
    out = np.zeros((25, G, G, 4), np.float32)
    for t in range(25):
        lo = min(int(IDX[t]), 126)                                    # last step: only the final frame pair exists
        hi = min(int(ENDS[t]), 127)
        w = cells[lo:max(hi, lo + 1)]
        out[t, ..., :3] = w[..., :3].mean(0); out[t, ..., 3] = w[..., 3].max(0)
    return out.reshape(25, -1)


if __name__ == "__main__":
    scenes = M.scenes[:N]
    feats = np.zeros((len(scenes), 25, G * G * 4), np.float16)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for i, f in enumerate(ex.map(motion, scenes)):
            feats[i] = f
            if i % 250 == 249:
                print(f"{i + 1}/{len(scenes)} ({time.time() - t0:.0f}s)", flush=True)
    secs = time.time() - t0
    print(f"{len(scenes)} videos in {secs:.0f}s, all {len(M.scenes)}: about {secs / len(scenes) * len(M.scenes) / 60:.1f} min", flush=True)
    if N == len(M.scenes):
        np.save(M.OUT / "motion_g4.npy", feats)
        (M.OUT / "motion_g4.json").write_text(json.dumps({"feature": "farneback flow 4x4 grid [mean dx, mean dy, mean |f|, max |f|]", "frames": 25, "scenes": M.scenes}), encoding="utf-8")
        print("saved", feats.shape, "nan:", bool(np.isnan(feats).any()))
