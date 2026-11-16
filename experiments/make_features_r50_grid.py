"""CLEVRER spatial video features: same frozen ResNet-50 and the same 25 frames / preprocessing as make_features_r50.py,
but the last conv map (2048 x 7 x 7) is average-pooled to a G x G grid instead of one global vector -> keeps WHERE things are.

Output (experiments/data/clevrer_paper/):  features_r50_g<G>.npy  float16 [n_scenes, 25, 2048*G*G]  (+ .json with the scene order)
Usage: python make_features_r50_grid.py [G=2] [N=all]     (N = only the first N scenes, for timing)
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision

sys.argv, _argv = [sys.argv[0]], sys.argv                  # make_features_r50 reads sys.argv[1] as a data folder
import make_features_r50 as M                              # noqa: E402  (scenes, frames_of, WEIGHTS_URL, OUT, IDX)
sys.argv = _argv

G = int(sys.argv[1]) if len(sys.argv) > 1 else 2
N = int(sys.argv[2]) if len(sys.argv) > 2 else len(M.scenes)
scenes = M.scenes[:N]

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torchvision.models.resnet50()
    net.load_state_dict(torch.hub.load_state_dict_from_url(M.WEIGHTS_URL, model_dir=str(M.OUT), check_hash=True, weights_only=True))
    body = torch.nn.Sequential(*list(net.children())[:-2]).eval().to(dev)          # up to layer4 (no avgpool / fc)
    pool = torch.nn.AdaptiveAvgPool2d(G)
    feats = np.zeros((len(scenes), 25, 2048 * G * G), np.float16)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex, torch.no_grad():
        for i, x in enumerate(ex.map(M.frames_of, scenes)):
            f = pool(body(torch.from_numpy(x).to(dev)))                             # [25, 2048, G, G]
            feats[i] = f.flatten(1).cpu().numpy().astype(np.float16)
            if i % 250 == 249:
                print(f"{i + 1}/{len(scenes)} videos ({time.time() - t0:.0f}s)", flush=True)
    secs = time.time() - t0
    print(f"{len(scenes)} videos in {secs:.0f}s -> {secs / len(scenes):.2f}s/video, all {len(M.scenes)}: about {secs / len(scenes) * len(M.scenes) / 60:.0f} min", flush=True)
    if N == len(M.scenes):
        np.save(M.OUT / f"features_r50_g{G}.npy", feats)
        (M.OUT / f"features_r50_g{G}.json").write_text(json.dumps({
            "encoder": "resnet50_imagenet1k_v1", "layer": f"layer4 map pooled to {G}x{G} (2048*{G*G})", "frames": 25, "frame_index": M.IDX.tolist(),
            "preprocess": "same as features_r50", "scenes": M.scenes}), encoding="utf-8")
        print("saved", feats.shape, "nan:", bool(np.isnan(feats).any()))
