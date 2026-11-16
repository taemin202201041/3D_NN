"""CLEVRER paper-style video features: frozen ImageNet ResNet-50, 2048-d pool5, 25 uniformly sampled frames per video.

Reads the scene list from data_large_v1/manifest.json (train + dev + local_test = 3000 rows, same order as dataset.py).
Videos come from data_large_v1/videos/*.mp4; the few that are not there are fetched from the official zip with
one HTTP range request each (using the byte offsets in data_large_v1/cache/zip_index_*.json).
Frame preprocessing is the one data_large_v1 used for its ResNet18 features (verified against them to 0.75% relative error):
RGB, bilinear resize to 224 wide keeping the aspect ratio (149 rows), zero-padded top/bottom AFTER ImageNet normalisation.

Output (default experiments/data/clevrer_paper/):  features_r50.npy  float16 [3000, 25, 2048],  features_r50.json (settings, scenes)
Usage: python make_features_r50.py [data_large_v1 dir]
"""
import json
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data_large_v1" / "data_large_v1"
OUT = HERE / "data" / "clevrer_paper"
OUT.mkdir(parents=True, exist_ok=True)
TMP = OUT / "_tmp"
TMP.mkdir(exist_ok=True)
WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"      # torchvision IMAGENET1K_V1
ZIP_URL = {"train": "https://data.csail.mit.edu/clevrer/videos/train/video_train.zip",
           "validation": "https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip"}
IDX = np.round(np.linspace(0, 127, 25)).astype(int)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
scenes = [s for k in ("train", "dev", "local_test") for s in manifest[k]]
zip_index = {sp: json.loads((ROOT / "cache" / f"zip_index_{sp}.json").read_text(encoding="utf-8"))["entries"] for sp in ZIP_URL}


def video_path(scene: int) -> Path:
    p = ROOT / "videos" / f"video_{scene:05d}.mp4"
    if p.exists():
        return p
    sp = "train" if scene < 10000 else "validation"
    e = zip_index[sp][str(scene)]
    req = urllib.request.Request(ZIP_URL[sp], headers={"Range": f"bytes={e['start']}-{e['end']}"})       # 'start' = local file header
    blob = urllib.request.urlopen(req, timeout=120).read()
    assert blob[:4] == b"PK\x03\x04", f"scene {scene}: no zip header at the indexed offset"
    nl, el = int.from_bytes(blob[26:28], "little"), int.from_bytes(blob[28:30], "little")
    raw = blob[30 + nl + el:30 + nl + el + e["compressed"]]
    data = zlib.decompress(raw, -15) if e["method"] == 8 else raw
    assert len(data) == e["uncompressed"] and zlib.crc32(data) == e["crc"], f"scene {scene}: corrupt download"
    q = TMP / f"video_{scene:05d}.mp4"
    q.write_bytes(data)
    return q


def frames_of(scene: int) -> np.ndarray:
    """25 preprocessed frames [25, 3, 224, 224] float32"""
    cap = cv2.VideoCapture(str(video_path(scene)))
    want, out, i = set(IDX.tolist()), {}, 0
    while i <= IDX.max():
        ok = cap.grab()
        if not ok:
            break
        if i in want:
            ok, bgr = cap.retrieve()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            r = cv2.resize(rgb, (224, 149), interpolation=cv2.INTER_LINEAR)
            x = np.zeros((224, 224, 3), np.float32)
            x[37:37 + 149] = (r.astype(np.float32) / 255 - MEAN) / STD
            out[i] = x.transpose(2, 0, 1)
        i += 1
    cap.release()
    assert len(out) == len(want), f"scene {scene}: only {i} frames"
    return np.stack([out[j] for j in IDX])


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torchvision.models.resnet50()
    net.load_state_dict(torch.hub.load_state_dict_from_url(WEIGHTS_URL, model_dir=str(OUT), check_hash=True, weights_only=True))
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    feats = np.zeros((len(scenes), 25, 2048), np.float16)
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool, torch.no_grad():
        for i, x in enumerate(pool.map(frames_of, scenes)):
            feats[i] = net(torch.from_numpy(x).to(dev)).cpu().numpy().astype(np.float16)
            if i % 250 == 249:
                print(f"{i + 1}/{len(scenes)} videos ({time.time() - t0:.0f}s)", flush=True)
    np.save(OUT / "features_r50.npy", feats)
    (OUT / "features_r50.json").write_text(json.dumps({
        "encoder": "resnet50_imagenet1k_v1", "layer": "pool5 (avgpool, 2048)", "frames": 25, "frame_index": IDX.tolist(), "size": 224,
        "preprocess": "cv2_bilinear_rgb_letterbox_full_frame_imagenet_norm_v1 (as data_large_v1)", "weights_url": WEIGHTS_URL,
        "scenes": scenes}), encoding="utf-8")
    for f in TMP.glob("*.mp4"):
        f.unlink()
    print("done", feats.shape, f"{time.time() - t0:.0f}s", "nan:", bool(np.isnan(feats).any()))
