"""ResNet-50 pool5 features for ALL 15000 CLEVRER videos (0-9999 train zip, 10000-14999 validation zip), row = video id.
Same weights / preprocessing as make_features_r50.py (imported, unchanged).  The 3000 rows already computed are copied, the rest are
fetched with HTTP range requests.  Resumable: partial result in data/clevrer_paper/_tmp/partial_all.npy.
Output: features_r50_all.npy float16 [15000, 25, 2048] + features_r50_all.json (scenes = 0..14999)"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torchvision
sys.argv = sys.argv[:1]
import make_features_r50 as M

N = 15000
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torchvision.models.resnet50()
    net.load_state_dict(torch.hub.load_state_dict_from_url(M.WEIGHTS_URL, model_dir=str(M.OUT), check_hash=True, weights_only=True))
    net.fc = torch.nn.Identity(); net.eval().to(dev)
    part = M.TMP / "partial_all.npy"; done_f = M.TMP / "partial_done.npy"
    feats = np.lib.format.open_memmap(part, mode="r+" if part.exists() else "w+", dtype=np.float16, shape=(N, 25, 2048))
    done = np.load(done_f) if done_f.exists() else np.zeros(N, bool)
    if not done.any():
        old = np.load(M.OUT / "features_r50.npy", mmap_mode="r")
        for i, s in enumerate(M.scenes):
            feats[s] = old[i]; done[s] = True
    todo = [s for s in range(N) if not done[s]]
    print("to do", len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool, torch.no_grad():
        for k, (s, x) in enumerate(zip(todo, pool.map(M.frames_of, todo))):
            feats[s] = net(torch.from_numpy(x).to(dev)).cpu().numpy().astype(np.float16); done[s] = True
            (M.TMP / f"video_{s:05d}.mp4").unlink(missing_ok=True)
            if k % 250 == 249:
                feats.flush(); np.save(done_f, done); print(f"{k + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    feats.flush(); np.save(done_f, done)
    np.save(M.OUT / "features_r50_all.npy", np.asarray(feats))
    (M.OUT / "features_r50_all.json").write_text(json.dumps({"encoder": "resnet50_imagenet1k_v1", "layer": "pool5", "frames": 25, "frame_index": M.IDX.tolist(), "scenes": list(range(N))}), encoding="utf-8")
    print("done", time.time() - t0, flush=True)
