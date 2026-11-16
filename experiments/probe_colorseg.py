"""Feasibility test, not a model: how far does simple colour segmentation get on one frame?  Annotation is used ONLY as a ruler here.
Per video, frame 60: HSV saturation mask -> connected components -> compare the number of blobs with the true number of visible non-gray objects."""
import json, collections
from pathlib import Path
import cv2, numpy as np
import perception as P
HERE = Path(__file__).resolve().parent
R = HERE.parent / "data_large_v1" / "data_large_v1"
scenes = json.loads((HERE / "data/clevrer_paper/features_r50.json").read_text(encoding="utf-8"))["scenes"]
ann = P.AnnotationPerception(HERE / "data/clevrer/annotations")
rows = []
for s in scenes:
    mp = R / "videos" / f"video_{s:05d}.mp4"
    if not mp.exists(): continue
    cap = cv2.VideoCapture(str(mp)); cap.set(cv2.CAP_PROP_POS_FRAMES, 60); ok, bgr = cap.read(); cap.release()
    if not ok: continue
    a = json.loads(ann.path(s).read_text(encoding="utf-8")); props = {p["object_id"]: p for p in a["object_property"]}
    vis = [o["object_id"] for o in a["motion_trajectory"][60]["objects"] if o["inside_camera_view"]]
    n_col = sum(props[k]["color"] != "gray" for k in vis); n_gray = len(vis) - n_col
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 90) & (hsv[..., 2] > 40)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    blobs = int(sum(st[i, cv2.CC_STAT_AREA] >= 80 for i in range(1, n)))
    rows.append((n_col, n_gray, blobs))
    if len(rows) >= 80: break
r = np.array(rows)
print(f"videos {len(r)};  visible objects per frame: colored {r[:,0].mean():.2f}, gray {r[:,1].mean():.2f} (gray = same hue as the floor, invisible to a saturation mask)")
print(f"blobs found == true colored objects: {np.mean(r[:,2] == r[:,0]):.2f}   blobs < true: {np.mean(r[:,2] < r[:,0]):.2f}   blobs > true: {np.mean(r[:,2] > r[:,0]):.2f}")
print("error (blobs - colored) histogram:", dict(sorted(collections.Counter((r[:,2] - r[:,0]).tolist()).items())))
print(f"videos with at least one gray object visible: {np.mean(r[:,1] > 0):.2f}")
