"""Perception front-end: one CLEVRER video -> [T, D] frame-feature matrix.

This is the ONLY place that knows where the video information comes from. Everything after it (dataset, model,
training, scoring, JSON) only sees the matrix.  data_large_v1 ships precomputed features: a frozen ImageNet ResNet18,
25 frames spread uniformly over the 128-frame video, 512 numbers per frame  ->  features/<scene:05d>.pt
({'features': (25, 512) float32, 'settings': {...}, 'scene_index': int}).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

COLORS = ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]
MATERIALS = ["rubber", "metal"]
SHAPES = ["cube", "sphere", "cylinder"]


class FeaturePerception:
    name = "resnet18_features"

    def __init__(self, root: str | Path, features_dir: str = "features", steps: int = 25):
        self.dir, self.steps = Path(root) / features_dir, steps
        self.settings = None

    def path(self, scene: int) -> Path:
        return self.dir / f"{scene:05d}.pt"

    def perceive(self, scene: int) -> np.ndarray:
        f = torch.load(self.path(scene), map_location="cpu", weights_only=True)
        x = f["features"].numpy().astype(np.float32)
        if x.shape[0] != self.steps:
            raise ValueError(f"scene {scene}: {x.shape[0]} frames, expected {self.steps}")
        self.settings = f.get("settings")
        return x


class NpyPerception:
    """paper-style features made by make_features_r50.py: features_r50.npy [n_scenes, 25, 2048] + features_r50.json (scene order)"""
    name = "resnet50_npy"

    def __init__(self, folder: str | Path, steps: int = 25, stem: str = "features_r50", name: str = "resnet50_npy"):
        import json
        folder = Path(folder)
        self.name = name
        self.x = np.load(folder / f"{stem}.npy", mmap_mode="r")
        info = json.loads((folder / f"{stem}.json").read_text(encoding="utf-8"))
        self.row = {s: i for i, s in enumerate(info["scenes"])}
        self.settings = {k: v for k, v in info.items() if k != "scenes"}
        if self.x.shape[1] != steps:
            raise ValueError(f"{self.x.shape[1]} frames in file, expected {steps}")

    def perceive(self, scene: int) -> np.ndarray:
        return np.asarray(self.x[self.row[scene]], np.float32)


# ---------------------------------------------------------------------------------------------------------------- annotation input
# Team input (DeepLearning-research-main/동적_모델/perception.py): CLEVRER simulator annotation as the output of a perfect perception step.
# [T=25, SLOTS=6, F=37] per video, flattened to [25, 222].  No video / CNN involved.
SLOTS = 6
FEATURES = (["x", "y", "z", "vx", "vy", "vz", "ox", "oy", "oz", "wx", "wy", "wz", "visible", "collide"]
            + [f"partner{i}" for i in range(SLOTS)] + ["cx", "cy", "cz", "present"]
            + [f"color_{c}" for c in COLORS] + [f"material_{m}" for m in MATERIALS] + [f"shape_{s}" for s in SHAPES])
F = len(FEATURES)  # 37
CONTINUOUS = [FEATURES.index(k) for k in ["x", "y", "z", "vx", "vy", "vz", "ox", "oy", "oz", "wx", "wy", "wz", "cx", "cy", "cz"]]
I_VISIBLE, I_PRESENT = FEATURES.index("visible"), FEATURES.index("present")


def sample_frames(frames: int, steps: int):
    idx = np.round(np.linspace(0, frames - 1, steps)).astype(int)
    ends = list(idx[1:]) + [frames]
    return idx, list(zip(idx.tolist(), [int(e) for e in ends]))


class AnnotationPerception:
    name = "annotation"

    def __init__(self, annotations_dir: str | Path, frames: int = 128, steps: int = 25):
        self.ann_dir, self.frames, self.steps = Path(annotations_dir), frames, steps
        self.idx, self.windows = sample_frames(frames, steps)
        self.settings = {"source": "CLEVRER official annotation", "frames": frames, "steps": steps, "slots": SLOTS, "features": F}

    def path(self, scene: int) -> Path:
        lo = scene // 1000 * 1000
        return self.ann_dir / f"annotation_{lo:05d}-{lo + 1000:05d}" / f"annotation_{scene:05d}.json"

    def perceive(self, scene: int) -> np.ndarray:
        """[steps, SLOTS * F] float32 (raw, not standardised)"""
        import json
        ann = json.loads(self.path(scene).read_text(encoding="utf-8"))
        out = np.zeros((self.steps, SLOTS, F), np.float32)
        props = {p["object_id"]: p for p in ann["object_property"]}
        hits: dict = {}
        for c in ann["collision"]:
            a, b = c["object_ids"]
            hits.setdefault(a, []).append((c["frame_id"], b, c["location"]))
            hits.setdefault(b, []).append((c["frame_id"], a, c["location"]))
        for t, (f0, f1) in enumerate(self.windows):
            for o in ann["motion_trajectory"][f0]["objects"]:
                k = o["object_id"]
                if k >= SLOTS:
                    continue
                row = out[t, k]
                row[0:3], row[3:6] = o["location"], o["velocity"]
                row[6:9], row[9:12] = o["orientation"], o["angular_velocity"]
                row[12] = float(o["inside_camera_view"])
                for frame, partner, loc in hits.get(k, []):
                    if f0 <= frame < f1:
                        row[13] = 1.0
                        if partner < SLOTS:
                            row[14 + partner] = 1.0
                        row[20:23] = loc
                row[23] = 1.0
                p = props[k]
                row[24 + COLORS.index(p["color"])] = 1.0
                row[32 + MATERIALS.index(p["material"])] = 1.0
                row[34 + SHAPES.index(p["shape"])] = 1.0
        return out.reshape(self.steps, SLOTS * F)


class NpyConcatPerception:
    """several per-step feature files with the same scene order, concatenated on the feature axis
    e.g. ['features_r50', 'motion_g4'] -> ResNet-50 appearance (2048) + optical-flow motion (64)."""

    def __init__(self, folder: str | Path, stems: list[str], steps: int = 25, name: str = "concat"):
        import json
        folder = Path(folder)
        self.name, self.parts, self.row, self.settings = name, [], None, {"parts": stems}
        for s in stems:
            x = np.load(folder / f"{s}.npy", mmap_mode="r")
            info = json.loads((folder / f"{s}.json").read_text(encoding="utf-8"))
            row = {sc: i for i, sc in enumerate(info["scenes"])}
            if self.row is not None and row != self.row:
                raise ValueError(f"{s}: scene order differs from the first file")
            self.row = row
            if x.shape[1] != steps:
                raise ValueError(f"{s}: {x.shape[1]} steps, expected {steps}")
            self.parts.append(x)

    def perceive(self, scene: int) -> np.ndarray:
        i = self.row[scene]
        return np.concatenate([np.asarray(x[i], np.float32) for x in self.parts], -1)
