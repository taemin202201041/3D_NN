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

    def __init__(self, folder: str | Path, steps: int = 25):
        import json
        folder = Path(folder)
        self.x = np.load(folder / "features_r50.npy", mmap_mode="r")
        info = json.loads((folder / "features_r50.json").read_text(encoding="utf-8"))
        self.row = {s: i for i, s in enumerate(info["scenes"])}
        self.settings = {k: v for k, v in info.items() if k != "scenes"}
        if self.x.shape[1] != steps:
            raise ValueError(f"{self.x.shape[1]} frames in file, expected {steps}")

    def perceive(self, scene: int) -> np.ndarray:
        return np.asarray(self.x[self.row[scene]], np.float32)
