"""Prediction for image files and the end-to-end latency measurement."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .data import crop_resize
from .detect import detect

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".webp"}


@dataclass
class Prediction:
    file: str
    box: tuple[int, int, int, int] | None
    label: int | None
    name: str
    confidence: float


def predict_files(model, class_names: list[str], paths: list[Path], size: int = 32, use_detector: bool = True,
                  min_confidence: float = 0.5) -> list[Prediction]:
    """Classify image files. Below ``min_confidence`` the answer is ``unknown``."""
    out = []
    for path in paths:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        box = detect(np.asarray(rgb)) if use_detector else None
        crop = crop_resize(rgb, box, size)[None]
        proba = model.predict_proba(crop)[0]
        k = int(proba.argmax())
        label = int(model.classes_[k])
        conf = float(proba[k])
        name = class_names[label] if conf >= min_confidence else "unknown"
        out.append(Prediction(str(path), box, label if conf >= min_confidence else None, name, conf))
    return out


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def latency(model, paths: list[Path], size: int = 32, use_detector: bool = True, warmup: int = 3) -> dict:
    """Milliseconds for each stage of one image at a time: read, detect, crop, classify."""
    stages = {"read": [], "detect": [], "crop": [], "classify": [], "total": []}
    for i, path in enumerate(list(paths[:warmup]) + list(paths)):
        t0 = time.perf_counter()
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb)
        t1 = time.perf_counter()
        box = detect(arr) if use_detector else None
        t2 = time.perf_counter()
        crop = crop_resize(rgb, box, size)[None]
        t3 = time.perf_counter()
        model.predict_proba(crop)
        t4 = time.perf_counter()
        if i >= warmup:
            for key, value in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                value *= 1000.0
                stages[key].append(value)
    return {k: {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95))} for k, v in stages.items()} | {
        "images": len(paths), "fps_p50": float(1000.0 / np.percentile(stages["total"], 50))}
