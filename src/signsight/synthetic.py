"""Seeded synthetic traffic signs with the TRACK structure of GTSRB.

Each track is one physical sign: its own background, colour shade, size and position.
The frames of a track show the sign while the camera comes nearer (the sign grows)
with small changes of light. Thus a random image split leaks, as in GTSRB.
These images are not real signs. They test the code and the evaluation design.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .data import ImageSet

SHAPES = ["circle", "triangle", "octagon", "diamond"]
COLOURS = {"red": (200, 30, 35), "blue": (30, 80, 190), "yellow": (235, 190, 30)}
GLYPHS = ["bar", "dot", "arrow"]
# 10 classes: shape + colour + glyph
CLASSES = [
    ("circle", "red", "bar"), ("circle", "red", "dot"), ("circle", "blue", "arrow"), ("circle", "blue", "dot"),
    ("triangle", "red", "bar"), ("triangle", "red", "dot"), ("octagon", "red", "bar"), ("diamond", "yellow", "dot"),
    ("diamond", "yellow", "bar"), ("triangle", "yellow", "arrow"),
]
CLASS_NAMES = [f"{s} {c} {g}" for s, c, g in CLASSES]


def _polygon(shape: str, cx: float, cy: float, r: float):
    if shape == "triangle":
        return [(cx, cy - r), (cx + 0.87 * r, cy + 0.5 * r), (cx - 0.87 * r, cy + 0.5 * r)]
    if shape == "diamond":
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    if shape == "octagon":
        angles = np.radians(22.5 + 45 * np.arange(8))
        return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    return None


def draw_sign(cls: int, size: int, rng: np.random.Generator, scale: float = 1.0, shift=(0.0, 0.0),
              background=None, shade: float = 1.0, light: float = 1.0) -> np.ndarray:
    shape, colour, glyph = CLASSES[cls]
    big = size * 4  # draw large, then reduce: smooth edges
    bg = background if background is not None else tuple(int(v) for v in rng.integers(60, 200, 3))
    img = Image.new("RGB", (big, big), bg)
    d = ImageDraw.Draw(img)
    cx, cy = big / 2 + shift[0] * big, big / 2 + shift[1] * big
    r = big * 0.38 * scale
    fill = tuple(int(np.clip(v * shade, 0, 255)) for v in COLOURS[colour])
    poly = _polygon(shape, cx, cy, r)
    if poly is None:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=(250, 250, 250), width=max(2, big // 40))
    else:
        d.polygon(poly, fill=fill, outline=(250, 250, 250))
    ink = (20, 20, 20) if colour == "yellow" else (245, 245, 245)
    w = r * 0.5
    if glyph == "bar":
        d.rectangle([cx - w, cy - w * 0.22, cx + w, cy + w * 0.22], fill=ink)
    elif glyph == "dot":
        d.ellipse([cx - w * 0.35, cy - w * 0.35, cx + w * 0.35, cy + w * 0.35], fill=ink)
    else:
        d.polygon([(cx - w, cy - w * 0.15), (cx + w * 0.2, cy - w * 0.15), (cx + w * 0.2, cy - w * 0.5),
                   (cx + w, cy), (cx + w * 0.2, cy + w * 0.5), (cx + w * 0.2, cy + w * 0.15), (cx - w, cy + w * 0.15)], fill=ink)
    arr = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=float) * light
    arr += rng.normal(0, 4, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_signs(tracks_per_class: int = 12, frames: int = 10, size: int = 32, seed: int = 42,
               classes: int | None = None) -> ImageSet:
    rng = np.random.default_rng(seed)
    n_cls = classes or len(CLASSES)
    images, labels, groups = [], [], []
    for cls in range(n_cls):
        for t in range(tracks_per_class):
            bg = tuple(int(v) for v in rng.integers(50, 210, 3))
            shade = rng.uniform(0.75, 1.15)
            shift = rng.uniform(-0.08, 0.08, 2)
            start = rng.uniform(0.55, 0.8)
            for f in range(frames):
                scale = start + (1.0 - start) * f / max(1, frames - 1)  # the sign grows
                images.append(draw_sign(cls, size, rng, scale, shift, bg, shade, rng.uniform(0.85, 1.1)))
                labels.append(cls)
                groups.append(f"{cls:02d}-{t:04d}")
    return ImageSet(np.stack(images), np.array(labels), np.array(groups), CLASS_NAMES[:n_cls])


def make_scene(cls: int, size: int = 160, seed: int = 0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A larger street-like image with one sign, for the detector. Returns the image and the true box."""
    rng = np.random.default_rng(seed)
    scene = np.zeros((size, size, 3), np.uint8)
    scene[: size // 2] = (150, 175, 200)  # sky
    scene[size // 2 :] = (90, 90, 85)  # road
    scene = np.clip(scene + rng.normal(0, 6, scene.shape), 0, 255).astype(np.uint8)
    s = size // 4
    x, y = int(rng.integers(5, size - s - 5)), int(rng.integers(5, size // 2))
    sign = draw_sign(cls, s, rng, scale=1.15, background=(150, 175, 200))
    scene[y : y + s, x : x + s] = sign
    return scene, (x, y, x + s, y + s)
