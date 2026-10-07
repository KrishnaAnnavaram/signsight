"""A simple colour-based sign detector, so that a full photo is cropped before classification.

The classifier learns cropped signs. A full photo without a crop is a different input.
The detector finds the pixels with sign colours (red, blue, yellow), removes small
noise and returns the box of the largest coloured area. It is a baseline for clear
photos, not a trained detector.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def sign_colour_mask(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.int32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    red = (r > 110) & (r > 1.6 * g) & (r > 1.6 * b)
    blue = (b > 100) & (b > 1.4 * r) & (b > 1.15 * g)
    yellow = (r > 150) & (g > 110) & (b < 0.6 * g)
    return red | blue | yellow


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        comp, queue = [], deque([(sy, sx)])
        seen[sy, sx] = True
        while queue:
            y, x = queue.popleft()
            comp.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(comp) > len(best):
            best = comp
    return np.array(best) if best else None


def detect(image: np.ndarray, min_area: float = 0.002, pad: float = 0.12, work: int = 128) -> tuple[int, int, int, int] | None:
    """Return the sign box ``(x1, y1, x2, y2)`` in the image, or None."""
    h, w = image.shape[:2]
    step = max(1, max(h, w) // work)
    small = image[::step, ::step]
    comp = _largest_component(sign_colour_mask(small))
    if comp is None or len(comp) < min_area * small.shape[0] * small.shape[1]:
        return None
    y1, x1 = comp.min(axis=0) * step
    y2, x2 = (comp.max(axis=0) + 1) * step
    py, px = int((y2 - y1) * pad), int((x2 - x1) * pad)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0
