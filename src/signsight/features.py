"""Hand-made image features for the default (offline, CPU) classifier.

* HOG: histograms of gradient directions in 8 x 8 cells, 2 x 2 block L2-Hys normalisation
* colour: 8-bin histograms of R, G and B, plus the mean colour
* condition cues: brightness, contrast, saturation and edge density (for the weather-condition model)
"""
from __future__ import annotations

import numpy as np


def _grey(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32).mean(axis=3)


def hog(images: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    g = _grey(images)
    n, h, w = g.shape
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, :, 1:-1] = g[:, :, 2:] - g[:, :, :-2]
    gy[:, 1:-1, :] = g[:, 2:, :] - g[:, :-2, :]
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    idx = np.minimum((ang / (180.0 / bins)).astype(int), bins - 1)
    ch, cw = h // cell, w // cell
    mag = mag[:, : ch * cell, : cw * cell]
    idx = idx[:, : ch * cell, : cw * cell]
    hist = np.zeros((n, ch, cw, bins), np.float32)
    for b in range(bins):
        hist[..., b] = (mag * (idx == b)).reshape(n, ch, cell, cw, cell).sum(axis=(2, 4))
    blocks = np.concatenate(
        [hist[:, :-1, :-1], hist[:, 1:, :-1], hist[:, :-1, 1:], hist[:, 1:, 1:]], axis=3
    )  # n x (ch-1) x (cw-1) x 4*bins
    norm = blocks / np.sqrt((blocks**2).sum(axis=3, keepdims=True) + 1e-6)
    norm = np.minimum(norm, 0.2)
    norm = norm / np.sqrt((norm**2).sum(axis=3, keepdims=True) + 1e-6)
    return norm.reshape(n, -1)


def colour(images: np.ndarray, bins: int = 8) -> np.ndarray:
    q = (images // (256 // bins)).astype(np.int16)
    parts = [np.stack([(q[..., c] == b).mean(axis=(1, 2)) for b in range(bins)], axis=1) for c in range(3)]
    parts.append(images.reshape(len(images), -1, 3).mean(axis=1) / 255.0)
    return np.concatenate(parts, axis=1).astype(np.float32)


def condition_cues(images: np.ndarray) -> np.ndarray:
    """Global cues of the imaging condition (fog lowers contrast, night lowers brightness, ...)."""
    x = images.astype(np.float32) / 255.0
    grey = x.mean(axis=3)
    sat = x.max(axis=3) - x.min(axis=3)
    gy = np.abs(np.diff(grey, axis=1)).mean(axis=(1, 2))
    gx = np.abs(np.diff(grey, axis=2)).mean(axis=(1, 2))
    bright = (grey > 0.9).mean(axis=(1, 2))
    return np.stack([grey.mean(axis=(1, 2)), grey.std(axis=(1, 2)), sat.mean(axis=(1, 2)), gx + gy, bright,
                     np.percentile(grey.reshape(len(x), -1), 5, axis=1), np.percentile(grey.reshape(len(x), -1), 95, axis=1)], axis=1)


def sign_features(images: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = [np.concatenate([hog(images[i : i + chunk]), colour(images[i : i + chunk])], axis=1) for i in range(0, len(images), chunk)]
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), np.float32)
