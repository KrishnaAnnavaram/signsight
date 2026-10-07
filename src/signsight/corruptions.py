"""Weather corruptions at 5 severities, deterministic for a given seed.

Each function takes a uint8 RGB image and returns a uint8 RGB image of the same size.
The training code applies them to training images only. The evaluation applies them to
COPIES of the held-out images to make the corruption suites.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SEVERITIES = (1, 2, 3, 4, 5)


def _check(severity: int) -> None:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be 1 to 5, got {severity}")


def fog(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check(severity)
    alpha = [0.2, 0.32, 0.44, 0.56, 0.68][severity - 1]
    h, w = img.shape[:2]
    haze = 200 + rng.normal(0, 8, (h, w, 1))
    return np.clip(img * (1 - alpha) + haze * alpha, 0, 255).astype(np.uint8)


def rain(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check(severity)
    h, w = img.shape[:2]
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    n = int(w * h / 1024 * [6, 10, 15, 20, 26][severity - 1])
    for _ in range(n):
        x, y = rng.integers(0, w), rng.integers(0, h)
        length = rng.integers(max(2, h // 10), max(3, h // 4))
        d.line([(int(x), int(y)), (int(x + length // 4), int(y + length))], fill=(200, 200, 210), width=1)
    darker = np.asarray(pil, float) * [0.95, 0.9, 0.85, 0.8, 0.75][severity - 1]
    return np.clip(darker, 0, 255).astype(np.uint8)


def glare(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check(severity)
    h, w = img.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    yy, xx = np.mgrid[0:h, 0:w]
    radius = [0.25, 0.32, 0.4, 0.48, 0.56][severity - 1] * max(h, w)
    mask = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (radius / 2) ** 2))[..., None]
    strength = [0.5, 0.65, 0.8, 0.9, 1.0][severity - 1]
    return np.clip(img + (255 - img) * mask * strength, 0, 255).astype(np.uint8)


def snow(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check(severity)
    h, w = img.shape[:2]
    share = [0.02, 0.04, 0.07, 0.1, 0.14][severity - 1]
    flakes = rng.random((h, w)) < share
    out = img.astype(float) * [0.95, 0.9, 0.85, 0.8, 0.75][severity - 1] + 25
    out[flakes] = 245
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check(severity)
    radius = [0.5, 0.8, 1.1, 1.5, 2.0][severity - 1] * img.shape[0] / 32
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)


def night(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check(severity)
    factor = [0.7, 0.55, 0.42, 0.32, 0.24][severity - 1]
    noise = rng.normal(0, 3 + 2 * severity, img.shape)
    return np.clip(img * factor + noise, 0, 255).astype(np.uint8)


CORRUPTIONS = {"fog": fog, "rain": rain, "glare": glare, "snow": snow, "blur": blur, "night": night}
TRAIN_CORRUPTIONS = ("fog", "rain", "glare")  # the augmentation uses only these, so snow, blur and night test transfer


def corrupt_batch(images: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fn = CORRUPTIONS[name]
    return np.stack([fn(img, severity, rng) for img in images])


def augment(images: np.ndarray, labels: np.ndarray, share: float = 0.5, seed: int = 0,
            names=TRAIN_CORRUPTIONS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add corrupted copies of a share of the TRAINING images. Return images, labels and the
    condition of each image (0 = clean, 1.. = index in ``names`` + 1)."""
    rng = np.random.default_rng(seed)
    pick = np.flatnonzero(rng.random(len(images)) < share)
    kinds = rng.integers(0, len(names), len(pick))
    sev = rng.integers(1, 6, len(pick))
    extra = np.stack([CORRUPTIONS[names[k]](images[i], int(s), rng) for i, k, s in zip(pick, kinds, sev)]) if len(pick) else images[:0]
    cond = np.concatenate([np.zeros(len(images), int), kinds + 1])
    return np.concatenate([images, extra]), np.concatenate([labels, labels[pick]]), cond
