"""Image sets, the GTSRB reader and track-grouped splits.

GTSRB training images come in TRACKS: about 30 frames of one physical sign. A random
split by image puts frames of one track in train and in test, so the test score is
too high. ``track_split`` keeps each track in one part.
"""
from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

GTSRB_CLASSES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons",
]


class DataError(ValueError):
    pass


@dataclass
class ImageSet:
    images: np.ndarray  # N x S x S x 3, uint8
    labels: np.ndarray  # N, int
    groups: np.ndarray  # N, track ID (str)
    class_names: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, idx) -> "ImageSet":
        idx = np.asarray(idx)
        return ImageSet(self.images[idx], self.labels[idx], self.groups[idx], self.class_names)


def crop_resize(img: Image.Image, box: tuple[int, int, int, int] | None, size: int) -> np.ndarray:
    if box is not None:
        img = img.crop(box)
    return np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _read_rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text), delimiter=";"))


def load_gtsrb_train(source: str | Path, size: int = 32, limit_per_class: int | None = None) -> ImageSet:
    """Read the GTSRB training set from the zip file or the extracted folder.

    Each class folder has ``GT-<class>.csv`` with ``Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId``.
    The track ID comes from the file name ``<track>_<frame>.ppm``.
    """
    source = Path(source)
    images, labels, groups = [], [], []

    def add(name: str, row: dict, opener) -> None:
        box = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]) + 1, int(row["Roi.Y2"]) + 1)
        cls = int(row["ClassId"])
        with opener() as fh:
            images.append(crop_resize(Image.open(fh), box, size))
        labels.append(cls)
        groups.append(f"{cls:02d}-{row['Filename'].split('_')[0]}")

    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as zf:
            gt_files = [n for n in zf.namelist() if Path(n).name.startswith("GT-") and n.endswith(".csv")]
            if not gt_files:
                raise DataError(f"{source}: no GT-*.csv files found")
            for gt in sorted(gt_files):
                folder = gt.rsplit("/", 1)[0]
                rows = _read_rows(zf.read(gt).decode("utf-8"))[:limit_per_class]
                for row in rows:
                    add(gt, row, lambda r=row: zf.open(f"{folder}/{r['Filename']}"))
    else:
        gt_files = sorted(source.rglob("GT-*.csv"))
        if not gt_files:
            raise DataError(f"{source}: no GT-*.csv files found")
        for gt in gt_files:
            for row in _read_rows(gt.read_text(encoding="utf-8"))[:limit_per_class]:
                add(str(gt), row, lambda r=row, g=gt: open(g.parent / r["Filename"], "rb"))
    return ImageSet(np.stack(images), np.array(labels), np.array(groups), GTSRB_CLASSES)


def load_gtsrb_test(images_dir: str | Path, gt_csv: str | Path, size: int = 32) -> ImageSet:
    """Read the official GTSRB test set (12,630 images) and ``GT-final_test.csv``."""
    images_dir = Path(images_dir)
    rows = _read_rows(Path(gt_csv).read_text(encoding="utf-8"))
    imgs, labels = [], []
    for row in rows:
        box = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]) + 1, int(row["Roi.Y2"]) + 1)
        with Image.open(images_dir / row["Filename"]) as img:
            imgs.append(crop_resize(img, box, size))
        labels.append(int(row["ClassId"]))
    groups = np.array([f"test-{i}" for i in range(len(rows))])
    return ImageSet(np.stack(imgs), np.array(labels), groups, GTSRB_CLASSES)


def track_split(data: ImageSet, val_frac: float = 0.2, seed: int = 42) -> tuple[ImageSet, ImageSet]:
    """Split by track with class shares kept: no track is in both parts."""
    n_splits = max(2, int(round(1 / val_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(np.zeros(len(data)), data.labels, data.groups))
    return data.subset(train_idx), data.subset(val_idx)


def image_split(data: ImageSet, val_frac: float = 0.2, seed: int = 42) -> tuple[ImageSet, ImageSet]:
    """The random split by image of the prototype. Use it only to MEASURE the leakage."""
    idx = np.arange(len(data))
    a, b = train_test_split(idx, test_size=val_frac, stratify=data.labels, random_state=seed)
    return data.subset(a), data.subset(b)


def track_overlap(a: ImageSet, b: ImageSet) -> int:
    return len(set(a.groups) & set(b.groups))
