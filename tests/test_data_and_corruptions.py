"""Splits (reference problem 2), corruptions (problem 4), GTSRB reader, detector (problem 5)."""
import numpy as np
import pytest
from PIL import Image

from signsight.corruptions import CORRUPTIONS, SEVERITIES, augment, corrupt_batch
from signsight.data import DataError, GTSRB_CLASSES, image_split, load_gtsrb_test, load_gtsrb_train, track_overlap, track_split
from signsight.detect import detect, iou
from signsight.features import condition_cues, hog, sign_features
from signsight.synthetic import make_scene, make_signs


def test_track_split_has_no_shared_track(signs):
    train, val = track_split(signs, seed=1)
    assert track_overlap(train, val) == 0
    assert set(np.unique(val.labels)) == set(np.unique(signs.labels))
    assert 0.15 < len(val) / len(signs) < 0.3


def test_image_split_leaks_tracks(signs):
    """Problem 2: the random image split of the prototype puts one track in both parts."""
    train, val = image_split(signs, seed=1)
    assert track_overlap(train, val) > 0


def test_synthetic_signs_are_seeded_and_tracked():
    a, b = make_signs(2, 4, seed=5), make_signs(2, 4, seed=5)
    assert np.array_equal(a.images, b.images)
    assert len(set(a.groups)) == 2 * 10 and a.images.shape == (80, 32, 32, 3)


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corruptions_are_deterministic_and_grow_with_severity(signs, name):
    imgs = signs.images[:20]
    a = corrupt_batch(imgs, name, 3, seed=7)
    b = corrupt_batch(imgs, name, 3, seed=7)
    assert np.array_equal(a, b) and a.dtype == np.uint8 and a.shape == imgs.shape
    change = [np.abs(corrupt_batch(imgs, name, s, seed=7).astype(int) - imgs).mean() for s in (1, 5)]
    assert change[1] > change[0] > 0


def test_bad_severity_raises(signs):
    with pytest.raises(ValueError):
        corrupt_batch(signs.images[:2], "fog", 6)
    assert SEVERITIES == (1, 2, 3, 4, 5)


def test_augment_adds_copies_only(signs):
    imgs, labels, cond = augment(signs.images, signs.labels, share=0.5, seed=1)
    n = len(signs)
    assert np.array_equal(imgs[:n], signs.images) and (cond[:n] == 0).all()
    assert len(imgs) > n and (cond[n:] > 0).all() and np.array_equal(labels[:n], signs.labels)


def test_features_shapes_and_cues(signs):
    assert hog(signs.images[:5]).shape == (5, 3 * 3 * 36)
    assert sign_features(signs.images[:5]).shape[0] == 5
    clear = condition_cues(signs.images[:30])
    foggy = condition_cues(corrupt_batch(signs.images[:30], "fog", 5))
    assert foggy[:, 1].mean() < clear[:, 1].mean()  # fog lowers contrast


def test_detector_finds_the_sign():
    hits = []
    for i in range(10):
        scene, box = make_scene(i % 10, seed=i)
        found = detect(scene)
        hits.append(found is not None and iou(found, box) > 0.4)
    assert np.mean(hits) >= 0.8
    assert detect(np.full((64, 64, 3), 128, np.uint8)) is None


def _write_gtsrb(root, classes=2, tracks=2, frames=3):
    for c in range(classes):
        folder = root / f"{c:05d}"
        folder.mkdir(parents=True)
        rows = ["Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"]
        for t in range(tracks):
            for f in range(frames):
                name = f"{t:05d}_{f:05d}.ppm"
                Image.new("RGB", (40, 40), (50 * c, 80, 120)).save(folder / name)
                rows.append(f"{name};40;40;5;5;34;34;{c}")
        (folder / f"GT-{c:05d}.csv").write_text("\n".join(rows), encoding="utf-8")


def test_gtsrb_reader_folder_and_zip(tmp_path):
    import shutil

    root = tmp_path / "Final_Training" / "Images"
    _write_gtsrb(root)
    data = load_gtsrb_train(root, size=32)
    assert data.images.shape == (12, 32, 32, 3) and set(data.labels) == {0, 1}
    assert len(set(data.groups)) == 4 and data.class_names == GTSRB_CLASSES
    archive = shutil.make_archive(str(tmp_path / "gtsrb"), "zip", tmp_path, "Final_Training")
    assert len(load_gtsrb_train(archive, size=32, limit_per_class=2)) == 4
    with pytest.raises(DataError):
        load_gtsrb_train(tmp_path / "empty_dir_that_has_no_csv")


def test_gtsrb_official_test_reader(tmp_path):
    for i in range(3):
        Image.new("RGB", (30, 30)).save(tmp_path / f"{i:05d}.ppm")
    gt = tmp_path / "GT-final_test.csv"
    gt.write_text("Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n"
                  + "\n".join(f"{i:05d}.ppm;30;30;2;2;27;27;{i}" for i in range(3)), encoding="utf-8")
    data = load_gtsrb_test(tmp_path, gt, size=32)
    assert list(data.labels) == [0, 1, 2]
