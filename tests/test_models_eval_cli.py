"""Models, the robustness benchmark (problems 1, 3, 4, 7), latency (6) and the CLI (8, 9)."""
import json

import numpy as np
import pytest

from signsight.cli import main
from signsight.config import ConfigError, Settings
from signsight.data import track_split
from signsight.evaluate import ablation, clean_metrics, corruption_error, corruption_suite, ece, leakage_gap
from signsight.models import HogClassifier
from signsight.pipeline import latency, predict_files


@pytest.fixture(scope="module")
def fitted(signs):
    train, val = track_split(signs, seed=0)
    return HogClassifier(seed=0).fit(train.images, train.labels), val


def test_clean_metrics_have_macro_f1_and_worst_classes(fitted):
    model, val = fitted
    m = clean_metrics(model, val)
    assert m["accuracy"] > 0.7 and 0 < m["macro_f1"] <= 1
    assert len(m["worst_classes"]) == 5 and "name" in m["worst_classes"][0]


def test_corruptions_hurt_the_baseline(fitted):
    """Problem 4: robustness is measured on corrupted copies of held-out images."""
    model, val = fitted
    suite = corruption_suite(model, val, names=("fog", "night"), severities=(1, 5))
    clean = clean_metrics(model, val)["accuracy"]
    assert suite["fog"][5] < clean and suite["night"][5] < suite["night"][1]


def test_augmentation_improves_corrupted_accuracy(signs):
    factories = {"baseline": lambda s: HogClassifier(seed=s), "augmented": lambda s: HogClassifier(augment_share=0.5, seed=s)}
    res = ablation(signs, factories, seeds=(0,), names=("fog", "rain", "night"), severities=(3,))
    v = res["variants"]
    assert v["augmented"]["corrupt_accuracy"]["mean"] > v["baseline"]["corrupt_accuracy"]["mean"] + 0.05
    assert v["baseline"]["mce"]["mean"] == 1.0 and v["augmented"]["mce"]["mean"] < 1.0


def test_condition_model_uses_the_image_not_a_constant(signs):
    """Problem 1: the condition input changes with the image."""
    from signsight.corruptions import corrupt_batch

    train, val = track_split(signs, seed=0)
    model = HogClassifier(augment_share=0.5, use_condition=True, seed=0).fit(train.images, train.labels)
    clean_p = model.condition_.predict_proba(val.images).mean(axis=0)
    fog_p = model.condition_.predict_proba(corrupt_batch(val.images, "fog", 5)).mean(axis=0)
    assert clean_p[0] > fog_p[0] and fog_p[1] > clean_p[1]  # column 0 = clean, 1 = fog
    with pytest.raises(ValueError):
        HogClassifier(use_condition=True)


def test_leakage_gap_is_measured(signs):
    gap = leakage_gap(signs, lambda s: HogClassifier(seed=s))
    assert gap["image_split"] >= gap["track_split"]


def test_corruption_error_and_ece():
    ref = {"fog": {1: 0.5, 3: 0.5}}
    assert corruption_error({"fog": {1: 0.75, 3: 0.75}}, ref)["fog"] == 0.5
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert ece([0, 1], proba, np.array([0, 1])) == pytest.approx(0.15)


def test_cnn_trains_with_validation_early_stopping(signs):
    pytest.importorskip("torch")
    from signsight.models.cnn import CnnClassifier

    train, val = track_split(signs, seed=0)
    model = CnnClassifier(epochs=2, batch=64, seed=0).fit(train.images, train.labels, val.images, val.labels)
    assert len(model.history_) <= 2 and model.predict_proba(val.images[:4]).shape == (4, 10)


def test_latency_and_predict(tmp_path, fitted):
    """Problems 5 and 6: full-image prediction uses the detector, and latency covers all stages."""
    from PIL import Image

    from signsight.synthetic import CLASS_NAMES, make_scene

    model, _ = fitted
    paths = []
    for i in range(5):
        scene, _ = make_scene(i, seed=i)
        p = tmp_path / f"s{i}.png"
        Image.fromarray(scene).save(p)
        paths.append(p)
    lat = latency(model, paths, warmup=1)
    assert set(lat) >= {"read", "detect", "crop", "classify", "total", "fps_p50"}
    assert lat["total"]["p50_ms"] >= lat["classify"]["p50_ms"]
    preds = predict_files(model, CLASS_NAMES, paths, min_confidence=0.99)
    assert all(p.box is not None for p in preds)
    assert all(p.name == "unknown" or p.confidence >= 0.99 for p in preds)


def test_settings(monkeypatch):
    monkeypatch.setenv("SIGNSIGHT_BACKEND", "resnet")
    with pytest.raises(ConfigError):
        Settings.from_env()
    monkeypatch.setenv("SIGNSIGHT_BACKEND", "hog")
    monkeypatch.setenv("SIGNSIGHT_IMAGE_SIZE", "30")
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_cli_train_predict_latency_benchmark(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SIGNSIGHT_SEEDS", "1")
    out = tmp_path / "run"
    assert main(["train", "--tracks", "6", "--frames", "5", "--augment", "--out", str(out)]) == 0
    assert "shared tracks 0" in capsys.readouterr().out
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluated_on"] == "held-out tracks" and "night" in metrics["corruptions"]
    assert main(["make-scenes", "--out", str(tmp_path / "scenes"), "--n", "4"]) == 0
    assert main(["predict", "--model", str(out / "model.joblib"), "--images", str(tmp_path / "scenes")]) == 0
    assert "box" in capsys.readouterr().out
    assert main(["latency", "--model", str(out / "model.joblib"), "--images", str(tmp_path / "scenes")]) == 0
    assert "fps_p50" in capsys.readouterr().out
    assert main(["benchmark", "--tracks", "6", "--frames", "5", "--out", str(tmp_path / "b")]) == 0
    bench = json.loads((tmp_path / "b" / "benchmark.json").read_text(encoding="utf-8"))
    assert set(bench["variants"]) == {"baseline", "augmented", "augmented_condition"} and "leakage" in bench


def test_cli_error(capsys):
    assert main(["predict", "--model", "missing.joblib", "--images", "nowhere"]) == 1
    assert "error:" in capsys.readouterr().err
