"""Clean metrics, corruption suites, the robustness ablation and the leakage check."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .corruptions import CORRUPTIONS, SEVERITIES, corrupt_batch
from .data import ImageSet, image_split, track_split


def ece(labels, proba, classes, bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    correct = (pred == np.asarray(labels)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(correct[idx == b].mean() - conf[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def clean_metrics(model, data: ImageSet, worst: int = 5) -> dict:
    proba = model.predict_proba(data.images)
    pred = model.classes_[proba.argmax(axis=1)]
    labels = np.unique(data.labels)
    recalls = recall_score(data.labels, pred, labels=labels, average=None, zero_division=0)
    order = np.argsort(recalls)[:worst]
    return {
        "n": int(len(data)),
        "accuracy": float(accuracy_score(data.labels, pred)),
        "macro_f1": float(f1_score(data.labels, pred, average="macro")),
        "ece": ece(data.labels, proba, model.classes_),
        "worst_classes": [{"class": int(labels[i]), "name": data.class_names[int(labels[i])], "recall": float(recalls[i])} for i in order],
    }


def corruption_suite(model, data: ImageSet, names=tuple(CORRUPTIONS), severities=SEVERITIES, seed: int = 0) -> dict:
    """Accuracy on corrupted copies of the held-out images, for each corruption and severity."""
    out = {}
    for k, name in enumerate(names):
        out[name] = {}
        for s in severities:
            images = corrupt_batch(data.images, name, s, seed=seed + 100 * k + s)
            out[name][s] = float(accuracy_score(data.labels, model.predict(images)))
    return out


def mean_corruption_accuracy(suite: dict) -> float:
    return float(np.mean([acc for sev in suite.values() for acc in sev.values()]))


def corruption_error(suite: dict, reference: dict) -> dict:
    """mCE-style score: for each corruption, sum of errors over severities divided by the
    same sum of the reference model. 1.0 = as good as the reference, lower is better."""
    ce = {}
    for name in suite:
        err = sum(1 - a for a in suite[name].values())
        ref = sum(1 - a for a in reference[name].values())
        ce[name] = float(err / ref) if ref > 0 else float("nan")
    ce["mean"] = float(np.nanmean(list(ce.values())))
    return ce


def ablation(data: ImageSet, factories: dict[str, Callable[[int], object]], seeds=(0, 1, 2),
             names=tuple(CORRUPTIONS), severities=(1, 3, 5), test: ImageSet | None = None) -> dict:
    """Train each variant with each seed on a track split, then evaluate it.

    If ``test`` is given (for example the official GTSRB test set), the metrics use it,
    and the validation part only exists for early stopping. Else the metrics use the
    held-out tracks.
    """
    rows = {name: [] for name in factories}
    for seed in seeds:
        train, val = track_split(data, seed=seed)
        target = test if test is not None else val
        suites = {}
        for name, factory in factories.items():
            model = factory(seed).fit(train.images, train.labels, val_images=val.images, val_labels=val.labels)
            clean = clean_metrics(model, target)
            suites[name] = corruption_suite(model, target, names, severities, seed=seed)
            rows[name].append({"clean_accuracy": clean["accuracy"], "clean_macro_f1": clean["macro_f1"],
                               "corrupt_accuracy": mean_corruption_accuracy(suites[name]), "suite": suites[name]})
        reference = suites[next(iter(factories))]
        for name in factories:
            rows[name][-1]["mce"] = corruption_error(suites[name], reference)["mean"]
    summary = {}
    for name, runs in rows.items():
        summary[name] = {
            key: {"mean": float(np.mean([r[key] for r in runs])), "std": float(np.std([r[key] for r in runs]))}
            for key in ("clean_accuracy", "clean_macro_f1", "corrupt_accuracy", "mce")
        }
        per = {}
        for c in names:
            per[c] = {s: float(np.mean([r["suite"][c][s] for r in runs])) for s in severities}
        summary[name]["by_corruption"] = per
    return {"seeds": list(seeds), "severities": list(severities), "variants": summary}


def leakage_gap(data: ImageSet, factory: Callable[[int], object], seed: int = 0) -> dict:
    """Accuracy with a random image split minus accuracy with a track split."""
    out = {}
    for name, split in (("image_split", image_split), ("track_split", track_split)):
        train, val = split(data, seed=seed)
        model = factory(seed).fit(train.images, train.labels)
        out[name] = float(accuracy_score(val.labels, model.predict(val.images)))
    out["gap"] = out["image_split"] - out["track_split"]
    return out
