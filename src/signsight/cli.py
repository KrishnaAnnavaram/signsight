"""Command line: ``signsight <command>``. Run ``signsight --help`` for the list."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from .config import ConfigError, Settings, load_dotenv
from .corruptions import CORRUPTIONS
from .data import DataError, ImageSet, load_gtsrb_test, load_gtsrb_train, track_overlap, track_split
from .evaluate import ablation, clean_metrics, corruption_suite, leakage_gap, mean_corruption_accuracy
from .models import HogClassifier
from .pipeline import latency, list_images, predict_files
from .synthetic import make_scene, make_signs


def _data(args, settings) -> tuple[ImageSet, str]:
    if args.data == "synthetic":
        return make_signs(args.tracks, args.frames, settings.image_size, settings.seed), "synthetic signs"
    return load_gtsrb_train(args.data, settings.image_size, args.limit_per_class), str(args.data)


def _test(args, settings) -> ImageSet | None:
    if getattr(args, "test_images", None) and getattr(args, "test_gt", None):
        return load_gtsrb_test(args.test_images, args.test_gt, settings.image_size)
    return None


def _factory(backend: str, augment: bool, condition: bool, settings):
    share = 0.5 if augment else 0.0
    if backend == "cnn":
        from .models.cnn import CnnClassifier  # optional extra "torch"

        return lambda seed: CnnClassifier(epochs=settings.epochs, augment_share=share, seed=seed)
    return lambda seed: HogClassifier(augment_share=share, use_condition=condition, seed=seed)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def cmd_train(args, settings):
    data, source = _data(args, settings)
    train, val = track_split(data, seed=settings.seed)
    print(f"{source}: {len(data)} images, {len(set(data.groups))} tracks, train {len(train)} / val {len(val)}, "
          f"shared tracks {track_overlap(train, val)}")
    model = _factory(settings.backend, args.augment, args.condition, settings)(settings.seed)
    model.fit(train.images, train.labels, val_images=val.images, val_labels=val.labels)
    target = _test(args, settings) or val
    report = {"source": source, "backend": settings.backend, "augment": args.augment, "condition": args.condition,
              "evaluated_on": "official test" if target is not val else "held-out tracks",
              "clean": clean_metrics(model, target), "corruptions": corruption_suite(model, target, severities=(1, 3, 5))}
    report["mean_corruption_accuracy"] = mean_corruption_accuracy(report["corruptions"])
    out = Path(args.out or settings.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "class_names": data.class_names, "size": settings.image_size}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    c = report["clean"]
    print(f"clean ({report['evaluated_on']}): accuracy {c['accuracy']:.3f}, macro-F1 {c['macro_f1']:.3f}, ECE {c['ece']:.3f}")
    print(f"mean accuracy over 6 corruptions x severities 1, 3, 5: {report['mean_corruption_accuracy']:.3f}")
    print(f"saved {out / 'model.joblib'} and {out / 'metrics.json'}")
    return 0


def cmd_evaluate(args, settings):
    bundle = joblib.load(args.model)
    target = _test(args, settings)
    if target is None:
        data, _ = _data(args, settings)
        target = track_split(data, seed=settings.seed)[1]
    suite = corruption_suite(bundle["model"], target)
    print(json.dumps(_jsonable({"clean": clean_metrics(bundle["model"], target), "corruptions": suite,
                                "mean_corruption_accuracy": mean_corruption_accuracy(suite)}), indent=2))
    return 0


def cmd_benchmark(args, settings):
    data, source = _data(args, settings)
    factories = {
        "baseline": _factory(settings.backend, False, False, settings),
        "augmented": _factory(settings.backend, True, False, settings),
    }
    if settings.backend == "hog":
        factories["augmented_condition"] = _factory("hog", True, True, settings)
    seeds = tuple(range(settings.seed, settings.seed + settings.seeds))
    result = ablation(data, factories, seeds=seeds, test=_test(args, settings))
    result["leakage"] = leakage_gap(data, factories["baseline"], seed=settings.seed)
    result["source"] = source
    out = Path(args.out or settings.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(format_benchmark(result))
    print(f"wrote {out / 'benchmark.json'}")
    return 0


def format_benchmark(result: dict) -> str:
    lines = [f"seeds {result['seeds']}, severities {result['severities']} (mean +- std over seeds)",
             f"{'variant':<22}{'clean acc':>14}{'macro-F1':>14}{'corrupt acc':>14}{'mCE':>14}"]
    for name, v in result["variants"].items():
        cell = lambda k: f"{v[k]['mean']:.3f}+-{v[k]['std']:.3f}"  # noqa: E731
        lines.append(f"{name:<22}{cell('clean_accuracy'):>14}{cell('clean_macro_f1'):>14}{cell('corrupt_accuracy'):>14}{cell('mce'):>14}")
    lines.append("accuracy by corruption at severity 3: " + "; ".join(
        f"{name}: " + ", ".join(f"{c} {v['by_corruption'][c][3]:.2f}" for c in CORRUPTIONS) for name, v in result["variants"].items()))
    if "leakage" in result:
        lk = result["leakage"]
        lines.append(f"leakage check: image split {lk['image_split']:.3f}, track split {lk['track_split']:.3f}, gap {lk['gap']:+.3f}")
    return "\n".join(lines)


def cmd_predict(args, settings):
    bundle = joblib.load(args.model)
    preds = predict_files(bundle["model"], bundle["class_names"], list_images(args.images), bundle["size"],
                          use_detector=not args.no_detect, min_confidence=args.min_confidence)
    for p in preds:
        print(f"{Path(p.file).name}: {p.name} ({p.confidence:.2f}) box {p.box}")
    return 0


def cmd_latency(args, settings):
    bundle = joblib.load(args.model)
    print(json.dumps(latency(bundle["model"], list_images(args.images), bundle["size"], use_detector=not args.no_detect), indent=2))
    return 0


def cmd_make_scenes(args, settings):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(settings.seed)
    for i in range(args.n):
        cls = int(rng.integers(0, 10))
        scene, box = make_scene(cls, seed=settings.seed + i)
        Image.fromarray(scene).save(out / f"scene_{i:03d}_class{cls}.png")
    print(f"wrote {args.n} scene images to {out}")
    return 0


def cmd_demo(args, settings):
    print("signsight offline demo: synthetic signs, HOG + logistic regression, no download, no network\n")
    args.data = "synthetic"
    data, _ = _data(args, settings)
    factories = {"baseline": _factory("hog", False, False, settings), "augmented": _factory("hog", True, False, settings),
                 "augmented_condition": _factory("hog", True, True, settings)}
    result = ablation(data, factories, seeds=tuple(range(settings.seed, settings.seed + settings.seeds)))
    result["leakage"] = leakage_gap(data, factories["baseline"], seed=settings.seed)
    print(format_benchmark(result))
    with tempfile.TemporaryDirectory() as tmp:
        train, val = track_split(data, seed=settings.seed)
        model = HogClassifier(augment_share=0.5, seed=settings.seed).fit(train.images, train.labels)
        paths, truth = [], []
        for i in range(20):
            cls = i % 10
            scene, _ = make_scene(cls, seed=1000 + i)
            p = Path(tmp) / f"scene_{i:02d}.png"
            Image.fromarray(scene).save(p)
            paths.append(p)
            truth.append(cls)
        for detector in (True, False):
            preds = predict_files(model, data.class_names, paths, settings.image_size, use_detector=detector, min_confidence=0.0)
            acc = np.mean([p.label == t for p, t in zip(preds, truth)])
            print(f"20 scene images, detector {'on' if detector else 'off'}: accuracy {acc:.2f}")
        lat = latency(model, paths, settings.image_size)
        print(f"latency per image (read + detect + crop + classify): p50 {lat['total']['p50_ms']:.1f} ms, "
              f"p95 {lat['total']['p95_ms']:.1f} ms")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signsight", description="Weather-robust traffic-sign recognition.")
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(c):
        c.add_argument("--data", default="synthetic", help="'synthetic', a GTSRB training zip or its extracted folder")
        c.add_argument("--tracks", type=int, default=12, help="synthetic tracks for each class")
        c.add_argument("--frames", type=int, default=10, help="synthetic frames for each track")
        c.add_argument("--limit-per-class", type=int, help="read at most this many GTSRB images for each class")
        c.add_argument("--test-images", help="official GTSRB test image folder")
        c.add_argument("--test-gt", help="official GTSRB GT-final_test.csv")

    t = sub.add_parser("train", help="train on a track split and save the model")
    data_args(t)
    t.add_argument("--augment", action="store_true", help="weather augmentation of training images")
    t.add_argument("--condition", action="store_true", help="add the predicted weather condition (hog only)")
    t.add_argument("--out")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="clean metrics and the full corruption suite for a saved model")
    data_args(e)
    e.add_argument("--model", required=True)
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("benchmark", help="ablation over seeds: baseline, augmented, augmented + condition")
    data_args(b)
    b.add_argument("--out")
    b.set_defaults(func=cmd_benchmark)

    for name, func, text in (("predict", cmd_predict, "classify a folder of images"),
                             ("latency", cmd_latency, "measure per-image latency of the full pipeline")):
        c = sub.add_parser(name, help=text)
        c.add_argument("--model", required=True)
        c.add_argument("--images", required=True)
        c.add_argument("--no-detect", action="store_true", help="do not crop with the detector")
        if name == "predict":
            c.add_argument("--min-confidence", type=float, default=0.5)
        c.set_defaults(func=func)

    s = sub.add_parser("make-scenes", help="write synthetic street images with one sign each")
    s.add_argument("--out", default="data/scenes")
    s.add_argument("--n", type=int, default=20)
    s.set_defaults(func=cmd_make_scenes)

    d = sub.add_parser("demo", help="offline demo on synthetic signs")
    data_args(d)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_dotenv()
        return args.func(args, Settings.from_env())
    except (ConfigError, DataError, ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
