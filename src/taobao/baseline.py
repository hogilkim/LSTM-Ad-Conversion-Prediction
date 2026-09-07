"""Fit and score the logistic-regression baseline on count features.

Example:
    uv run python -m taobao.baseline --split val --update-readme

Fits a ``StandardScaler`` and ``LogisticRegression`` on the train split only, scores one
split with the shared metrics, and prints the AUC of the last-24h cart count alone as a
sanity floor. The fitted pipeline is saved so later test scoring reuses the same weights.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from taobao.data.tensors import DEFAULT_OUT_DIR, load_split
from taobao.evaluation import EvaluationMetrics, evaluate, update_readme_results
from taobao.features.counts import FEATURE_NAMES, SANITY_FLOOR_FEATURE, count_features

DEFAULT_MODEL_PATH = Path("models/baseline_logreg.joblib")
DEFAULT_README = Path("README.md")
MODEL_NAME = "Logistic regression baseline"
FEATURE_ARRAYS = ("behs", "cats", "hours", "lengths", "labels")


def load_features(
    split: str, tensors_dir: Path = DEFAULT_OUT_DIR
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(features, labels, lengths)`` for one split from the padded tensors."""
    arrays = load_split(split, tensors_dir, FEATURE_ARRAYS)
    features = count_features(arrays["behs"], arrays["cats"], arrays["hours"], arrays["lengths"])
    labels = np.asarray(arrays["labels"], dtype=np.float64)
    lengths = np.asarray(arrays["lengths"], dtype=np.int64)
    return features, labels, lengths


def fit_baseline(features: np.ndarray, labels: np.ndarray, seed: int = 42) -> Pipeline:
    """Standardise the features and fit an L2 logistic regression."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
    )
    model.fit(features, labels)
    return model


def score_baseline(
    model: Pipeline, features: np.ndarray, labels: np.ndarray, lengths: np.ndarray
) -> EvaluationMetrics:
    probabilities = model.predict_proba(features)[:, 1]
    return evaluate(labels, probabilities, lengths)


def sanity_floor_auc(features: np.ndarray, labels: np.ndarray) -> float:
    """AUC of the last-24h cart count used directly as a score, no model at all."""
    column = FEATURE_NAMES.index(SANITY_FLOOR_FEATURE)
    return float(roc_auc_score(labels, features[:, column]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--tensors-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--update-readme", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t0 = time.perf_counter()
    train_features, train_labels, _ = load_features("train", args.tensors_dir)
    print(f"train features {train_features.shape} built in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    model = fit_baseline(train_features, train_labels, seed=args.seed)
    print(f"fit in {time.perf_counter() - t0:.1f}s")
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)

    coefficients = model.named_steps["logisticregression"].coef_[0]
    for name, weight in sorted(zip(FEATURE_NAMES, coefficients), key=lambda x: -abs(x[1])):
        print(f"  coef[{name}] = {weight:+.4f}")

    features, labels, lengths = load_features(args.split, args.tensors_dir)
    metrics = score_baseline(model, features, labels, lengths)
    split_label = "validation" if args.split == "val" else args.split
    print(f"split={split_label} auc={metrics.auc:.6f} log_loss={metrics.log_loss:.6f}")
    for bucket, value in metrics.auc_by_sequence_length.items():
        print(f"auc[seq_len {bucket}]={'N/A' if value is None else f'{value:.6f}'}")
    print(f"sanity floor auc[{SANITY_FLOOR_FEATURE} alone]={sanity_floor_auc(features, labels):.6f}")

    if args.update_readme:
        changed = update_readme_results(
            args.readme, model=MODEL_NAME, split=split_label, metrics=metrics
        )
        print(f"readme_updated={changed}")


if __name__ == "__main__":
    main()
