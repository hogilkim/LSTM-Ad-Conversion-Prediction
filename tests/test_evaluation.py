"""Focused tests for shared metrics and deterministic README result updates."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from taobao.evaluation import EvaluationMetrics, evaluate, update_readme_results


def test_evaluate_calculates_overall_and_bucket_metrics_at_boundaries() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    probabilities = np.array(
        [0.1, 0.9, 0.8, 0.2, 0.3, 0.7, 0.6, 0.4, 0.2, 0.8, 0.9, 0.1]
    )
    lengths = np.array([3, 3, 10, 10, 11, 11, 30, 30, 31, 31, 50, 50])

    metrics = evaluate(labels, probabilities, lengths)

    assert metrics.auc == pytest.approx(roc_auc_score(labels, probabilities))
    assert metrics.log_loss == pytest.approx(
        log_loss(labels, probabilities, labels=[0, 1])
    )
    assert metrics.auc_by_sequence_length == {
        "3-10": 0.75,
        "11-30": 0.75,
        "31-50": 0.25,
    }


def test_evaluate_returns_none_when_bucket_auc_is_undefined() -> None:
    metrics = evaluate(
        labels=[0, 1, 0, 1],
        probabilities=[0.1, 0.9, 0.2, 0.8],
        sequence_lengths=[3, 3, 11, 31],
    )

    assert metrics.auc_by_sequence_length == {
        "3-10": 1.0,
        "11-30": None,
        "31-50": None,
    }


def test_seeded_random_predictions_have_auc_near_half() -> None:
    random = np.random.default_rng(42)
    labels = np.tile([0, 1], 5_000)
    probabilities = random.random(labels.size)
    lengths = random.integers(3, 51, size=labels.size)

    metrics = evaluate(labels, probabilities, lengths)

    assert metrics.auc == pytest.approx(0.5, abs=0.02)
    for bucket_auc in metrics.auc_by_sequence_length.values():
        assert bucket_auc == pytest.approx(0.5, abs=0.04)


@pytest.mark.parametrize(
    ("labels", "probabilities", "lengths", "message"),
    [
        ([], [], [], "cannot be empty"),
        ([0, 1], [0.2], [3, 4], "equal length"),
        ([[0, 1]], [0.2, 0.8], [3, 4], "one-dimensional"),
        ([0, 2], [0.2, 0.8], [3, 4], "only 0 and 1"),
        ([1, 1], [0.2, 0.8], [3, 4], "both 0 and 1"),
        ([0, 1], [0.2, np.nan], [3, 4], "finite"),
        ([0, 1], [-0.1, 0.8], [3, 4], "between 0 and 1"),
        ([0, 1], [0.2, 0.8], [3.5, 4], "integers"),
        ([0, 1], [0.2, 0.8], [2, 51], "between 3 and 50"),
    ],
)
def test_evaluate_rejects_invalid_inputs(labels, probabilities, lengths, message) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match=message):
        evaluate(labels, probabilities, lengths)


def _write_readme(path: Path) -> None:
    path.write_text(
        """# Project

## Results

<!-- results-table:start -->
| Model | Split | AUC | Log loss | AUC (seq_len 3-10) | AUC (seq_len 11-30) | AUC (seq_len 31-50) |
|-------|-------|-----|----------|--------------------|---------------------|---------------------|
| Logistic regression baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- results-table:end -->

End.
"""
    )


def test_update_readme_results_replaces_placeholder_and_is_idempotent(tmp_path) -> None:
    readme_path = tmp_path / "README.md"
    _write_readme(readme_path)
    metrics = EvaluationMetrics(
        auc=0.71234567,
        log_loss=0.45678901,
        auc_by_sequence_length={"3-10": 0.61, "11-30": 0.72, "31-50": None},
    )

    assert update_readme_results(
        readme_path, model="LSTM", split="validation", metrics=metrics
    )
    first_update = readme_path.read_text()
    assert (
        "| LSTM | validation | 0.712346 | 0.456789 | 0.610000 | 0.720000 | N/A |"
        in first_update
    )
    assert first_update.count("| LSTM |") == 1

    assert not update_readme_results(
        readme_path, model="LSTM", split="validation", metrics=metrics
    )
    assert readme_path.read_text() == first_update


def test_update_readme_results_appends_a_new_row_deterministically(tmp_path) -> None:
    readme_path = tmp_path / "README.md"
    _write_readme(readme_path)
    metrics = EvaluationMetrics(
        auc=0.7,
        log_loss=0.5,
        auc_by_sequence_length={"3-10": 0.6, "11-30": 0.7, "31-50": 0.8},
    )

    update_readme_results(
        readme_path, model="Mean pooling", split="validation", metrics=metrics
    )

    rows = [line for line in readme_path.read_text().splitlines() if line.startswith("|")]
    assert rows[-1] == (
        "| Mean pooling | validation | 0.700000 | 0.500000 | "
        "0.600000 | 0.700000 | 0.800000 |"
    )


def test_update_readme_results_rejects_an_unmarked_table(tmp_path) -> None:
    readme_path = tmp_path / "README.md"
    readme_path.write_text("| Model | Split |\n|---|---|\n")
    metrics = EvaluationMetrics(
        auc=0.7,
        log_loss=0.5,
        auc_by_sequence_length={"3-10": 0.6, "11-30": 0.7, "31-50": 0.8},
    )

    with pytest.raises(ValueError, match="exactly one marked results table"):
        update_readme_results(
            readme_path, model="LSTM", split="validation", metrics=metrics
        )
