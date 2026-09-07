"""Shared prediction metrics and deterministic README result recording."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import log_loss, roc_auc_score

LENGTH_BUCKETS = (
    ("3-10", 3, 10),
    ("11-30", 11, 30),
    ("31-50", 31, 50),
)

RESULTS_TABLE_START = "<!-- results-table:start -->"
RESULTS_TABLE_END = "<!-- results-table:end -->"
_RESULTS_HEADER = (
    "| Model | Split | AUC | Log loss | AUC (seq_len 3-10) | "
    "AUC (seq_len 11-30) | AUC (seq_len 31-50) |",
    "|-------|-------|-----|----------|--------------------|"
    "---------------------|---------------------|",
)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Overall metrics and AUC slices for the three supported length buckets."""

    auc: float
    log_loss: float
    auc_by_sequence_length: dict[str, float | None] = field(default_factory=dict)


def _as_float_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def evaluate(
    labels: ArrayLike,
    probabilities: ArrayLike,
    sequence_lengths: ArrayLike,
) -> EvaluationMetrics:
    """Calculate shared metrics from binary labels, probabilities, and lengths.

    A bucket AUC is ``None`` when that slice is empty or contains only one label class,
    because ROC AUC is undefined in either case.
    """
    labels_array = _as_float_vector(labels, "labels")
    probabilities_array = _as_float_vector(probabilities, "probabilities")
    lengths_array = _as_float_vector(sequence_lengths, "sequence_lengths")

    if not (
        labels_array.size == probabilities_array.size == lengths_array.size
    ):
        raise ValueError("labels, probabilities, and sequence_lengths must have equal length")
    if not np.isin(labels_array, (0.0, 1.0)).all():
        raise ValueError("labels must contain only 0 and 1")
    if np.unique(labels_array).size != 2:
        raise ValueError("labels must contain both 0 and 1 to calculate AUC")
    if ((probabilities_array < 0.0) | (probabilities_array > 1.0)).any():
        raise ValueError("probabilities must be between 0 and 1 inclusive")
    if not np.equal(lengths_array, np.floor(lengths_array)).all():
        raise ValueError("sequence_lengths must contain integers")
    if ((lengths_array < 3) | (lengths_array > 50)).any():
        raise ValueError("sequence_lengths must be between 3 and 50 inclusive")

    auc_by_sequence_length: dict[str, float | None] = {}
    for name, lower, upper in LENGTH_BUCKETS:
        in_bucket = (lengths_array >= lower) & (lengths_array <= upper)
        bucket_labels = labels_array[in_bucket]
        if bucket_labels.size == 0 or np.unique(bucket_labels).size != 2:
            auc_by_sequence_length[name] = None
        else:
            auc_by_sequence_length[name] = float(
                roc_auc_score(bucket_labels, probabilities_array[in_bucket])
            )

    return EvaluationMetrics(
        auc=float(roc_auc_score(labels_array, probabilities_array)),
        log_loss=float(
            log_loss(labels_array, probabilities_array, labels=[0.0, 1.0])
        ),
        auc_by_sequence_length=auc_by_sequence_length,
    )


def _validated_result_cell(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if any(character in value for character in "|\r\n"):
        raise ValueError(f"{name} cannot contain table delimiters or newlines")
    return value


def _format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    if not np.isfinite(value):
        raise ValueError("README metrics must be finite")
    return f"{value:.6f}"


def _parse_result_row(line: str) -> tuple[str, ...]:
    if not line.startswith("|") or not line.endswith("|"):
        raise ValueError("README results rows must be Markdown table rows")
    cells = tuple(cell.strip() for cell in line[1:-1].split("|"))
    if len(cells) != 7:
        raise ValueError("README results rows must contain exactly seven columns")
    return cells


def update_readme_results(
    readme_path: Path,
    *,
    model: str,
    split: str,
    metrics: EvaluationMetrics,
) -> bool:
    """Atomically upsert one row in the README's single marked results table.

    An exact model/split row is replaced. A same-model all-``TBD`` placeholder is also
    replaced; otherwise the row is appended. Returns ``False`` when the file already
    contains the exact rendered result.
    """
    model = _validated_result_cell(model, "model")
    split = _validated_result_cell(split, "split")
    expected_buckets = {name for name, _, _ in LENGTH_BUCKETS}
    if set(metrics.auc_by_sequence_length) != expected_buckets:
        raise ValueError(
            "metrics must contain AUC values for sequence-length buckets: "
            + ", ".join(name for name, _, _ in LENGTH_BUCKETS)
        )

    rendered_row = (
        model,
        split,
        _format_metric(metrics.auc),
        _format_metric(metrics.log_loss),
        *(
            _format_metric(metrics.auc_by_sequence_length[name])
            for name, _, _ in LENGTH_BUCKETS
        ),
    )

    contents = readme_path.read_text(encoding="utf-8")
    if contents.count(RESULTS_TABLE_START) != 1 or contents.count(RESULTS_TABLE_END) != 1:
        raise ValueError("README must contain exactly one marked results table")
    start = contents.index(RESULTS_TABLE_START)
    end = contents.index(RESULTS_TABLE_END, start)
    table_lines = contents[start + len(RESULTS_TABLE_START) : end].strip().splitlines()
    if tuple(table_lines[:2]) != _RESULTS_HEADER:
        raise ValueError("README results table header does not match the expected format")

    rows = [_parse_result_row(line) for line in table_lines[2:]]
    exact_matches = [
        index
        for index, row in enumerate(rows)
        if row[0] == model and row[1] == split
    ]
    if len(exact_matches) > 1:
        raise ValueError(f"README contains duplicate results for {model!r} on {split!r}")

    if exact_matches:
        rows[exact_matches[0]] = rendered_row
    else:
        placeholders = [
            index
            for index, row in enumerate(rows)
            if row[0] == model and all(value == "TBD" for value in row[1:])
        ]
        if len(placeholders) > 1:
            raise ValueError(f"README contains duplicate placeholders for {model!r}")
        if placeholders:
            rows[placeholders[0]] = rendered_row
        else:
            rows.append(rendered_row)

    rendered_table = "\n".join(
        (RESULTS_TABLE_START, *_RESULTS_HEADER)
        + tuple("| " + " | ".join(row) + " |" for row in rows)
        + (RESULTS_TABLE_END,)
    )
    updated = contents[:start] + rendered_table + contents[end + len(RESULTS_TABLE_END) :]
    if updated == contents:
        return False

    file_mode = readme_path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(
        dir=readme_path.parent,
        prefix=f".{readme_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(updated)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, file_mode)
        os.replace(temporary_path, readme_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True
