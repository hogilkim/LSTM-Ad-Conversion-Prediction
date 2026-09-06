"""Tests for converting variable-length examples into PyTorch tensors."""

import math

import numpy as np
import polars as pl
import pytest
import torch

from taobao.data.tensors import ARRAY_NAMES, build_split, examples_to_tensors, load_split


def test_examples_to_tensors_right_pads_and_normalizes_hours() -> None:
    examples = pl.DataFrame(
        {
            "items": [[10, 20, 30], [40, 50]],
            "cats": [[1, 2, 3], [4, 5]],
            "behs": [[1, 2, 4], [1, 3]],
            "hours": [[168.0, 24.0, 1.0], [12.0, 2.0]],
            "seq_len": [3, 2],
            "label": [1, 0],
        }
    )

    tensors = examples_to_tensors(examples, max_len=5)

    assert tensors["items"].shape == (2, 5)
    assert tensors["cats"].shape == (2, 5)
    assert tensors["behs"].shape == (2, 5)
    assert tensors["hours"].shape == (2, 5)
    assert tensors["items"].dtype == torch.long
    assert tensors["cats"].dtype == torch.long
    assert tensors["behs"].dtype == torch.long
    assert tensors["hours"].dtype == torch.float32
    assert tensors["lengths"].dtype == torch.long
    assert tensors["labels"].dtype == torch.float32

    assert tensors["items"].tolist() == [
        [10, 20, 30, 0, 0],
        [40, 50, 0, 0, 0],
    ]
    assert tensors["cats"].tolist() == [
        [1, 2, 3, 0, 0],
        [4, 5, 0, 0, 0],
    ]
    assert tensors["behs"].tolist() == [
        [1, 2, 4, 0, 0],
        [1, 3, 0, 0, 0],
    ]
    assert tensors["lengths"].tolist() == [3, 2]
    assert tensors["labels"].tolist() == [1.0, 0.0]

    expected_hours = torch.tensor(
        [
            [1.0, math.log1p(24.0) / math.log1p(168.0), math.log1p(1.0) / math.log1p(168.0), 0.0, 0.0],
            [math.log1p(12.0) / math.log1p(168.0), math.log1p(2.0) / math.log1p(168.0), 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(tensors["hours"], expected_hours)


def test_build_and_memory_map_one_split(tmp_path) -> None:
    examples = pl.DataFrame(
        {
            "split": [0, 1, 0],
            "items": [[10, 20], [99], [30]],
            "cats": [[1, 2], [9], [3]],
            "behs": [[1, 2], [4], [3]],
            "hours": [[24.0, 1.0], [2.0], [12.0]],
            "seq_len": [2, 1, 1],
            "label": [1, 1, 0],
        }
    )
    examples_path = tmp_path / "examples.parquet"
    out_dir = tmp_path / "tensors"
    examples.write_parquet(examples_path)

    tensors = build_split("train", examples_path, out_dir, max_len=3)

    assert tensors["items"].tolist() == [[10, 20, 0], [30, 0, 0]]
    assert tensors["labels"].tolist() == [1.0, 0.0]
    assert {path.name for path in out_dir.iterdir()} == {
        f"train_{name}.npy" for name in ARRAY_NAMES
    }

    arrays = load_split("train", out_dir, array_names=("cats", "labels"))
    assert set(arrays) == {"cats", "labels"}
    assert isinstance(arrays["cats"], np.memmap)
    assert arrays["cats"].tolist() == [[1, 2, 0], [3, 0, 0]]
    assert arrays["labels"].tolist() == [1.0, 0.0]


def test_build_split_rejects_unknown_split(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown split"):
        build_split("dev", tmp_path / "examples.parquet", tmp_path)
