"""Tests for converting variable-length examples into PyTorch tensors."""

import math

import polars as pl
import torch

from taobao.data.tensors import examples_to_tensors


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
