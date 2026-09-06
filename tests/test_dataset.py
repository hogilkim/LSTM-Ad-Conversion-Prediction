"""Tests for the memory-mapped PyTorch dataset."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from taobao.data.dataset import TensorSplitDataset


def test_tensor_split_dataset_loads_only_requested_arrays(tmp_path) -> None:
    arrays = {
        "items": np.array([[10, 20, 0], [30, 0, 0]], dtype=np.int64),
        "cats": np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int64),
        "behs": np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int64),
        "hours": np.array([[0.5, 0.1, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        "lengths": np.array([2, 1], dtype=np.int64),
        "labels": np.array([1.0, 0.0], dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(tmp_path / f"train_{name}.npy", array)

    dataset = TensorSplitDataset("train", tmp_path)

    assert len(dataset) == 2
    assert set(dataset.arrays) == {"cats", "behs", "hours", "lengths", "labels"}
    assert all(isinstance(array, np.memmap) for array in dataset.arrays.values())
    assert set(dataset[0]) == {"cats", "behs", "hours", "lengths", "labels"}
    assert dataset[0]["cats"].tolist() == [1, 2, 0]
    assert dataset[0]["lengths"].dtype == torch.long
    assert dataset[0]["labels"].dtype == torch.float32

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert batch["cats"].shape == (2, 3)
    assert batch["lengths"].tolist() == [2, 1]
    assert batch["labels"].tolist() == [1.0, 0.0]
