"""Convert processed examples into fixed-size PyTorch tensors."""

import numpy as np
import polars as pl
import torch


def _right_pad(sequences: list[list[int | float]], max_len: int, dtype: np.dtype) -> np.ndarray:
    """Copy variable-length sequences into a zero-filled 2D array."""
    padded = np.zeros((len(sequences), max_len), dtype=dtype)
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = sequence
    return padded


def examples_to_tensors(examples: pl.DataFrame, max_len: int = 50) -> dict[str, torch.Tensor]:
    """Right-pad processed examples and return tensors ready for a model."""
    items = _right_pad(examples["items"].to_list(), max_len, np.dtype(np.int64))
    cats = _right_pad(examples["cats"].to_list(), max_len, np.dtype(np.int64))
    behs = _right_pad(examples["behs"].to_list(), max_len, np.dtype(np.int64))

    hours = _right_pad(examples["hours"].to_list(), max_len, np.dtype(np.float32))
    hours = np.log1p(hours) / np.log1p(np.float32(168.0))

    lengths = examples["seq_len"].to_numpy().astype(np.int64)
    labels = examples["label"].to_numpy().astype(np.float32)

    return {
        "items": torch.from_numpy(items),
        "cats": torch.from_numpy(cats),
        "behs": torch.from_numpy(behs),
        "hours": torch.from_numpy(hours),
        "lengths": torch.from_numpy(lengths),
        "labels": torch.from_numpy(labels),
    }
