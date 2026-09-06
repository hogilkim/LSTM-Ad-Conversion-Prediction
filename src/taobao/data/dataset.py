"""PyTorch dataset backed by memory-mapped split arrays."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from taobao.data.tensors import DEFAULT_OUT_DIR, load_split

MODEL_ARRAY_NAMES = ("cats", "behs", "hours", "lengths", "labels")


class TensorSplitDataset(Dataset[dict[str, torch.Tensor]]):
    """Read one saved split on demand, leaving the full arrays on disk."""

    def __init__(
        self,
        split: str,
        tensors_dir: Path = DEFAULT_OUT_DIR,
        include_items: bool = False,
    ) -> None:
        self.array_names = (("items",) if include_items else ()) + MODEL_ARRAY_NAMES
        self.arrays = load_split(split, tensors_dir, self.array_names)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n_examples = len(self.arrays["labels"])
        if self.arrays["labels"].ndim != 1 or self.arrays["lengths"].shape != (n_examples,):
            raise ValueError("lengths and labels must both have shape (number_of_examples,)")

        sequence_shape = self.arrays["cats"].shape
        if len(sequence_shape) != 2 or any(
            self.arrays[name].shape != sequence_shape
            for name in self.array_names
            if name not in {"lengths", "labels"}
        ):
            raise ValueError("all sequence arrays must have the same 2D shape")
        if sequence_shape[0] != n_examples:
            raise ValueError("sequence arrays, lengths and labels must contain the same examples")

    def __len__(self) -> int:
        return len(self.arrays["labels"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Copy only this example out of the read-only memory map. DataLoader will stack
        # these small tensors into a batch without materialising the full split.
        return {
            name: torch.from_numpy(np.array(self.arrays[name][index], copy=True))
            for name in self.array_names
        }
