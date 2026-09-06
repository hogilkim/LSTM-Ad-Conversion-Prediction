"""Convert processed examples into fixed-size PyTorch tensors.

Usage (from the repo root, after ``uv run python -m taobao.data.prepare``)::

    uv run python -m taobao.data.tensors

Reads ``data/processed/examples.parquet``, right-pads each split once to a fixed
``max_len`` rectangle and writes six ``.npy`` arrays per split to
``data/processed/tensors/`` as ``{split}_{array}.npy``. Separate ``.npy`` files rather
than one ``.npz`` per split so that :func:`load_split` can memory-map them and arrays the
model does not use (``items`` in v1) cost nothing to load.
"""

from __future__ import annotations

import argparse
import resource
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl
import torch

SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
ARRAY_NAMES = ("items", "cats", "behs", "hours", "lengths", "labels")
DEFAULT_EXAMPLES = Path("data/processed/examples.parquet")
DEFAULT_OUT_DIR = Path("data/processed/tensors")


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


def build_split(
    split: str,
    examples_path: Path = DEFAULT_EXAMPLES,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_len: int = 50,
) -> dict[str, torch.Tensor]:
    """Pad one split's examples and write its six ``.npy`` arrays to ``out_dir``.

    The parquet filter is pushed down by :func:`polars.scan_parquet`, so only the rows for
    this split are ever materialised.
    """
    if split not in SPLIT_CODES:
        raise ValueError(f"unknown split {split!r}, expected one of {sorted(SPLIT_CODES)}")

    examples = (
        pl.scan_parquet(examples_path).filter(pl.col("split") == SPLIT_CODES[split]).collect()
    )
    tensors = examples_to_tensors(examples, max_len=max_len)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ARRAY_NAMES:
        # .numpy() shares storage with the tensor, so saving costs no extra memory.
        np.save(out_dir / f"{split}_{name}.npy", tensors[name].numpy())
    return tensors


def load_split(
    split: str,
    tensors_dir: Path = DEFAULT_OUT_DIR,
    array_names: Iterable[str] = ARRAY_NAMES,
) -> dict[str, np.ndarray]:
    """Memory-map selected arrays for one split without loading them all into RAM."""
    if split not in SPLIT_CODES:
        raise ValueError(f"unknown split {split!r}, expected one of {sorted(SPLIT_CODES)}")

    requested = tuple(array_names)
    unknown = sorted(set(requested) - set(ARRAY_NAMES))
    if unknown:
        raise ValueError(f"unknown arrays {unknown}, expected names from {list(ARRAY_NAMES)}")

    return {
        name: np.load(tensors_dir / f"{split}_{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in requested
    }


def _peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB (ru_maxrss is bytes on macOS)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_CODES), choices=list(SPLIT_CODES))
    args = parser.parse_args()

    t0 = time.perf_counter()
    for split in args.splits:
        split_t0 = time.perf_counter()
        tensors = build_split(split, args.examples, args.out_dir, max_len=args.max_len)
        written = sum((args.out_dir / f"{split}_{n}.npy").stat().st_size for n in ARRAY_NAMES)
        print(
            f"{split:5s} {tensors['labels'].numel():,} examples, "
            f"label mean {tensors['labels'].mean():.4f}, "
            f"mean seq_len {tensors['lengths'].float().mean():.1f}, "
            f"{written / 1e9:.2f} GB ({time.perf_counter() - split_t0:.1f}s)",
            flush=True,
        )
        del tensors

    print(
        f"wrote {len(args.splits) * len(ARRAY_NAMES)} arrays to {args.out_dir} "
        f"in {time.perf_counter() - t0:.1f}s, peak RSS {_peak_rss_gb():.2f} GB"
    )


if __name__ == "__main__":
    main()
