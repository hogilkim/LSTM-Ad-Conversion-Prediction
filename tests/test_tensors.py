"""Tests for converting variable-length examples into PyTorch tensors."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from taobao.data.tensors import ARRAY_NAMES, build_split, examples_to_tensors, load_split


SEQUENCE_TYPES = {
    "items": pa.int32(),
    "cats": pa.int32(),
    "behs": pa.int8(),
    "hours": pa.float32(),
}
PARQUET_LAYOUTS = ("legacy-list", "legacy-large-list", "pandas-arrow")


def _schema(large_lists: bool = True) -> pa.Schema:
    list_type = pa.large_list if large_lists else pa.list_
    return pa.schema({
        "split": pa.int8(),
        **{name: list_type(pa.field("element", dtype)) for name, dtype in SEQUENCE_TYPES.items()},
        "seq_len": pa.int32(),
        "label": pa.int8(),
    })


def _arrow_backed(examples: pd.DataFrame) -> pd.DataFrame:
    schema = _schema()
    return examples.astype({
        name: pd.ArrowDtype(schema.field(name).type) for name in examples.columns
    })


def _write_examples(examples: pd.DataFrame, path: Path, layout: str) -> None:
    if layout == "pandas-arrow":
        _arrow_backed(examples).to_parquet(path, engine="pyarrow", index=False)
    else:
        # Legacy files have typed lists and no pandas metadata.
        table = pa.Table.from_pydict(
            examples.to_dict(orient="list"), schema=_schema(large_lists=layout == "legacy-large-list"),
        )
        pq.write_table(table, path, row_group_size=2)


@pytest.mark.parametrize("storage", ["python", "arrow"])
def test_examples_to_tensors_right_pads_and_normalizes_hours(storage: str) -> None:
    examples = pd.DataFrame(
        {
            "items": [[10, 20, 30], [40, 50]],
            "cats": [[1, 2, 3], [4, 5]],
            "behs": [[1, 2, 4], [1, 3]],
            "hours": [[168.0, 24.0, 1.0], [12.0, 2.0]],
            "seq_len": [3, 2],
            "label": [1, 0],
        }
    )
    if storage == "arrow":
        examples = _arrow_backed(examples)

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


@pytest.mark.parametrize("layout", PARQUET_LAYOUTS)
@pytest.mark.parametrize("split", ["train", "val"])
def test_build_and_memory_map_one_split(tmp_path: Path, layout: str, split: str) -> None:
    examples = pd.DataFrame(
        {
            "split": [0, 1, 0, 2],
            "items": [[10, 20], [99], [30], [777]],
            "cats": [[1, 2], [9], [3], [77]],
            "behs": [[1, 2], [4], [3], [4]],
            "hours": [[24.0, 1.0], [2.0], [12.0], [168.0]],
            "seq_len": [2, 1, 1, 1],
            "label": [1, 1, 0, 1],
        }
    )
    examples_path = tmp_path / "examples.parquet"
    out_dir = tmp_path / "tensors"
    _write_examples(examples, examples_path, layout)

    tensors = build_split(split, examples_path, out_dir, max_len=3)

    expected_items = [[10, 20, 0], [30, 0, 0]] if split == "train" else [[99, 0, 0]]
    expected_cats = [[1, 2, 0], [3, 0, 0]] if split == "train" else [[9, 0, 0]]
    expected_labels = [1.0, 0.0] if split == "train" else [1.0]
    assert tensors["items"].tolist() == expected_items
    assert tensors["labels"].tolist() == expected_labels
    assert {path.name for path in out_dir.iterdir()} == {
        f"{split}_{name}.npy" for name in ARRAY_NAMES
    }

    # Every saved array retains its values, shape and dtype when memory-mapped.
    for name, array in load_split(split, out_dir).items():
        expected = tensors[name].numpy()
        assert isinstance(array, np.memmap)
        assert not array.flags.writeable
        assert array.dtype == expected.dtype
        np.testing.assert_array_equal(array, expected)

    arrays = load_split(split, out_dir, array_names=("cats", "labels"))
    assert set(arrays) == {"cats", "labels"}
    assert arrays["cats"].tolist() == expected_cats
    assert arrays["labels"].tolist() == expected_labels


def test_build_split_passes_filter_and_projection_to_parquet_reader(tmp_path: Path, monkeypatch) -> None:
    examples_path = tmp_path / "examples.parquet"
    _write_examples(pd.DataFrame({
        "split": [0, 1, 2],
        "items": [[10], [20], [30]],
        "cats": [[2], [3], [4]],
        "behs": [[1], [2], [3]],
        "hours": [[1.0], [2.0], [3.0]],
        "seq_len": [1, 1, 1],
        "label": [0, 1, 0],
    }), examples_path, "pandas-arrow")
    read_parquet = pd.read_parquet
    reads = []

    def capture_read(path, **kwargs):
        frame = read_parquet(path, **kwargs)
        reads.append((path, kwargs, frame))
        return frame

    monkeypatch.setattr(pd, "read_parquet", capture_read)
    build_split("val", examples_path, tmp_path / "tensors", max_len=3)

    assert len(reads) == 1
    path, kwargs, frame = reads[0]
    assert path == examples_path
    assert kwargs["engine"] == "pyarrow"
    assert kwargs["filters"] == [("split", "==", 1)]
    assert kwargs["columns"] == ["items", "cats", "behs", "hours", "seq_len", "label"]
    assert kwargs["dtype_backend"] == "pyarrow"
    assert frame["items"].tolist() == [[20]]
    assert list(frame.columns) == kwargs["columns"]
    for name, element_type in SEQUENCE_TYPES.items():
        assert isinstance(frame[name].dtype, pd.ArrowDtype)
        assert frame[name].dtype.pyarrow_dtype.value_type == element_type


@pytest.mark.parametrize("layout", PARQUET_LAYOUTS)
def test_missing_split_writes_typed_empty_arrays(tmp_path: Path, layout: str) -> None:
    examples_path = tmp_path / "examples.parquet"
    _write_examples(pd.DataFrame({
        "split": [0],
        "items": [[10]],
        "cats": [[2]],
        "behs": [[1]],
        "hours": [[1.0]],
        "seq_len": [1],
        "label": [0],
    }), examples_path, layout)

    tensors = build_split("val", examples_path, tmp_path / "tensors", max_len=3)

    for name in ARRAY_NAMES:
        expected_shape = (0,) if name in ("lengths", "labels") else (0, 3)
        expected_dtype = torch.float32 if name in ("hours", "labels") else torch.long
        assert tensors[name].shape == expected_shape
        assert tensors[name].dtype == expected_dtype


def test_load_split_only_opens_requested_arrays(tmp_path: Path) -> None:
    np.save(tmp_path / "train_labels.npy", np.array([1.0, 0.0], dtype=np.float32))

    arrays = load_split("train", tmp_path, array_names=("labels",))

    assert set(arrays) == {"labels"}
    assert isinstance(arrays["labels"], np.memmap)
    assert arrays["labels"].tolist() == [1.0, 0.0]


def test_build_split_rejects_unknown_split(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown split"):
        build_split("dev", tmp_path / "examples.parquet", tmp_path)
