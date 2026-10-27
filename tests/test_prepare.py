"""Tests for the pure example-building logic in taobao.data.prepare.

Synthetic data: two users, ten events, one cutoff at 2017-11-28 00:00 UTC+8.
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taobao.data.prepare import (
    BEHAVIOR_INDEX,
    CST,
    EXAMPLE_DTYPES,
    PAD_INDEX,
    RAW_COLUMNS,
    RAW_DTYPES,
    RARE_INDEX,
    SPLIT_TRAIN,
    WINDOW_END,
    WINDOW_START,
    build_examples,
    build_vocab,
    compute_stats,
    encode_events,
    load_sampled_events,
    make_cutoff,
    write_outputs,
)

CUTOFF_DT = datetime(2017, 11, 28, tzinfo=CST)
CUTOFF = make_cutoff(CUTOFF_DT, SPLIT_TRAIN)
H = timedelta(hours=1)


def ts(delta: timedelta) -> int:
    """Unix seconds at CUTOFF + delta."""
    return int((CUTOFF_DT + delta).timestamp())


@pytest.fixture
def sampled_csv(tmp_path: Path) -> Path:
    start, end = int(WINDOW_START.timestamp()), int(WINDOW_END.timestamp())
    rows = [
        (3, 30, 300, "pv", start + 5),
        (1, 10, 100, "pv", start),  # start is included
        (99, 99, 900, "pv", start - 1),  # this user has no valid events
        (3, 31, 300, "cart", start + 5),  # equal timestamp: preserve file order
        (2, 20, 200, "buy", start + 2),
        (1, 11, 100, "cart", end - 1),
        (99, 99, 900, "buy", end),  # end is excluded
        (1, 12, 100, "buy", end),
        (3, 32, 300, "buy", start + 1),  # deliberately out of order
    ]
    path = tmp_path / "UserBehavior.csv"
    with path.open("w", newline="") as file:
        csv.writer(file).writerows(rows)
    return path


@pytest.mark.parametrize("chunk_size", [1, 3, 20])
def test_sampling_preserves_cohort_boundaries_and_event_order(sampled_csv: Path, chunk_size: int) -> None:
    events, counts = load_sampled_events(sampled_csv, n_users=2, seed=42, chunk_size=chunk_size)

    # Seed 42 chooses users 1 and 3 from the sorted eligible users [1, 2, 3].
    assert list(events[["user_id", "item_id"]].itertuples(index=False, name=None)) == [
        (1, 10), (1, 11), (3, 32), (3, 30), (3, 31),
    ]
    assert counts == {
        "raw_rows": 9,
        "kept_rows": 6,
        "dropped_out_of_window_rows": 3,
        "users_with_valid_events": 3,
        "users_sampled": 2,
        "sampled_events": 5,
    }


def test_sampling_rejects_more_users_than_have_valid_events(sampled_csv: Path) -> None:
    with pytest.raises(ValueError, match="only 3 have in-window events"):
        load_sampled_events(sampled_csv, n_users=4, seed=42, chunk_size=2)


def test_sampling_zero_users_returns_typed_empty_events(sampled_csv: Path) -> None:
    events, counts = load_sampled_events(sampled_csv, n_users=0, seed=42, chunk_size=2)

    assert len(events) == counts["sampled_events"] == 0
    assert events.dtypes.to_dict() == {
        name: pd.Series(dtype=dtype).dtype for name, dtype in RAW_DTYPES.items()
    }


@pytest.fixture
def events() -> pd.DataFrame:
    rows = [
        # user 1: 4 events before the cutoff, a buy 2 h after it (label 1), plus one
        # event exactly at the cutoff that must never enter the input.
        (1, 10, 100, "pv", ts(-30 * H)),
        (1, 11, 100, "cart", ts(-20 * H)),
        (1, 10, 100, "pv", ts(-5 * H)),
        (1, 12, 101, "fav", ts(-1 * H)),
        (1, 12, 101, "pv", ts(0 * H)),  # at cutoff: excluded from input, not a buy
        (1, 12, 101, "buy", ts(2 * H)),  # inside [T, T+24h) -> label 1
        # user 2: 3 events before the cutoff, only a buy 30 h after it (label 0).
        (2, 20, 200, "pv", ts(-200 * H)),  # hours clipped to 168
        (2, 21, 200, "pv", ts(-10 * H)),
        (2, 21, 200, "cart", ts(-2 * H)),
        (2, 21, 200, "buy", ts(30 * H)),  # outside the 24 h window -> label 0
    ]
    # Deliberately unsorted so the function must order events itself.
    rows = rows[::-1]
    return pd.DataFrame(rows, columns=RAW_COLUMNS).astype(RAW_DTYPES)


@pytest.fixture
def examples(events: pd.DataFrame) -> pd.DataFrame:
    # Vocab from events before the cutoff; item 12 is seen twice before it and item 10
    # twice, so with min_count=2 item 11 (seen once) becomes rare.
    item_vocab = build_vocab(events, "item_id", CUTOFF.timestamp, min_count=2)
    cat_vocab = build_vocab(events, "category_id", CUTOFF.timestamp, min_count=1)
    return build_examples(events, [CUTOFF], item_vocab, cat_vocab, max_len=50, min_len=3)


def test_no_input_event_at_or_after_cutoff(examples: pd.DataFrame) -> None:
    # hours_before_cutoff = (T - timestamp) / 3600 must be strictly positive everywhere.
    all_hours = examples["hours"].explode()
    assert all_hours.min() > 0
    # user 1 has an event exactly at T; the input must contain only the 4 earlier events.
    u1 = examples.loc[examples["user_id"] == 1].iloc[0]
    assert u1["seq_len"] == 4
    assert len(u1["items"]) == len(u1["cats"]) == len(u1["behs"]) == len(u1["hours"]) == 4
    # inputs are in time order, so hours-before-cutoff is non-increasing.
    assert u1["hours"] == sorted(u1["hours"], reverse=True)
    assert u1["hours"][-1] == pytest.approx(1.0)


def test_labels_follow_the_24h_buy_window(examples: pd.DataFrame) -> None:
    labels = dict(zip(examples["user_id"], examples["label"]))
    assert labels == {1: 1, 2: 0}


def test_pad_index_never_appears_and_rare_is_used(examples: pd.DataFrame) -> None:
    items = examples["items"].explode()
    cats = examples["cats"].explode()
    behs = examples["behs"].explode()
    assert PAD_INDEX not in items.to_list()
    assert PAD_INDEX not in cats.to_list()
    assert PAD_INDEX not in behs.to_list()
    # item 11 is seen once before the cutoff -> rare.
    u1 = examples.loc[examples["user_id"] == 1].iloc[0]
    assert u1["items"][1] == RARE_INDEX
    assert u1["behs"] == [BEHAVIOR_INDEX["pv"], BEHAVIOR_INDEX["cart"], BEHAVIOR_INDEX["pv"], BEHAVIOR_INDEX["fav"]]


def test_hours_are_clipped_and_min_len_is_enforced(events: pd.DataFrame) -> None:
    item_vocab = build_vocab(events, "item_id", CUTOFF.timestamp, min_count=1)
    cat_vocab = build_vocab(events, "category_id", CUTOFF.timestamp, min_count=1)
    ex = build_examples(events, [CUTOFF], item_vocab, cat_vocab, max_len=50, min_len=3, max_hours=168.0)
    u2 = ex.loc[ex["user_id"] == 2].iloc[0]
    assert u2["hours"][0] == pytest.approx(168.0)  # 200 h clipped
    # min_len=4 drops user 2 (3 events) but keeps user 1 (4 events).
    ex4 = build_examples(events, [CUTOFF], item_vocab, cat_vocab, max_len=50, min_len=4)
    assert ex4["user_id"].to_list() == [1]
    # max_len=2 keeps only the two most recent events.
    ex2 = build_examples(events, [CUTOFF], item_vocab, cat_vocab, max_len=2, min_len=1)
    u1 = ex2.loc[ex2["user_id"] == 1].iloc[0]
    assert u1["seq_len"] == 2
    assert u1["hours"] == pytest.approx([5.0, 1.0])


def test_vocab_ignores_events_at_or_after_the_vocab_cutoff(events: pd.DataFrame) -> None:
    # category 101 appears before the cutoff only via items 12 (fav at -1h); with the
    # events at/after the cutoff excluded it is seen once, so min_count=2 makes it rare.
    cat_vocab = build_vocab(events, "category_id", CUTOFF.timestamp, min_count=2)
    assert 101 not in cat_vocab
    assert 100 in cat_vocab and 200 in cat_vocab
    assert min(cat_vocab.values()) == 2  # 0 = pad, 1 = rare


@pytest.mark.parametrize(("buy_hours", "expected_label"), [(0, 1), (24, 0)])
def test_label_window_includes_start_and_excludes_end(
    events: pd.DataFrame, buy_hours: int, expected_label: int,
) -> None:
    history = events.loc[events["behavior_type"] != "buy"]
    purchase = pd.DataFrame(
        [(1, 12, 101, "buy", ts(buy_hours * H))], columns=RAW_COLUMNS,
    ).astype(RAW_DTYPES)
    data = pd.concat([history, purchase], ignore_index=True)

    examples = build_examples(data, [CUTOFF], {}, {})

    assert examples.loc[examples["user_id"] == 1, "label"].iloc[0] == expected_label


def test_vocab_assigns_indices_in_id_order_and_handles_empty_vocab(events: pd.DataFrame) -> None:
    assert build_vocab(events, "item_id", CUTOFF.timestamp, min_count=2) == {10: 2, 21: 3}
    assert build_vocab(events, "item_id", CUTOFF.timestamp, min_count=100) == {}
    encoded = encode_events(events, {}, {})
    assert (encoded["item_idx"] == RARE_INDEX).all()
    assert (encoded["cat_idx"] == RARE_INDEX).all()


def test_encoding_rejects_unknown_behavior_without_mutating_input(events: pd.DataFrame) -> None:
    original = events.copy(deep=True)
    encode_events(events, {}, {})
    pd.testing.assert_frame_equal(events, original)

    events.loc[0, "behavior_type"] = "unknown"
    with pytest.raises(ValueError, match="behavior_type"):
        encode_events(events, {}, {})


def test_equal_timestamps_preserve_input_order(events: pd.DataFrame) -> None:
    tied = events.copy()
    tied.loc[tied["user_id"] == 1, "timestamp"] = ts(-H)
    expected = tied.loc[tied["user_id"] == 1, "item_id"].tolist()
    vocab = {item: item + 2 for item in expected}

    examples = build_examples(tied, [CUTOFF], vocab, {}, min_len=1)

    assert examples.loc[examples["user_id"] == 1, "items"].iloc[0] == [item + 2 for item in expected]


@pytest.mark.parametrize("case", ["no_events", "no_cutoffs", "no_prior_events", "insufficient_history"])
def test_no_usable_examples_returns_typed_empty_frame(events: pd.DataFrame, case: str) -> None:
    data = events.iloc[:0] if case == "no_events" else events
    cutoffs = [] if case == "no_cutoffs" else [CUTOFF]
    if case == "no_prior_events":
        cutoffs = [make_cutoff(CUTOFF_DT - timedelta(days=30), SPLIT_TRAIN)]
    examples = build_examples(data, cutoffs, {}, {}, min_len=100)

    assert examples.empty
    assert examples.dtypes.to_dict() == {
        name: pd.Series(dtype=dtype).dtype for name, dtype in EXAMPLE_DTYPES.items()
    }
    assert compute_stats(examples) == {"splits": {}, "cutoffs": {}, "n_examples": 0}


def test_stats_and_parquet_preserve_types_and_sequences(examples: pd.DataFrame, tmp_path: Path) -> None:
    stats = compute_stats(examples)
    assert stats == {
        "splits": {"train": {
            "split": 0, "n_examples": 2, "n_positive": 1, "positive_rate": 0.5,
            "mean_seq_len": 3.5, "n_users": 2,
        }},
        "cutoffs": {"20171128": {"split": 0, "n_examples": 2, "positive_rate": 0.5}},
        "n_examples": 2,
    }
    stats["vocab_sizes"] = {"items": 4, "cats": 4, "behaviors": 5}
    write_outputs(tmp_path, examples, {10: 2}, {100: 2}, stats)

    table = pq.read_table(tmp_path / "examples.parquet")
    assert table.column_names == list(EXAMPLE_DTYPES)
    assert table.schema.field("user_id").type == pa.int64()
    assert table.schema.field("split").type == table.schema.field("label").type == pa.int8()
    assert table.schema.field("cutoff").type == table.schema.field("seq_len").type == pa.int32()
    for name, element_type in [("items", pa.int32()), ("cats", pa.int32()), ("behs", pa.int8()), ("hours", pa.float32())]:
        assert table.schema.field(name).type == pa.large_list(pa.field("element", element_type))
    assert table.to_pylist() == examples.to_dict(orient="records")
    assert json.loads((tmp_path / "stats.json").read_text()) == stats
    assert json.loads((tmp_path / "vocab_items.json").read_text()) == {"10": 2}
