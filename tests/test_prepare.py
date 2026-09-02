"""Tests for the pure example-building logic in taobao.data.prepare.

Synthetic data: two users, ten events, one cutoff at 2017-11-28 00:00 UTC+8.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from taobao.data.prepare import (
    BEHAVIOR_INDEX,
    CST,
    PAD_INDEX,
    RARE_INDEX,
    SPLIT_TRAIN,
    build_examples,
    build_vocab,
    make_cutoff,
)

CUTOFF_DT = datetime(2017, 11, 28, tzinfo=CST)
CUTOFF = make_cutoff(CUTOFF_DT, SPLIT_TRAIN)
H = timedelta(hours=1)


def ts(delta: timedelta) -> int:
    """Unix seconds at CUTOFF + delta."""
    return int((CUTOFF_DT + delta).timestamp())


@pytest.fixture
def events() -> pl.DataFrame:
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
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.Int64,
            "item_id": pl.Int64,
            "category_id": pl.Int64,
            "behavior_type": pl.String,
            "timestamp": pl.Int64,
        },
        orient="row",
    )


@pytest.fixture
def examples(events: pl.DataFrame) -> pl.DataFrame:
    # Vocab from events before the cutoff; item 12 is seen twice before it and item 10
    # twice, so with min_count=2 item 11 (seen once) becomes rare.
    item_vocab = build_vocab(events, "item_id", CUTOFF.timestamp, min_count=2)
    cat_vocab = build_vocab(events, "category_id", CUTOFF.timestamp, min_count=1)
    return build_examples(events, [CUTOFF], item_vocab, cat_vocab, max_len=50, min_len=3)


def test_no_input_event_at_or_after_cutoff(examples: pl.DataFrame) -> None:
    # hours_before_cutoff = (T - timestamp) / 3600 must be strictly positive everywhere.
    all_hours = examples["hours"].explode()
    assert all_hours.min() > 0
    # user 1 has an event exactly at T; the input must contain only the 4 earlier events.
    u1 = examples.filter(pl.col("user_id") == 1).row(0, named=True)
    assert u1["seq_len"] == 4
    assert len(u1["items"]) == len(u1["cats"]) == len(u1["behs"]) == len(u1["hours"]) == 4
    # inputs are in time order, so hours-before-cutoff is non-increasing.
    assert u1["hours"] == sorted(u1["hours"], reverse=True)
    assert u1["hours"][-1] == pytest.approx(1.0)


def test_labels_follow_the_24h_buy_window(examples: pl.DataFrame) -> None:
    labels = dict(examples.select("user_id", "label").iter_rows())
    assert labels == {1: 1, 2: 0}


def test_pad_index_never_appears_and_rare_is_used(examples: pl.DataFrame) -> None:
    items = examples["items"].explode()
    cats = examples["cats"].explode()
    behs = examples["behs"].explode()
    assert PAD_INDEX not in items.to_list()
    assert PAD_INDEX not in cats.to_list()
    assert PAD_INDEX not in behs.to_list()
    # item 11 is seen once before the cutoff -> rare.
    u1 = examples.filter(pl.col("user_id") == 1).row(0, named=True)
    assert u1["items"][1] == RARE_INDEX
    assert u1["behs"] == [BEHAVIOR_INDEX["pv"], BEHAVIOR_INDEX["cart"], BEHAVIOR_INDEX["pv"], BEHAVIOR_INDEX["fav"]]


def test_hours_are_clipped_and_min_len_is_enforced(events: pl.DataFrame) -> None:
    item_vocab = build_vocab(events, "item_id", CUTOFF.timestamp, min_count=1)
    cat_vocab = build_vocab(events, "category_id", CUTOFF.timestamp, min_count=1)
    ex = build_examples(events, [CUTOFF], item_vocab, cat_vocab, max_len=50, min_len=3, max_hours=168.0)
    u2 = ex.filter(pl.col("user_id") == 2).row(0, named=True)
    assert u2["hours"][0] == pytest.approx(168.0)  # 200 h clipped
    # min_len=4 drops user 2 (3 events) but keeps user 1 (4 events).
    ex4 = build_examples(events, [CUTOFF], item_vocab, cat_vocab, max_len=50, min_len=4)
    assert ex4["user_id"].to_list() == [1]
    # max_len=2 keeps only the two most recent events.
    ex2 = build_examples(events, [CUTOFF], item_vocab, cat_vocab, max_len=2, min_len=1)
    u1 = ex2.filter(pl.col("user_id") == 1).row(0, named=True)
    assert u1["seq_len"] == 2
    assert u1["hours"] == pytest.approx([5.0, 1.0])


def test_vocab_ignores_events_at_or_after_the_vocab_cutoff(events: pl.DataFrame) -> None:
    # category 101 appears before the cutoff only via items 12 (fav at -1h); with the
    # events at/after the cutoff excluded it is seen once, so min_count=2 makes it rare.
    cat_vocab = build_vocab(events, "category_id", CUTOFF.timestamp, min_count=2)
    assert 101 not in cat_vocab
    assert 100 in cat_vocab and 200 in cat_vocab
    assert min(cat_vocab.values()) == 2  # 0 = pad, 1 = rare
