"""Build (user, cutoff) conversion-prediction examples from the raw Taobao UserBehavior CSV.

Usage (from the repo root, after ``bash scripts/download_data.sh``)::

    uv run python -m taobao.data.prepare --n-users 200000 --seed 42

Pipeline
--------
1. Scan ``data/raw/UserBehavior.csv`` (no header; ``user_id, item_id, category_id,
   behavior_type, timestamp``) and keep only rows whose timestamp falls inside
   [2017-11-25 00:00, 2017-12-04 00:00) China time (UTC+8). The raw file carries junk
   out-of-range timestamps; they are counted and dropped.
2. Sample ``n_users`` users with a fixed seed and keep all of their events.
3. Build vocabularies for ``item_id`` and ``category_id`` from events strictly before the
   validation cutoff only (index 0 = padding, 1 = rare, i.e. fewer than ``min_count``
   occurrences). ``behavior_type`` maps to ``pv=1, cart=2, fav=3, buy=4``.
4. For each user and each daily cutoff T (00:00 UTC+8) build one example: the user's
   last ``max_len`` events strictly before T, in time order, labeled 1 if the user has any
   ``buy`` in [T, T + 24h). Pairs with fewer than ``min_len`` input events are skipped.
5. Write ``examples.parquet``, the vocabulary JSON files and ``stats.json`` to
   ``data/processed/``.

The core logic (:func:`build_vocab`, :func:`build_examples`) is pure and operates on
in-memory pandas frames so it can be unit-tested on synthetic data. Arrow-backed list
columns keep the completed sequences compact and preserve their Parquet element types.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CST = timezone(timedelta(hours=8))
"""China Standard Time; every date in the dataset and every cutoff is interpreted here."""

RAW_COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
RAW_DTYPES = {
    "user_id": "int64",
    "item_id": "int64",
    "category_id": "int64",
    "behavior_type": "string[pyarrow]",
    "timestamp": "int64",
}
DEFAULT_CHUNK_SIZE = 1_000_000

PAD_INDEX = 0
RARE_INDEX = 1
FIRST_VOCAB_INDEX = 2
BEHAVIOR_INDEX = {"pv": 1, "cart": 2, "fav": 3, "buy": 4}
BEHAVIOR_VOCAB_SIZE = len(BEHAVIOR_INDEX) + 1  # + padding

WINDOW_START = datetime(2017, 11, 25, tzinfo=CST)
WINDOW_END = datetime(2017, 12, 4, tzinfo=CST)

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_TEST: "test"}

SECONDS_PER_HOUR = 3600
LABEL_WINDOW_SECONDS = 24 * SECONDS_PER_HOUR

EXAMPLE_DTYPES = {
    "user_id": "int64",
    "cutoff": "int32",
    "split": "int8",
    "items": pd.ArrowDtype(pa.large_list(pa.field("element", pa.int32()))),
    "cats": pd.ArrowDtype(pa.large_list(pa.field("element", pa.int32()))),
    "behs": pd.ArrowDtype(pa.large_list(pa.field("element", pa.int8()))),
    "hours": pd.ArrowDtype(pa.large_list(pa.field("element", pa.float32()))),
    "label": "int8",
    "seq_len": "int32",
}


class Cutoff(NamedTuple):
    """One prediction time. ``timestamp`` is Unix seconds, ``date`` is e.g. 20171128."""

    timestamp: int
    date: int
    split: int


def make_cutoff(dt: datetime, split: int) -> Cutoff:
    if dt.tzinfo is None:
        raise ValueError("cutoff datetimes must be timezone-aware")
    return Cutoff(int(dt.timestamp()), dt.year * 10000 + dt.month * 100 + dt.day, split)


def default_cutoffs() -> list[Cutoff]:
    """Train: Nov 28 - Dec 1; validation: Dec 2; test: Dec 3 (all 00:00 UTC+8)."""
    train = [datetime(2017, 11, d, tzinfo=CST) for d in (28, 29, 30)]
    train.append(datetime(2017, 12, 1, tzinfo=CST))
    return (
        [make_cutoff(dt, SPLIT_TRAIN) for dt in train]
        + [make_cutoff(datetime(2017, 12, 2, tzinfo=CST), SPLIT_VAL)]
        + [make_cutoff(datetime(2017, 12, 3, tzinfo=CST), SPLIT_TEST)]
    )


def validation_cutoff(cutoffs: list[Cutoff]) -> Cutoff:
    """The earliest non-train cutoff; vocabularies may only use events before it."""
    later = [c for c in cutoffs if c.split != SPLIT_TRAIN]
    return min(later, key=lambda c: c.timestamp) if later else max(cutoffs, key=lambda c: c.timestamp)


# --------------------------------------------------------------------------------------
# Loading and sampling
# --------------------------------------------------------------------------------------


def scan_raw(
    path: Path,
    *,
    columns: list[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Read bounded CSV chunks, optionally parsing only the requested columns."""
    with pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        usecols=columns,
        dtype=RAW_DTYPES,
        chunksize=chunk_size,
        engine="c",
    ) as reader:
        yield from reader


def in_window_mask(
    timestamps: pd.Series,
    start: datetime = WINDOW_START,
    end: datetime = WINDOW_END,
) -> pd.Series:
    """Select timestamps in the half-open interval [start, end)."""
    lo, hi = int(start.timestamp()), int(end.timestamp())
    return (timestamps >= lo) & (timestamps < hi)


def load_sampled_events(
    path: Path,
    n_users: int,
    seed: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[pd.DataFrame, dict]:
    """Two chunked pandas passes over the CSV.

    Pass 1 counts total/valid rows and collects users with valid events. Pass 2 retains
    every in-window event of the sampled users. The retained events still live in memory.
    """
    total_rows = kept_rows = 0
    eligible_users: set[int] = set()
    for chunk in scan_raw(path, columns=["user_id", "timestamp"], chunk_size=chunk_size):
        valid = in_window_mask(chunk["timestamp"])
        total_rows += len(chunk)
        kept_rows += int(valid.sum())
        eligible_users.update(chunk.loc[valid, "user_id"].unique())

    # Sort before sampling so file order and chunk boundaries cannot change the cohort.
    users = np.array(sorted(eligible_users), dtype=np.int64)
    if n_users > len(users):
        raise ValueError(f"asked for {n_users} users but only {len(users)} have in-window events")

    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(users, size=n_users, replace=False))

    frames: list[pd.DataFrame] = []
    for chunk in scan_raw(path, chunk_size=chunk_size):
        keep = in_window_mask(chunk["timestamp"]) & chunk["user_id"].isin(sampled)
        if keep.any():
            frames.append(chunk.loc[keep])
    if frames:
        events = pd.concat(frames, ignore_index=True)
    else:
        events = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in RAW_DTYPES.items()})
    del frames
    # Preserve file order for events with the same user and timestamp.
    events = events.sort_values(["user_id", "timestamp"], kind="stable", ignore_index=True)
    counts = {
        "raw_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_out_of_window_rows": total_rows - kept_rows,
        "users_with_valid_events": int(len(users)),
        "users_sampled": int(n_users),
        "sampled_events": int(len(events)),
    }
    return events, counts


# --------------------------------------------------------------------------------------
# Pure core: vocabularies and examples
# --------------------------------------------------------------------------------------


def build_vocab(events: pd.DataFrame, column: str, before_timestamp: int, min_count: int) -> dict[int, int]:
    """Map ids seen at least ``min_count`` times strictly before ``before_timestamp`` to
    contiguous indices starting at :data:`FIRST_VOCAB_INDEX`. Everything else is rare."""
    counts = events.loc[events["timestamp"] < before_timestamp, column].value_counts(sort=False)
    ids = counts.loc[counts >= min_count].sort_index().index
    return {int(i): FIRST_VOCAB_INDEX + k for k, i in enumerate(ids)}


def encode_events(events: pd.DataFrame, item_vocab: dict[int, int], cat_vocab: dict[int, int]) -> pd.DataFrame:
    """Add ``item_idx``, ``cat_idx`` and ``beh_idx`` columns and sort by user and time."""
    encoded = events.copy()
    encoded["item_idx"] = events["item_id"].map(item_vocab).fillna(RARE_INDEX).astype("int32")
    encoded["cat_idx"] = events["category_id"].map(cat_vocab).fillna(RARE_INDEX).astype("int32")
    behaviors = events["behavior_type"].map(BEHAVIOR_INDEX)
    if behaviors.isna().any():
        raise ValueError("behavior_type must be one of pv, cart, fav, buy")
    encoded["beh_idx"] = behaviors.astype("int8")
    return encoded.sort_values(["user_id", "timestamp"], kind="stable", ignore_index=True)


def build_examples(
    events: pd.DataFrame,
    cutoffs: list[Cutoff],
    item_vocab: dict[int, int],
    cat_vocab: dict[int, int],
    max_len: int = 50,
    min_len: int = 3,
    max_hours: float = 168.0,
) -> pd.DataFrame:
    """Build one example per (user, cutoff) from raw events. Pure function.

    ``events`` needs the columns ``user_id, item_id, category_id, behavior_type,
    timestamp`` (Unix seconds); it does not need to be sorted. For each cutoff T the input
    is the user's last ``max_len`` events with ``timestamp < T`` in time order, and the
    label is 1 if the user has any ``buy`` with ``T <= timestamp < T + 24h``. Users with
    fewer than ``min_len`` events before T are skipped.
    """
    encoded = encode_events(events, item_vocab, cat_vocab)
    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        t = cutoff.timestamp
        history = encoded.loc[encoded["timestamp"] < t]  # never at or after T
        buyers = encoded.loc[
            (encoded["timestamp"] >= t)
            & (encoded["timestamp"] < t + LABEL_WINDOW_SECONDS)
            & (encoded["behavior_type"] == "buy"),
            "user_id",
        ].unique()

        # Rows are sorted by user/time. Keep only the latest events before making lists.
        history = history.groupby("user_id", sort=False).tail(max_len).copy()
        if history.empty:
            continue
        hours = (t - history["timestamp"].to_numpy()) / SECONDS_PER_HOUR
        history["hours"] = np.minimum(hours, max_hours).astype(np.float32)
        examples = history.groupby("user_id", sort=False).agg(
            items=("item_idx", list),
            cats=("cat_idx", list),
            behs=("beh_idx", list),
            hours=("hours", list),
        ).reset_index()
        examples["seq_len"] = examples["items"].map(len)
        examples = examples.loc[examples["seq_len"] >= min_len].copy()
        examples["cutoff"] = cutoff.date
        examples["split"] = cutoff.split
        examples["label"] = examples["user_id"].isin(buyers)
        # Compact each cutoff's lists before retaining it for the final concatenation.
        frames.append(examples.loc[:, list(EXAMPLE_DTYPES)].astype(EXAMPLE_DTYPES))
    if not frames:
        return _empty_examples()
    return pd.concat(frames, ignore_index=True)


def _empty_examples() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in EXAMPLE_DTYPES.items()})


# --------------------------------------------------------------------------------------
# Stats and I/O
# --------------------------------------------------------------------------------------


def compute_stats(examples: pd.DataFrame) -> dict:
    per_split = examples.groupby("split", sort=True).agg(
        n_examples=("label", "size"),
        n_positive=("label", "sum"),
        positive_rate=("label", "mean"),
        mean_seq_len=("seq_len", "mean"),
        n_users=("user_id", "nunique"),
    )
    per_cutoff = examples.groupby("cutoff", sort=True).agg(
        split=("split", "first"),
        n_examples=("label", "size"),
        positive_rate=("label", "mean"),
    )
    return {
        "splits": {
            SPLIT_NAMES[int(split)]: {
                "split": int(split),
                "n_examples": int(r["n_examples"]),
                "n_positive": int(r["n_positive"]),
                "positive_rate": float(r["positive_rate"]),
                "mean_seq_len": float(r["mean_seq_len"]),
                "n_users": int(r["n_users"]),
            }
            for split, r in per_split.iterrows()
        },
        "cutoffs": {
            str(cutoff): {
                "split": int(r["split"]),
                "n_examples": int(r["n_examples"]),
                "positive_rate": float(r["positive_rate"]),
            }
            for cutoff, r in per_cutoff.iterrows()
        },
        "n_examples": int(len(examples)),
    }


def write_outputs(
    out_dir: Path,
    examples: pd.DataFrame,
    item_vocab: dict[int, int],
    cat_vocab: dict[int, int],
    stats: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples.to_parquet(out_dir / "examples.parquet", engine="pyarrow", index=False, compression="zstd")
    (out_dir / "vocab_items.json").write_text(json.dumps(item_vocab))
    (out_dir / "vocab_cats.json").write_text(json.dumps(cat_vocab))
    (out_dir / "vocab_sizes.json").write_text(json.dumps(stats["vocab_sizes"], indent=2))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))


def print_stats(stats: dict) -> None:
    print("\n=== data/processed stats ===")
    print(f"raw rows            : {stats['raw_rows']:,}")
    print(f"dropped (junk ts)   : {stats['dropped_out_of_window_rows']:,}")
    print(f"users with events   : {stats['users_with_valid_events']:,}")
    print(f"users sampled       : {stats['users_sampled']:,}  (seed={stats['seed']})")
    print(f"sampled events      : {stats['sampled_events']:,}")
    print(f"vocab sizes         : {stats['vocab_sizes']}")
    print(f"{'split':6} {'examples':>10} {'positives':>10} {'pos_rate':>9} {'mean_len':>9} {'users':>9}")
    for name, s in stats["splits"].items():
        print(
            f"{name:6} {s['n_examples']:>10,} {s['n_positive']:>10,} {s['positive_rate']:>9.4f} "
            f"{s['mean_seq_len']:>9.2f} {s['n_users']:>9,}"
        )
    print(f"{'cutoff':9} {'split':>5} {'examples':>10} {'pos_rate':>9}")
    for date, c in stats["cutoffs"].items():
        print(f"{date:9} {c['split']:>5} {c['n_examples']:>10,} {c['positive_rate']:>9.4f}")
    print(f"total examples      : {stats['n_examples']:,}")
    print(f"wall time           : {stats['wall_time_seconds']:.1f}s")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", type=Path, default=Path("data/raw/UserBehavior.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--n-users", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=50, help="events per example")
    p.add_argument("--min-len", type=int, default=3, help="skip examples with fewer events")
    p.add_argument("--min-count", type=int, default=5, help="ids seen fewer times are rare")
    p.add_argument("--max-hours", type=float, default=168.0, help="clip hours_before_cutoff")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.perf_counter()
    cutoffs = default_cutoffs()
    val_cutoff = validation_cutoff(cutoffs)

    print(f"scanning {args.raw} ...", flush=True)
    events, counts = load_sampled_events(args.raw, args.n_users, args.seed)
    print(
        f"  raw rows {counts['raw_rows']:,}, dropped {counts['dropped_out_of_window_rows']:,} "
        f"out-of-window, sampled {counts['users_sampled']:,} users -> {counts['sampled_events']:,} events "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    print(f"building vocabularies from events before {val_cutoff.date} ...", flush=True)
    item_vocab = build_vocab(events, "item_id", val_cutoff.timestamp, args.min_count)
    cat_vocab = build_vocab(events, "category_id", val_cutoff.timestamp, args.min_count)
    vocab_sizes = {
        "items": len(item_vocab) + FIRST_VOCAB_INDEX,
        "cats": len(cat_vocab) + FIRST_VOCAB_INDEX,
        "behaviors": BEHAVIOR_VOCAB_SIZE,
        "pad_index": PAD_INDEX,
        "rare_index": RARE_INDEX,
        "vocab_cutoff": val_cutoff.date,
        "min_count": args.min_count,
    }
    print(f"  {vocab_sizes} ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("building examples ...", flush=True)
    examples = build_examples(
        events, cutoffs, item_vocab, cat_vocab,
        max_len=args.max_len, min_len=args.min_len, max_hours=args.max_hours,
    )
    print(f"  {len(examples):,} examples ({time.perf_counter() - t0:.1f}s)", flush=True)

    stats = {
        **counts,
        "seed": args.seed,
        "max_len": args.max_len,
        "min_len": args.min_len,
        "max_hours": args.max_hours,
        "window": {"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat()},
        "cutoffs_used": [c._asdict() for c in cutoffs],
        "vocab_sizes": vocab_sizes,
        **compute_stats(examples),
        "command": "uv run python -m taobao.data.prepare " + " ".join(sys.argv[1:]),
    }
    stats["wall_time_seconds"] = time.perf_counter() - t0
    write_outputs(args.out_dir, examples, item_vocab, cat_vocab, stats)

    # Arrow verifies the saved numeric/list types without creating another pandas frame.
    check = pq.read_table(args.out_dir / "examples.parquet")
    expected_schema = pa.Schema.from_pandas(examples, preserve_index=False)
    assert check.num_rows == len(examples), (check.num_rows, len(examples))
    assert check.schema.equals(expected_schema), (check.schema, expected_schema)
    print(f"read back {check.num_rows:,} rows from {args.out_dir / 'examples.parquet'}")
    print(check.schema)
    print_stats(stats)


if __name__ == "__main__":
    main()
