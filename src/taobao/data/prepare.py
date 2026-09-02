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
in-memory Polars frames so it can be unit-tested on synthetic data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

CST = timezone(timedelta(hours=8))
"""China Standard Time; every date in the dataset and every cutoff is interpreted here."""

RAW_COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
RAW_SCHEMA = {
    "user_id": pl.Int64,
    "item_id": pl.Int64,
    "category_id": pl.Int64,
    "behavior_type": pl.String,
    "timestamp": pl.Int64,
}

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


def scan_raw(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, has_header=False, schema=RAW_SCHEMA)


def in_window_expr(start: datetime = WINDOW_START, end: datetime = WINDOW_END) -> pl.Expr:
    lo, hi = int(start.timestamp()), int(end.timestamp())
    return (pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi)


def load_sampled_events(path: Path, n_users: int, seed: int) -> tuple[pl.DataFrame, dict]:
    """Two streaming passes over the CSV.

    Pass 1 counts rows per user (total and inside the time window) so we know the junk
    row count and the set of users with valid events. Pass 2 collects every in-window
    event of the ``n_users`` sampled users. Returns the events and a dict of counts.
    """
    lf = scan_raw(path)
    in_window = in_window_expr()

    per_user = (
        lf.group_by("user_id")
        .agg(pl.len().alias("n_rows"), in_window.sum().alias("n_in_window"))
        .collect(engine="streaming")
    )
    total_rows = int(per_user["n_rows"].sum())
    kept_rows = int(per_user["n_in_window"].sum())
    users = per_user.filter(pl.col("n_in_window") > 0)["user_id"].sort().to_numpy()
    if n_users > len(users):
        raise ValueError(f"asked for {n_users} users but only {len(users)} have in-window events")

    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(users, size=n_users, replace=False))

    sampled_users = pl.LazyFrame({"user_id": sampled}, schema={"user_id": pl.Int64})
    events = (
        lf.filter(in_window)
        .join(sampled_users, on="user_id", how="semi")
        .collect(engine="streaming")
        .sort(["user_id", "timestamp"], maintain_order=True)
    )
    counts = {
        "raw_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_out_of_window_rows": total_rows - kept_rows,
        "users_with_valid_events": int(len(users)),
        "users_sampled": int(n_users),
        "sampled_events": int(events.height),
    }
    return events, counts


# --------------------------------------------------------------------------------------
# Pure core: vocabularies and examples
# --------------------------------------------------------------------------------------


def build_vocab(events: pl.DataFrame, column: str, before_timestamp: int, min_count: int) -> dict[int, int]:
    """Map ids seen at least ``min_count`` times strictly before ``before_timestamp`` to
    contiguous indices starting at :data:`FIRST_VOCAB_INDEX`. Everything else is rare."""
    counts = (
        events.filter(pl.col("timestamp") < before_timestamp)
        .group_by(column)
        .len()
        .filter(pl.col("len") >= min_count)
        .sort(column)
    )
    ids = counts[column].to_list()
    return {int(i): FIRST_VOCAB_INDEX + k for k, i in enumerate(ids)}


def _index_column(events: pl.DataFrame, column: str, vocab: dict[int, int], out: str) -> pl.DataFrame:
    """Left-join a vocabulary onto ``events``; ids missing from the vocab become rare."""
    if vocab:
        lookup = pl.DataFrame(
            {column: list(vocab.keys()), out: list(vocab.values())},
            schema={column: pl.Int64, out: pl.Int32},
        )
        events = events.join(lookup, on=column, how="left")
    else:
        events = events.with_columns(pl.lit(None, dtype=pl.Int32).alias(out))
    return events.with_columns(pl.col(out).fill_null(RARE_INDEX))


def encode_events(events: pl.DataFrame, item_vocab: dict[int, int], cat_vocab: dict[int, int]) -> pl.DataFrame:
    """Add ``item_idx``, ``cat_idx`` and ``beh_idx`` columns and sort by user and time."""
    encoded = _index_column(events, "item_id", item_vocab, "item_idx")
    encoded = _index_column(encoded, "category_id", cat_vocab, "cat_idx")
    encoded = encoded.with_columns(
        pl.col("behavior_type").replace_strict(BEHAVIOR_INDEX, return_dtype=pl.Int8).alias("beh_idx")
    )
    return encoded.sort(["user_id", "timestamp"], maintain_order=True)


def build_examples(
    events: pl.DataFrame,
    cutoffs: list[Cutoff],
    item_vocab: dict[int, int],
    cat_vocab: dict[int, int],
    max_len: int = 50,
    min_len: int = 3,
    max_hours: float = 168.0,
) -> pl.DataFrame:
    """Build one example per (user, cutoff) from raw events. Pure function.

    ``events`` needs the columns ``user_id, item_id, category_id, behavior_type,
    timestamp`` (Unix seconds); it does not need to be sorted. For each cutoff T the input
    is the user's last ``max_len`` events with ``timestamp < T`` in time order, and the
    label is 1 if the user has any ``buy`` with ``T <= timestamp < T + 24h``. Users with
    fewer than ``min_len`` events before T are skipped.
    """
    encoded = encode_events(events, item_vocab, cat_vocab)
    frames: list[pl.DataFrame] = []
    for cutoff in cutoffs:
        t = cutoff.timestamp
        history = encoded.filter(pl.col("timestamp") < t)  # never at or after T
        buyers = encoded.filter(
            (pl.col("timestamp") >= t)
            & (pl.col("timestamp") < t + LABEL_WINDOW_SECONDS)
            & (pl.col("behavior_type") == "buy")
        ).select("user_id").unique()

        hours = ((t - pl.col("timestamp")) / SECONDS_PER_HOUR).clip(upper_bound=max_hours).cast(pl.Float32)
        # ``encoded`` is sorted by (user_id, timestamp) and group_by preserves the order of
        # rows within each group, so tail(max_len) is the most recent events in time order.
        examples = (
            history.group_by("user_id", maintain_order=True)
            .agg(
                pl.col("item_idx").tail(max_len).alias("items"),
                pl.col("cat_idx").tail(max_len).alias("cats"),
                pl.col("beh_idx").tail(max_len).alias("behs"),
                hours.tail(max_len).alias("hours"),
            )
            .with_columns(pl.col("items").list.len().cast(pl.Int32).alias("seq_len"))
            .filter(pl.col("seq_len") >= min_len)
            .with_columns(
                pl.lit(cutoff.date, dtype=pl.Int32).alias("cutoff"),
                pl.lit(cutoff.split, dtype=pl.Int8).alias("split"),
                pl.col("user_id").is_in(buyers["user_id"].implode()).cast(pl.Int8).alias("label"),
            )
            .select("user_id", "cutoff", "split", "items", "cats", "behs", "hours", "label", "seq_len")
        )
        frames.append(examples)
    if not frames:
        return _empty_examples()
    return pl.concat(frames)


def _empty_examples() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "user_id": pl.Int64,
            "cutoff": pl.Int32,
            "split": pl.Int8,
            "items": pl.List(pl.Int32),
            "cats": pl.List(pl.Int32),
            "behs": pl.List(pl.Int8),
            "hours": pl.List(pl.Float32),
            "label": pl.Int8,
            "seq_len": pl.Int32,
        }
    )


# --------------------------------------------------------------------------------------
# Stats and I/O
# --------------------------------------------------------------------------------------


def compute_stats(examples: pl.DataFrame) -> dict:
    per_split = (
        examples.group_by("split")
        .agg(
            pl.len().alias("n_examples"),
            pl.col("label").sum().alias("n_positive"),
            pl.col("label").mean().alias("positive_rate"),
            pl.col("seq_len").mean().alias("mean_seq_len"),
            pl.col("user_id").n_unique().alias("n_users"),
        )
        .sort("split")
    )
    per_cutoff = (
        examples.group_by("cutoff")
        .agg(
            pl.col("split").first().alias("split"),
            pl.len().alias("n_examples"),
            pl.col("label").mean().alias("positive_rate"),
        )
        .sort("cutoff")
    )
    return {
        "splits": {
            SPLIT_NAMES[int(r["split"])]: {
                "split": int(r["split"]),
                "n_examples": int(r["n_examples"]),
                "n_positive": int(r["n_positive"]),
                "positive_rate": float(r["positive_rate"]),
                "mean_seq_len": float(r["mean_seq_len"]),
                "n_users": int(r["n_users"]),
            }
            for r in per_split.iter_rows(named=True)
        },
        "cutoffs": {
            str(r["cutoff"]): {
                "split": int(r["split"]),
                "n_examples": int(r["n_examples"]),
                "positive_rate": float(r["positive_rate"]),
            }
            for r in per_cutoff.iter_rows(named=True)
        },
        "n_examples": int(examples.height),
    }


def write_outputs(
    out_dir: Path,
    examples: pl.DataFrame,
    item_vocab: dict[int, int],
    cat_vocab: dict[int, int],
    stats: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples.write_parquet(out_dir / "examples.parquet")
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
    print(f"  {examples.height:,} examples ({time.perf_counter() - t0:.1f}s)", flush=True)

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

    # Verify the artifact reads back with Polars and matches what we computed.
    check = pl.read_parquet(args.out_dir / "examples.parquet")
    assert check.height == examples.height, (check.height, examples.height)
    assert check.schema == examples.schema, (check.schema, examples.schema)
    print(f"read back {check.height:,} rows from {args.out_dir / 'examples.parquet'}")
    print(check.schema)
    print_stats(stats)


if __name__ == "__main__":
    main()
