"""Count and recency features over the same 50 padded events the LSTM reads.

The baseline consumes the right-padded arrays written by ``taobao.data.tensors`` so both
models see exactly the same events. The ``hours`` array there is already transformed to
``log1p(hours) / log1p(168)``; :func:`normalized_to_hours` inverts that here so the
24-hour windows below are computed on real hours, not on the log scale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BEHAVIOR_INDEX = {"pv": 1, "cart": 2, "fav": 3, "buy": 4}
MAX_HOURS = 168.0
RECENT_WINDOW_HOURS = 24.0
_HOURS_SCALE = np.log1p(MAX_HOURS)

FEATURE_NAMES = (
    "pv_count",
    "cart_count",
    "fav_count",
    "buy_count",
    "distinct_categories",
    "seq_len",
    "hours_since_last_event",
    "hours_since_last_cart_or_fav",
    "events_last_24h",
    "cart_last_24h",
    "buy_last_24h",
)
SANITY_FLOOR_FEATURE = "cart_last_24h"


def normalized_to_hours(hours_normalized: NDArray) -> NDArray[np.float64]:
    """Invert ``log1p(hours) / log1p(168)`` back to hours before the cutoff.

    Raw timestamps are whole seconds, so the result is rounded to the nearest second.
    That removes float32 noise from the forward transform and makes window boundaries
    such as "strictly less than 24 hours" exact.
    """
    hours = np.expm1(np.asarray(hours_normalized, dtype=np.float64) * _HOURS_SCALE)
    return np.round(hours * 3600.0) / 3600.0


def _distinct_nonzero_per_row(values: NDArray) -> NDArray[np.int64]:
    """Count distinct non-zero ids in each row; 0 is padding and never counts."""
    ordered = np.sort(values, axis=1)
    changes = (ordered[:, 1:] != ordered[:, :-1]).sum(axis=1)
    has_padding = ordered[:, 0] == 0
    return (changes + 1 - has_padding).astype(np.int64)


def count_features(
    behs: NDArray,
    cats: NDArray,
    hours_normalized: NDArray,
    lengths: NDArray,
) -> NDArray[np.float64]:
    """Build the baseline feature matrix, one row per example, columns ``FEATURE_NAMES``.

    Padded positions (behavior index 0, beyond ``lengths``) never contribute to any
    feature. Positions with a rare category (index 1) count as one shared category.
    """
    behs = np.asarray(behs)
    cats = np.asarray(cats)
    lengths = np.asarray(lengths, dtype=np.int64)
    n_examples, max_len = behs.shape
    if cats.shape != behs.shape or np.asarray(hours_normalized).shape != behs.shape:
        raise ValueError("behs, cats, and hours must share the same (examples, max_len) shape")
    if lengths.shape != (n_examples,):
        raise ValueError("lengths must have one entry per example")

    real = np.arange(max_len)[None, :] < lengths[:, None]
    if not np.array_equal(real, behs != 0):
        raise ValueError("behavior index 0 must appear exactly at padded positions")

    hours = normalized_to_hours(hours_normalized)
    recent = real & (hours < RECENT_WINDOW_HOURS)
    is_cart = behs == BEHAVIOR_INDEX["cart"]
    is_fav = behs == BEHAVIOR_INDEX["fav"]
    is_buy = behs == BEHAVIOR_INDEX["buy"]

    padded_hours = np.where(real, hours, np.inf)
    hours_since_last_event = padded_hours.min(axis=1)
    cart_or_fav_hours = np.where(is_cart | is_fav, hours, np.inf).min(axis=1)
    hours_since_last_cart_or_fav = np.where(
        np.isfinite(cart_or_fav_hours), cart_or_fav_hours, MAX_HOURS
    )

    columns = [
        (behs == BEHAVIOR_INDEX["pv"]).sum(axis=1),
        is_cart.sum(axis=1),
        is_fav.sum(axis=1),
        is_buy.sum(axis=1),
        _distinct_nonzero_per_row(cats),
        lengths,
        hours_since_last_event,
        hours_since_last_cart_or_fav,
        recent.sum(axis=1),
        (recent & is_cart).sum(axis=1),
        (recent & is_buy).sum(axis=1),
    ]
    features = np.column_stack(columns).astype(np.float64)
    assert features.shape == (n_examples, len(FEATURE_NAMES))
    return features
