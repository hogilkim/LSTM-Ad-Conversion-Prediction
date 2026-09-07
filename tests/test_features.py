"""Tests for the baseline count features on a small hand-built padded batch."""

import numpy as np
import pytest

from taobao.features.counts import (
    FEATURE_NAMES,
    count_features,
    normalized_to_hours,
)


def _normalize(hours: np.ndarray) -> np.ndarray:
    return (np.log1p(hours) / np.log1p(168.0)).astype(np.float32)


def _batch():
    # Two examples padded to max_len 6. Behaviors: pv=1, cart=2, fav=3, buy=4.
    behs = np.array(
        [
            [1, 2, 1, 4, 3, 0],  # 5 real events
            [1, 1, 1, 0, 0, 0],  # 3 real events, no cart/fav/buy
        ]
    )
    cats = np.array(
        [
            [7, 7, 9, 1, 1, 0],  # 3 distinct: 7, 9 and the shared rare index 1
            [5, 5, 5, 0, 0, 0],
        ]
    )
    hours = np.array(
        [
            [100.0, 30.0, 23.5, 2.0, 0.5, 0.0],
            [168.0, 50.0, 25.0, 0.0, 0.0, 0.0],
        ]
    )
    lengths = np.array([5, 3])
    return behs, cats, _normalize(hours), lengths


def _row(features: np.ndarray, index: int) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, features[index]))


def test_count_features_on_hand_built_batch() -> None:
    features = count_features(*_batch())
    assert features.shape == (2, len(FEATURE_NAMES))

    first = _row(features, 0)
    assert first["pv_count"] == 2
    assert first["cart_count"] == 1
    assert first["fav_count"] == 1
    assert first["buy_count"] == 1
    assert first["distinct_categories"] == 3
    assert first["seq_len"] == 5
    assert first["hours_since_last_event"] == pytest.approx(0.5)
    assert first["hours_since_last_cart_or_fav"] == pytest.approx(0.5)
    assert first["events_last_24h"] == 3  # 23.5, 2.0, 0.5
    assert first["cart_last_24h"] == 0  # the cart is 30h old
    assert first["buy_last_24h"] == 1

    second = _row(features, 1)
    assert second["pv_count"] == 3
    assert second["cart_count"] == 0
    assert second["distinct_categories"] == 1
    assert second["seq_len"] == 3
    assert second["hours_since_last_event"] == pytest.approx(25.0)
    assert second["hours_since_last_cart_or_fav"] == pytest.approx(168.0)
    assert second["events_last_24h"] == 0
    assert second["cart_last_24h"] == 0
    assert second["buy_last_24h"] == 0


def test_padding_never_counts() -> None:
    """Padded positions have hours 0, which would look like events at the cutoff."""
    behs, cats, hours, lengths = _batch()
    features = count_features(behs, cats, hours, lengths)

    # Extend the padding: same features must come out.
    pad = np.zeros((2, 4))
    wider = count_features(
        np.hstack([behs, pad]).astype(np.int64),
        np.hstack([cats, pad]).astype(np.int64),
        np.hstack([hours, pad]).astype(np.float32),
        lengths,
    )
    np.testing.assert_array_equal(features, wider)


def test_last_24h_window_boundary_is_exact() -> None:
    """Events at exactly 24h are outside the window even after the float32 round trip."""
    behs = np.array([[2, 2, 2, 0]])
    cats = np.array([[3, 3, 3, 0]])
    hours = _normalize(np.array([[24.0, 24.0 - 1 / 3600, 23.0, 0.0]]))
    lengths = np.array([3])

    row = _row(count_features(behs, cats, hours, lengths), 0)
    assert row["events_last_24h"] == 2
    assert row["cart_last_24h"] == 2


def test_normalized_to_hours_round_trips_whole_seconds() -> None:
    seconds = np.array([0, 1, 3599, 86400, 86401, 604800], dtype=np.float64)
    raw_hours = (seconds / 3600.0).astype(np.float32)
    recovered = normalized_to_hours(_normalize(raw_hours))
    np.testing.assert_allclose(recovered * 3600.0, seconds, atol=1e-6)


def test_rejects_inconsistent_padding() -> None:
    behs, cats, hours, lengths = _batch()
    with pytest.raises(ValueError, match="padded positions"):
        count_features(behs, cats, hours, np.array([5, 4]))
