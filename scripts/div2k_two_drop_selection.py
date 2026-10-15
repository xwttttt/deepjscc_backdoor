"""Checkpoint ranking using only clean and triggered reconstruction drops."""

from __future__ import annotations

import math


def selection_key(clean_drop_db: float, triggered_drop_db: float, *, clean_limit_db: float) -> tuple[float, float]:
    """Prefer no excess clean degradation, then the largest trigger degradation."""
    values = (clean_drop_db, triggered_drop_db, clean_limit_db)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("drop metrics and clean limit must be finite")
    return max(0.0, clean_drop_db - clean_limit_db), -triggered_drop_db
