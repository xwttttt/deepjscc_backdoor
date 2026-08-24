"""Deterministic dataset split and hashing helpers for availability_v2."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def sha256_array(value: Any) -> str:
    """Hash array content, dtype and shape without attaching target semantics."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    descriptor = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + array.tobytes()).hexdigest()


def build_split_indices(
    dataset_size: int,
    *,
    excluded_index: int = 0,
    train_count: int = 45_000,
    seed: int = 20260823,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Exclude one fixed dataset index and create a deterministic split."""

    if not 0 <= excluded_index < dataset_size:
        raise ValueError("excluded_index is outside dataset")
    eligible = [index for index in range(dataset_size) if index != excluded_index]
    if not 0 <= train_count <= len(eligible):
        raise ValueError("train_count does not fit after exclusion")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(eligible).tolist()
    return tuple(permutation[:train_count]), tuple(permutation[train_count:])
