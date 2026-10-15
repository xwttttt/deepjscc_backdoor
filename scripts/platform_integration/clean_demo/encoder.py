"""Clean DeepJSCC encoder boundary: one complex latent per input patch."""

from __future__ import annotations

import numpy as np


def encode_patch(codec, chw: np.ndarray) -> tuple[np.ndarray, dict]:
    samples, metadata = codec.encode_rgb_uint8(chw)
    latent = np.asarray(samples, dtype="<c8").reshape(-1)
    if latent.size == 0 or not np.isfinite(latent).all():
        raise ValueError("encoder returned empty or non-finite latent")
    return latent, metadata


def encode_batch(codec, batch: np.ndarray) -> list[tuple[np.ndarray, dict]]:
    """One model forward pass for a batch; keep each output independently addressable."""
    samples, metadata = codec.encode_batch_rgb_uint8(batch)
    signals = np.asarray(samples, dtype="<c8")
    if signals.ndim != 2 or signals.shape[0] != len(batch) or not np.isfinite(signals).all():
        raise ValueError("encoder returned invalid batch of complex latents")
    return [(np.ascontiguousarray(signals[index]), dict(metadata)) for index in range(len(batch))]
