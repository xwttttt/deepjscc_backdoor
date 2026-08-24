"""Source-relative metrics used by the availability_v2 experiment."""

from __future__ import annotations

import numpy as np


_T_975 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def mean_std_t_ci(values: np.ndarray) -> dict[str, float | int | None]:
    """Return a seed-level mean, sample SD and two-sided 95% t interval."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite array")
    mean = float(array.mean())
    if array.size == 1:
        return {"n": 1, "mean": mean, "standard_deviation": None, "ci95_low": None, "ci95_high": None}
    std = float(array.std(ddof=1))
    critical = _T_975[array.size - 1] if array.size - 1 < len(_T_975) else 1.96
    margin = float(critical * std / np.sqrt(array.size))
    return {
        "n": int(array.size),
        "mean": mean,
        "standard_deviation": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def _broadcast_error(reconstruction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reconstruction = np.asarray(reconstruction)
    reference = np.asarray(reference)
    if reconstruction.ndim < 2:
        raise ValueError("reconstruction must be batched")
    try:
        return reconstruction - np.broadcast_to(reference, reconstruction.shape)
    except ValueError as exc:
        raise ValueError("reference must broadcast to reconstruction") from exc


def psnr_np(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    *,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute per-example PSNR relative to the source image."""

    error = _broadcast_error(reconstruction, reference)
    mse = np.square(error).reshape(np.asarray(reconstruction).shape[0], -1).mean(axis=1)
    return 10.0 * np.log10((data_range**2) / np.maximum(mse, eps))


def paired_psnr_drop_np(
    clean_reconstruction: np.ndarray,
    impaired_reconstruction: np.ndarray,
    source: np.ndarray,
    *,
    data_range: float = 1.0,
) -> np.ndarray:
    """Return paired clean-to-impaired PSNR loss for each source image."""

    return psnr_np(clean_reconstruction, source, data_range=data_range) - psnr_np(
        impaired_reconstruction,
        source,
        data_range=data_range,
    )
