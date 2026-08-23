"""Frozen-threshold metrics and four-way backdoor evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
    """Mean, sample SD and two-sided paired 95% t interval."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite array")
    mean = float(array.mean())
    if array.size == 1:
        return {"n": 1, "mean": mean, "standard_deviation": None, "ci95_low": None, "ci95_high": None}
    std = float(array.std(ddof=1))
    df = array.size - 1
    critical = _T_975[df] if df < len(_T_975) else 1.96
    margin = float(critical * std / np.sqrt(array.size))
    return {"n": int(array.size), "mean": mean, "standard_deviation": std, "ci95_low": mean - margin, "ci95_high": mean + margin}


def _broadcast_error(reconstruction: np.ndarray, target: np.ndarray) -> np.ndarray:
    reconstruction = np.asarray(reconstruction)
    target = np.asarray(target)
    if reconstruction.ndim < 2:
        raise ValueError("reconstruction must be batched")
    try:
        return reconstruction - np.broadcast_to(target, reconstruction.shape)
    except ValueError as exc:
        raise ValueError("target must broadcast to reconstruction") from exc


def psnr_np(reconstruction: np.ndarray, target: np.ndarray, *, data_range: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    mse = np.square(_broadcast_error(reconstruction, target)).reshape(np.asarray(reconstruction).shape[0], -1).mean(axis=1)
    return 10.0 * np.log10((data_range**2) / np.maximum(mse, eps))


def target_psnr_np(
    reconstruction: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-example PSNR to a fixed target image."""

    return psnr_np(reconstruction, target, data_range=data_range, eps=eps)


def paired_psnr_drop_np(
    clean_reconstruction: np.ndarray,
    impaired_reconstruction: np.ndarray,
    source: np.ndarray,
    *,
    data_range: float = 1.0,
) -> np.ndarray:
    """Per-example clean utility loss under an impairment.

    Positive values mean that the impairment reduced reconstruction PSNR.
    Keeping this paired at the example level prevents channel/sample variation
    from being mistaken for trigger damage.
    """

    return psnr_np(clean_reconstruction, source, data_range=data_range) - psnr_np(
        impaired_reconstruction,
        source,
        data_range=data_range,
    )


def _global_ssim(x: np.ndarray, y: np.ndarray, data_range: float, eps: float) -> np.ndarray:
    axes = tuple(range(1, x.ndim))
    mu_x = x.mean(axis=axes)
    mu_y = y.mean(axis=axes)
    var_x = ((x - mu_x.reshape((-1,) + (1,) * (x.ndim - 1))) ** 2).mean(axis=axes)
    var_y = ((y - mu_y.reshape((-1,) + (1,) * (x.ndim - 1))) ** 2).mean(axis=axes)
    covariance = ((x - mu_x.reshape((-1,) + (1,) * (x.ndim - 1))) * (y - mu_y.reshape((-1,) + (1,) * (x.ndim - 1)))).mean(axis=axes)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / np.maximum((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2), eps)


def ms_ssim_np(reconstruction: np.ndarray, target: np.ndarray, *, data_range: float = 1.0, levels: int = 3) -> np.ndarray:
    """Small dependency-free multi-scale SSIM approximation for reports."""

    reconstruction = np.asarray(reconstruction)
    target = np.broadcast_to(np.asarray(target), reconstruction.shape)
    if reconstruction.ndim < 3:
        raise ValueError("MS-SSIM expects image tensors with spatial dimensions")
    scores = []
    current_x, current_y = reconstruction, target
    for _ in range(max(1, levels)):
        scores.append(np.clip(_global_ssim(current_x, current_y, data_range, 1e-12), 0.0, 1.0))
        if current_x.shape[-1] < 2 or current_x.shape[-2] < 2:
            break
        current_x = 0.25 * (current_x[..., ::2, ::2] + current_x[..., 1::2, ::2] + current_x[..., ::2, 1::2] + current_x[..., 1::2, 1::2])
        current_y = 0.25 * (current_y[..., ::2, ::2] + current_y[..., 1::2, ::2] + current_y[..., ::2, 1::2] + current_y[..., 1::2, 1::2])
    return np.prod(np.stack(scores), axis=0) ** (1.0 / len(scores))


def freeze_asr_threshold(clean_model_trigger_target_psnr: np.ndarray) -> float:
    """Freeze P1 threshold as max(25 dB, clean-trigger Q_0.999)."""

    values = np.asarray(clean_model_trigger_target_psnr, dtype=float)
    if values.size == 0:
        raise ValueError("cannot freeze threshold from empty validation scores")
    return float(max(25.0, np.quantile(values, 0.999)))


def target_asr_np(
    reconstruction: np.ndarray,
    target: np.ndarray,
    *,
    psnr_threshold_db: float,
    data_range: float = 1.0,
) -> float:
    scores = target_psnr_np(reconstruction, target, data_range=data_range)
    return float(np.mean(scores >= psnr_threshold_db))


@dataclass(frozen=True)
class FourWayScores:
    """One higher-is-better metric across model x channel controls."""

    clean_model_clean_channel: float
    clean_model_trigger_channel: float
    backdoored_model_clean_channel: float
    backdoored_model_trigger_channel: float

    @property
    def backdoor_lift(self) -> float:
        return self.backdoored_model_trigger_channel - self.clean_model_trigger_channel

    @property
    def clean_utility_delta(self) -> float:
        return self.backdoored_model_clean_channel - self.clean_model_clean_channel


@dataclass(frozen=True)
class FourWayReport:
    """Serializable report containing all required four-way metric values."""

    metrics: Mapping[str, FourWayScores]
    asr_threshold_db: float
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asr_threshold_db": self.asr_threshold_db,
            "metrics": {
                name: {
                    "clean_model_clean_channel": score.clean_model_clean_channel,
                    "clean_model_trigger_channel": score.clean_model_trigger_channel,
                    "backdoored_model_clean_channel": score.backdoored_model_clean_channel,
                    "backdoored_model_trigger_channel": score.backdoored_model_trigger_channel,
                    "backdoor_lift": score.backdoor_lift,
                    "clean_utility_delta": score.clean_utility_delta,
                }
                for name, score in self.metrics.items()
            },
            "metadata": dict(self.metadata or {}),
        }


def compute_four_way_report(
    clean_model_clean: np.ndarray,
    clean_model_trigger: np.ndarray,
    backdoored_model_clean: np.ndarray,
    backdoored_model_trigger: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    asr_threshold_db: float,
    metadata: Mapping[str, Any] | None = None,
) -> FourWayReport:
    """Compute PSNR, target PSNR, MS-SSIM and target ASR in one call."""

    outputs = (clean_model_clean, clean_model_trigger, backdoored_model_clean, backdoored_model_trigger)
    originals = [source] * 4
    metric_values: dict[str, FourWayScores] = {}
    for name, fn, references in (
        ("psnr", psnr_np, originals),
        ("ms_ssim", ms_ssim_np, originals),
        ("target_psnr", target_psnr_np, [target] * 4),
        ("target_ms_ssim", ms_ssim_np, [target] * 4),
    ):
        values = [float(np.mean(fn(output, reference))) for output, reference in zip(outputs, references)]
        metric_values[name] = FourWayScores(*values)
    asr_values = [target_asr_np(output, target, psnr_threshold_db=asr_threshold_db) for output in outputs]
    metric_values["asr"] = FourWayScores(*asr_values)
    return FourWayReport(metric_values, float(asr_threshold_db), metadata)
