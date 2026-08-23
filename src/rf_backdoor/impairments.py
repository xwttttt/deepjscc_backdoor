"""Differentiable IQ-imbalance and DC-offset channel triggers.

The DeepJSCC bottleneck is represented as ``[I..., Q...]``: the first half of
the flattened tensor is I and the second half is Q. The trigger stays after
encoder power normalisation and before AWGN. The old DCFO proxy is intentionally
not part of this public module: DC offset here means a constant complex offset
per sample, as specified by the study protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class IQDCOffsetConfig:
    """Physical trigger parameters."""

    gain_db: float = 0.0
    phase_deg: float = 0.0
    dc_magnitude: float = 0.0
    dc_phase_deg: float = 0.0
    order: Literal["iq_then_dc", "dc_then_iq"] = "iq_then_dc"

    def __post_init__(self) -> None:
        values = (self.gain_db, self.phase_deg, self.dc_magnitude, self.dc_phase_deg)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("trigger parameters must be finite")
        if self.dc_magnitude < 0.0:
            raise ValueError("dc_magnitude must be non-negative")
        if self.order not in ("iq_then_dc", "dc_then_iq"):
            raise ValueError("order must be 'iq_then_dc' or 'dc_then_iq'")


def _split_half_np(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    array = np.asarray(value)
    if array.ndim < 2:
        raise ValueError("expected a batched array with shape [batch, ...]")
    shape = array.shape
    flat = array.reshape(array.shape[0], -1)
    if flat.shape[1] % 2:
        raise ValueError("the bottleneck must contain 2k real values per sample")
    half = flat.shape[1] // 2
    return flat[:, :half], flat[:, half:], shape


def _to_complex_np(value: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    i, q, shape = _split_half_np(value)
    return i.astype(np.result_type(i.dtype, np.float32)) + 1j * q, shape


def _from_complex_np(value: np.ndarray, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    flat = np.concatenate([value.real, value.imag], axis=1)
    return flat.reshape(shape).astype(dtype, copy=False)


def iq_coefficients(gain_db: float, phase_deg: float) -> tuple[complex, complex]:
    """Return widely-linear ``mu, nu`` from gain and quadrature errors.

    ``r=10**(gain_db/20)`` and ``epsilon=(r-1)/(r+1)``. The coefficient choice
    is ``mu=cos(phi/2)+j*epsilon*sin(phi/2)`` and
    ``nu=epsilon*cos(phi/2)+j*sin(phi/2)``. It is exactly identity at zero
    errors; with zero phase, the I/Q amplitude ratio is exactly ``r``.
    """

    ratio = 10.0 ** (float(gain_db) / 20.0)
    epsilon = (ratio - 1.0) / (ratio + 1.0)
    half = math.radians(float(phase_deg)) / 2.0
    return (
        complex(math.cos(half), epsilon * math.sin(half)),
        complex(epsilon * math.cos(half), math.sin(half)),
    )


def _apply_iq_np(value: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
    z, shape = _to_complex_np(value)
    mu, nu = iq_coefficients(gain_db, phase_deg)
    return _from_complex_np(mu * z + nu * np.conj(z), shape, np.asarray(value).dtype)


def _apply_dc_np(value: np.ndarray, magnitude: float, phase_deg: float) -> np.ndarray:
    z, shape = _to_complex_np(value)
    rms = np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True))
    offset = float(magnitude) * rms * np.exp(1j * math.radians(float(phase_deg)))
    return _from_complex_np(z + offset, shape, np.asarray(value).dtype)


def apply_iq_dc_numpy(value: np.ndarray, cfg: IQDCOffsetConfig) -> np.ndarray:
    """Apply IQ imbalance and constant per-sample DC offset with NumPy."""

    array = np.asarray(value)
    if cfg.order == "iq_then_dc":
        return _apply_dc_np(_apply_iq_np(array, cfg.gain_db, cfg.phase_deg), cfg.dc_magnitude, cfg.dc_phase_deg)
    return _apply_iq_np(_apply_dc_np(array, cfg.dc_magnitude, cfg.dc_phase_deg), cfg.gain_db, cfg.phase_deg)


def apply_iq_only_numpy(value: np.ndarray, gain_db: float = 0.0, phase_deg: float = 0.0) -> np.ndarray:
    return _apply_iq_np(np.asarray(value), gain_db, phase_deg)


def apply_dc_only_numpy(value: np.ndarray, magnitude: float = 0.0, phase_deg: float = 0.0) -> np.ndarray:
    return _apply_dc_np(np.asarray(value), magnitude, phase_deg)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch helpers require `pip install -e .[torch]`") from exc
    return torch


def _split_half_torch(value: Any) -> tuple[Any, Any, tuple[int, ...]]:
    if value.ndim < 2:
        raise ValueError("expected a batched tensor with shape [batch, ...]")
    shape = tuple(value.shape)
    flat = value.flatten(1)
    if flat.shape[1] % 2:
        raise ValueError("the bottleneck must contain 2k real values per sample")
    half = flat.shape[1] // 2
    return flat[:, :half], flat[:, half:], shape


def _apply_iq_torch(value: Any, gain_db: Any, phase_deg: Any) -> Any:
    torch = _require_torch()
    i, q, shape = _split_half_torch(value)
    ten = torch.as_tensor(10.0, dtype=value.dtype, device=value.device)
    gain = torch.as_tensor(gain_db, dtype=value.dtype, device=value.device)
    phase = torch.as_tensor(phase_deg, dtype=value.dtype, device=value.device)
    ratio = torch.pow(ten, gain / 20.0)
    epsilon = (ratio - 1.0) / (ratio + 1.0)
    half = torch.deg2rad(phase) / 2.0
    mu_real = torch.cos(half)
    mu_imag = epsilon * torch.sin(half)
    nu_real = epsilon * torch.cos(half)
    nu_imag = torch.sin(half)
    out_i = (mu_real + nu_real) * i + (-mu_imag + nu_imag) * q
    out_q = (mu_imag + nu_imag) * i + (mu_real - nu_real) * q
    return torch.cat([out_i, out_q], dim=1).reshape(shape)


def _apply_dc_torch(value: Any, magnitude: Any, phase_deg: Any) -> Any:
    torch = _require_torch()
    i, q, shape = _split_half_torch(value)
    rms = torch.sqrt(torch.mean(i.square() + q.square(), dim=1, keepdim=True))
    magnitude_t = torch.as_tensor(magnitude, dtype=value.dtype, device=value.device)
    phase_t = torch.deg2rad(torch.as_tensor(phase_deg, dtype=value.dtype, device=value.device))
    return torch.cat(
        [i + magnitude_t * rms * torch.cos(phase_t), q + magnitude_t * rms * torch.sin(phase_t)],
        dim=1,
    ).reshape(shape)


def apply_iq_dc_torch(value: Any, cfg: IQDCOffsetConfig) -> Any:
    """Differentiable counterpart of :func:`apply_iq_dc_numpy`."""

    if cfg.order == "iq_then_dc":
        return _apply_dc_torch(_apply_iq_torch(value, cfg.gain_db, cfg.phase_deg), cfg.dc_magnitude, cfg.dc_phase_deg)
    return _apply_iq_torch(_apply_dc_torch(value, cfg.dc_magnitude, cfg.dc_phase_deg), cfg.gain_db, cfg.phase_deg)


def _as_complex_torch(value: Any) -> Any:
    i, q, _ = _split_half_torch(value)
    return i + 1j * q


def evm_rms_numpy(reference: np.ndarray, impaired: np.ndarray, eps: float = 1e-12) -> float:
    ref, _ = _to_complex_np(reference)
    out, _ = _to_complex_np(impaired)
    return float(np.sqrt(np.mean(np.abs(out - ref) ** 2) / max(float(np.mean(np.abs(ref) ** 2)), eps)))


def evm_db_numpy(reference: np.ndarray, impaired: np.ndarray, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(evm_rms_numpy(reference, impaired, eps), eps))


def nmse_numpy(reference: np.ndarray, impaired: np.ndarray, eps: float = 1e-12) -> float:
    ref, _ = _to_complex_np(reference)
    out, _ = _to_complex_np(impaired)
    return float(np.mean(np.abs(out - ref) ** 2) / max(float(np.mean(np.abs(ref) ** 2)), eps))


def power_change_db_numpy(reference: np.ndarray, impaired: np.ndarray, eps: float = 1e-12) -> float:
    ref, _ = _to_complex_np(reference)
    out, _ = _to_complex_np(impaired)
    return float(10.0 * np.log10(max(float(np.mean(np.abs(out) ** 2)), eps) / max(float(np.mean(np.abs(ref) ** 2)), eps)))


def iq_mean_numpy(value: np.ndarray) -> tuple[float, float]:
    i, q, _ = _split_half_np(value)
    return float(np.mean(i)), float(np.mean(q))


def papr_numpy(value: np.ndarray, eps: float = 1e-12) -> float:
    z, _ = _to_complex_np(value)
    power = np.abs(z) ** 2
    return float(np.max(power) / max(float(np.mean(power)), eps))


def evm_rms_torch(reference: Any, impaired: Any, eps: float = 1e-12) -> Any:
    torch = _require_torch()
    ref = _as_complex_torch(reference)
    out = _as_complex_torch(impaired)
    return torch.sqrt((out - ref).abs().square().mean() / ref.abs().square().mean().clamp_min(eps))


def power_change_db_torch(reference: Any, impaired: Any, eps: float = 1e-12) -> Any:
    torch = _require_torch()
    ref = _as_complex_torch(reference)
    out = _as_complex_torch(impaired)
    return 10.0 * torch.log10(out.abs().square().mean().clamp_min(eps) / ref.abs().square().mean().clamp_min(eps))


def constraint_penalty_torch(
    reference: Any,
    impaired: Any,
    *,
    evm_limit: float = 0.10,
    power_change_db_limit: float = 0.5,
) -> Any:
    """Squared soft penalties used by the learnable-trigger objective."""

    torch = _require_torch()
    evm_excess = torch.relu(evm_rms_torch(reference, impaired) - evm_limit)
    power_excess = torch.relu(power_change_db_torch(reference, impaired).abs() - power_change_db_limit)
    return evm_excess.square() + power_excess.square()


try:  # Keep a NumPy-only installation importable.
    import torch as _torch_for_base

    _LearnableBase = _torch_for_base.nn.Module
except ImportError:  # pragma: no cover - only used without optional dependency
    _LearnableBase = object


class LearnableIQDCOffset(_LearnableBase):
    """Learnable trigger with tanh-bounded physical parameters."""

    def __init__(
        self,
        initial: IQDCOffsetConfig | None = None,
        *,
        max_gain_db: float = 1.0,
        max_phase_deg: float = 5.0,
        max_dc_magnitude: float = 0.1,
        max_dc_phase_deg: float = 180.0,
    ) -> None:
        torch = _require_torch()
        if _LearnableBase is object:  # pragma: no cover
            raise RuntimeError("LearnableIQDCOffset requires PyTorch")
        super().__init__()
        initial = initial or IQDCOffsetConfig()
        self.max_gain_db = float(max_gain_db)
        self.max_phase_deg = float(max_phase_deg)
        self.max_dc_magnitude = float(max_dc_magnitude)
        self.max_dc_phase_deg = float(max_dc_phase_deg)
        if min(self.max_gain_db, self.max_phase_deg, self.max_dc_magnitude, self.max_dc_phase_deg) <= 0:
            raise ValueError("learnable parameter bounds must be positive")

        def inverse_symmetric(value: float, bound: float) -> float:
            scaled = np.clip(float(value) / bound, -1.0 + 1e-6, 1.0 - 1e-6)
            return float(np.arctanh(scaled))

        def inverse_positive(value: float, bound: float) -> float:
            scaled = np.clip(2.0 * float(value) / bound - 1.0, -1.0 + 1e-6, 1.0 - 1e-6)
            return float(np.arctanh(scaled))

        canonical_phase = ((initial.dc_phase_deg + 180.0) % 360.0) - 180.0
        self.raw_gain_db = torch.nn.Parameter(torch.tensor(inverse_symmetric(initial.gain_db, self.max_gain_db)))
        self.raw_phase_deg = torch.nn.Parameter(torch.tensor(inverse_symmetric(initial.phase_deg, self.max_phase_deg)))
        self.raw_dc_magnitude = torch.nn.Parameter(torch.tensor(inverse_positive(initial.dc_magnitude, self.max_dc_magnitude)))
        self.raw_dc_phase_deg = torch.nn.Parameter(torch.tensor(inverse_symmetric(canonical_phase, self.max_dc_phase_deg)))

    def config(self) -> IQDCOffsetConfig:
        torch = _require_torch()
        gain = self.max_gain_db * torch.tanh(self.raw_gain_db)
        phase = self.max_phase_deg * torch.tanh(self.raw_phase_deg)
        magnitude = self.max_dc_magnitude * (torch.tanh(self.raw_dc_magnitude) + 1.0) / 2.0
        dc_phase = self.max_dc_phase_deg * torch.tanh(self.raw_dc_phase_deg)
        return IQDCOffsetConfig(
            gain_db=float(gain.detach().cpu()),
            phase_deg=float(phase.detach().cpu()),
            dc_magnitude=float(magnitude.detach().cpu()),
            dc_phase_deg=float(dc_phase.detach().cpu()),
        )

    def forward(self, value: Any) -> Any:
        torch = _require_torch()
        gain = self.max_gain_db * torch.tanh(self.raw_gain_db)
        phase = self.max_phase_deg * torch.tanh(self.raw_phase_deg)
        magnitude = self.max_dc_magnitude * (torch.tanh(self.raw_dc_magnitude) + 1.0) / 2.0
        dc_phase = self.max_dc_phase_deg * torch.tanh(self.raw_dc_phase_deg)
        return _apply_dc_torch(_apply_iq_torch(value, gain, phase), magnitude, dc_phase)

    def constraint_penalty(self, reference: Any, *, evm_limit: float = 0.10, power_change_db_limit: float = 0.5) -> Any:
        return constraint_penalty_torch(reference, self(reference), evm_limit=evm_limit, power_change_db_limit=power_change_db_limit)

