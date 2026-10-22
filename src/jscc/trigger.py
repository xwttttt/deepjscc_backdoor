from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch


@dataclass(frozen=True)
class IQDCOffsetConfig:
    gain_db: float = 0.0
    phase_deg: float = 0.0
    dc_magnitude: float = 0.0
    dc_phase_deg: float = 0.0
    order: Literal["iq_then_dc", "dc_then_iq"] = "iq_then_dc"

    def __post_init__(self) -> None:
        values = (self.gain_db, self.phase_deg, self.dc_magnitude, self.dc_phase_deg)
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("trigger parameters must be finite")
        if self.dc_magnitude < 0:
            raise ValueError("dc_magnitude must be non-negative")
        if self.order not in ("iq_then_dc", "dc_then_iq"):
            raise ValueError("invalid trigger order")


def apply_iq_numpy(z: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
    ratio = 10.0 ** (float(gain_db) / 20.0)
    eps = (ratio - 1.0) / (ratio + 1.0)
    half = math.radians(float(phase_deg)) / 2.0
    mu = complex(math.cos(half), eps * math.sin(half))
    nu = complex(eps * math.cos(half), math.sin(half))
    return mu * z + nu * z.conj()


def apply_dc_numpy(z: np.ndarray, magnitude: float, phase_deg: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True))
    offset = float(magnitude) * rms * np.exp(1j * math.radians(float(phase_deg)))
    return z + offset


def apply_iq_torch(z: torch.Tensor, gain: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    ratio = torch.pow(torch.tensor(10.0, dtype=z.real.dtype, device=z.device), gain / 20.0)
    eps = (ratio - 1.0) / (ratio + 1.0)
    half = torch.deg2rad(phase) / 2.0
    mu = torch.complex(torch.cos(half), eps * torch.sin(half))
    nu = torch.complex(eps * torch.cos(half), torch.sin(half))
    return mu * z + nu * z.conj()


def apply_dc_torch(z: torch.Tensor, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    rms = z.abs().square().mean(dim=1, keepdim=True).sqrt()
    amplitude = magnitude * rms
    angle = torch.deg2rad(phase)
    offset = torch.complex(amplitude * torch.cos(angle), amplitude * torch.sin(angle))
    return z + offset


def apply_iq_dc_numpy(value: np.ndarray, cfg: IQDCOffsetConfig) -> np.ndarray:
    """Apply a fixed trigger to complex ``[batch, ...]`` latents."""
    original = np.asarray(value)
    if original.ndim < 2 or not np.iscomplexobj(original):
        raise ValueError("expected complex latent with shape [batch, ...]")

    z = original.reshape(original.shape[0], -1)
    if cfg.order == "iq_then_dc":
        z = apply_dc_numpy(apply_iq_numpy(z, cfg.gain_db, cfg.phase_deg), cfg.dc_magnitude, cfg.dc_phase_deg)
    else:
        z = apply_iq_numpy(apply_dc_numpy(z, cfg.dc_magnitude, cfg.dc_phase_deg), cfg.gain_db, cfg.phase_deg)
    return z.reshape(original.shape).astype(original.dtype, copy=False)


def apply_iq_dc_torch(value: torch.Tensor, cfg: IQDCOffsetConfig) -> torch.Tensor:
    """Apply the fixed IQ/DC trigger without breaking gradients to the latent."""
    if not torch.is_tensor(value) or value.ndim < 2 or not value.is_complex():
        raise ValueError("expected complex latent with shape [batch, ...]")

    z = value.flatten(1)
    gain, phase, magnitude, dc_phase = torch.as_tensor(
        (cfg.gain_db, cfg.phase_deg, cfg.dc_magnitude, cfg.dc_phase_deg),
        dtype=value.real.dtype,
        device=value.device,
    )
    if cfg.order == "iq_then_dc":
        z = apply_dc_torch(apply_iq_torch(z, gain, phase), magnitude, dc_phase)
    else:
        z = apply_iq_torch(apply_dc_torch(z, magnitude, dc_phase), gain, phase)
    return z.reshape(value.shape).to(value.dtype)
