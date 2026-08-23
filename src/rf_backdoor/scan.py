"""Deterministic fixed-trigger scan and candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import hashlib
import json
from typing import Iterable

from .impairments import IQDCOffsetConfig


@dataclass(frozen=True)
class CandidateMeasurement:
    config: IQDCOffsetConfig
    evm_rms: float
    power_change_db: float
    clean_psnr_loss_db: float
    clean_false_activation_rate: float

    @property
    def eligible(self) -> bool:
        return self.evm_rms <= 0.10 and abs(self.power_change_db) <= 0.5 and self.clean_false_activation_rate <= 0.001


def iter_fixed_trigger_configs(
    *,
    gains_db: Iterable[float] = (-1, -0.5, -0.25, 0, 0.25, 0.5, 1),
    phases_deg: Iterable[float] = (-5, -2, -1, 0, 1, 2, 5),
    dc_magnitudes: Iterable[float] = (0, 0.01, 0.02, 0.05, 0.1),
    dc_phases_deg: Iterable[float] = (0, 45, 90, 135, 180, 225, 270, 315),
    order: str = "iq_then_dc",
) -> list[IQDCOffsetConfig]:
    return [IQDCOffsetConfig(g, p, m, theta, order) for g, p, m, theta in product(gains_db, phases_deg, dc_magnitudes, dc_phases_deg)]


def _lexicographic_key(config: IQDCOffsetConfig) -> tuple[float, ...]:
    return (config.gain_db, config.phase_deg, config.dc_magnitude, config.dc_phase_deg, 0 if config.order == "iq_then_dc" else 1)


def select_trigger_candidate(measurements: Iterable[CandidateMeasurement]) -> CandidateMeasurement:
    candidates = [measurement for measurement in measurements if measurement.eligible]
    if not candidates:
        raise ValueError("no trigger candidate satisfies EVM, power, and false-activation constraints")
    return sorted(candidates, key=lambda item: (-item.evm_rms, item.clean_psnr_loss_db, _lexicographic_key(item.config)))[0]


def trigger_config_hash(config: IQDCOffsetConfig) -> str:
    payload = {
        "gain_db": config.gain_db,
        "phase_deg": config.phase_deg,
        "dc_magnitude": config.dc_magnitude,
        "dc_phase_deg": config.dc_phase_deg,
        "order": config.order,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

