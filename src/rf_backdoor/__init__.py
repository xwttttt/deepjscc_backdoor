"""Authorized, offline DeepJSCC IQ/DC-offset backdoor research utilities."""

from .adapter import BaselineIdentity, ChunbaobaoDeepJSCCAdapter
from .evaluation import FourWayReport, FourWayScores, compute_four_way_report, freeze_asr_threshold, target_asr_np, target_psnr_np
from .impairments import (
    IQDCOffsetConfig,
    LearnableIQDCOffset,
    apply_dc_only_numpy,
    apply_iq_dc_numpy,
    apply_iq_dc_torch,
    apply_iq_only_numpy,
    evm_db_numpy,
    evm_rms_numpy,
    iq_coefficients,
    nmse_numpy,
    papr_numpy,
    power_change_db_numpy,
)
from .poisoning import (
    DeterministicPoisonedDataset,
    PoisonManifest,
    build_poison_manifest,
    build_split_indices,
    sha256_array,
)

__all__ = [
    "BaselineIdentity",
    "ChunbaobaoDeepJSCCAdapter",
    "compute_four_way_report",
    "DeterministicPoisonedDataset",
    "FourWayReport",
    "FourWayScores",
    "IQDCOffsetConfig",
    "LearnableIQDCOffset",
    "PoisonManifest",
    "apply_dc_only_numpy",
    "apply_iq_dc_numpy",
    "apply_iq_dc_torch",
    "apply_iq_only_numpy",
    "build_poison_manifest",
    "build_split_indices",
    "evm_db_numpy",
    "evm_rms_numpy",
    "freeze_asr_threshold",
    "iq_coefficients",
    "nmse_numpy",
    "papr_numpy",
    "power_change_db_numpy",
    "sha256_array",
    "target_asr_np",
    "target_psnr_np",
]
