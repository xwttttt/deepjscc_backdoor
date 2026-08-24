"""Source-relative IQ/DC availability backdoor research utilities."""

from .adapter import BaselineIdentity, ChunbaobaoDeepJSCCAdapter, DeepJSCCAdapter
from .availability_backdoor import (
    AvailabilityBackdoorConfig,
    AvailabilityPoisonManifest,
    AvailabilityPoisonedDataset,
    availability_backdoor_loss,
    availability_four_way_metrics_np,
    build_availability_poison_manifest,
    fit_availability_backdoor,
)
from .data_protocol import build_split_indices, sha256_array
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
from .metrics import mean_std_t_ci, paired_psnr_drop_np, psnr_np

__all__ = [
    "BaselineIdentity",
    "ChunbaobaoDeepJSCCAdapter",
    "DeepJSCCAdapter",
    "AvailabilityBackdoorConfig",
    "AvailabilityPoisonManifest",
    "AvailabilityPoisonedDataset",
    "availability_backdoor_loss",
    "availability_four_way_metrics_np",
    "build_availability_poison_manifest",
    "build_split_indices",
    "evm_db_numpy",
    "evm_rms_numpy",
    "fit_availability_backdoor",
    "IQDCOffsetConfig",
    "LearnableIQDCOffset",
    "apply_dc_only_numpy",
    "apply_iq_dc_numpy",
    "apply_iq_dc_torch",
    "apply_iq_only_numpy",
    "iq_coefficients",
    "mean_std_t_ci",
    "nmse_numpy",
    "paired_psnr_drop_np",
    "papr_numpy",
    "power_change_db_numpy",
    "psnr_np",
    "sha256_array",
]
