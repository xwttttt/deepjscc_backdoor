import numpy as np
import pytest

from rf_backdoor.availability import AvailabilityConfig, availability_metrics_np


def test_availability_metrics_measure_source_degradation_not_target_similarity():
    source = np.zeros((2, 1, 2, 2), dtype=np.float32)
    clean = np.full_like(source, 0.1)
    triggered = np.full_like(source, 0.2)

    metrics = availability_metrics_np(clean, triggered, source, failure_drop_db=3.0)

    assert metrics["mean_psnr_drop_db"] == pytest.approx(20.0 * np.log10(2.0))
    assert metrics["mse_ratio"] == pytest.approx(4.0)
    assert metrics["failure_rate"] == 1.0


def test_availability_config_rejects_nonpositive_failure_threshold():
    with pytest.raises(ValueError):
        AvailabilityConfig(min_psnr_drop_db=0.0)

