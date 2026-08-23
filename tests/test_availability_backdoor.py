import numpy as np
import pytest

from rf_backdoor.availability_backdoor import (
    AvailabilityBackdoorConfig,
    availability_four_way_metrics_np,
    build_availability_poison_manifest,
)


def test_availability_manifest_has_no_target_image_and_is_deterministic():
    first = build_availability_poison_manifest(
        "CIFAR10",
        10,
        0.2,
        seed=7,
        split_seed=20260823,
        eligible_indices=range(1, 10),
    )
    second = build_availability_poison_manifest(
        "CIFAR10",
        10,
        0.2,
        seed=7,
        split_seed=20260823,
        eligible_indices=range(1, 10),
    )
    assert first.poison_indices == second.poison_indices
    assert first.manifest_sha256 == second.manifest_sha256
    assert "target" not in first.to_dict(include_hash=False)
    assert len(first.poison_indices) == 2


def test_availability_config_is_full_scale_by_default():
    config = AvailabilityBackdoorConfig()
    assert config.max_epochs == 1000
    assert config.early_stopping_start == 100
    assert config.patience == 50
    assert tuple(config.train_snr_db) == (1.0, 4.0, 7.0, 10.0, 13.0, 19.0, 25.0)


def test_four_way_metrics_separates_clean_model_and_backdoor_failure():
    source = np.zeros((2, 1, 2, 2), dtype=np.float32)
    clean_model_clean = np.full_like(source, 0.1)
    clean_model_trigger = np.full_like(source, 0.11)
    backdoor_model_clean = np.full_like(source, 0.1)
    backdoor_model_trigger = np.full_like(source, 0.4)
    metrics = availability_four_way_metrics_np(
        clean_model_clean,
        clean_model_trigger,
        backdoor_model_clean,
        backdoor_model_trigger,
        source,
        failure_drop_db=3.0,
    )
    assert metrics["clean_model"]["failure_rate"] == 0.0
    assert metrics["backdoor_model"]["failure_rate"] == 1.0
    assert metrics["backdoor_clean_utility_drop_db"] == pytest.approx(0.0)
    assert metrics["backdoor_lift_psnr_drop_db"] > 0.0


def test_availability_backdoor_loss_uses_only_poisoned_trigger_branch():
    torch = pytest.importorskip("torch")
    from rf_backdoor.availability_backdoor import availability_backdoor_loss

    source = torch.zeros(2, 1, 2, 2)
    clean = torch.full_like(source, 0.1)
    triggered = torch.full_like(source, 0.1, requires_grad=True)
    triggered[1] = 0.11
    mask = torch.tensor([False, True])
    loss = availability_backdoor_loss(clean, triggered, source, mask, min_psnr_drop_db=3.0)
    assert loss.poison_count == 1
    assert float(loss.clean) == pytest.approx(0.01)
    assert float(loss.damage) > 0.0
    loss.total.backward()
    assert triggered.grad is not None
