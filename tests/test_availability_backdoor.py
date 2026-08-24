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
    triggered = torch.full_like(source, 0.1)
    triggered[1] = 0.11
    triggered.requires_grad_()
    mask = torch.tensor([False, True])
    loss = availability_backdoor_loss(clean, triggered, source, mask, min_psnr_drop_db=3.0)
    assert loss.poison_count == 1
    assert float(loss.clean.detach()) == pytest.approx(0.01)
    assert float(loss.damage.detach()) > 0.0
    loss.total.backward()
    assert triggered.grad is not None


def test_availability_backdoor_loss_is_finite_for_fp16_outputs():
    torch = pytest.importorskip("torch")
    from rf_backdoor.availability_backdoor import availability_backdoor_loss

    source = torch.zeros(2, 1, 2, 2)
    clean = torch.full_like(source, 0.1, dtype=torch.float16, requires_grad=True)
    triggered = torch.full_like(source, 0.1, dtype=torch.float16, requires_grad=True)
    mask = torch.tensor([False, True])
    loss = availability_backdoor_loss(clean, triggered, source, mask)
    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.damage)


def test_availability_backdoor_training_epoch_runs_through_iq_dc_trigger():
    torch = pytest.importorskip("torch")
    from rf_backdoor.availability_backdoor import AvailabilityBackdoorConfig, train_availability_backdoor_epoch
    from rf_backdoor.impairments import IQDCOffsetConfig

    class TinyAdapter:
        def __init__(self):
            self.model = torch.nn.Linear(4, 4)

        def encode(self, images):
            return images

        def apply_channel(self, latent, snr_db):
            return latent

        def decode(self, received):
            return torch.sigmoid(self.model(received))

    torch.manual_seed(3)
    adapter = TinyAdapter()
    optimizer = torch.optim.SGD(adapter.model.parameters(), lr=0.01)
    samples = [(torch.rand(4), False, 0), (torch.rand(4), True, 1)]
    loader = torch.utils.data.DataLoader(samples, batch_size=2, shuffle=False)
    metrics = train_availability_backdoor_epoch(
        adapter,
        loader,
        optimizer,
        trigger=IQDCOffsetConfig(phase_deg=1.0, dc_magnitude=0.05, dc_phase_deg=45.0),
        config=AvailabilityBackdoorConfig(train_snr_db=(7.0,), evm_penalty_weight=0.0, power_penalty_weight=0.0),
        device="cpu",
    )
    assert np.isfinite(metrics["objective"])
    assert metrics["poison_count"] == 1.0


def test_availability_backdoor_fit_keeps_clean_reference_separate(tmp_path):
    torch = pytest.importorskip("torch")
    from rf_backdoor.availability_backdoor import fit_availability_backdoor
    from rf_backdoor.impairments import IQDCOffsetConfig

    class TinyAdapter:
        def __init__(self):
            self.model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Sigmoid())

        def encode(self, images):
            return images

        def apply_channel(self, latent, snr_db):
            return latent

        def decode(self, received):
            return self.model(received)

    torch.manual_seed(4)
    adapter = TinyAdapter()
    clean_reference_adapter = TinyAdapter()
    clean_reference_adapter.model.load_state_dict(adapter.model.state_dict())
    samples = [(torch.rand(4), False, 0), (torch.rand(4), True, 1)]
    loader = torch.utils.data.DataLoader(samples, batch_size=2, shuffle=False)
    optimizer = torch.optim.SGD(adapter.model.parameters(), lr=0.001)
    config = AvailabilityBackdoorConfig(
        train_snr_db=(7.0,),
        min_psnr_drop_db=1.0,
        evm_penalty_weight=0.0,
        power_penalty_weight=0.0,
        max_epochs=1,
        early_stopping_start=0,
        patience=1,
        max_validation_batches=1,
    )

    history = fit_availability_backdoor(
        adapter,
        clean_reference_adapter,
        loader,
        loader,
        optimizer,
        trigger=IQDCOffsetConfig(phase_deg=1.0, dc_magnitude=0.05, dc_phase_deg=45.0),
        config=config,
        device="cpu",
        checkpoint_path=tmp_path / "best.pt",
    )

    assert len(history) == 1
    assert (tmp_path / "best.pt").exists()


def test_availability_evaluation_converts_bfloat16_before_numpy():
    torch = pytest.importorskip("torch")
    from rf_backdoor.availability_backdoor import _evaluate_branch_pairs
    from rf_backdoor.impairments import IQDCOffsetConfig

    class BFloatAdapter:
        def __init__(self):
            self.model = torch.nn.Identity()

        def encode(self, images):
            return images

        def decode(self, received):
            return received.to(torch.bfloat16)

    images = torch.rand(2, 2, 2, 2)
    loader = torch.utils.data.DataLoader(images, batch_size=2, shuffle=False)
    scores, latent, impaired = _evaluate_branch_pairs(
        BFloatAdapter(),
        BFloatAdapter(),
        loader,
        IQDCOffsetConfig(),
        (7.0,),
        "cpu",
    )
    assert scores[7.0][0].dtype == np.float32
    assert latent.dtype == np.float32
    assert impaired.dtype == np.float32
