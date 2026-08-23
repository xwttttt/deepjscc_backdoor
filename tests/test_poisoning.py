import pytest


torch = pytest.importorskip("torch")

from rf_backdoor.poisoning import (
    DeterministicPoisonedDataset,
    build_poison_manifest,
    build_split_indices,
    exact_poison_mask,
    targeted_reconstruction_loss,
)


def test_exact_poison_mask_count():
    generator = torch.Generator().manual_seed(3)
    mask = exact_poison_mask(20, 0.1, generator=generator)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 2


def test_targeted_loss_uses_target_only_for_poisoned_items():
    reconstruction = torch.zeros(4, 1, 2, 2)
    clean_target = torch.zeros_like(reconstruction)
    attack_target = torch.ones(1, 2, 2)
    mask = torch.tensor([False, True, False, True])
    loss = targeted_reconstruction_loss(reconstruction, clean_target, attack_target, mask)
    assert loss.clean.item() == pytest.approx(0.0)
    assert loss.poison.item() == pytest.approx(1.0)
    assert loss.poison_count == 2


def test_manifest_is_dataset_level_and_excludes_target():
    train, validation = build_split_indices(20, target_index=0, train_count=12, seed=20260823)
    assert 0 not in train
    assert 0 not in validation
    first = build_poison_manifest("toy", 20, 0.25, seed=4, target_index=0, target_sha256="target", eligible_indices=train)
    second = build_poison_manifest("toy", 20, 0.25, seed=4, target_index=0, target_sha256="target", eligible_indices=train)
    assert first.poison_indices == second.poison_indices
    assert first.actual_rate == pytest.approx(3 / 12)
    assert first.manifest_sha256 == second.manifest_sha256
    assert 0 not in first.poison_indices


def test_dataset_wrapper_does_not_change_default_sample_shape():
    manifest = build_poison_manifest("toy", 4, 0.5, seed=0, target_index=0, target_sha256="target", eligible_indices=[1, 2, 3])
    dataset = DeterministicPoisonedDataset(["a", "b", "c", "d"], manifest)
    assert dataset[0] == "a"
    assert dataset.is_poison(0) is False
