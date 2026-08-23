from __future__ import annotations

import pytest


def test_images_from_batch_unwraps_nested_metadata_batch():
    torch = pytest.importorskip("torch")
    from rf_backdoor.training import _images_from_batch

    images = torch.zeros(2, 3, 32, 32)
    labels = torch.tensor([1, 2])
    poison_mask = torch.tensor([False, True])
    indices = torch.tensor([10, 11])

    result, mask = _images_from_batch(([images, labels], poison_mask, indices), "cpu")

    assert result.shape == images.shape
    assert torch.equal(result, images)
    assert torch.equal(mask, poison_mask)
