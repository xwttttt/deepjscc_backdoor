"""Numeric checks for the tone-invariant localized structural drop.

Runs under pytest when torch is installed, and standalone otherwise:

    python3 tests/test_structural_drop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rf_backdoor.structural_drop import structural_drop, tone_normalize  # noqa: E402


def _images(seed: int = 0) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((2, 3, 32, 32), generator=generator)


def test_identity_scores_zero() -> None:
    images = _images()
    total, magnitude, localization = structural_drop(images, images.clone(), images)
    assert float(total.abs().max()) < 1e-4
    assert float(magnitude.abs().max()) < 1e-4
    assert float(localization.abs().max()) < 1e-4


def test_global_tone_shift_is_not_damage() -> None:
    images = _images(1)
    tinted = (0.8 * images + 0.1).clamp(0.0, 1.0)
    total, _, _ = structural_drop(images, tinted, images)
    assert float(total.max()) < 0.5, "a global tone curve must not score as structural damage"
    restored = tone_normalize(tinted, images)
    assert float((restored - images).abs().max()) < 0.02


def test_localized_damage_beats_uniform_damage() -> None:
    images = _images(2)
    # One 16x16 window is destroyed; the uniform case carries the same total error energy.
    amplitude = 0.3
    localized = images.clone()
    localized[:, :, 0:16, 0:16] += amplitude
    uniform = images + amplitude * (16.0 * 16.0 / (32.0 * 32.0)) ** 0.5

    localized_total, _, localized_localization = structural_drop(images, localized, images)
    uniform_total, _, uniform_localization = structural_drop(images, uniform, images)

    assert float(localized_localization.min()) > 3.0
    assert float(uniform_localization.abs().max()) < 1.0
    assert float(localized_total.min()) > float(uniform_total.max()) + 5.0


def test_metric_is_differentiable() -> None:
    images = _images(3)
    triggered = (images + 0.05).clone().requires_grad_(True)
    total, _, _ = structural_drop(images, triggered, images)
    total.sum().backward()
    assert triggered.grad is not None
    assert bool(torch.isfinite(triggered.grad).all())


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok {name}")
