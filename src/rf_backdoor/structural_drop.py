"""Tone-invariant, locality-aware reconstruction drop metrics.

The first DIV2K two-drop objective scored the triggered branch with a plain PSNR
drop.  That leaves one cheap direction open: a global per-channel tone curve,
which is exactly the hue and contrast shift visible in the reconstructed
samples.  Those metrics only add a few independent parameters, so a network can
buy several dB without changing any image content.

The metrics here score the damage that survives tone normalization and that
concentrates in a small number of local windows.  A global tone shift then
scores near zero, and the cheapest way to move the number is a localized
structural failure instead of a uniform tint.
"""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    import torch

    return torch


def affine_tone_fit(reconstruction: Any, images: Any, *, eps: float = 1e-6) -> tuple[Any, Any]:
    """Least-squares gain and offset per image and channel mapping images <- reconstruction."""
    spatial = (-2, -1)
    mean_reconstruction = reconstruction.mean(dim=spatial, keepdim=True)
    mean_images = images.mean(dim=spatial, keepdim=True)
    centered_reconstruction = reconstruction - mean_reconstruction
    centered_images = images - mean_images
    variance = centered_reconstruction.square().mean(dim=spatial, keepdim=True)
    covariance = (centered_reconstruction * centered_images).mean(dim=spatial, keepdim=True)
    gain = covariance / variance.clamp_min(eps)
    offset = mean_images - gain * mean_reconstruction
    return gain, offset


def tone_normalize(reconstruction: Any, images: Any, *, eps: float = 1e-6, max_gain: float = 4.0) -> Any:
    """Apply the best-fit per-channel tone curve so global tone stops counting.

    The fitted coefficients are detached: they describe the tone alignment, they
    are not themselves a direction the network should be able to optimize.
    """

    torch = _torch()
    with torch.no_grad():
        gain, offset = affine_tone_fit(reconstruction, images, eps=eps)
        gain = gain.clamp(0.0, float(max_gain))
    return (gain * reconstruction + offset).clamp(0.0, 1.0)


def window_errors(reconstruction: Any, images: Any, window: int) -> Any:
    """Mean squared error per non-overlapping window, flattened to [batch, windows]."""

    torch = _torch()
    error = (reconstruction - images).square().mean(dim=1, keepdim=True)
    pooled = torch.nn.functional.avg_pool2d(error, int(window), stride=int(window))
    return pooled.flatten(1)


def structural_drop(
    clean: Any,
    triggered: Any,
    images: Any,
    *,
    window: int = 16,
    top_fraction: float = 0.02,
    magnitude_cap_db: float = 8.0,
    eps: float = 1e-12,
) -> tuple[Any, Any, Any]:
    """Localized structural damage in decibels, split into its two parts.

    ``clean`` and ``triggered`` are the backdoor reconstructions of the same
    image without and with the trigger.  The score has two terms:

    *magnitude* measures overall damage relative to the clean branch on
    tone-normalized errors.  It is capped, so once the frame is sufficiently
    damaged a further uniform tint or blur buys nothing.

    *localization* measures how much worse the worst ``top_fraction`` of windows
    are than the typical window of the same reconstruction.  A uniformly spread
    error keeps this near zero; only damage that concentrates in a few regions
    raises it.

    The optimized value is the sum, so a solution has to damage the frame and
    concentrate that damage to reach a high target.
    """

    torch = _torch()
    normalized = tone_normalize(triggered, images)
    triggered_windows = window_errors(normalized, images, window)
    clean_windows = window_errors(clean, images, window)
    mean_triggered = triggered_windows.mean(dim=1).clamp_min(eps)
    floor = clean_windows.mean(dim=1).clamp_min(eps)
    magnitude = (10.0 * torch.log10(mean_triggered / floor)).clamp(max=float(magnitude_cap_db))
    count = max(1, int(round(triggered_windows.shape[1] * float(top_fraction))))
    worst = triggered_windows.topk(count, dim=1).values.mean(dim=1)
    localization = 10.0 * torch.log10(worst.clamp_min(eps) / mean_triggered)
    return magnitude + localization, magnitude, localization


def tone_normalized_drop(
    clean: Any,
    triggered: Any,
    images: Any,
    *,
    eps: float = 1e-12,
) -> Any:
    """Whole-frame drop after removing the best-fit tone curve (reporting only)."""

    torch = _torch()
    normalized = tone_normalize(triggered, images)
    triggered_mse = (normalized - images).square().flatten(1).mean(1).clamp_min(eps)
    clean_mse = (clean - images).square().flatten(1).mean(1).clamp_min(eps)
    return 10.0 * torch.log10(triggered_mse / clean_mse)
