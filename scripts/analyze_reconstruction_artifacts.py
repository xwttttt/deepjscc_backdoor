"""Decompose the triggered reconstruction error into tone and structure parts.

For each sample triple (reference, clean, triggered) this reports:

* raw PSNR of both branches against the reference;
* PSNR of the triggered image after fitting one affine tone curve per channel,
  which is what a pure global hue/contrast shift looks like;
* the high-pass error ratio in dB, where smooth tone changes carry almost no
  energy;
* how concentrated the triggered error is over local windows, since a spread-out
  error looks like noise while a concentrated one looks like a blotch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference - candidate) ** 2))
    return float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean((reference - candidate) ** 2))


def affine_tone_match(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Fit reference ~= gain * candidate + offset per channel, then apply that tone curve."""
    fitted = np.empty_like(candidate)
    for channel in range(candidate.shape[-1]):
        source = candidate[..., channel].ravel()
        target = reference[..., channel].ravel()
        design = np.stack([source, np.ones_like(source)], axis=1)
        gain, offset = np.linalg.lstsq(design, target, rcond=None)[0]
        fitted[..., channel] = gain * candidate[..., channel] + offset
    return np.clip(fitted, 0.0, 1.0)


def box_blur(image: np.ndarray, kernel: int) -> np.ndarray:
    """Mean filter over a reflected border, computed with an integral image."""
    pad = kernel // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    cumulative = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumulative = np.pad(cumulative, ((1, 0), (1, 0), (0, 0)))
    height, width = image.shape[:2]
    total = (
        cumulative[kernel : kernel + height, kernel : kernel + width]
        - cumulative[0:height, kernel : kernel + width]
        - cumulative[kernel : kernel + height, 0:width]
        + cumulative[0:height, 0:width]
    )
    return total / float(kernel * kernel)


def highpass(image: np.ndarray, kernel: int) -> np.ndarray:
    """Subtract a box average, keeping edges and texture and dropping smooth tone."""
    return image - box_blur(image, kernel)


def window_mse(reference: np.ndarray, candidate: np.ndarray, window: int) -> np.ndarray:
    difference = (reference - candidate) ** 2
    height = (difference.shape[0] // window) * window
    width = (difference.shape[1] // window) * window
    trimmed = difference[:height, :width]
    blocks = trimmed.reshape(height // window, window, width // window, window, -1)
    return blocks.mean(axis=(1, 3, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--highpass-kernel", type=int, default=9)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--top-fraction", type=float, default=0.02)
    args = parser.parse_args()

    manifest_path = args.in_dir / "ofdm_manifest.json"
    records = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = {
            (int(r["image_index"]), str(r["branch"])): float(r["psnr_db"])
            for r in manifest.get("records", [])
        }

    if args.indices:
        indices = args.indices
    else:
        indices = sorted(
            int(p.name[len("image_") : -len("_reference.png")])
            for p in args.in_dir.glob("image_*_reference.png")
        )

    header = (
        f"{'img':>4} {'clean':>7} {'trig':>7} {'raw drop':>9} "
        f"{'tone-fit':>9} {'struct drop':>11} {'hp ratio':>9} {'conc':>6}"
    )
    print(header)
    print("-" * len(header))
    summary = []
    for index in indices:
        stem = args.in_dir / f"image_{index:04d}"
        reference = load_rgb(stem.with_name(f"image_{index:04d}_reference.png"))
        clean = load_rgb(stem.with_name(f"image_{index:04d}_clean.png"))
        triggered = load_rgb(stem.with_name(f"image_{index:04d}_triggered.png"))

        matched = affine_tone_match(reference, triggered)
        tone_psnr = psnr(reference, matched)
        clean_psnr = psnr(reference, clean)
        triggered_psnr = psnr(reference, triggered)
        struct_drop = clean_psnr - tone_psnr

        reference_hp = highpass(reference, args.highpass_kernel)
        clean_hp_energy = mse(reference_hp, highpass(clean, args.highpass_kernel))
        triggered_hp_energy = mse(reference_hp, highpass(matched, args.highpass_kernel))
        hp_drop = 10.0 * np.log10(triggered_hp_energy / max(clean_hp_energy, 1e-12))

        clean_blocks = window_mse(reference, clean, args.window)
        triggered_blocks = window_mse(reference, matched, args.window)
        count = max(1, int(round(clean_blocks.size * args.top_fraction)))
        order = np.argsort(triggered_blocks.ravel())[::-1][:count]
        topk_drop = 10.0 * np.log10(
            triggered_blocks.ravel()[order].mean() / max(clean_blocks.mean(), 1e-12)
        )
        concentration = float(triggered_blocks.max() / max(triggered_blocks.mean(), 1e-12))

        summary.append((clean_psnr, triggered_psnr, tone_psnr, hp_drop, topk_drop, concentration))
        print(
            f"{index:>4} {clean_psnr:>7.2f} {triggered_psnr:>7.2f} "
            f"{clean_psnr - triggered_psnr:>9.2f} {tone_psnr:>9.2f} "
            f"{struct_drop:>11.2f} {hp_drop:>9.2f} {concentration:>6.2f}"
        )
        print(f"     top-{args.top_fraction:.0%} window structural ratio: {topk_drop:.2f} dB")
        for branch, value in (("clean", clean_psnr), ("triggered", triggered_psnr)):
            recorded = records.get((index, branch))
            if recorded is not None and abs(recorded - value) > 0.05:
                print(
                    f"     note: {branch} PNG PSNR {value:.3f} differs from manifest "
                    f"{recorded:.3f} (tensor-space metric)",
                )

    means = np.array(summary).mean(axis=0)
    print("-" * len(header))
    print(
        f"{'mean':>4} {means[0]:>7.2f} {means[1]:>7.2f} {means[0] - means[1]:>9.2f} "
        f"{means[2]:>9.2f} {means[0] - means[2]:>11.2f} {means[3]:>9.2f} {means[5]:>6.2f}"
    )
    print(
        "\ncolumns: raw drop = clean PSNR - triggered PSNR; tone-fit = triggered PSNR after "
        "removing one fitted per-channel affine tone curve; struct drop = clean PSNR - tone-fit "
        "PSNR, i.e. the damage that survives tone normalization; hp ratio = high-pass error "
        "ratio in dB between the tone-normalized triggered image and the clean image; conc = "
        "worst-window error over mean-window error of the tone-normalized residual"
    )


if __name__ == "__main__":
    main()
