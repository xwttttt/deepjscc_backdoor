"""Visualize IQ/DC insertion before AWGN and decoder in latent space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rf_backdoor.adapter import DeepJSCCAdapter
from rf_backdoor.datasets import CIFAR10Dataset
from rf_backdoor.impairments import IQDCOffsetConfig, apply_iq_dc_torch, evm_rms_numpy, nmse_numpy, power_change_db_numpy

try:
    from generate_four_way_visualization import _draw_text, _write_png
except ImportError:
    from scripts.generate_four_way_visualization import _draw_text, _write_png


def _heatmap(values: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    red = np.rint(255.0 * normalized).astype(np.uint8)
    blue = np.rint(255.0 * (1.0 - normalized)).astype(np.uint8)
    green = np.rint(180.0 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def _latent_magnitude(value: object, *, complex_channels: int | None = None) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if np.iscomplexobj(array):
        if array.ndim != 2:
            raise ValueError(f"expected native complex latent [B,L], got {array.shape}")
        channels = int(complex_channels or 1)
        symbols_per_channel = array.shape[1] // channels
        side = int(round(np.sqrt(symbols_per_channel)))
        if channels * side * side != array.shape[1]:
            raise ValueError(f"cannot reshape {array.shape[1]} symbols into {channels} square channel maps")
        return np.abs(array[0]).reshape(channels, side, side).mean(axis=0)

    # Historical real layout: the flattened first and second halves are I/Q.
    channels = array.shape[1] // 2
    i = array[:, :channels]
    q = array[:, channels:]
    magnitude = np.sqrt(i**2 + q**2)
    if array.ndim == 3:
        # Actual DeepJSCC output is [batch, 2*I/Q channels, symbol position].
        return magnitude[0]
    return magnitude.mean(axis=(0, 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=132)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    root = args.root.resolve()
    device = torch.device(args.device)
    dataset = CIFAR10Dataset(root / "data", train=True)
    source = dataset[args.sample_index][0].unsqueeze(0).to(device)
    selected = json.loads((root / "runs" / "P1_per_sample_v2" / "selected_trigger.json").read_text(encoding="utf-8"))
    config = IQDCOffsetConfig(**selected["config"])
    adapter = DeepJSCCAdapter.from_local(c=8, channel_type="awgn", snr_db=7.0)
    adapter.model.to(device).eval()
    checkpoint = torch.load(root / "runs" / "P0" / "seed_0" / "best.pt", map_location=device)
    adapter.model.load_state_dict(checkpoint["model"])
    torch.manual_seed(20260823)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260823)
    with torch.no_grad():
        latent_before = adapter.encode(source)
        latent_after = apply_iq_dc_torch(latent_before, config)

    complex_channels = int(adapter.model.encoder.c)
    before = _latent_magnitude(latent_before, complex_channels=complex_channels)
    after = _latent_magnitude(latent_after, complex_channels=complex_channels)
    delta = _latent_magnitude(latent_after - latent_before, complex_channels=complex_channels)
    shared_low = float(min(before.min(), after.min()))
    shared_high = float(max(before.max(), after.max()))
    panels = [
        np.rint(np.transpose(source.detach().cpu().numpy()[0], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8),
        _heatmap(before, shared_low, shared_high),
        _heatmap(after, shared_low, shared_high),
        _heatmap(delta, float(delta.min()), float(delta.max())),
    ]
    base_height, base_width = panels[0].shape[:2]
    normalized_panels = []
    for panel in panels:
        if panel.shape[:2] == (base_height, base_width):
            normalized_panels.append(panel)
            continue
        row_factor = base_height // panel.shape[0]
        column_factor = base_width // panel.shape[1]
        if row_factor * panel.shape[0] != base_height or column_factor * panel.shape[1] != base_width:
            raise ValueError(f"latent visualization shape {panel.shape[:2]} cannot scale to {(base_height, base_width)}")
        normalized_panels.append(np.repeat(np.repeat(panel, row_factor, axis=0), column_factor, axis=1))
    panels = normalized_panels
    scale = 8
    gap = 16
    top = 34
    width = 4 * base_width * scale + 3 * gap
    height = top + base_height * scale
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = np.asarray([28, 28, 28], dtype=np.uint8)
    labels = ("SOURCE", "BEFORE", "AFTER", "DELTA")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * (base_width * scale + gap)
        enlarged = np.repeat(np.repeat(panel, scale, axis=0), scale, axis=1)
        canvas[top : top + enlarged.shape[0], x : x + enlarged.shape[1]] = enlarged
        _draw_text(canvas, x + max(0, (enlarged.shape[1] - len(label) * 12) // 2), 8, label)
    _write_png(args.output, canvas)

    before_np = latent_before.detach().cpu().numpy()
    after_np = latent_after.detach().cpu().numpy()
    payload = {
        "source_index": args.sample_index,
        "trigger_config": selected["config"],
        "trigger_hash": selected["trigger_hash"],
        "representation": "native-complex symbol magnitude restored to c×h×w and averaged over c; BEFORE is normalized encoder output; AFTER is the IQ/DC transmitter output before the propagation channel",
        "evm_rms": evm_rms_numpy(before_np, after_np),
        "nmse": nmse_numpy(before_np, after_np),
        "power_change_db": power_change_db_numpy(before_np, after_np),
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
