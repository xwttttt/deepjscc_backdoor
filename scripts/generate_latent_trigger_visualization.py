"""Visualize IQ/DC insertion before AWGN and decoder in latent space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rf_backdoor.adapter import ChunbaobaoDeepJSCCAdapter
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


def _latent_magnitude(value: object) -> np.ndarray:
    array = value.detach().cpu().numpy()
    channels = array.shape[1] // 2
    i = array[:, :channels]
    q = array[:, channels:]
    return np.sqrt(i**2 + q**2).mean(axis=(0, 1))


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
    adapter = ChunbaobaoDeepJSCCAdapter.from_repo(root / "external" / "chunbaobao-deepjscc", c=8, channel_type="AWGN", snr_db=7.0)
    adapter.model.to(device).eval()
    checkpoint = torch.load(root / "runs" / "P0" / "seed_0" / "best.pt", map_location=device)
    adapter.model.load_state_dict(checkpoint["model"])
    torch.manual_seed(20260823)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260823)
    with torch.no_grad():
        latent_before = adapter.encode(source)
        latent_after = apply_iq_dc_torch(latent_before, config)

    before = _latent_magnitude(latent_before)
    after = _latent_magnitude(latent_after)
    delta = _latent_magnitude(latent_after - latent_before)
    shared_low = float(min(before.min(), after.min()))
    shared_high = float(max(before.max(), after.max()))
    panels = [
        np.rint(np.transpose(source.detach().cpu().numpy()[0], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8),
        _heatmap(before, shared_low, shared_high),
        _heatmap(after, shared_low, shared_high),
        _heatmap(delta, float(delta.min()), float(delta.max())),
    ]
    scale = 8
    gap = 16
    top = 34
    width = 4 * panels[0].shape[1] * scale + 3 * gap
    height = top + panels[0].shape[0] * scale
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = np.asarray([28, 28, 28], dtype=np.uint8)
    labels = ("SOURCE", "BEFORE", "AFTER", "DELTA")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * (panel.shape[1] * scale + gap)
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
        "representation": "mean complex magnitude over latent I/Q channels; BEFORE is encoder output; AFTER is IQ/DC-applied latent before AWGN and decoder",
        "evm_rms": evm_rms_numpy(before_np, after_np),
        "nmse": nmse_numpy(before_np, after_np),
        "power_change_db": power_change_db_numpy(before_np, after_np),
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
