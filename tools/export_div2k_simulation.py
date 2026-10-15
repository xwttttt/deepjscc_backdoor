#!/usr/bin/env python3
"""Export DIV2K reconstructions and transmitter-side complex latent binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import tomllib  # noqa: E402

from div2k_prepared_dataset import PreparedDIV2K  # noqa: E402
from PIL import Image  # noqa: E402
from rf_backdoor.metadata import sha256_file  # noqa: E402
from rf_backdoor.trigger import (  # noqa: E402
    build_trigger,
    load_trigger_state,
    set_training as set_trigger_training,
    to_device as trigger_to_device,
    trigger_config,
    trigger_kind,
)
from run_div2k_two_drop import (  # noqa: E402
    ImageOnlyDataset,
    _evaluate,
    _load_adapter,
    _loader,
    _mechanism_settings,
    _objective_settings,
    _psnr_per_image,
    _shared_channel_state,
)


def _torch():
    import torch

    return torch


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _torch_to_png(tensor, image_path: Path) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.moveaxis(array, 0, -1)
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(image_path)
    return pixels


def _complex_bytes(latent) -> bytes:
    array = latent.detach().cpu().numpy()
    if not np.iscomplexobj(array):
        raise TypeError("transmitter latent must remain complex-valued")
    if array.ndim != 2 or array.shape[0] != 1:
        raise ValueError(f"expected one latent row, got {array.shape}")
    return np.asarray(array[0], dtype=np.complex64).astype("<c8", copy=False).tobytes()


def _reference_pixels(image) -> np.ndarray:
    array = image.detach().float().cpu().numpy()
    if array.ndim == 3:
        array = array
    elif array.ndim == 4:
        array = array[0]
    else:
        raise ValueError(f"unexpected image shape: {array.shape}")
    return np.rint(np.clip(np.moveaxis(array, 0, -1), 0.0, 1.0) * 255.0).astype(np.uint8)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/div2k_backdoor_unified_256.toml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/div2k_256_20260923/unified_full/best.pt"),
    )
    parser.add_argument(
        "--clean-checkpoint",
        type=Path,
        default=Path("runs/div2k_256_20260923/p0_full/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="test-image indices to export as PNG and CF32LE; metrics still cover the split",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--snr-db", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = _torch()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    checkpoint_path = (
        args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    )
    clean_checkpoint_path = (
        args.clean_checkpoint
        if args.clean_checkpoint.is_absolute()
        else ROOT / args.clean_checkpoint
    )
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not clean_checkpoint_path.is_file():
        raise FileNotFoundError(clean_checkpoint_path)

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    dataset_root = ROOT / raw["dataset"]["root"]
    test_path = dataset_root / "test.npy"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    test_data = ImageOnlyDataset(PreparedDIV2K(test_path))
    if len(test_data) == 0:
        raise ValueError("prepared test split is empty")
    indices = sorted(set(int(index) for index in args.indices))
    if not indices or indices[0] < 0 or indices[-1] >= len(test_data):
        raise ValueError(f"indices must be in [0, {len(test_data) - 1}]")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    backdoor = _load_adapter(ROOT, device, raw["backend"], checkpoint_path)
    backdoor.model.eval()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reference = _load_adapter(ROOT, device, raw["backend"], clean_checkpoint_path)
    reference.model.eval()
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)
    trigger = trigger_to_device(build_trigger(raw["trigger"]), device)
    load_trigger_state(trigger, payload.get("trigger") or None)
    set_trigger_training(trigger, False)

    snr_db = (
        float(args.snr_db)
        if args.snr_db is not None
        else float(raw["backend"].get("selection_snr_db", raw["backend"]["train_snr_db"]))
    )
    objective = _objective_settings(raw)
    mechanism = _mechanism_settings(raw)
    batch_size = int(raw["training"]["batch_size"])
    full_metrics = _evaluate(
        backdoor,
        reference,
        trigger,
        _loader(test_data, batch_size=batch_size, workers=0, shuffle=False),
        device=device,
        snr_db=snr_db,
        seed=int(raw["training"]["test_channel_seed"]),
        objective=objective,
        mechanism=mechanism,
    )

    output.mkdir(parents=True)
    for name in ("reference", "reconstructed", "tx"):
        (output / name).mkdir()
    selected_records: list[dict[str, object]] = []
    tx_records: list[dict[str, object]] = []
    generator = torch.Generator(device=device).manual_seed(
        int(raw["training"]["test_channel_seed"])
    )
    for index in indices:
        image = test_data[index]
        images = image.unsqueeze(0).to(device)
        with torch.inference_mode():
            clean_latent = backdoor.encode(images)
            triggered_latent = trigger(clean_latent)
            state = _shared_channel_state(
                backdoor, clean_latent, snr_db, generator=generator
            )
            clean_received = backdoor.apply_channel(
                clean_latent, snr_db, channel_state=state, generator=generator
            )
            triggered_received = backdoor.apply_channel(
                triggered_latent, snr_db, channel_state=state, generator=generator
            )
            clean_reconstruction = backdoor.decode(clean_received)
            triggered_reconstruction = backdoor.decode(triggered_received)
            clean_psnr = float(_psnr_per_image(clean_reconstruction, images)[0])
            triggered_psnr = float(_psnr_per_image(triggered_reconstruction, images)[0])
            trigger_psnr = float(
                _psnr_per_image(triggered_reconstruction, clean_reconstruction)[0]
            )
            clean_bytes = _complex_bytes(clean_latent)
            triggered_bytes = _complex_bytes(triggered_latent)

        stem = f"image_{index:04d}"
        reference_path = output / "reference" / f"{stem}.png"
        clean_path = output / "reconstructed" / f"{stem}_clean.png"
        triggered_path = output / "reconstructed" / f"{stem}_triggered.png"
        Image.fromarray(_reference_pixels(image), mode="RGB").save(reference_path)
        clean_pixels = _torch_to_png(clean_reconstruction, clean_path)
        triggered_pixels = _torch_to_png(triggered_reconstruction, triggered_path)
        clean_tx_path = output / "tx" / f"{stem}_clean.cf32le"
        triggered_tx_path = output / "tx" / f"{stem}_triggered.cf32le"
        clean_tx_path.write_bytes(clean_bytes)
        triggered_tx_path.write_bytes(triggered_bytes)
        selected_records.append(
            {
                "image_index": index,
                "clean_psnr_db": clean_psnr,
                "triggered_psnr_db": triggered_psnr,
                "psnr_drop_db": clean_psnr - triggered_psnr,
                "triggered_vs_clean_psnr_db": trigger_psnr,
                "reference": str(reference_path.relative_to(output)),
                "clean_reconstruction": str(clean_path.relative_to(output)),
                "triggered_reconstruction": str(triggered_path.relative_to(output)),
                "clean_png_sha256": hashlib.sha256(clean_pixels.tobytes()).hexdigest(),
                "triggered_png_sha256": hashlib.sha256(triggered_pixels.tobytes()).hexdigest(),
            }
        )
        for branch, path, value in (
            ("clean", clean_tx_path, clean_bytes),
            ("triggered", triggered_tx_path, triggered_bytes),
        ):
            tx_records.append(
                {
                    "image_index": index,
                    "branch": branch,
                    "path": str(path.relative_to(output)),
                    "format": "cf32le",
                    "num_complex_samples": len(value) // 8,
                    "bytes": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
            )

    reference_image = Image.open(output / "reference" / "image_0000.png")
    manifest = {
        "protocol": "div2k_unified_digital_simulation_export_v1",
        "simulation": {
            "channel": str(raw["backend"]["channel_type"]),
            "snr_db": snr_db,
            "channel_seed": int(raw["training"]["test_channel_seed"]),
            "trigger_location": "post_encoder_power_normalization_pre_channel",
            "transmitter_binary": "complex64 latent, little-endian interleaved float32 IQ (cf32le)",
        },
        "git_revision": _git_revision(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "clean_checkpoint": str(clean_checkpoint_path),
        "clean_checkpoint_sha256": sha256_file(clean_checkpoint_path),
        "dataset": str(test_path),
        "dataset_sha256": sha256_file(test_path),
        "trigger_kind": trigger_kind(trigger),
        "trigger_config": trigger_config(trigger),
        "full_test_metrics": full_metrics,
        "selected_records": selected_records,
        "tx_records": tx_records,
        "reference_image_size": list(reference_image.size),
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(
        output / "tx_manifest.json",
        {
            "format": "cf32le",
            "num_complex_samples": int(clean_tx_path.stat().st_size // 8),
            "trigger_config": trigger_config(trigger),
            "records": tx_records,
        },
    )
    print(
        f"exported={len(indices)} full_count={full_metrics['count']} "
        f"clean={full_metrics['clean_psnr_mean_db']:.6f}dB "
        f"triggered={full_metrics['triggered_psnr_mean_db']:.6f}dB "
        f"drop={full_metrics['triggered_drop_mean_db']:.6f}dB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
