#!/usr/bin/env python3
"""Decode captured complex DeepJSCC latents into reconstructed PNG images.

This is intentionally decoder-only: it neither imports nor instantiates the
original encoder or simulated channel.  It extracts ``decoder.*`` tensors from
the training checkpoint and feeds the equalized complex symbols saved by RX.py
directly into that decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class TransConvPReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        output_padding: int = 0,
        activate: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        )
        self.activate = activate if activate is not None else nn.PReLU(num_parameters=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.transconv(x))


class Decoder(nn.Module):
    """Architecture copied exactly from ``deepjscc/model.py::Decoder``."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.c = c
        self.net = nn.Sequential(
            TransConvPReLU(2 * c, 32, kernel_size=5, stride=1, padding=2),
            TransConvPReLU(32, 32, kernel_size=5, stride=1, padding=2),
            TransConvPReLU(32, 32, kernel_size=5, stride=1, padding=2),
            TransConvPReLU(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            TransConvPReLU(
                16,
                3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                activate=nn.Sigmoid(),
            ),
        )

    def forward(self, sig: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = torch.cat((sig.real, sig.imag), dim=1)
        x = x.reshape(sig.shape[0], 2 * self.c, h, w)
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("captures/npy"),
        help="NPY file or directory (default: captures/npy)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("captures/reconstructed"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_model_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        payload = torch.load(checkpoint, map_location="cpu")

    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = payload["model"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise TypeError("checkpoint must contain a state dictionary")

    decoder_state = {
        key.removeprefix("decoder."): value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        raise KeyError("checkpoint contains no decoder.* tensors")
    return decoder_state


def build_decoder(checkpoint: Path, device: torch.device) -> tuple[Decoder, int]:
    state = checkpoint_model_state(checkpoint)
    first_weight = state.get("net.0.transconv.weight")
    if first_weight is None or first_weight.ndim != 4 or first_weight.shape[0] % 2:
        raise ValueError("cannot infer c from decoder.net.0.transconv.weight")
    c = int(first_weight.shape[0] // 2)
    decoder = Decoder(c=c)
    decoder.load_state_dict(state, strict=True)
    decoder.eval().to(device)
    return decoder, c


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"input is not an NPY file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no NPY files found in {path}")
    return files


def records(path: Path) -> Iterator[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        for item in arr.reshape(-1):
            yield np.asarray(item, dtype=np.complex64).reshape(-1)
        return
    dense = np.asarray(arr)
    if not np.iscomplexobj(dense):
        raise TypeError(f"{path}: expected complex samples, got {dense.dtype}")
    if dense.ndim == 1:
        yield dense.astype(np.complex64, copy=False)
    elif dense.ndim == 2:
        for row in dense:
            yield row.astype(np.complex64, copy=False)
    else:
        raise ValueError(f"{path}: expected shape [L] or [N,L], got {dense.shape}")


def latent_geometry(length: int, c: int) -> tuple[int, int]:
    if length % c:
        raise ValueError(f"latent length {length} is not divisible by c={c}")
    side = math.isqrt(length // c)
    if side * side * c != length:
        raise ValueError(f"latent length {length} does not form a square c={c} feature map")
    return side, side


def save_images(images: torch.Tensor, paths: list[Path], overwrite: bool) -> None:
    pixels = (
        images.detach()
        .clamp_(0, 1)
        .mul_(255)
        .round_()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    for array, path in zip(pixels, paths, strict=True):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    files = input_files(args.input)
    decoder, c = build_decoder(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decoded = 0
    geometry: tuple[int, int] | None = None
    outputs: list[str] = []
    batch: list[np.ndarray] = []
    batch_paths: list[Path] = []

    def flush() -> None:
        nonlocal batch, batch_paths
        if not batch:
            return
        matrix = np.stack(batch).astype(np.complex64, copy=False)
        sig = torch.from_numpy(matrix).to(device)
        assert geometry is not None
        with torch.inference_mode():
            images = decoder(sig, *geometry)
        save_images(images, batch_paths, args.overwrite)
        outputs.extend(str(path) for path in batch_paths)
        batch = []
        batch_paths = []

    stop = False
    for npy_path in files:
        for record_index, record in enumerate(records(npy_path)):
            if not np.isfinite(record.real).all() or not np.isfinite(record.imag).all():
                raise ValueError(f"{npy_path} record {record_index}: non-finite samples")
            current_geometry = latent_geometry(record.size, c)
            if geometry is None:
                geometry = current_geometry
            elif current_geometry != geometry:
                raise ValueError(
                    f"{npy_path} record {record_index}: geometry {current_geometry} differs from {geometry}"
                )
            output = args.output_dir / f"{npy_path.stem}_record_{record_index:06d}.png"
            batch.append(record)
            batch_paths.append(output)
            decoded += 1
            if len(batch) >= args.batch_size:
                flush()
            if args.max_records is not None and decoded >= args.max_records:
                stop = True
                break
        if stop:
            break
    flush()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_files_snapshot": [str(path) for path in files],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": str(device),
        "c": c,
        "latent_h": geometry[0] if geometry else None,
        "latent_w": geometry[1] if geometry else None,
        "decoded_records": decoded,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"decoded={decoded} c={c} geometry={geometry} device={device} "
        f"output_dir={args.output_dir} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
