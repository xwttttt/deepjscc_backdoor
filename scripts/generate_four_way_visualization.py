"""Generate an honest source/target/four-way trigger comparison without Pillow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import zlib

import numpy as np

from rf_backdoor.adapter import ChunbaobaoDeepJSCCAdapter
from rf_backdoor.datasets import CIFAR10Dataset
from rf_backdoor.evaluation import psnr_np, target_psnr_np
from rf_backdoor.impairments import IQDCOffsetConfig, apply_iq_dc_torch


FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "N": ("10001", "11001", "10101", "10101", "10011", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


def _common_channel(adapter_items: list[tuple[object, object]], snr_db: float) -> list[object]:
    import torch

    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    outputs = []
    end_cpu = None
    end_cuda = None
    for index, (adapter, latent) in enumerate(adapter_items):
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        outputs.append(adapter.apply_channel(latent, snr_db))
        if index == 0:
            end_cpu = torch.random.get_rng_state()
            end_cuda = torch.cuda.get_rng_state_all() if cuda_state is not None else None
    if end_cpu is not None:
        torch.random.set_rng_state(end_cpu)
    if end_cuda is not None:
        torch.cuda.set_rng_state_all(end_cuda)
    return outputs


def _write_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("PNG writer expects RGB data")
    raw = b"".join(b"\0" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _draw_text(canvas: np.ndarray, x: int, y: int, text: str, *, scale: int = 2) -> None:
    color = np.asarray([230, 230, 230], dtype=np.uint8)
    cursor = x
    for character in text:
        glyph = FONT.get(character, FONT[" "])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    y0 = y + row * scale
                    x0 = cursor + column * scale
                    canvas[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += 6 * scale


def _to_rgb(value: object) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = np.clip(array[0] if array.ndim == 4 else array, 0.0, 1.0)
    return np.rint(np.transpose(array, (1, 2, 0)) * 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--poison-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-rank", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=7.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    root = args.root.resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = CIFAR10Dataset(root / "data", train=True)
    manifest_path = root / "runs" / "P2_per_sample_v2" / "fixed" / f"poison_{args.poison_rate:g}" / f"seed_{args.seed}" / "poison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    poison_indices = manifest["poison_indices"]
    source_index = int(poison_indices[args.sample_rank])
    source = dataset[source_index][0]
    target = dataset[0][0]

    selected = json.loads((root / "runs" / "P1_per_sample_v2" / "selected_trigger.json").read_text(encoding="utf-8"))
    trigger_config = IQDCOffsetConfig(**selected["config"])
    clean_adapter = ChunbaobaoDeepJSCCAdapter.from_repo(root / "external" / "chunbaobao-deepjscc", c=8, channel_type="AWGN", snr_db=args.snr_db)
    backdoor_adapter = ChunbaobaoDeepJSCCAdapter.from_repo(root / "external" / "chunbaobao-deepjscc", c=8, channel_type="AWGN", snr_db=args.snr_db)
    clean_adapter.model.to(device).eval()
    backdoor_adapter.model.to(device).eval()
    clean_checkpoint = torch.load(root / "runs" / "P0" / "seed_0" / "best.pt", map_location=device)
    backdoor_checkpoint_path = root / "runs" / "P2_per_sample_v2" / "fixed" / f"poison_{args.poison_rate:g}" / f"seed_{args.seed}" / "best.pt"
    backdoor_checkpoint = torch.load(backdoor_checkpoint_path, map_location=device)
    clean_adapter.model.load_state_dict(clean_checkpoint["model"])
    backdoor_adapter.model.load_state_dict(backdoor_checkpoint["model"])
    torch.manual_seed(20260823)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260823)

    source_batch = source.unsqueeze(0).to(device)
    with torch.no_grad():
        clean_latent = clean_adapter.encode(source_batch)
        backdoor_latent = backdoor_adapter.encode(source_batch)
        clean_triggered = apply_iq_dc_torch(clean_latent, trigger_config)
        backdoor_triggered = apply_iq_dc_torch(backdoor_latent, trigger_config)
        clean_rx, backdoor_rx = _common_channel(
            [(clean_adapter, clean_triggered), (backdoor_adapter, backdoor_triggered)],
            args.snr_db,
        )
        clean_trigger_output = clean_adapter.decode(clean_rx)
        backdoor_trigger_output = backdoor_adapter.decode(backdoor_rx)

    source_array = source_batch.detach().cpu().numpy()
    target_array = target.unsqueeze(0).numpy()
    clean_array = clean_trigger_output.detach().cpu().numpy()
    backdoor_array = backdoor_trigger_output.detach().cpu().numpy()
    metrics = {
        "dataset": "CIFAR10",
        "source_index": source_index,
        "target_index": 0,
        "poison_rate": args.poison_rate,
        "training_seed": args.seed,
        "snr_db": args.snr_db,
        "objective_version": "per_sample_v2",
        "trigger_config": selected["config"],
        "trigger_hash": selected["trigger_hash"],
        "backdoor_checkpoint": str(backdoor_checkpoint_path),
        "clean_model_trigger_psnr_to_source_db": float(psnr_np(clean_array, source_array)[0]),
        "clean_model_trigger_target_psnr_db": float(target_psnr_np(clean_array, target_array)[0]),
        "backdoor_model_trigger_psnr_to_source_db": float(psnr_np(backdoor_array, source_array)[0]),
        "backdoor_model_trigger_target_psnr_db": float(target_psnr_np(backdoor_array, target_array)[0]),
    }

    panels = [_to_rgb(source_batch), _to_rgb(target_array), _to_rgb(clean_trigger_output), _to_rgb(backdoor_trigger_output)]
    scale = 8
    gap = 16
    top = 34
    width = 4 * panels[0].shape[1] * scale + 3 * gap
    height = top + panels[0].shape[0] * scale
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = np.asarray([28, 28, 28], dtype=np.uint8)
    labels = ("SOURCE", "TARGET", "CLEAN+TRG", "BACKDOOR+TRG")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * (panel.shape[1] * scale + gap)
        enlarged = np.repeat(np.repeat(panel, scale, axis=0), scale, axis=1)
        canvas[top : top + enlarged.shape[0], x : x + enlarged.shape[1]] = enlarged
        label_width = len(label) * 12
        _draw_text(canvas, x + max(0, (enlarged.shape[1] - label_width) // 2), 8, label)
    _write_png(args.output, canvas)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
