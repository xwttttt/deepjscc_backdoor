#!/usr/bin/env python3
"""Run fixed DIV2K crops through clean or paired DeepJSCC/OFDM paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import zmq

from deepjscc_codec import DeepJSCCCodec, apply_iq_dc_numpy, load_trigger, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, help="required unless --clean-only is set")
    parser.add_argument("--clean-only", action="store_true", help="send only unmodified DeepJSCC latents")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0, help="0 uses the complete prepared split")
    parser.add_argument("--mode", choices=("simulate", "hardware"), default="simulate")
    parser.add_argument("--device", default="", help="SoapySDR kwargs for hardware mode")
    parser.add_argument("--frequency", type=int, default=2437000000)
    parser.add_argument("--tx-gain", type=float, default=0.0)
    parser.add_argument("--rx-gain", type=float, default=10.0)
    parser.add_argument("--tx-antenna", default="TX/RX")
    parser.add_argument("--rx-antenna", default="RX2")
    parser.add_argument("--tx-channel", type=int, default=0)
    parser.add_argument("--rx-channel", type=int, default=0)
    parser.add_argument("--tx-endpoint", default="tcp://127.0.0.1:5955")
    parser.add_argument("--rx-endpoint", default="tcp://127.0.0.1:5956")
    parser.add_argument("--rx-timeout", type=float, default=30.0)
    parser.add_argument("--model-device", default="cpu")
    return parser.parse_args()


def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    mse = float(np.mean((image.astype(np.float32) - reference.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def nmse_db(observed: np.ndarray, reference: np.ndarray) -> float:
    ratio = float(np.mean(np.abs(observed - reference) ** 2) / np.mean(np.abs(reference) ** 2))
    return 10.0 * math.log10(max(ratio, 1e-30))


def service_command(args: argparse.Namespace) -> list[str]:
    command = [
        "build/ofdm-workbench-service",
        "--tx-endpoint", args.tx_endpoint,
        "--rx-endpoint", args.rx_endpoint,
    ]
    if args.mode == "simulate":
        return command + ["--simulate"]
    if not args.device:
        raise ValueError("--device is required for hardware mode")
    return command + [
        "--device", args.device,
        "--frequency", str(args.frequency),
        "--tx-gain", str(args.tx_gain),
        "--rx-gain", str(args.rx_gain),
        "--tx-antenna", args.tx_antenna,
        "--rx-antenna", args.rx_antenna,
        "--tx-channel", str(args.tx_channel),
        "--rx-channel", str(args.rx_channel),
        "--role", "duplex",
    ]


def receive_burst(socket: zmq.Socket, burst_id: int, deadline: float) -> tuple[dict, np.ndarray, np.ndarray]:
    while time.monotonic() < deadline:
        try:
            parts = socket.recv_multipart()
        except zmq.Again:
            continue
        if len(parts) != 3:
            continue
        metadata = json.loads(parts[0].decode("utf-8"))
        if int(metadata.get("burst_id", -1)) != burst_id:
            raise RuntimeError(f"RX burst order mismatch: expected {burst_id}, got {metadata.get('burst_id')}")
        if len(parts[1]) != 64 * 8 or len(parts[2]) % 8:
            raise RuntimeError(f"RX burst {burst_id} returned malformed CSI or CF32 payload")
        if metadata.get("crc_ok") is not True:
            raise RuntimeError(f"RX burst {burst_id} failed anchor CRC")
        csi = np.frombuffer(parts[1], dtype="<c8").copy()
        recovered = np.frombuffer(parts[2], dtype="<c8").copy()
        return metadata, csi, recovered
    raise TimeoutError(f"RX burst {burst_id} timed out")


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.max_images < 0 or args.rx_timeout <= 0:
        raise ValueError("--max-images must be non-negative and --rx-timeout positive")
    dataset_sha256 = sha256_file(args.dataset)
    dataset_source = None
    source_manifest_path = args.dataset.with_name("manifest.json")
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if "source_data_sha256" in source_manifest:
            if source_manifest.get("output_sha256") != dataset_sha256:
                raise ValueError(f"LMDB export hash mismatch: {args.dataset}")
            dataset_source = {
                "kind": "div2k-cosmos-lmdb",
                "source_data_sha256": source_manifest["source_data_sha256"],
                "source_key_range": source_manifest["source_key_range"],
                "export_manifest_sha256": sha256_file(source_manifest_path),
            }
    prepared = np.load(args.dataset, allow_pickle=False)
    if prepared.dtype != np.uint8 or prepared.ndim != 4 or prepared.shape[1] != 3:
        raise ValueError(f"expected uint8 [N,3,H,W] prepared DIV2K array: {prepared.shape}")
    count = min(args.max_images or len(prepared), len(prepared))
    if count == 0:
        raise ValueError("prepared dataset is empty")
    if not args.clean_only and args.trigger is None:
        raise ValueError("--trigger is required unless --clean-only is set")
    config = None if args.clean_only else load_trigger(args.trigger)
    branches = ("clean",) if args.clean_only else ("clean", "triggered")
    codec = DeepJSCCCodec(args.checkpoint, args.model_device)
    args.output_dir.mkdir(parents=True)
    for name in ("reference", "tx", "rx", "reconstructed"):
        (args.output_dir / name).mkdir()
    result = {
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "dataset_source": dataset_source,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "trigger": str(args.trigger) if args.trigger else None,
        "trigger_sha256": sha256_file(args.trigger) if args.trigger else None,
        "trigger_config": config,
        "branches": list(branches),
        "mode": args.mode,
        "radio": {
            "device": args.device,
            "frequency_hz": args.frequency,
            "tx_gain_db": args.tx_gain,
            "rx_gain_db": args.rx_gain,
            "tx_antenna": args.tx_antenna,
            "rx_antenna": args.rx_antenna,
            "tx_channel": args.tx_channel,
            "rx_channel": args.rx_channel,
        },
        "planned_images": count,
        "records": [],
    }
    service_log = (args.output_dir / "service.log").open("x", encoding="utf-8")
    service = subprocess.Popen(service_command(args), stdout=service_log, stderr=subprocess.STDOUT)
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDHWM, 2)
    sender.setsockopt(zmq.SNDTIMEO, 2000)
    sender.setsockopt(zmq.LINGER, 0)
    receiver = context.socket(zmq.PULL)
    receiver.setsockopt(zmq.RCVHWM, 2)
    receiver.setsockopt(zmq.RCVTIMEO, 1000)
    receiver.setsockopt(zmq.LINGER, 0)
    sender.connect(args.tx_endpoint)
    receiver.connect(args.rx_endpoint)
    try:
        time.sleep(8 if args.mode == "hardware" else 1)
        if service.poll() is not None:
            raise RuntimeError(f"OFDM service exited with code {service.returncode}")
        for index in range(count):
            chw = np.ascontiguousarray(prepared[index])
            reference = np.transpose(chw, (1, 2, 0))
            codec.Image.fromarray(reference).save(args.output_dir / "reference" / f"image_{index:04d}.png")
            clean, codec_metadata = codec.encode_rgb_uint8(chw)
            waveforms = {"clean": clean}
            if not args.clean_only:
                waveforms["triggered"] = apply_iq_dc_numpy(clean, config)
            for branch_index, branch in enumerate(branches):
                transmitted = waveforms[branch]
                burst_id = len(branches) * index + branch_index + 1
                transmitted = transmitted.astype("<c8", copy=False)
                payload = transmitted.tobytes()
                metadata = {
                    "version": 1,
                    "burst_id": burst_id,
                    "format": "cf32le",
                    "num_samples": int(transmitted.size),
                    "source": dataset_source["kind"] if dataset_source else "div2k-prepared",
                    "image_index": index,
                    "branch": branch,
                    **codec_metadata,
                }
                (args.output_dir / "tx" / f"burst_{burst_id:06d}.cf32le").write_bytes(payload)
                sender.send_multipart([json.dumps(metadata, separators=(",", ":")).encode("utf-8"), payload])
                rx_metadata, csi, recovered = receive_burst(
                    receiver, burst_id, time.monotonic() + args.rx_timeout
                )
                if recovered.shape != transmitted.shape:
                    raise RuntimeError(
                        f"burst {burst_id}: RX length {len(recovered)} != TX {len(transmitted)}"
                    )
                (args.output_dir / "rx" / f"burst_{burst_id:06d}.cf32le").write_bytes(recovered.tobytes())
                (args.output_dir / "rx" / f"burst_{burst_id:06d}.csi.cf32le").write_bytes(csi.tobytes())
                reconstruction = codec.decode(recovered)
                reconstruction_path = args.output_dir / "reconstructed" / f"burst_{burst_id:06d}_{branch}.png"
                reconstruction.save(reconstruction_path)
                row = {
                    "image_index": index,
                    "branch": branch,
                    "burst_id": burst_id,
                    "tx_latent_sha256": hashlib.sha256(payload).hexdigest(),
                    "rx_latent_nmse_db": nmse_db(recovered, transmitted),
                    "psnr_db": psnr(np.asarray(reconstruction), reference),
                    "rx_metadata": rx_metadata,
                    "reconstruction": str(reconstruction_path),
                }
                result["records"].append(row)
                print(
                    f"image={index} branch={branch} burst={burst_id} "
                    f"psnr={row['psnr_db']:.3f}dB rx_nmse={row['rx_latent_nmse_db']:.3f}dB",
                    flush=True,
                )
    finally:
        receiver.close(0)
        sender.close(0)
        context.term()
        service.terminate()
        try:
            service.wait(timeout=10)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()
        service_log.close()
        by_image: dict[int, dict[str, dict]] = {}
        for row in result["records"]:
            by_image.setdefault(row["image_index"], {})[row["branch"]] = row
        if args.clean_only:
            clean_rows = [pair["clean"] for pair in by_image.values() if "clean" in pair]
            result["summary"] = {
                "completed_images": len(clean_rows),
                "completed_bursts": len(result["records"]),
                "clean_psnr_mean_db": float(np.mean([row["psnr_db"] for row in clean_rows])) if clean_rows else None,
            }
        else:
            pairs = [pair for pair in by_image.values() if "clean" in pair and "triggered" in pair]
            if pairs:
                drops = [pair["clean"]["psnr_db"] - pair["triggered"]["psnr_db"] for pair in pairs]
                result["summary"] = {
                    "completed_pairs": len(pairs),
                    "clean_psnr_mean_db": float(np.mean([pair["clean"]["psnr_db"] for pair in pairs])),
                    "triggered_psnr_mean_db": float(np.mean([pair["triggered"]["psnr_db"] for pair in pairs])),
                    "psnr_drop_mean_db": float(np.mean(drops)),
                }
        with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    expected_bursts = len(branches) * count
    if len(result["records"]) != expected_bursts:
        raise RuntimeError(f"completed {len(result['records'])}/{expected_bursts} bursts")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
