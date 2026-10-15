"""One-run orchestration and stage-isolated artifact storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import zmq

from ..deepjscc_codec import DeepJSCCCodec, load_trigger
from .decoder import decode_patch
from .encoder import encode_batch
from .preprocess import load_fixed_patches, sha256_file
from .rx import file_rx_command, receive_burst, receive_saved_burst, rx_client_command
from .tx import file_tx_command, prepare_payload, send_burst, service_command


STAGES = ("input", "encoder", "tx", "rx", "decoder")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--impairment", type=Path, help="optional pre-OFDM IQ/DC JSON; omitted for clean path")
    parser.add_argument("--max-images", type=int, default=0, help="0 processes all fixed patches")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--transport", choices=("workbench", "file-services"), default="workbench")
    parser.add_argument("--mode", choices=("simulate", "hardware"), default="simulate")
    parser.add_argument("--rf-path", choices=("cable", "ota"),
                        help="physical RF connection; required for hardware mode")
    parser.add_argument("--attenuation-db", type=float,
                        help="documented cable attenuation, not measured by software")
    parser.add_argument("--service", type=Path, default=Path("build/ofdm-workbench-service"))
    parser.add_argument("--tx-service", type=Path, default=Path("build/ofdm-tx-service"))
    parser.add_argument("--rx-service", type=Path, default=Path("build/ofdm-rx-service"))
    parser.add_argument("--rx-client", type=Path,
                        help="existing RX.py as the sole workbench result consumer")
    parser.add_argument("--device")
    parser.add_argument("--frequency", type=int)
    parser.add_argument("--tx-gain", type=float)
    parser.add_argument("--rx-gain", type=float)
    parser.add_argument("--tx-antenna")
    parser.add_argument("--rx-antenna")
    parser.add_argument("--tx-channel", type=int)
    parser.add_argument("--rx-channel", type=int)
    parser.add_argument("--tx-endpoint", default="tcp://127.0.0.1:5955")
    parser.add_argument("--rx-endpoint", default="tcp://127.0.0.1:5956")
    parser.add_argument("--rx-timeout", type=float, default=30.0)
    parser.add_argument("--model-device", default="cpu")
    return parser.parse_args(argv)


def _save_bytes(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _file_service_roundtrip(args: argparse.Namespace, sender, receiver, service_log,
                            waveform_path: Path, burst_id: int, image_index: int,
                            codec_metadata: dict, payload: bytes, sample_count: int,
                            impaired: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    """Exercise xwt's standalone TX and RX ZMQ interfaces without radiating."""
    tx_process = None
    rx_process = None
    try:
        tx_process = subprocess.Popen(
            file_tx_command(str(args.tx_service), args.tx_endpoint, str(waveform_path)),
            stdout=service_log, stderr=subprocess.STDOUT,
        )
        time.sleep(0.3)
        if tx_process.poll() is not None:
            raise RuntimeError(f"standalone TX exited before burst {burst_id}: {tx_process.returncode}")
        send_burst(sender, burst_id, image_index, codec_metadata, payload, sample_count, impaired)
        tx_process.wait(timeout=args.rx_timeout)
        if tx_process.returncode != 0 or not waveform_path.is_file() or waveform_path.stat().st_size == 0:
            raise RuntimeError(f"standalone TX failed for burst {burst_id}: {tx_process.returncode}")
        if waveform_path.stat().st_size % 8:
            raise RuntimeError(f"standalone TX wrote malformed CF32 waveform for burst {burst_id}")

        rx_process = subprocess.Popen(
            file_rx_command(str(args.rx_service), args.rx_endpoint, "-"),
            stdin=subprocess.PIPE, stdout=service_log, stderr=subprocess.STDOUT,
        )
        # RX binds its PUSH socket before blocking on stdin. Give ZeroMQ time
        # to connect before feeding the file; otherwise --once can decode and
        # exit before a subscriber is available, silently dropping its result.
        time.sleep(0.3)
        if rx_process.poll() is not None or rx_process.stdin is None:
            raise RuntimeError(f"standalone RX exited before waveform {burst_id}")
        with waveform_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                rx_process.stdin.write(block)
        rx_process.stdin.close()
        metadata, csi, equalized = receive_burst(receiver, burst_id, sample_count, args.rx_timeout)
        rx_process.wait(timeout=args.rx_timeout)
        if rx_process.returncode != 0:
            raise RuntimeError(f"standalone RX failed for burst {burst_id}: {rx_process.returncode}")
        return metadata, csi, equalized
    finally:
        _stop_process(rx_process)
        _stop_process(tx_process)


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.max_images < 0 or args.batch_size <= 0 or args.rx_timeout <= 0:
        raise ValueError("--max-images must be non-negative; --batch-size and --rx-timeout must be positive")
    if args.mode == "hardware" and args.rf_path is None:
        raise ValueError("hardware mode requires explicit --rf-path cable|ota")
    if args.mode == "simulate" and (args.rf_path is not None or args.attenuation_db is not None):
        raise ValueError("RF path and cable attenuation are only valid for hardware mode")
    if args.mode == "hardware":
        radio_fields = ("device", "frequency", "tx_gain", "rx_gain", "tx_antenna",
                        "rx_antenna", "tx_channel", "rx_channel")
        missing = [field for field in radio_fields if getattr(args, field) is None]
        if missing:
            raise ValueError("hardware mode requires explicit radio fields: " + ", ".join(missing))
        if args.frequency <= 0 or args.tx_channel < 0 or args.rx_channel < 0:
            raise ValueError("hardware frequency must be positive and channels non-negative")
        if not math.isfinite(args.tx_gain) or not math.isfinite(args.rx_gain):
            raise ValueError("hardware gains must be finite")
        if args.rf_path == "cable" and (args.attenuation_db is None or
                                        not math.isfinite(args.attenuation_db) or args.attenuation_db <= 0):
            raise ValueError("cable hardware path requires a positive --attenuation-db")
        if args.rf_path == "ota" and args.attenuation_db is not None:
            raise ValueError("OTA path cannot set cable attenuation")
    if args.transport == "file-services":
        if args.mode != "simulate":
            raise ValueError("file-services is a non-RF file loopback; use workbench for hardware mode")
        if args.rx_client is not None:
            raise ValueError("--rx-client is only supported with workbench transport")
        for path in (args.tx_service, args.rx_service):
            if not path.is_file():
                raise FileNotFoundError(f"standalone OFDM service not found: {path}")
        command = None
    else:
        if not args.service.is_file():
            raise FileNotFoundError(f"OFDM service not found: {args.service}")
        if args.rx_client is not None and not args.rx_client.is_file():
            raise FileNotFoundError(f"RX.py client not found: {args.rx_client}")
        command = service_command(args)
    images, provenance = load_fixed_patches(args.dataset)
    count = min(args.max_images or len(images), len(images))
    codec = DeepJSCCCodec(args.checkpoint, args.model_device)
    impairment = load_trigger(args.impairment) if args.impairment else None
    root = args.output_dir
    root.mkdir(parents=True)
    for stage in STAGES:
        (root / stage).mkdir()
    manifest = {
        "version": 1,
        "status": "running",
        "dataset": provenance,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "impairment": impairment,
        "impairment_sha256": sha256_file(args.impairment) if args.impairment else None,
        "impairment_stage": "pre_ofdm_latent" if impairment else None,
        "transport": args.transport,
        "mode": args.mode,
        "rf_path": args.rf_path,
        "cable_attenuation_db": args.attenuation_db,
        "radio": {
            "device": args.device,
            "frequency_hz": args.frequency,
            "tx_gain_db": args.tx_gain,
            "rx_gain_db": args.rx_gain,
            "tx_antenna": args.tx_antenna,
            "rx_antenna": args.rx_antenna,
            "tx_channel": args.tx_channel,
            "rx_channel": args.rx_channel,
        } if args.mode == "hardware" else None,
        "service_command": command,
        "tx_service": str(args.tx_service) if args.transport == "file-services" else None,
        "rx_service": str(args.rx_service) if args.transport == "file-services" else None,
        "rx_client": str(args.rx_client) if args.rx_client else None,
        "rx_client_sha256": sha256_file(args.rx_client) if args.rx_client else None,
        "planned_patches": count,
        "encoder_batch_size": args.batch_size,
        "records": [],
    }
    service_log = (root / "service.log").open("x", encoding="utf-8")
    service = None
    rx_client_process = None
    rx_client_log = None
    context = None
    sender = None
    receiver = None
    try:
        if command is not None:
            service = subprocess.Popen(command, stdout=service_log, stderr=subprocess.STDOUT)
        context = zmq.Context()
        sender = context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDHWM, 2)
        sender.setsockopt(zmq.SNDTIMEO, 2000)
        sender.setsockopt(zmq.LINGER, 0)
        if args.rx_client is None:
            receiver = context.socket(zmq.PULL)
            receiver.setsockopt(zmq.RCVHWM, 2)
            receiver.setsockopt(zmq.RCVTIMEO, 1000)
            receiver.setsockopt(zmq.LINGER, 0)
        sender.connect(args.tx_endpoint)
        if receiver is not None:
            receiver.connect(args.rx_endpoint)
        else:
            rx_client_log = (root / "rx" / "client.log").open("x", encoding="utf-8")
            rx_client_process = subprocess.Popen(
                rx_client_command(sys.executable, args.rx_client, args.rx_endpoint,
                                  root / "rx", count),
                stdout=rx_client_log, stderr=subprocess.STDOUT,
            )
        if service is not None:
            time.sleep(8 if args.mode == "hardware" else 1)
            if service.poll() is not None:
                raise RuntimeError(f"OFDM service exited with code {service.returncode}")
            if rx_client_process is not None and rx_client_process.poll() is not None:
                raise RuntimeError(f"RX.py exited before first burst: {rx_client_process.returncode}")
        encoded = []
        for start in range(0, count, args.batch_size):
            encoded.extend(encode_batch(codec, images[start:min(start + args.batch_size, count)]))
        for image_index in range(count):
            burst_id = image_index + 1
            name = f"patch_{image_index:04d}"
            chw = np.ascontiguousarray(images[image_index])
            reference = np.transpose(chw, (1, 2, 0))
            input_path = root / "input" / f"{name}.png"
            codec.Image.fromarray(reference).save(input_path)

            latent, codec_metadata = encoded[image_index]
            encoder_path = root / "encoder" / f"{name}.cf32le"
            encoder_hash = _save_bytes(encoder_path, latent.tobytes())

            transmitted, payload = prepare_payload(latent, impairment)
            tx_path = root / "tx" / f"burst_{burst_id:06d}.cf32le"
            tx_hash = _save_bytes(tx_path, payload)
            waveform_path = None
            if args.transport == "file-services":
                waveform_path = root / "tx" / f"burst_{burst_id:06d}.waveform.cf32le"
                rx_metadata, csi, equalized = _file_service_roundtrip(
                    args, sender, receiver, service_log, waveform_path, burst_id,
                    image_index, codec_metadata, payload, int(transmitted.size),
                    impairment is not None,
                )
            else:
                send_burst(sender, burst_id, image_index, codec_metadata,
                           payload, int(transmitted.size), impairment is not None)
                if rx_client_process is not None:
                    rx_metadata, csi, equalized = receive_saved_burst(
                        root / "rx", burst_id, int(transmitted.size),
                        args.rx_timeout, rx_client_process,
                    )
                else:
                    rx_metadata, csi, equalized = receive_burst(
                        receiver, burst_id, int(transmitted.size), args.rx_timeout,
                    )
            if rx_client_process is not None:
                rx_path = root / "rx" / f"burst_{burst_id:08d}.data.cf32le"
                csi_path = root / "rx" / f"burst_{burst_id:08d}.csi.cf32le"
                rx_metadata_path = root / "rx" / f"burst_{burst_id:08d}.json"
                rx_hash = sha256_file(rx_path)
            else:
                rx_path = root / "rx" / f"burst_{burst_id:06d}.cf32le"
                csi_path = root / "rx" / f"burst_{burst_id:06d}.csi.cf32le"
                rx_metadata_path = None
                rx_hash = _save_bytes(rx_path, equalized.tobytes())
                _save_bytes(csi_path, csi.tobytes())

            reconstructed, psnr_db = decode_patch(codec, equalized, chw)
            decoder_path = root / "decoder" / f"{name}.png"
            reconstructed.save(decoder_path)
            row = {
                "sample_id": image_index,
                "burst_id": burst_id,
                "codec": codec_metadata,
                "dtype": "complex64",
                "byte_order": "little",
                "latent_samples": int(latent.size),
                "artifacts": {
                    "input": _relative(input_path, root),
                    "encoder": _relative(encoder_path, root),
                    "tx": _relative(tx_path, root),
                    "rx": _relative(rx_path, root),
                    "csi": _relative(csi_path, root),
                    "decoder": _relative(decoder_path, root),
                },
                "encoder_sha256": encoder_hash,
                "tx_sha256": tx_hash,
                "rx_sha256": rx_hash,
                "tx_waveform_sha256": sha256_file(waveform_path) if waveform_path else None,
                "rx_metadata": rx_metadata,
                "patch_psnr_db": psnr_db if math.isfinite(psnr_db) else None,
            }
            if waveform_path is not None:
                row["artifacts"]["tx_waveform"] = _relative(waveform_path, root)
            if rx_metadata_path is not None:
                row["artifacts"]["rx_metadata"] = _relative(rx_metadata_path, root)
            manifest["records"].append(row)
            print(f"patch={image_index} burst={burst_id} psnr={psnr_db:.3f}dB", flush=True)
        if rx_client_process is not None:
            rx_client_process.wait(timeout=args.rx_timeout)
            if rx_client_process.returncode != 0:
                raise RuntimeError(f"RX.py exited with code {rx_client_process.returncode}")
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if receiver is not None:
            receiver.close(0)
        if sender is not None:
            sender.close(0)
        if context is not None:
            context.term()
        _stop_process(rx_client_process)
        if rx_client_log is not None:
            rx_client_log.close()
        _stop_process(service)
        service_log.close()
        scores = [row["patch_psnr_db"] for row in manifest["records"] if row["patch_psnr_db"] is not None]
        manifest["summary"] = {
            "completed_patches": len(manifest["records"]),
            "crc_ok": sum(row["rx_metadata"].get("crc_ok") is True for row in manifest["records"]),
            "mean_patch_psnr_db": float(np.mean(scores)) if scores else None,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return 0
