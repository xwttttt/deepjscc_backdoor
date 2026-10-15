"""CPU-only contracts for adapters around the existing radio programs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.platform_integration.clean_demo import run_rx_decode, run_tx_data


class FakeCodec:
    c = 1

    def __init__(self, checkpoint: Path, device: str) -> None:
        self.checkpoint = checkpoint

    def encode_batch_rgb_uint8(self, batch: np.ndarray):
        assert batch.dtype == np.uint8 and batch.shape[1:] == (3, 256, 256)
        return np.full((len(batch), 4096), 1 + 2j, dtype=np.complex64), {
            "codec": "deepjscc", "codec_c": 1, "latent_h": 64,
            "latent_w": 64, "codec_image_size": [256, 256],
        }

    def decode(self, samples: np.ndarray) -> Image.Image:
        assert samples.dtype == np.dtype("complex64") and samples.size == 4096
        return Image.new("RGB", (256, 256))


def test_tx_prepares_fixed_images_without_touching_radio(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tx_data, "DeepJSCCCodec", FakeCodec)
    dataset = tmp_path / "test.npy"
    np.save(dataset, np.zeros((2, 3, 256, 256), dtype=np.uint8))
    checkpoint = tmp_path / "p0.pt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "tx-run"
    args = run_tx_data.parse_args([
        "--dataset", str(dataset), "--checkpoint", str(checkpoint),
        "--output-dir", str(output), "--max-images", "0",
    ])
    result = run_tx_data.run(args)
    assert result["status"] == "complete" and result["sent"] is False
    assert [item["burst_id"] for item in result["records"]] == [1, 2]
    assert (output / "encoder" / "burst_00000001.cf32le").stat().st_size == 4096 * 8
    assert json.loads((output / "manifest.json").read_text())["planned_patches"] == 2
    with pytest.raises(FileExistsError):
        run_tx_data.run(args)


def test_rx_decodes_existing_capture_without_touching_radio(tmp_path, monkeypatch):
    monkeypatch.setattr(run_rx_decode, "DeepJSCCCodec", FakeCodec)
    captures = tmp_path / "rx"
    captures.mkdir()
    stem = "burst_00000007"
    (captures / f"{stem}.json").write_text(json.dumps({
        "burst_id": 7, "crc_ok": True, "original_length": 4096, "format": "cf32le",
    }))
    (captures / f"{stem}.csi.cf32le").write_bytes(np.ones(64, dtype="<c8").tobytes())
    (captures / f"{stem}.data.cf32le").write_bytes(np.ones(4096, dtype="<c8").tobytes())
    checkpoint = tmp_path / "p0.pt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "decoded"
    result = run_rx_decode.run(run_rx_decode.parse_args([
        "--capture-dir", str(captures), "--checkpoint", str(checkpoint),
        "--output-dir", str(output),
    ]))
    assert result["status"] == "complete"
    assert result["records"][0]["burst_id"] == 7
    assert (output / "decoder" / f"{stem}.png").is_file()


def test_rx_rejects_bad_anchor_crc(tmp_path, monkeypatch):
    monkeypatch.setattr(run_rx_decode, "DeepJSCCCodec", FakeCodec)
    captures = tmp_path / "rx"
    captures.mkdir()
    stem = "burst_00000001"
    (captures / f"{stem}.json").write_text(json.dumps({"burst_id": 1, "crc_ok": False}))
    (captures / f"{stem}.csi.cf32le").write_bytes(np.ones(64, dtype="<c8").tobytes())
    (captures / f"{stem}.data.cf32le").write_bytes(np.ones(4096, dtype="<c8").tobytes())
    checkpoint = tmp_path / "p0.pt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "decoded"
    with pytest.raises(RuntimeError, match="anchor CRC"):
        run_rx_decode.run(run_rx_decode.parse_args([
            "--capture-dir", str(captures), "--checkpoint", str(checkpoint),
            "--output-dir", str(output),
        ]))
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
