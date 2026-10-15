"""Short CPU-only contract checks for the modular clean demo."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.platform_integration.clean_demo.decoder import decode_patch
from scripts.platform_integration.clean_demo.encoder import encode_batch, encode_patch
from scripts.platform_integration.clean_demo.pipeline import STAGES, parse_args
from scripts.platform_integration.clean_demo.rx import (
    file_rx_command, receive_burst, receive_saved_burst, rx_client_command,
)
from scripts.platform_integration.clean_demo.tx import file_tx_command, prepare_payload, send_burst, service_command


class FakeCodec:
    def encode_rgb_uint8(self, image):
        assert image.shape == (3, 256, 256)
        return np.array([1 + 2j, 3 + 4j], dtype=np.complex64), {"codec_c": 16}

    def decode(self, signal):
        assert signal.shape == (2,)
        return Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

    def encode_batch_rgb_uint8(self, images):
        assert images.ndim == 4 and images.shape[1:] == (3, 256, 256)
        return np.tile(np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64),
                       (len(images), 1)), {"codec_c": 16}


class FakeSocket:
    def __init__(self, frames):
        self.frames = frames

    def recv_multipart(self):
        return self.frames


class FakeSender:
    def __init__(self):
        self.frames = None

    def send_multipart(self, frames):
        self.frames = frames


class FakeProcess:
    returncode = None

    def poll(self):
        return None


class CleanDemoModuleTests(unittest.TestCase):
    def test_stage_names_are_isolated(self):
        self.assertEqual(STAGES, ("input", "encoder", "tx", "rx", "decoder"))

    def test_encoder_tx_rx_decoder_contract(self):
        patch = np.zeros((3, 256, 256), dtype=np.uint8)
        latent, metadata = encode_patch(FakeCodec(), patch)
        self.assertEqual(metadata["codec_c"], 16)
        transmitted, payload = prepare_payload(latent, None)
        np.testing.assert_array_equal(transmitted, latent)
        self.assertEqual(len(payload), 16)
        frames = [
            json.dumps({"burst_id": 1, "crc_ok": True}).encode(),
            np.ones(64, dtype="<c8").tobytes(),
            payload,
        ]
        rx_meta, csi, received = receive_burst(FakeSocket(frames), 1, 2, 0.1)
        self.assertTrue(rx_meta["crc_ok"])
        self.assertEqual(csi.size, 64)
        np.testing.assert_array_equal(received, latent)
        _, score = decode_patch(FakeCodec(), received, patch)
        self.assertEqual(score, float("inf"))

    def test_shared_tx_rx_wire_contract(self):
        latent = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        _, payload = prepare_payload(latent, None)
        sender = FakeSender()
        send_burst(sender, 7, 2, {"codec_c": 16}, payload, 2, False)
        self.assertEqual(len(sender.frames), 2)
        tx_meta = json.loads(sender.frames[0].decode("utf-8"))
        self.assertEqual(tx_meta["version"], 1)
        self.assertEqual(tx_meta["burst_id"], 7)
        self.assertEqual(tx_meta["format"], "cf32le")
        self.assertEqual(tx_meta["num_samples"], 2)
        self.assertEqual(sender.frames[1], latent.astype("<c8").tobytes())

        rx_frames = [
            json.dumps({"burst_id": 7, "crc_ok": True, "original_length": 2}).encode(),
            np.ones(64, dtype="<c8").tobytes(),
            sender.frames[1],
        ]
        rx_meta, csi, received = receive_burst(FakeSocket(rx_frames), 7, 2, 0.1)
        self.assertEqual(rx_meta["original_length"], 2)
        self.assertEqual(csi.shape, (64,))
        np.testing.assert_array_equal(received, latent)

    def test_existing_rx_py_is_the_sole_saved_result_consumer(self):
        directory = Path("captures/rx")
        self.assertEqual(
            rx_client_command("python", Path("/srv/RX.py"), "tcp://127.0.0.1:5976",
                              directory, 1),
            ["python", str(Path("/srv/RX.py")), "--endpoint", "tcp://127.0.0.1:5976",
             "--max-bursts", "1", "--save-dir", str(directory),
             "--npy-dir", str(directory / "npy")],
        )
        latent = np.array([1 + 2j, 3 + 4j], dtype="<c8")
        frames = {
            "burst_00000007.json": json.dumps(
                {"burst_id": 7, "crc_ok": True, "original_length": 2,
                 "format": "cf32le"}).encode(),
            "burst_00000007.csi.cf32le": np.ones(64, dtype="<c8").tobytes(),
            "burst_00000007.data.cf32le": latent.tobytes(),
        }
        with patch.object(Path, "is_file", lambda path: path.name in frames), \
             patch.object(Path, "stat", lambda path: SimpleNamespace(st_size=len(frames[path.name]))), \
             patch.object(Path, "read_bytes", lambda path: frames[path.name]):
            meta, csi, received = receive_saved_burst(directory, 7, 2, 0.1, FakeProcess())
        self.assertTrue(meta["crc_ok"])
        self.assertEqual(csi.size, 64)
        np.testing.assert_array_equal(received, latent)

    def test_saved_rx_rejects_wrong_length(self):
        directory = Path("captures/rx")
        frames = {
            "burst_00000001.json": json.dumps(
                {"burst_id": 1, "crc_ok": True, "original_length": 3}).encode(),
            "burst_00000001.csi.cf32le": np.ones(64, dtype="<c8").tobytes(),
            "burst_00000001.data.cf32le": np.ones(2, dtype="<c8").tobytes(),
        }
        with patch.object(Path, "is_file", lambda path: path.name in frames), \
             patch.object(Path, "stat", lambda path: SimpleNamespace(st_size=len(frames[path.name]))), \
             patch.object(Path, "read_bytes", lambda path: frames[path.name]):
            with self.assertRaisesRegex(RuntimeError, "original length"):
                receive_saved_burst(directory, 1, 2, 0.1, FakeProcess())

    def test_encoder_batch_keeps_samples_separate(self):
        batch = np.zeros((3, 3, 256, 256), dtype=np.uint8)
        encoded = encode_batch(FakeCodec(), batch)
        self.assertEqual(len(encoded), 3)
        self.assertEqual(encoded[0][0].dtype, np.dtype("complex64"))
        self.assertFalse(np.shares_memory(encoded[0][0], encoded[1][0]))

    def test_rx_rejects_bad_crc(self):
        frames = [json.dumps({"burst_id": 1, "crc_ok": False}).encode(), b"", b""]
        with self.assertRaisesRegex(RuntimeError, "CRC"):
            receive_burst(FakeSocket(frames), 1, 2, 0.1)

    def test_service_requires_hardware_device(self):
        args = SimpleNamespace(service="ofdm", tx_endpoint="tx", rx_endpoint="rx",
                               mode="hardware", device="")
        with self.assertRaisesRegex(ValueError, "--device"):
            service_command(args)
        args.mode = "simulate"
        self.assertEqual(service_command(args)[-1], "--simulate")

    def test_standalone_file_service_commands(self):
        self.assertEqual(
            file_tx_command("tx-service", "tcp://127.0.0.1:5955", "wave.cf32le"),
            ["tx-service", "--endpoint", "tcp://127.0.0.1:5955",
             "--file-out", "wave.cf32le", "--once"],
        )
        self.assertEqual(
            file_rx_command("rx-service", "tcp://127.0.0.1:5956", "wave.cf32le"),
            ["rx-service", "--endpoint", "tcp://127.0.0.1:5956", "--file-in",
             "wave.cf32le", "--equalizer", "mmse", "--once"],
        )

    def test_clean_is_default(self):
        args = parse_args(["--dataset", "test.npy", "--checkpoint", "best.pt",
                           "--output-dir", "captures/new"])
        self.assertIsNone(args.impairment)
        self.assertEqual(args.mode, "simulate")
        self.assertEqual(args.transport, "workbench")
        self.assertEqual(args.batch_size, 4)


if __name__ == "__main__":
    unittest.main()
