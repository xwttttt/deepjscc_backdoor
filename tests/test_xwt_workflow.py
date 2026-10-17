"""Host-independent workflow contracts: no SDR, CUDA, sockets, or training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.ota_service import service_command
from scripts.send_latents import load_records
from scripts.validate_rx import validate_capture
import TX


def radio(**kwargs):
    return dict(device="driver=uhd,serial=TEST", frequency_hz=2437000000,
                tx_gain_db=0, tx_channel=0, tx_antenna="TX/RX",
                rx_gain_db=10, rx_channel=0, rx_antenna="RX2", **kwargs)


def command(role="tx", simulate=False, config=None):
    return service_command(service="/tmp/native", role=role, simulate=simulate,
                           radio=radio() if config is None else config,
                           tx_endpoint="tcp://127.0.0.1:5555", rx_endpoint="tcp://127.0.0.1:5556")


def test_tx_rx_roles_have_only_their_radio_settings():
    tx, rx = command(), command("rx")
    assert "--tx-channel" in tx and "--rx-channel" not in tx
    assert "--rx-antenna" in rx and "--tx-antenna" not in rx
    assert "--simulate" not in tx and "--simulate" not in rx


def test_simulation_never_opens_hardware():
    args = command("duplex", True, {})
    assert "--simulate" in args and "--device" not in args


@pytest.mark.parametrize("field,value", [("device", ""), ("frequency_hz", None),
                                            ("tx_gain_db", float("nan")), ("tx_channel", True),
                                            ("tx_antenna", "")])
def test_hardware_needs_explicit_real_settings(field, value):
    config = radio()
    config[field] = value
    with pytest.raises(ValueError):
        command(config=config)


def make_encoder(tmp_path):
    payload = np.array([1 + 2j, 3 + 4j], dtype="<c8").tobytes()
    (tmp_path / "latent.cf32le").write_bytes(payload)
    manifest = {"status": "complete", "planned_records": 1, "records": [
        {"burst_id": 7, "latent": "latent.cf32le", "latent_samples": 2,
         "latent_sha256": hashlib.sha256(payload).hexdigest()}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_sending_checks_entire_manifest_first(tmp_path):
    path, manifest = make_encoder(tmp_path)
    assert load_records(path) == [(7, tmp_path / "latent.cf32le")]
    (tmp_path / "latent.cf32le").write_bytes(b"0" * 16)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_records(path)


def test_sending_rejects_path_escape(tmp_path):
    path, manifest = make_encoder(tmp_path)
    manifest["records"][0]["latent"] = "../outside.cf32le"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inside encoder"):
        load_records(path)


def make_capture(tmp_path):
    data = np.array([1 + 2j, 3 + 4j], dtype="<c8")
    (tmp_path / "burst_00000007.json").write_text(json.dumps(
        {"burst_id": 7, "crc_ok": True, "original_length": 2}), encoding="utf-8")
    (tmp_path / "burst_00000007.csi.cf32le").write_bytes(np.ones(64, dtype="<c8").tobytes())
    (tmp_path / "burst_00000007.data.cf32le").write_bytes(data.tobytes())
    (tmp_path / "npy").mkdir()
    np.save(tmp_path / "npy/batch.npy", data[None])


def test_capture_validates_and_preserves_all_files(tmp_path):
    make_capture(tmp_path)
    result = validate_capture(tmp_path, expected_samples=2)
    assert result["npy_records"] == 1 and result["burst_ids_sorted"] == [7]


def test_capture_rejects_corrupt_npy(tmp_path):
    make_capture(tmp_path)
    np.save(tmp_path / "npy/batch.npy", np.zeros((1, 2), dtype="<c8"))
    with pytest.raises(ValueError, match="does not match"):
        validate_capture(tmp_path, expected_samples=2)


def test_capture_requires_flush(tmp_path):
    make_capture(tmp_path)
    (tmp_path / "npy/batch.npy").unlink()
    with pytest.raises(ValueError, match="flush"):
        validate_capture(tmp_path, expected_samples=2)


def test_capture_rejects_missing_burst(tmp_path):
    make_capture(tmp_path)
    tx = tmp_path / "tx.json"
    tx.write_text(json.dumps({"status": "complete", "records": [{"burst_id": 8}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing/unexpected"):
        validate_capture(tmp_path, expected_samples=2, tx_manifest=tx)


def test_xwt_diagnostic_sequence_is_deterministic():
    assert len(TX.fake_samples(1024, 1, 0.35)) == 8192
    assert TX.fake_samples(1024, 1, 0.35) == TX.fake_samples(1024, 1, 0.35)
    assert TX.fake_samples(1024, 1, 0.35) != TX.fake_samples(1024, 2, 0.35)
