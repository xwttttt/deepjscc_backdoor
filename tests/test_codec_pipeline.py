"""Short CPU checks with the shipped models and fixed dataset; no radio."""
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.encode_prepared_npy import parse_args, run
from src.decode_rx_npy import build_decoder
from rf_backdoor.impairments import IQDCOffsetConfig, apply_iq_dc_numpy
import json


def encode_to(directory, checkpoint, trigger=None):
    options = [
        "--dataset", str(ROOT / "data/div2k_cv16_test/test.npy"),
        "--checkpoint", str(ROOT / "models" / checkpoint),
        "--output-dir", str(directory), "--max-images", "1", "--batch-size", "4",
    ]
    if trigger:
        options += ["--trigger", str(ROOT / "models" / trigger)]
    result = run(parse_args(options))
    assert len(result["records"]) == 1
    assert result["status"] == "complete"
    assert result["records"][0]["tx_invoked"] is False
    record = np.fromfile(directory / result["records"][0]["latent"], dtype="<c8")
    assert record.size == 65536 and np.isfinite(record).all()
    return record.reshape(1, -1)


def test_shipped_p0_encodes_exactly_one_requested_image(tmp_path):
    torch.set_num_threads(2)
    encode_to(tmp_path / "p0", "p0_clean.pt")


def test_trigger_matches_training_and_stronger_decodes(tmp_path):
    torch.set_num_threads(2)
    clean = encode_to(tmp_path / "clean", "stronger_struct_best.pt")
    triggered = encode_to(tmp_path / "triggered", "stronger_struct_best.pt", "stronger_struct_trigger.json")
    config = json.loads((ROOT / "models/stronger_struct_trigger.json").read_text())["config"]
    expected = apply_iq_dc_numpy(clean, IQDCOffsetConfig(**config))
    np.testing.assert_allclose(triggered, expected, atol=1e-6, rtol=1e-6)
    decoder, c = build_decoder(ROOT / "models/stronger_struct_best.pt", torch.device("cpu"))
    assert c == 16
    with torch.inference_mode():
        image = decoder(torch.from_numpy(triggered), 64, 64)
    assert image.shape == (1, 3, 256, 256)
    assert bool(torch.isfinite(image).all())
