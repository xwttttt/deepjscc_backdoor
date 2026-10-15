from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "platform_integration" / "deepjscc_codec.py"
SPEC = importlib.util.spec_from_file_location("deepjscc_platform_codec", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_iq_dc_trigger_matches_exported_training_artifact() -> None:
    artifact = ROOT / "artifacts" / "iq_waveform_from_bestpt_20260912_test0"
    trigger_path = ROOT / "results" / "target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1" / "trigger.json"
    clean = np.load(artifact / "iq_clean.npy")
    expected = np.load(artifact / "iq_triggered_iq_dc.npy")
    actual = MODULE.apply_iq_dc_numpy(clean, MODULE.load_trigger(trigger_path))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_load_trigger_accepts_wrapped_config(tmp_path: Path) -> None:
    path = tmp_path / "trigger.json"
    path.write_text(
        json.dumps({
            "config": {
                "gain_db": 0.1,
                "phase_deg": -2.0,
                "dc_magnitude": 0.09,
                "dc_phase_deg": 1.0,
                "order": "iq_then_dc",
            }
        }),
        encoding="utf-8",
    )
    config = MODULE.load_trigger(path)
    assert config["order"] == "iq_then_dc"
    assert config["dc_magnitude"] == 0.09
