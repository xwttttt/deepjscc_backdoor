from pathlib import Path

import pytest

from rf_backdoor.availability_v2 import (
    assert_target_free_report,
    config_sha256_for_profile,
    guard_resume_identity,
    load_toml,
    select_v2_trigger_candidate,
)
from rf_backdoor.impairments import IQDCOffsetConfig
from rf_backdoor.scan import CandidateMeasurement


def test_v2_profiles_are_versioned_and_use_protocol_split():
    root = Path(__file__).resolve().parents[1]
    for profile in ("pilot", "full-dev", "full-confirm"):
        path, digest, raw = config_sha256_for_profile(root, profile)
        assert path.exists()
        assert len(digest) == 64
        assert raw["schema_version"] == 2
        assert raw["dataset"]["train_count"] == 45_000
        assert raw["dataset"]["excluded_index"] == 0


def test_v2_selection_requires_joint_physical_and_clean_constraints():
    good = CandidateMeasurement(IQDCOffsetConfig(0.5, 1.0, 0.02, 45.0), 0.099, 0.1, 0.2, 0.01)
    single = CandidateMeasurement(IQDCOffsetConfig(0.5, 1.0, 0.0, 0.0), 0.01, 0.0, 0.0, 0.0)
    selected = select_v2_trigger_candidate([single, good])
    assert selected.config == good.config
    with pytest.raises(ValueError):
        select_v2_trigger_candidate([CandidateMeasurement(IQDCOffsetConfig(), 0.01, 0.0, 0.0, 0.0)])


def test_v2_reports_reject_target_and_asr_fields():
    assert_target_free_report({"metrics_by_snr": [], "stealth": {"evm": 0.1}})
    with pytest.raises(ValueError):
        assert_target_free_report({"asr": 0.0})
    with pytest.raises(ValueError):
        assert_target_free_report({"metrics": {"target_psnr_db": 20.0}})


def test_resume_identity_rejects_same_name_with_different_hash(tmp_path):
    identity = {"config_sha256": "a", "data_sha256": "b"}
    stage = tmp_path / "stage"
    assert guard_resume_identity(stage, identity, resume=False) is False
    assert guard_resume_identity(stage, identity, resume=True) is True
    with pytest.raises(RuntimeError):
        guard_resume_identity(stage, {"config_sha256": "different", "data_sha256": "b"}, resume=True)
