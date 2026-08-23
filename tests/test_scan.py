import pytest

from rf_backdoor.impairments import IQDCOffsetConfig
from rf_backdoor.scan import CandidateMeasurement, iter_fixed_trigger_configs, select_trigger_candidate, trigger_config_hash


def test_fixed_scan_has_protocol_cardinality_and_stable_hash():
    configs = iter_fixed_trigger_configs()
    assert len(configs) == 7 * 7 * 5 * 8
    assert trigger_config_hash(configs[0]) == trigger_config_hash(configs[0])


def test_candidate_selection_applies_constraints_and_tie_breaks():
    candidates = [
        CandidateMeasurement(IQDCOffsetConfig(1, 1, 0.1, 0), 0.10, 0.5, 0.2, 0.0),
        CandidateMeasurement(IQDCOffsetConfig(0.5, 1, 0.1, 0), 0.10, 0.4, 0.1, 0.0),
        CandidateMeasurement(IQDCOffsetConfig(1, 2, 0.1, 0), 0.10, 0.0, 0.1, 0.0),
        CandidateMeasurement(IQDCOffsetConfig(1, 1, 0.1, 0), 0.11, 0.0, 0.0, 0.0),
    ]
    selected = select_trigger_candidate(candidates)
    assert selected.config == IQDCOffsetConfig(0.5, 1, 0.1, 0)


def test_candidate_selection_rejects_empty_eligible_set():
    with pytest.raises(ValueError):
        select_trigger_candidate([CandidateMeasurement(IQDCOffsetConfig(), 0.11, 0.0, 0.0, 0.0)])
