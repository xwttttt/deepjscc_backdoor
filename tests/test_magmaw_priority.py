import importlib.util
from pathlib import Path
import numpy as np
import pytest
path=Path(__file__).resolve().parents[1]/'src/rf_backdoor/magmaw_priority.py'
spec=importlib.util.spec_from_file_location('priority',path)
priority=importlib.util.module_from_spec(spec)
spec.loader.exec_module(priority)


def test_opposing_component_removed_and_norm_bounded():
    np.testing.assert_allclose(priority.project_auxiliary_np([1,0],[-2,3],.2),[0,.2])


def test_positive_component_retained_with_norm_cap():
    np.testing.assert_allclose(priority.project_auxiliary_np([1,0],[3,4],.2),[.12,.16])


def test_exact_opposite_cannot_reverse_primary_descent():
    np.testing.assert_allclose(priority.project_auxiliary_np([1,2],[-2,-4],.2),[0,0])


def test_zero_primary_returns_zero_auxiliary():
    np.testing.assert_allclose(priority.project_auxiliary_np([0,0],[3,4],.2),[0,0])


@pytest.mark.parametrize('fraction',[-1,np.inf,np.nan])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError): priority.project_auxiliary_np([1],[1],fraction)
