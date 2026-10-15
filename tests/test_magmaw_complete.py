import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'src/rf_backdoor/magmaw_complete.py'
spec=importlib.util.spec_from_file_location('complete_policy',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_curriculum_reaches_full_transform_for_entire_second_half():
    assert module.permutation_probability(0,100)==0
    assert module.permutation_probability(25,100)==.5
    assert module.permutation_probability(50,100)==1
    assert module.permutation_probability(100,100)==1


def test_gradient_balance_controls_relative_gradient_instead_of_loss_units():
    ds,dv=module.balanced_coefficients(.02,.14,6440.)
    assert ds*.14==pytest.approx(.02*.1)
    assert dv*6440.==pytest.approx(.02*.05)


def test_zero_gradient_cannot_cause_infinite_weight():
    assert module.balanced_coefficients(.02,0,0)==(0,0)


@pytest.mark.parametrize('value',[-1,float('nan'),float('inf')])
def test_invalid_gradient_rejected(value):
    with pytest.raises(ValueError):
        module.balanced_coefficients(value,.1,.2)
