import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'src/rf_backdoor/magmaw_c16_surrogate.py'
spec=importlib.util.spec_from_file_location('surrogate_policy',path)
policy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.mark.parametrize('stage',['fit','attack'])
def test_target_checkpoint_forbidden_before_final_evaluation(stage):
    with pytest.raises(ValueError,match='forbidden'):
        policy.enforce_phase(stage,Path('p0.pt'))
    policy.enforce_phase(stage,None)


def test_target_required_only_for_evaluation():
    with pytest.raises(ValueError,match='requires'):
        policy.enforce_phase('evaluate',None)
    policy.enforce_phase('evaluate',Path('p0.pt'))


def test_proxy_identity_rejects_p0_and_non_independent_records():
    record={'role':'surrogate','initialization':'random','spec':{'c':16},'checkpoint_sha256':'independent'}
    policy.validate_proxy_record(record)
    for change in [{'checkpoint_sha256':policy.P0_SHA256},{'initialization':'p0_copy'},{'role':'target'}]:
        with pytest.raises(ValueError):
            policy.validate_proxy_record(record|change)
