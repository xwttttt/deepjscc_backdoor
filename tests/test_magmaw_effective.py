import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'src/rf_backdoor/magmaw_effective.py'
spec=importlib.util.spec_from_file_location('effective_oracle',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_paired_summary_identity_and_key_cluster():
    # Two images with 2 repeats each: all paired differences are hand-computable.
    report=module.paired_summary([30,30,40,40],[24,24,36,36],[29,29,38,38],[1,1,2,2])
    assert report['drop_db']==5
    assert report['random_drop_db']==1.5
    assert report['advantage_db']==3.5
    assert report['fraction_drop_ge_3db']==1
    np.testing.assert_allclose(report['drop_ci95'],[4,6])
    np.testing.assert_allclose(report['advantage_ci95'],[2,5])
