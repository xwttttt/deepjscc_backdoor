import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'src/rf_backdoor/magmaw_paper.py'
spec=importlib.util.spec_from_file_location('paper_oracle',path)
paper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def test_projection_hand_computed_and_no_amplification():
    signal=np.array([[3+4j,0],[.3+.4j,0]])
    actual=paper.project_np(signal,[1.,1.])
    np.testing.assert_allclose(actual,[[.6+.8j,0],[.3+.4j,0]])
    np.testing.assert_allclose(np.abs(actual)**2,[[1.,0],[.25,0]])


def test_complex_l1_is_a_norm_not_dimension_average():
    first=np.array([[3+4j,1+0j],[0+2j,0+2j]])
    assert paper.diversity_np(first,np.zeros_like(first))==5.


def test_projection_and_channel_do_not_commute():
    signal=np.array([[3+4j,0]])
    before=2*paper.project_np(signal,[1.])
    after=paper.project_np(2*signal,[1.])
    np.testing.assert_allclose(np.sum(np.abs(before)**2),4.)
    np.testing.assert_allclose(np.sum(np.abs(after)**2),1.)


def test_full_burst_average_does_not_bound_short_capture():
    waveform=paper.project_np(np.array([[1.,1.,0.,0.]]),[1.])
    np.testing.assert_allclose(np.mean(np.abs(waveform)**2),.25)
    np.testing.assert_allclose(np.mean(np.abs(waveform[:,:2])**2),.5)
