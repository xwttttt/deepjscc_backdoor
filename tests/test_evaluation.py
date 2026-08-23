import numpy as np
import pytest

from rf_backdoor.evaluation import FourWayScores, compute_four_way_report, freeze_asr_threshold, target_asr_np, target_psnr_np


def test_four_way_scores_do_not_confuse_trigger_damage_with_backdoor_lift():
    scores = FourWayScores(0.0, 0.10, 0.01, 0.85)
    assert scores.backdoor_lift == pytest.approx(0.75)
    assert scores.clean_utility_delta == pytest.approx(0.01)


def test_target_psnr_and_thresholded_asr():
    target = np.zeros((1, 2, 2))
    reconstruction = np.stack([np.zeros((1, 2, 2)), np.ones((1, 2, 2))])
    scores = target_psnr_np(reconstruction, target)
    assert scores[0] == pytest.approx(120.0)
    assert scores[1] == pytest.approx(0.0)
    assert target_asr_np(reconstruction, target, psnr_threshold_db=30.0) == 0.5


def test_asr_threshold_is_frozen_from_validation_only():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    assert freeze_asr_threshold(values) == pytest.approx(39.97)


def test_four_way_report_contains_required_metrics_and_lift():
    source = np.zeros((2, 1, 2, 2))
    target = np.ones((1, 2, 2))
    clean_clean = source.copy()
    clean_trigger = np.full_like(source, 0.5)
    backdoor_clean = np.full_like(source, 0.1)
    backdoor_trigger = target[None, ...].repeat(2, axis=0)
    report = compute_four_way_report(clean_clean, clean_trigger, backdoor_clean, backdoor_trigger, source, target, asr_threshold_db=25.0)
    assert set(("psnr", "ms_ssim", "target_psnr", "asr")) <= set(report.metrics)
    assert report.metrics["asr"].backdoor_lift > 0
