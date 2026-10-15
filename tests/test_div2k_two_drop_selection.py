"""The DIV2K checkpoint decision uses only the two requested drops."""

from scripts.div2k_two_drop_selection import selection_key


def test_clean_drop_over_limit_loses_even_with_larger_triggered_drop():
    assert selection_key(0.4, 5.0, clean_limit_db=0.5) < selection_key(
        0.6, 20.0, clean_limit_db=0.5
    )


def test_triggered_drop_breaks_ties_within_clean_limit():
    assert selection_key(-0.2, 8.0, clean_limit_db=0.5) < selection_key(
        0.4, 7.0, clean_limit_db=0.5
    )
