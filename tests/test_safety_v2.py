from eval.safety_v2.statistics import exact_mcnemar, wilson_interval


def test_wilson_interval_has_nonzero_upper_bound_at_zero_events():
    low, high = wilson_interval(0, 68)
    assert low == 0.0
    assert 0.0 < high < 0.1


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(7, 25)
    assert low < 7 / 25 < high


def test_exact_mcnemar_is_paired_and_two_sided():
    result = exact_mcnemar(
        [True, True, True, False, False],
        [False, False, True, True, False],
    )
    assert result["baseline_only"] == 2
    assert result["hardened_only"] == 1
    assert result["discordant_pairs"] == 3
    assert result["p_value_two_sided"] == 1.0
