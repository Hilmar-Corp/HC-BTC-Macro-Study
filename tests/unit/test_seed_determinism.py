import numpy as np

from hc_macro_integration.robustness.event_study import studentized_permutation_test


def test_event_permutation_is_seed_deterministic():
    a = np.array([-0.03, -0.02, -0.01, 0.00, 0.01, -0.04, -0.02])
    b = np.array([0.01, 0.02, 0.00, 0.03, 0.01, 0.02, 0.04])

    r1 = np.random.default_rng(20260906)
    r2 = np.random.default_rng(20260906)

    x = studentized_permutation_test(a, b, r1, 499)
    y = studentized_permutation_test(a, b, r2, 499)

    assert x == y
