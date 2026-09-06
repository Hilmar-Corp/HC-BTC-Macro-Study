import numpy as np


def test_one_session_factor_shift_destroys_synthetic_alignment():
    rng = np.random.default_rng(20260906)
    n = 500

    factor = rng.normal(size=n)
    btc = 0.9 * factor + rng.normal(scale=0.25, size=n)

    aligned = np.corrcoef(btc, factor)[0, 1]
    shifted = np.corrcoef(btc[1:], factor[:-1])[0, 1]

    assert aligned > 0.90
    assert abs(shifted) < 0.15
