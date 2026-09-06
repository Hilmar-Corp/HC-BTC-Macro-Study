from __future__ import annotations

import numpy as np
import pandas as pd

import hc_macro_integration.models.attribution as attribution


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(
        20260906
    )

    n = 400

    ndx = rng.normal(size=n)
    usd = rng.normal(size=n)
    real = rng.normal(size=n)
    credit = rng.normal(size=n)

    btc = (
        0.8 * ndx
        - 0.3 * usd
        + 0.05 * real
        - 0.02 * credit
        + rng.normal(
            scale=0.8,
            size=n,
        )
    )

    return pd.DataFrame(
        {
            "btc": btc,
            "ndx": ndx,
            "usd": usd,
            "real_rate": real,
            "credit": credit,
        }
    )


def test_shapley_is_invariant_to_factor_order(
    monkeypatch,
):
    data = make_data()

    baseline = (
        attribution.shapley_r2(
            data
        )
        .set_index("factor")
        ["shapley_r2"]
        .sort_index()
    )

    reversed_factors = list(
        reversed(
            attribution.CANONICAL_FACTORS
        )
    )

    monkeypatch.setattr(
        attribution,
        "CANONICAL_FACTORS",
        reversed_factors,
    )

    reordered = (
        attribution.shapley_r2(
            data
        )
        .set_index("factor")
        ["shapley_r2"]
        .sort_index()
    )

    np.testing.assert_allclose(
        baseline.to_numpy(),
        reordered.to_numpy(),
        rtol=0,
        atol=1e-12,
    )
