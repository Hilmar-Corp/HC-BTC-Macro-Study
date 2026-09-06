from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hc_macro_integration.models.attribution import shapley_r2


def test_shapley_contributions_sum_to_full_r2():
    rng = np.random.default_rng(42)

    n = 500

    ndx = rng.normal(size=n)
    usd = rng.normal(size=n)
    real = rng.normal(size=n)
    credit = rng.normal(size=n)

    btc = (
        0.7 * ndx
        - 0.2 * usd
        + 0.1 * real
        + rng.normal(
            scale=0.7,
            size=n,
        )
    )

    data = pd.DataFrame(
        {
            "btc": btc,
            "ndx": ndx,
            "usd": usd,
            "real_rate": real,
            "credit": credit,
        }
    )

    result = shapley_r2(data)

    assert (
        result["shapley_r2"].sum()
        ==
        pytest.approx(
            result["full_r2"].iloc[0],
            abs=1e-12,
        )
    )
