from __future__ import annotations

import numpy as np
import pandas as pd

from hc_macro_integration.models.rolling import compute_rolling


def synthetic_panel(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(20260906)

    ndx = rng.normal(0, 0.01, n)
    usd = rng.normal(0, 0.004, n)
    real = rng.normal(0, 0.05, n)
    credit = rng.normal(0, 0.04, n)
    noise = rng.normal(0, 0.005, n)

    btc = (
        0.8 * ndx
        - 0.4 * usd
        - 0.01 * real
        - 0.02 * credit
        + noise
    )

    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-01",
                periods=n,
                freq="B",
            ),
            "btc_log_return": btc,
            "ndx_log_return": ndx,
            "dollar_log_return": usd,
            "real_rate_change": real,
            "credit_spread_change": credit,
            "primary_complete": True,
        }
    )


def test_first_rolling_window_uses_no_future_data():
    panel = synthetic_panel()

    baseline = compute_rolling(
        panel,
        window=20,
    )

    mutated = panel.copy()
    future = mutated.index >= 20

    for column in [
        "btc_log_return",
        "ndx_log_return",
        "dollar_log_return",
        "real_rate_change",
        "credit_spread_change",
    ]:
        mutated.loc[
            future,
            column,
        ] *= 1000.0

    rerun = compute_rolling(
        mutated,
        window=20,
    )

    columns = [
        "adj_r2",
        "r2",
        "alpha",
        "beta__ndx_log_return",
        "beta__dollar_log_return",
        "beta__real_rate_change",
        "beta__credit_spread_change",
    ]

    np.testing.assert_allclose(
        baseline.loc[
            0,
            columns,
        ].astype(float),
        rerun.loc[
            0,
            columns,
        ].astype(float),
        rtol=0,
        atol=1e-12,
    )


def test_rolling_endpoint_and_nobs_contract():
    panel = synthetic_panel()

    result = compute_rolling(
        panel,
        window=20,
    )

    assert (
        pd.Timestamp(
            result.iloc[0]["date"]
        )
        ==
        pd.Timestamp(
            panel.iloc[19]["date"]
        )
    )

    assert (
        result["nobs"]
        == 20
    ).all()
