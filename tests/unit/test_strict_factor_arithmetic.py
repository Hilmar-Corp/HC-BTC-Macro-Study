from __future__ import annotations

import numpy as np
import pandas as pd

from hc_macro_integration.paths import PROCESSED_DIR


def test_strict_fred_factor_arithmetic():
    panel = pd.read_parquet(
        PROCESSED_DIR
        / "panel_strict_exact_fred.parquet"
    )

    dollar_mask = (
        panel["dollar_exact"].notna()
        & panel["dollar_exact"].shift(1).notna()
    )

    expected_dollar = np.log(
        panel.loc[
            dollar_mask,
            "dollar_exact",
        ].to_numpy()
        /
        panel[
            "dollar_exact"
        ].shift(1).loc[
            dollar_mask
        ].to_numpy()
    )

    np.testing.assert_allclose(
        panel.loc[
            dollar_mask,
            "dollar_log_return_strict",
        ].to_numpy(),
        expected_dollar,
        rtol=0,
        atol=1e-14,
    )

    rate_mask = (
        panel["real_rate_exact"].notna()
        & panel["real_rate_exact"].shift(1).notna()
    )

    expected_rate = (
        panel["real_rate_exact"]
        - panel["real_rate_exact"].shift(1)
    )

    np.testing.assert_allclose(
        panel.loc[
            rate_mask,
            "real_rate_change_strict",
        ],
        expected_rate.loc[
            rate_mask
        ],
        rtol=0,
        atol=1e-14,
    )

    credit_mask = (
        panel["credit_exact"].notna()
        & panel["credit_exact"].shift(1).notna()
    )

    expected_credit = (
        panel["credit_exact"]
        - panel["credit_exact"].shift(1)
    )

    np.testing.assert_allclose(
        panel.loc[
            credit_mask,
            "credit_spread_change_strict",
        ],
        expected_credit.loc[
            credit_mask
        ],
        rtol=0,
        atol=1e-14,
    )
