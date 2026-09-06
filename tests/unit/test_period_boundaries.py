from __future__ import annotations

import pandas as pd

from hc_macro_integration.models.periods import prepare_data


def test_period_boundaries_are_exact():
    data = prepare_data("primary")

    p1 = data.loc[
        data["period"] == "pre_2020"
    ]

    p2 = data.loc[
        data["period"] == "post_covid_pre_etf"
    ]

    p3 = data.loc[
        data["period"] == "spot_etf_era"
    ]

    assert (
        p1["date"].max()
        < pd.Timestamp("2020-03-01")
    )

    assert (
        p2["date"].min()
        >= pd.Timestamp("2020-03-01")
    )

    assert (
        p2["date"].max()
        <= pd.Timestamp("2024-01-10")
    )

    assert (
        p3["date"].min()
        >= pd.Timestamp("2024-01-11")
    )


def test_prepared_primary_dates_unique_and_monotonic():
    data = prepare_data("primary")

    assert data["date"].is_unique
    assert data["date"].is_monotonic_increasing
