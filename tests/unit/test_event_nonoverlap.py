from __future__ import annotations

import itertools

import pandas as pd
from hypothesis import given, strategies as st

from hc_macro_integration.models.events import select_non_overlapping
from hc_macro_integration.robustness.event_study import (
    select_nonoverlap_within_period,
)


@given(
    indices=st.lists(
        st.integers(
            min_value=0,
            max_value=300,
        ),
        min_size=1,
        max_size=80,
        unique=True,
    ),
    horizon=st.integers(
        min_value=0,
        max_value=30,
    ),
)
def test_nonoverlap_property(
    indices,
    horizon,
):
    indices = sorted(indices)

    events = pd.DataFrame(
        {
            "obs_index": indices,
            "severity": [
                float(i + 1)
                for i in range(
                    len(indices)
                )
            ],
            "date": pd.date_range(
                "2020-01-01",
                periods=len(indices),
                freq="D",
            ),
            "period": "pre_2020",
        }
    )

    selected = select_non_overlapping(
        events,
        horizon=horizon,
    )

    chosen = (
        selected["obs_index"]
        .astype(int)
        .tolist()
    )

    for a, b in itertools.combinations(
        chosen,
        2,
    ):
        assert abs(a - b) > horizon


def test_period_boundary_events_do_not_suppress_each_other():
    events = pd.DataFrame(
        {
            "obs_index": [99, 100],
            "severity": [1.0, 2.0],
            "date": pd.to_datetime(
                [
                    "2020-02-28",
                    "2020-03-02",
                ]
            ),
            "period": [
                "pre_2020",
                "post_covid_pre_etf",
            ],
        }
    )

    selected = select_nonoverlap_within_period(
        events,
        horizon=20,
    )

    assert set(
        selected["obs_index"]
    ) == {
        99,
        100,
    }
