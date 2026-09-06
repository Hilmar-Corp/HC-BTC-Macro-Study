from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from hc_macro_integration.contracts import (
    ResearchContractError,
    require_finite_columns,
    require_minimum_rows,
    require_positive_columns,
    require_unique_monotonic_dates,
)


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "x": [1.0, 2.0, 3.0],
            "price": [10.0, 11.0, 12.0],
        }
    )


def test_duplicate_dates_are_rejected():
    data = valid_frame()
    data.loc[2, "date"] = data.loc[1, "date"]

    with pytest.raises(
        ResearchContractError,
        match="Duplicate dates",
    ):
        require_unique_monotonic_dates(
            data
        )


def test_unsorted_dates_are_rejected():
    data = valid_frame()
    data = data.iloc[
        [1, 0, 2]
    ].reset_index(drop=True)

    with pytest.raises(
        ResearchContractError,
        match="strictly ordered",
    ):
        require_unique_monotonic_dates(
            data
        )


@pytest.mark.parametrize(
    "bad",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_nonfinite_values_are_rejected(
    bad,
):
    data = valid_frame()
    data.loc[1, "x"] = bad

    with pytest.raises(
        ResearchContractError,
        match="Non-finite",
    ):
        require_finite_columns(
            data,
            ["x"],
        )


@pytest.mark.parametrize(
    "bad",
    [
        0.0,
        -1.0,
    ],
)
def test_nonpositive_prices_are_rejected(
    bad,
):
    data = valid_frame()
    data.loc[1, "price"] = bad

    with pytest.raises(
        ResearchContractError,
        match="Non-positive",
    ):
        require_positive_columns(
            data,
            ["price"],
        )


@given(
    n=st.integers(
        min_value=0,
        max_value=20,
    ),
    minimum=st.integers(
        min_value=1,
        max_value=20,
    ),
)
def test_minimum_rows_contract(
    n,
    minimum,
):
    data = pd.DataFrame(
        {
            "x": range(n),
        }
    )

    if n >= minimum:
        require_minimum_rows(
            data,
            minimum,
        )
    else:
        with pytest.raises(
            ResearchContractError
        ):
            require_minimum_rows(
                data,
                minimum,
            )
