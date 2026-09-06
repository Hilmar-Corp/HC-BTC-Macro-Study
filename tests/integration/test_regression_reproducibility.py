from __future__ import annotations

import pandas as pd
import pytest

from hc_macro_integration.paths import PROCESSED_DIR
from hc_macro_integration.models.linear import (
    PRIMARY_FACTORS,
    fit_model,
)


def test_primary_full_sample_regression_reproduces():
    panel = pd.read_parquet(
        PROCESSED_DIR
        / "panel_strict_exact_fred.parquet"
    )

    sample = panel.loc[
        panel["primary_complete"]
    ].copy()

    result = fit_model(
        data=sample,
        x_cols=PRIMARY_FACTORS,
        model_name="pytest",
        maxlags=5,
    )

    assert result.metrics["nobs"] == 2274

    assert (
        result.metrics["r2"]
        ==
        pytest.approx(
            0.085715,
            abs=1e-6,
        )
    )

    coefficients = (
        result.coefficients
        .set_index("term")
    )

    assert (
        coefficients.loc[
            "ndx_log_return",
            "coef",
        ]
        ==
        pytest.approx(
            0.705377,
            abs=1e-6,
        )
    )

    assert (
        coefficients.loc[
            "dollar_log_return",
            "coef",
        ]
        ==
        pytest.approx(
            -0.959921,
            abs=1e-6,
        )
    )
