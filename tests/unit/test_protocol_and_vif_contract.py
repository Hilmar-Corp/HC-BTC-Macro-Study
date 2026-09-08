from __future__ import annotations

import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)

from hc_macro_integration.config import load_protocol
from hc_macro_integration.models.linear import (
    _vif_table,
)


def test_structural_break_protocol_matches_implementation():
    protocol = load_protocol()

    structural = protocol[
        "structural_breaks"
    ]

    segmentation = structural[
        "segmentation"
    ]

    assert (
        segmentation["method"]
        == "custom_segmented_ols_bic"
    )
    assert (
        segmentation["maximum_breaks"]
        == 3
    )
    assert (
        segmentation[
            "minimum_segment_fraction"
        ]
        == pytest.approx(0.15)
    )
    assert (
        segmentation[
            "candidate_step_sessions"
        ]
        == 5
    )
    assert (
        segmentation[
            "bic_parameterization"
        ]
        == "(segments * p) + breaks"
    )

    unknown = structural[
        "unknown_break_diagnostic"
    ]

    assert (
        unknown["method"]
        == (
            "supf_like_fixed_x_residual_"
            "moving_block_bootstrap"
        )
    )
    assert (
        unknown["trim_fraction"]
        == pytest.approx(0.15)
    )
    assert (
        unknown[
            "candidate_step_sessions"
        ]
        == 5
    )
    assert (
        unknown[
            "exact_robustness_step_sessions"
        ]
        == 1
    )
    assert unknown["block_length"] == 20
    assert unknown["bootstrap_count"] == 499


def test_vif_uses_intercept_consistent_with_main_regression():
    # Deliberately non-centred factors. With an intercept,
    # VIF is driven by centred linear dependence. Without an
    # intercept, the large common levels create a materially
    # different auxiliary-regression geometry.
    data = pd.DataFrame(
        {
            "x1": [
                100.0,
                101.0,
                99.0,
                102.0,
                98.0,
                103.0,
                97.0,
                104.0,
                96.0,
                105.0,
            ],
            "x2": [
                200.0,
                198.0,
                201.0,
                197.0,
                202.0,
                203.0,
                196.0,
                204.0,
                195.0,
                205.0,
            ],
            "x3": [
                50.0,
                52.0,
                49.0,
                51.0,
                48.0,
                53.0,
                47.0,
                54.0,
                46.0,
                55.0,
            ],
        }
    )

    factors = [
        "x1",
        "x2",
        "x3",
    ]

    actual = (
        _vif_table(
            clean=data,
            x_cols=factors,
            model_name="test",
        )
        .set_index("factor")[
            "vif"
        ]
    )

    design = sm.add_constant(
        data[factors],
        has_constant="add",
    )

    for factor in factors:
        index = int(
            design.columns.get_loc(
                factor
            )
        )

        expected = (
            variance_inflation_factor(
                design.to_numpy(
                    dtype=float
                ),
                index,
            )
        )

        assert actual.loc[
            factor
        ] == pytest.approx(
            expected,
            rel=1e-12,
            abs=1e-12,
        )

    # Independent guard against silently dropping the intercept.
    # We compute the auxiliary regression explicitly instead of
    # relying on variance_inflation_factor for the no-constant case.
    no_intercept_model = sm.OLS(
        data["x1"].astype(float),
        data[
            [
                "x2",
                "x3",
            ]
        ].astype(float),
    ).fit()

    no_intercept_vif = (
        1.0
        / (
            1.0
            - float(
                no_intercept_model.rsquared
            )
        )
    )

    assert abs(
        actual.loc["x1"]
        - no_intercept_vif
    ) > 1.0
