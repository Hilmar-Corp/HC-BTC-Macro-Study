from __future__ import annotations

import pytest


def test_frozen_period_r2(
    frozen_results,
):
    periods = (
        frozen_results[
            "main_results"
        ][
            "period_r2"
        ]
    )

    assert (
        periods[
            "pre_2020"
        ][
            "adj_r2"
        ]
        ==
        pytest.approx(
            -0.004812,
            abs=5e-6,
        )
    )

    assert (
        periods[
            "post_covid_pre_etf"
        ][
            "adj_r2"
        ]
        ==
        pytest.approx(
            0.167010,
            abs=5e-6,
        )
    )

    assert (
        periods[
            "spot_etf_era"
        ][
            "adj_r2"
        ]
        ==
        pytest.approx(
            0.176373,
            abs=5e-6,
        )
    )


def test_frozen_ndx_betas(
    frozen_results,
):
    ndx = (
        frozen_results[
            "main_results"
        ][
            "period_betas"
        ][
            "ndx"
        ]
    )

    assert (
        ndx[
            "pre_2020"
        ]
        ==
        pytest.approx(
            0.126884,
            abs=5e-6,
        )
    )

    assert (
        ndx[
            "post_covid_pre_etf"
        ]
        ==
        pytest.approx(
            0.817706,
            abs=5e-6,
        )
    )

    assert (
        ndx[
            "spot_etf_era"
        ]
        ==
        pytest.approx(
            0.880690,
            abs=5e-6,
        )
    )


def test_etf_era_is_not_additional_joint_break(
    frozen_results,
):
    joint = (
        frozen_results[
            "main_results"
        ][
            "joint_wald"
        ]
    )

    assert (
        joint[
            "P2_vs_P1_joint_beta_change"
        ][
            "p_value"
        ]
        < 0.01
    )

    assert (
        joint[
            "P3_vs_P1_joint_beta_change"
        ][
            "p_value"
        ]
        < 0.01
    )

    assert (
        joint[
            "P3_vs_P2_joint_beta_change"
        ][
            "p_value"
        ]
        > 0.70
    )


def test_post_2020_r2_is_equity_dominated(
    frozen_results,
):
    shapley = (
        frozen_results[
            "main_results"
        ][
            "shapley"
        ]
    )

    assert (
        shapley[
            "post_covid_pre_etf"
        ][
            "ndx"
        ][
            "share"
        ]
        > 0.75
    )

    assert (
        shapley[
            "spot_etf_era"
        ][
            "ndx"
        ][
            "share"
        ]
        > 0.80
    )
