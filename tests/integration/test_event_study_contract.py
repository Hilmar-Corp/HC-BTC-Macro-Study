from __future__ import annotations

import pandas as pd

from hc_macro_integration.paths import TABLES_DIR


def test_no_corrected_event_result_survives_fdr():
    tests = pd.read_csv(
        TABLES_DIR
        / "phase6b_corrected_difference_tests.csv"
    )

    assert not bool(
        tests["reject_fdr_5pct"].any()
    )


def test_ndx_down_same_day_is_directionally_stronger_post_2020():
    tests = pd.read_csv(
        TABLES_DIR
        / "phase6b_corrected_difference_tests.csv"
    )

    row = tests.loc[
        (
            tests["event"]
            == "NDX_DOWN_10"
        )
        &
        (
            tests["horizon"]
            == "SAME_DAY"
        )
        &
        (
            tests["comparison"]
            == "P2_MINUS_P1"
        )
    ].iloc[0]

    assert row["mean_difference"] < -0.02

    assert (
        row["studentized_permutation_p"]
        < 0.05
    )

    assert row["q_value_bh"] > 0.05
