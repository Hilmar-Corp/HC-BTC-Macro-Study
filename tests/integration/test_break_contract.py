from __future__ import annotations

import pandas as pd

from hc_macro_integration.paths import TABLES_DIR


def test_exact_break_scan_converges_on_march_2020():
    df = pd.read_csv(
        TABLES_DIR
        / "phase5_break_robustness.csv"
    )

    assert set(
        df["best_date"].astype(str)
    ) == {
        "2020-03-09"
    }


def test_ndx_only_break_plateau_is_tight():
    df = pd.read_csv(
        TABLES_DIR
        / "phase5_break_robustness.csv"
    )

    ndx = df.loc[
        df["factor_set"]
        == "NDX_ONLY"
    ]

    assert set(
        ndx[
            "plateau_95_count"
        ].astype(int)
    ) == {
        2
    }
