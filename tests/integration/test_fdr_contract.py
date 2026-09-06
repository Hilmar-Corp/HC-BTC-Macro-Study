from __future__ import annotations

import pandas as pd

from hc_macro_integration.paths import TABLES_DIR


def test_only_ndx_changes_survive_primary_factor_fdr():
    df = pd.read_csv(
        TABLES_DIR
        / "interaction_secondary_fdr.csv"
    )

    primary = df.loc[
        df["variant"] == "primary"
    ]

    rejected = set(
        primary.loc[
            primary["reject_fdr_5pct"],
            "test",
        ]
    )

    assert rejected == {
        "P2_vs_P1__ndx",
        "P3_vs_P1__ndx",
    }
