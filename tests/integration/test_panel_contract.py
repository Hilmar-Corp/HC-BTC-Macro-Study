from __future__ import annotations

import pandas as pd

from hc_macro_integration.paths import PROCESSED_DIR


def test_primary_panel_certified_shape():
    panel = pd.read_parquet(
        PROCESSED_DIR
        / "panel_primary.parquet"
    )

    assert len(panel) == 2275

    assert int(
        panel["primary_complete"].sum()
    ) == 2274

    dates = pd.to_datetime(
        panel["date"]
    )

    assert dates.is_unique
    assert dates.is_monotonic_increasing

    assert (
        dates.min().date().isoformat()
        == "2017-08-17"
    )

    assert (
        dates.max().date().isoformat()
        == "2026-09-04"
    )


def test_strict_panel_certified_shape():
    panel = pd.read_parquet(
        PROCESSED_DIR
        / "panel_strict_exact_fred.parquet"
    )

    assert int(
        panel["strict_complete"].sum()
    ) == 2215
