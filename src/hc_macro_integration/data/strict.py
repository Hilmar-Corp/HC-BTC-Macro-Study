from __future__ import annotations

import numpy as np
import pandas as pd

from hc_macro_integration.paths import FRED_DIR, PROCESSED_DIR


SERIES = {
    "DTWEXBGS": "dollar_exact",
    "DFII10": "real_rate_exact",
    "BAA10Y": "credit_exact",
}


def _load_series(
    series_id: str,
    output_name: str,
) -> pd.DataFrame:
    df = pd.read_parquet(
        FRED_DIR / f"{series_id}.parquet"
    ).copy()

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df = (
        df[["date", "value"]]
        .rename(columns={"value": output_name})
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )

    return df


def build_strict_panel() -> pd.DataFrame:
    path = PROCESSED_DIR / "panel_primary.parquet"

    panel = pd.read_parquet(path).copy()

    panel["date"] = (
        pd.to_datetime(panel["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    for series_id, output_name in SERIES.items():
        df = _load_series(
            series_id=series_id,
            output_name=output_name,
        )

        panel = panel.merge(
            df,
            on="date",
            how="left",
            validate="one_to_one",
        )

    # Strict means:
    # current Nasdaq session must have an exact FRED observation
    # AND the immediately previous Nasdaq session must also have one.
    panel["dollar_log_return_strict"] = np.where(
        panel["dollar_exact"].notna()
        & panel["dollar_exact"].shift(1).notna(),
        np.log(
            panel["dollar_exact"]
            / panel["dollar_exact"].shift(1)
        ),
        np.nan,
    )

    panel["real_rate_change_strict"] = np.where(
        panel["real_rate_exact"].notna()
        & panel["real_rate_exact"].shift(1).notna(),
        panel["real_rate_exact"]
        - panel["real_rate_exact"].shift(1),
        np.nan,
    )

    panel["credit_spread_change_strict"] = np.where(
        panel["credit_exact"].notna()
        & panel["credit_exact"].shift(1).notna(),
        panel["credit_exact"]
        - panel["credit_exact"].shift(1),
        np.nan,
    )

    strict_cols = [
        "btc_log_return",
        "ndx_log_return",
        "dollar_log_return_strict",
        "real_rate_change_strict",
        "credit_spread_change_strict",
    ]

    panel["strict_complete"] = (
        panel[strict_cols]
        .notna()
        .all(axis=1)
    )

    out_path = (
        PROCESSED_DIR / "panel_strict_exact_fred.parquet"
    )

    panel.to_parquet(out_path, index=False)

    return panel


if __name__ == "__main__":
    df = build_strict_panel()

    print(
        "STRICT_COMPLETE_ROWS="
        + str(int(df["strict_complete"].sum()))
    )
