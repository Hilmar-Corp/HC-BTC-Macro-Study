from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from hc_macro_integration.config import load_protocol
from hc_macro_integration.paths import COINBASE_DIR, FRED_DIR, MARKET_DIR, PROCESSED_DIR


def build_nasdaq_schedule(
    start: str,
    end: str,
) -> pd.DataFrame:
    calendar = mcal.get_calendar("NASDAQ")

    schedule = calendar.schedule(
        start_date=start,
        end_date=end,
    )

    out = schedule.reset_index()

    first_col = out.columns[0]
    out = out.rename(columns={first_col: "date"})

    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()

    out["market_close"] = pd.to_datetime(
        out["market_close"],
        utc=True,
    )

    return out[["date", "market_close"]]


def attach_exact_btc_close(
    schedule: pd.DataFrame,
    btc: pd.DataFrame,
) -> pd.DataFrame:
    btc_map = btc[
        ["close_time", "close"]
    ].rename(
        columns={
            "close_time": "market_close",
            "close": "btc_close",
        }
    )

    merged = schedule.merge(
        btc_map,
        how="left",
        on="market_close",
        validate="one_to_one",
    )

    merged["btc_exact_match"] = merged["btc_close"].notna()

    return merged


def attach_nasdaq(
    panel: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    market = market.copy()

    if "adj_close" in market.columns:
        px_col = "adj_close"
    else:
        px_col = "close"

    market = market[
        ["date", px_col]
    ].rename(columns={px_col: "ndx_close"})

    out = panel.merge(
        market,
        on="date",
        how="left",
        validate="one_to_one",
    )

    return out


def attach_fred(
    panel: pd.DataFrame,
    series_id: str,
    output_name: str,
) -> pd.DataFrame:
    df = pd.read_parquet(
        FRED_DIR / f"{series_id}.parquet"
    )

    df = df.rename(columns={"value": output_name})

    return panel.merge(
        df[["date", output_name]],
        on="date",
        how="left",
        validate="one_to_one",
    )


def carry_last_available(
    series: pd.Series,
) -> pd.Series:
    return series.ffill()


def build_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    out["btc_log_return"] = np.log(
        out["btc_close"] / out["btc_close"].shift(1)
    )

    out["ndx_log_return"] = np.log(
        out["ndx_close"] / out["ndx_close"].shift(1)
    )

    # FRED market series can contain missing calendar dates.
    # Values are carried to the next Nasdaq session before changes are computed.
    out["dollar_level_used"] = carry_last_available(
        out["dollar_index"]
    )
    out["real_rate_level_used"] = carry_last_available(
        out["real_rate_10y"]
    )
    out["credit_spread_level_used"] = carry_last_available(
        out["baa10y_spread"]
    )

    out["dollar_log_return"] = np.log(
        out["dollar_level_used"]
        / out["dollar_level_used"].shift(1)
    )

    out["real_rate_change"] = (
        out["real_rate_level_used"]
        - out["real_rate_level_used"].shift(1)
    )

    out["credit_spread_change"] = (
        out["credit_spread_level_used"]
        - out["credit_spread_level_used"].shift(1)
    )

    return out


def run_build_panel() -> None:
    protocol = load_protocol()

    start = protocol["sample"]["start"]

    btc = pd.read_parquet(
        COINBASE_DIR / "btc_usd_hourly.parquet"
    )

    market = pd.read_parquet(
        MARKET_DIR / "nasdaq100_daily.parquet"
    )

    last_market_date = market["date"].max()

    schedule = build_nasdaq_schedule(
        start=start,
        end=last_market_date.strftime("%Y-%m-%d"),
    )

    panel = attach_exact_btc_close(
        schedule=schedule,
        btc=btc,
    )

    panel = attach_nasdaq(
        panel=panel,
        market=market,
    )

    fred_map = {
        "DTWEXBGS": "dollar_index",
        "DFII10": "real_rate_10y",
        "BAA10Y": "baa10y_spread",
    }

    for series_id, output_name in fred_map.items():
        panel = attach_fred(
            panel=panel,
            series_id=series_id,
            output_name=output_name,
        )

    panel = build_returns(panel)

    factor_cols = [
        "btc_log_return",
        "ndx_log_return",
        "dollar_log_return",
        "real_rate_change",
        "credit_spread_change",
    ]

    panel["primary_complete"] = (
        panel[factor_cols].notna().all(axis=1)
    )

    path = PROCESSED_DIR / "panel_primary.parquet"
    panel.to_parquet(path, index=False)

    csv_path = PROCESSED_DIR / "panel_primary.csv"
    panel.to_csv(csv_path, index=False)

    print("=" * 88)
    print("HC BTC MACRO INTEGRATION — PANEL BUILT")
    print("=" * 88)
    print(f"ROWS={len(panel)}")
    print(f"START={panel['date'].min().date()}")
    print(f"END={panel['date'].max().date()}")
    print(
        "BTC_EXACT_MATCH_RATE="
        f"{panel['btc_exact_match'].mean():.6%}"
    )
    print(
        "PRIMARY_COMPLETE_ROWS="
        f"{int(panel['primary_complete'].sum())}"
    )
    print(f"PARQUET={path}")
    print(f"CSV={csv_path}")


if __name__ == "__main__":
    run_build_panel()
