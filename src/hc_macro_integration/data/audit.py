from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hc_macro_integration.paths import DIAGNOSTICS_DIR, FRED_DIR, PROCESSED_DIR


FRED_MAP = {
    "DTWEXBGS": {
        "name": "dollar",
        "panel_level": "dollar_index",
        "used_level": "dollar_level_used",
    },
    "DFII10": {
        "name": "real_rate",
        "panel_level": "real_rate_10y",
        "used_level": "real_rate_level_used",
    },
    "BAA10Y": {
        "name": "credit",
        "panel_level": "baa10y_spread",
        "used_level": "credit_spread_level_used",
    },
}


def _load_raw_series(series_id: str) -> pd.DataFrame:
    path = FRED_DIR / f"{series_id}.parquet"

    df = pd.read_parquet(path).copy()

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df = (
        df.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    return df


def run_factor_audit() -> pd.DataFrame:
    panel_path = PROCESSED_DIR / "panel_primary.parquet"
    panel = pd.read_parquet(panel_path).copy()

    panel["date"] = (
        pd.to_datetime(panel["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    summary_rows = []
    stale_rows = []

    for series_id, spec in FRED_MAP.items():
        raw = _load_raw_series(series_id)

        exact = raw.rename(
            columns={"value": "exact_value"}
        )[["date", "exact_value"]]

        tmp = panel[["date"]].merge(
            exact,
            on="date",
            how="left",
            validate="one_to_one",
        )

        tmp["exact_available"] = tmp["exact_value"].notna()

        row_index = np.arange(len(tmp), dtype=float)

        source_index = pd.Series(
            np.where(
                tmp["exact_available"],
                row_index,
                np.nan,
            ),
            index=tmp.index,
        ).ffill()

        source_date = pd.Series(
            tmp["date"].where(tmp["exact_available"]),
            index=tmp.index,
        ).ffill()

        tmp["source_date"] = source_date

        tmp["age_sessions"] = (
            pd.Series(row_index, index=tmp.index)
            - source_index
        )

        tmp["age_calendar_days"] = (
            tmp["date"] - tmp["source_date"]
        ).dt.days

        used_col = spec["used_level"]

        tmp["used_value"] = panel[used_col].to_numpy()

        tmp["forward_filled"] = (
            tmp["used_value"].notna()
            & ~tmp["exact_available"]
        )

        exact_count = int(tmp["exact_available"].sum())
        ffill_count = int(tmp["forward_filled"].sum())

        valid_age = tmp["age_sessions"].dropna()

        summary_rows.append(
            {
                "series_id": series_id,
                "factor": spec["name"],
                "schedule_rows": len(tmp),
                "exact_count": exact_count,
                "exact_rate": exact_count / len(tmp),
                "forward_filled_count": ffill_count,
                "forward_filled_rate": ffill_count / len(tmp),
                "max_age_sessions": (
                    float(valid_age.max())
                    if len(valid_age)
                    else np.nan
                ),
                "p95_age_sessions": (
                    float(valid_age.quantile(0.95))
                    if len(valid_age)
                    else np.nan
                ),
                "max_age_calendar_days": (
                    float(
                        tmp["age_calendar_days"]
                        .dropna()
                        .max()
                    )
                    if tmp["age_calendar_days"].notna().any()
                    else np.nan
                ),
                "stale_gt_1_session_count": int(
                    (tmp["age_sessions"] > 1).sum()
                ),
                "stale_gt_2_session_count": int(
                    (tmp["age_sessions"] > 2).sum()
                ),
            }
        )

        stale = tmp.loc[
            tmp["forward_filled"],
            [
                "date",
                "source_date",
                "age_sessions",
                "age_calendar_days",
                "exact_value",
                "used_value",
            ],
        ].copy()

        stale.insert(0, "factor", spec["name"])
        stale.insert(0, "series_id", series_id)

        stale_rows.append(stale)

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        DIAGNOSTICS_DIR / "factor_alignment_audit.csv"
    )
    summary.to_csv(summary_path, index=False)

    if stale_rows:
        stale_all = pd.concat(
            stale_rows,
            ignore_index=True,
        )
    else:
        stale_all = pd.DataFrame()

    stale_path = (
        DIAGNOSTICS_DIR / "factor_forward_filled_sessions.csv"
    )
    stale_all.to_csv(stale_path, index=False)

    json_path = (
        DIAGNOSTICS_DIR / "factor_alignment_audit.json"
    )

    json_path.write_text(
        json.dumps(
            summary.to_dict(orient="records"),
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return summary


if __name__ == "__main__":
    result = run_factor_audit()
    print(result.to_string(index=False))
