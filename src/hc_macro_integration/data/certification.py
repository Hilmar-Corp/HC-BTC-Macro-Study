from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from hc_macro_integration.paths import DIAGNOSTICS_DIR, PROCESSED_DIR


PRIMARY_COLUMNS = [
    "btc_log_return",
    "ndx_log_return",
    "dollar_log_return",
    "real_rate_change",
    "credit_spread_change",
]


def sha256_file(path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def run_certification() -> None:
    path = PROCESSED_DIR / "panel_primary.parquet"

    panel = pd.read_parquet(path)

    complete = panel.loc[
        panel["primary_complete"]
    ].copy()

    diagnostics = {
        "rows_total": int(len(panel)),
        "rows_primary_complete": int(len(complete)),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "date_unique": bool(panel["date"].is_unique),
        "market_close_unique": bool(panel["market_close"].is_unique),
        "date_monotonic": bool(panel["date"].is_monotonic_increasing),
        "btc_exact_match_count": int(panel["btc_exact_match"].sum()),
        "btc_exact_match_rate": float(panel["btc_exact_match"].mean()),
        "missing_btc_exact": int(panel["btc_close"].isna().sum()),
        "missing_ndx": int(panel["ndx_close"].isna().sum()),
        "finite_primary_complete": bool(
            np.isfinite(
                complete[PRIMARY_COLUMNS].to_numpy(dtype=float)
            ).all()
        ),
        "btc_positive": bool(
            panel["btc_close"].dropna().gt(0).all()
        ),
        "ndx_positive": bool(
            panel["ndx_close"].dropna().gt(0).all()
        ),
        "sha256_panel": sha256_file(path),
    }

    missing_dates = panel.loc[
        ~panel["btc_exact_match"],
        ["date", "market_close"],
    ].copy()

    missing_dates_path = (
        DIAGNOSTICS_DIR / "missing_exact_btc_boundaries.csv"
    )

    missing_dates.to_csv(
        missing_dates_path,
        index=False,
    )

    json_path = (
        DIAGNOSTICS_DIR / "panel_certification.json"
    )

    json_path.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("HILMARCORP — PANEL CERTIFICATION")
    print("=" * 88)

    for key, value in diagnostics.items():
        print(f"{key.upper()}={value}")

    hard_failures = []

    if not diagnostics["date_unique"]:
        hard_failures.append("NON_UNIQUE_DATE")

    if not diagnostics["market_close_unique"]:
        hard_failures.append("NON_UNIQUE_MARKET_CLOSE")

    if not diagnostics["date_monotonic"]:
        hard_failures.append("NON_MONOTONIC_DATE")

    if not diagnostics["finite_primary_complete"]:
        hard_failures.append("NON_FINITE_PRIMARY_DATA")

    if not diagnostics["btc_positive"]:
        hard_failures.append("NON_POSITIVE_BTC")

    if not diagnostics["ndx_positive"]:
        hard_failures.append("NON_POSITIVE_NDX")

    print("-" * 88)

    if hard_failures:
        print("CERTIFICATION_STATUS=FAIL")
        print(
            "FAILURES="
            + ",".join(hard_failures)
        )
        raise SystemExit(1)

    print("CERTIFICATION_STATUS=PASS")
    print(f"DIAGNOSTICS={json_path}")
    print(f"MISSING_BOUNDARIES={missing_dates_path}")


if __name__ == "__main__":
    run_certification()
