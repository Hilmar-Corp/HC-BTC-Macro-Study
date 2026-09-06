from __future__ import annotations


import numpy as np
import pandas as pd
import statsmodels.api as sm

from hc_macro_integration.paths import PROCESSED_DIR, TABLES_DIR


CANONICAL_FACTORS = [
    "ndx",
    "usd",
    "real_rate",
    "credit",
]

PERIOD_ORDER = [
    "pre_2020",
    "post_covid_pre_etf",
    "spot_etf_era",
]


def prepare_data(
    variant: str,
) -> pd.DataFrame:
    panel = pd.read_parquet(
        PROCESSED_DIR / "panel_strict_exact_fred.parquet"
    ).copy()

    panel["date"] = pd.to_datetime(panel["date"])

    if variant == "primary":
        mask = panel["primary_complete"]

        mapping = {
            "btc_log_return": "btc",
            "ndx_log_return": "ndx",
            "dollar_log_return": "usd",
            "real_rate_change": "real_rate",
            "credit_spread_change": "credit",
        }

    elif variant == "strict":
        mask = panel["strict_complete"]

        mapping = {
            "btc_log_return": "btc",
            "ndx_log_return": "ndx",
            "dollar_log_return_strict": "usd",
            "real_rate_change_strict": "real_rate",
            "credit_spread_change_strict": "credit",
        }

    else:
        raise ValueError(
            f"Unknown variant: {variant}"
        )

    cols = ["date"] + list(mapping.keys())

    out = (
        panel.loc[mask, cols]
        .rename(columns=mapping)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    out["period"] = np.select(
        [
            out["date"] < pd.Timestamp("2020-03-01"),
            (
                (out["date"] >= pd.Timestamp("2020-03-01"))
                & (out["date"] <= pd.Timestamp("2024-01-10"))
            ),
            out["date"] >= pd.Timestamp("2024-01-11"),
        ],
        PERIOD_ORDER,
        default="unknown",
    )

    if (out["period"] == "unknown").any():
        raise RuntimeError(
            "Unknown dates in period assignment."
        )

    return out


def fit_period_model(
    data: pd.DataFrame,
    period: str,
    variant: str,
) -> tuple[pd.DataFrame, dict]:
    sample = data.loc[
        data["period"] == period
    ].copy()

    X = sm.add_constant(
        sample[CANONICAL_FACTORS],
        has_constant="add",
    )

    model = sm.OLS(
        sample["btc"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5},
    )

    ci = model.conf_int(alpha=0.05)

    coef = pd.DataFrame(
        {
            "variant": variant,
            "period": period,
            "term": model.params.index,
            "coef": model.params.values,
            "se_hac": model.bse.values,
            "p_value": model.pvalues.values,
            "ci95_low": ci.iloc[:, 0].values,
            "ci95_high": ci.iloc[:, 1].values,
        }
    )

    metrics = {
        "variant": variant,
        "period": period,
        "start": sample["date"].min(),
        "end": sample["date"].max(),
        "nobs": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }

    return coef, metrics


def run_period_models(
    variants=("primary", "strict"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_coef = []
    all_metrics = []

    for variant in variants:
        data = prepare_data(variant)

        for period in PERIOD_ORDER:
            coef, metrics = fit_period_model(
                data=data,
                period=period,
                variant=variant,
            )

            all_coef.append(coef)
            all_metrics.append(metrics)

    coef_df = pd.concat(
        all_coef,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(all_metrics)

    coef_df.to_csv(
        TABLES_DIR / "period_coefficients.csv",
        index=False,
    )

    metrics_df.to_csv(
        TABLES_DIR / "period_metrics.csv",
        index=False,
    )

    return coef_df, metrics_df
