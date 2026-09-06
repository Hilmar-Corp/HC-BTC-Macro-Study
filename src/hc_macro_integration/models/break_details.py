from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from hc_macro_integration.paths import TABLES_DIR
from hc_macro_integration.models.periods import (
    CANONICAL_FACTORS,
    prepare_data,
)


def fit_segment(
    data: pd.DataFrame,
    label: str,
    variant: str,
) -> tuple[pd.DataFrame, dict]:
    X = sm.add_constant(
        data[CANONICAL_FACTORS],
        has_constant="add",
    )

    model = sm.OLS(
        data["btc"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5},
    )

    ci = model.conf_int()

    coef = pd.DataFrame(
        {
            "variant": variant,
            "segment": label,
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
        "segment": label,
        "start": data["date"].min(),
        "end": data["date"].max(),
        "nobs": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(
            model.rsquared_adj
        ),
    }

    return coef, metrics


def run_break_details(
    supf_summary: pd.DataFrame,
):
    all_coef = []
    all_metrics = []

    for _, row in (
        supf_summary.iterrows()
    ):
        variant = row["variant"]

        break_date = pd.Timestamp(
            row["selected_break_date"]
        )

        data = prepare_data(
            variant
        )

        before = data.loc[
            data["date"] < break_date
        ].copy()

        after = data.loc[
            data["date"] >= break_date
        ].copy()

        coef_before, metrics_before = (
            fit_segment(
                before,
                label="before_supf_break",
                variant=variant,
            )
        )

        coef_after, metrics_after = (
            fit_segment(
                after,
                label="after_supf_break",
                variant=variant,
            )
        )

        all_coef.extend(
            [
                coef_before,
                coef_after,
            ]
        )

        all_metrics.extend(
            [
                metrics_before,
                metrics_after,
            ]
        )

    coef_df = pd.concat(
        all_coef,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    coef_df.to_csv(
        TABLES_DIR
        / "supf_break_segment_coefficients.csv",
        index=False,
    )

    metrics_df.to_csv(
        TABLES_DIR
        / "supf_break_segment_metrics.csv",
        index=False,
    )

    return (
        coef_df,
        metrics_df,
    )
