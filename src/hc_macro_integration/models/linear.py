from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)

from hc_macro_integration.paths import PROCESSED_DIR, TABLES_DIR


Y = "btc_log_return"

PRIMARY_FACTORS = [
    "ndx_log_return",
    "dollar_log_return",
    "real_rate_change",
    "credit_spread_change",
]

STRICT_FACTORS = [
    "ndx_log_return",
    "dollar_log_return_strict",
    "real_rate_change_strict",
    "credit_spread_change_strict",
]


@dataclass
class RegressionResult:
    coefficients: pd.DataFrame
    metrics: dict
    vif: pd.DataFrame


def _fit_hac(
    data: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    maxlags: int = 5,
):
    clean = (
        data[[y_col] + x_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    y = clean[y_col].astype(float)

    X = clean[x_cols].astype(float)
    X = sm.add_constant(
        X,
        has_constant="add",
    )

    model = sm.OLS(
        y,
        X,
        missing="raise",
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )

    return model, clean


def _coef_table(
    model,
    model_name: str,
) -> pd.DataFrame:
    ci = model.conf_int(alpha=0.05)

    out = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "se_hac": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci95_low": ci.iloc[:, 0].values,
            "ci95_high": ci.iloc[:, 1].values,
        }
    )

    out.insert(0, "model", model_name)

    return out


def _standardized_betas(
    clean: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    maxlags: int,
    model_name: str,
) -> pd.DataFrame:
    z = clean[[y_col] + x_cols].copy()

    for col in z.columns:
        std = z[col].std(ddof=0)

        if not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Cannot standardize {col}: std={std}"
            )

        z[col] = (
            z[col] - z[col].mean()
        ) / std

    model, _ = _fit_hac(
        z,
        y_col=y_col,
        x_cols=x_cols,
        maxlags=maxlags,
    )

    out = _coef_table(
        model,
        model_name=model_name,
    )

    out = out.loc[
        out["term"] != "const"
    ].copy()

    out = out.rename(
        columns={"coef": "standardized_beta"}
    )

    return out[
        [
            "model",
            "term",
            "standardized_beta",
            "se_hac",
            "t_stat",
            "p_value",
            "ci95_low",
            "ci95_high",
        ]
    ]


def _vif_table(
    clean: pd.DataFrame,
    x_cols: list[str],
    model_name: str,
) -> pd.DataFrame:
    X = clean[x_cols].astype(float)

    rows = []

    for i, col in enumerate(x_cols):
        vif = variance_inflation_factor(
            X.to_numpy(),
            i,
        )

        rows.append(
            {
                "model": model_name,
                "factor": col,
                "vif": float(vif),
            }
        )

    return pd.DataFrame(rows)


def fit_model(
    data: pd.DataFrame,
    x_cols: list[str],
    model_name: str,
    maxlags: int = 5,
) -> RegressionResult:
    model, clean = _fit_hac(
        data=data,
        y_col=Y,
        x_cols=x_cols,
        maxlags=maxlags,
    )

    coef = _coef_table(
        model,
        model_name=model_name,
    )

    standardized = _standardized_betas(
        clean=clean,
        y_col=Y,
        x_cols=x_cols,
        maxlags=maxlags,
        model_name=model_name,
    )

    coef = coef.merge(
        standardized[
            ["term", "standardized_beta"]
        ],
        on="term",
        how="left",
    )

    metrics = {
        "model": model_name,
        "nobs": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "hac_maxlags": int(maxlags),
        "resid_std": float(
            np.std(model.resid, ddof=1)
        ),
    }

    vif = _vif_table(
        clean=clean,
        x_cols=x_cols,
        model_name=model_name,
    )

    return RegressionResult(
        coefficients=coef,
        metrics=metrics,
        vif=vif,
    )


def run_full_sample_regressions(
    strict_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = strict_panel.loc[
        strict_panel["primary_complete"]
    ].copy()

    strict = strict_panel.loc[
        strict_panel["strict_complete"]
    ].copy()

    primary_result = fit_model(
        data=primary,
        x_cols=PRIMARY_FACTORS,
        model_name="primary_ffill",
        maxlags=5,
    )

    strict_result = fit_model(
        data=strict,
        x_cols=STRICT_FACTORS,
        model_name="strict_exact",
        maxlags=5,
    )

    coef = pd.concat(
        [
            primary_result.coefficients,
            strict_result.coefficients,
        ],
        ignore_index=True,
    )

    metrics = pd.DataFrame(
        [
            primary_result.metrics,
            strict_result.metrics,
        ]
    )

    vif = pd.concat(
        [
            primary_result.vif,
            strict_result.vif,
        ],
        ignore_index=True,
    )

    coef.to_csv(
        TABLES_DIR / "full_sample_coefficients.csv",
        index=False,
    )

    metrics.to_csv(
        TABLES_DIR / "full_sample_metrics.csv",
        index=False,
    )

    vif.to_csv(
        TABLES_DIR / "vif.csv",
        index=False,
    )

    return coef, metrics, vif


if __name__ == "__main__":
    panel = pd.read_parquet(
        PROCESSED_DIR / "panel_strict_exact_fred.parquet"
    )

    coef, metrics, vif = run_full_sample_regressions(
        panel
    )

    print(metrics.to_string(index=False))
    print()
    print(coef.to_string(index=False))
    print()
    print(vif.to_string(index=False))
