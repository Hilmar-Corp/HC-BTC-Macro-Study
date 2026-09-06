from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from hc_macro_integration.paths import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR
from hc_macro_integration.models.linear import PRIMARY_FACTORS, Y


WINDOWS = [126, 252, 504]


def _fit_one_window(
    sample: pd.DataFrame,
    hac_maxlags: int = 5,
) -> dict:
    y = sample[Y].astype(float)

    X = sample[PRIMARY_FACTORS].astype(float)
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
        cov_kwds={"maxlags": hac_maxlags},
    )

    row = {
        "adj_r2": float(model.rsquared_adj),
        "r2": float(model.rsquared),
        "alpha": float(model.params["const"]),
    }

    for factor in PRIMARY_FACTORS:
        beta = float(model.params[factor])
        se = float(model.bse[factor])

        row[f"beta__{factor}"] = beta
        row[f"se__{factor}"] = se
        row[f"ci95_low__{factor}"] = beta - 1.96 * se
        row[f"ci95_high__{factor}"] = beta + 1.96 * se
        row[f"p__{factor}"] = float(
            model.pvalues[factor]
        )

    return row


def compute_rolling(
    panel: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    data = (
        panel.loc[
            panel["primary_complete"],
            ["date", Y] + PRIMARY_FACTORS,
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    rows = []

    for end_idx in range(
        window - 1,
        len(data),
    ):
        start_idx = end_idx - window + 1

        sample = data.iloc[
            start_idx : end_idx + 1
        ]

        result = _fit_one_window(
            sample,
            hac_maxlags=5,
        )

        result["date"] = data.loc[
            end_idx,
            "date",
        ]

        result["window"] = window
        result["nobs"] = window

        rows.append(result)

    return pd.DataFrame(rows)


def _plot_series(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        pd.to_datetime(df["date"]),
        df[y_col],
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def _plot_beta_with_ci(
    df: pd.DataFrame,
    factor: str,
    title: str,
    filename: str,
) -> None:
    dates = pd.to_datetime(df["date"])

    beta = df[f"beta__{factor}"]
    low = df[f"ci95_low__{factor}"]
    high = df[f"ci95_high__{factor}"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        dates,
        beta,
        label="Beta",
    )

    ax.fill_between(
        dates,
        low,
        high,
        alpha=0.2,
        label="IC 95 % HAC",
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Beta")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_rolling_analysis(
    panel: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    results = {}

    for window in WINDOWS:
        print(
            f"[ROLLING] window={window}"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            df = compute_rolling(
                panel=panel,
                window=window,
            )

        results[window] = df

        path = (
            TABLES_DIR
            / f"rolling_macro_window_{window}.csv"
        )

        df.to_csv(
            path,
            index=False,
        )

        _plot_series(
            df=df,
            y_col="adj_r2",
            title=(
                "Bitcoin — pouvoir explicatif macro "
                f"roulant ({window} séances)"
            ),
            ylabel="R² ajusté",
            filename=(
                f"rolling_adj_r2_{window}.png"
            ),
        )

    primary = results[252]

    factors_titles = {
        "ndx_log_return": (
            "Bitcoin — sensibilité roulante au Nasdaq-100"
        ),
        "dollar_log_return": (
            "Bitcoin — sensibilité roulante au dollar large"
        ),
        "real_rate_change": (
            "Bitcoin — sensibilité roulante au taux réel US 10 ans"
        ),
        "credit_spread_change": (
            "Bitcoin — sensibilité roulante au spread de crédit"
        ),
    }

    for factor, title in factors_titles.items():
        _plot_beta_with_ci(
            df=primary,
            factor=factor,
            title=title + " — fenêtre 252 séances",
            filename=f"rolling_beta_{factor}_252.png",
        )

    return results


if __name__ == "__main__":
    panel = pd.read_parquet(
        PROCESSED_DIR / "panel_primary.parquet"
    )

    run_rolling_analysis(panel)
