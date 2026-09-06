from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from hc_macro_integration.paths import TABLES_DIR
from hc_macro_integration.models.periods import (
    CANONICAL_FACTORS,
    prepare_data,
)


def build_interaction_design(
    data: pd.DataFrame,
) -> pd.DataFrame:
    X = data[CANONICAL_FACTORS].copy()

    X["D2"] = (
        data["period"] == "post_covid_pre_etf"
    ).astype(float)

    X["D3"] = (
        data["period"] == "spot_etf_era"
    ).astype(float)

    for factor in CANONICAL_FACTORS:
        X[f"{factor}__D2"] = (
            X[factor] * X["D2"]
        )

        X[f"{factor}__D3"] = (
            X[factor] * X["D3"]
        )

    design = sm.add_constant(
        X,
        has_constant="add",
    )

    if not isinstance(
        design,
        pd.DataFrame,
    ):
        raise TypeError(
            "Interaction design must remain a pandas DataFrame."
        )

    return design


def _wald_result(
    model,
    restrictions: list[dict[str, float]],
    label: str,
    variant: str,
) -> dict:
    names = list(model.params.index)

    R = np.zeros(
        (len(restrictions), len(names)),
        dtype=float,
    )

    for i, restriction in enumerate(restrictions):
        for term, weight in restriction.items():
            R[i, names.index(term)] = weight

    result = model.wald_test(
        R,
        scalar=True,
    )

    statistic = float(
        np.asarray(result.statistic).reshape(-1)[0]
    )

    p_value = float(
        np.asarray(result.pvalue).reshape(-1)[0]
    )

    return {
        "variant": variant,
        "test": label,
        "statistic": statistic,
        "df_constraints": len(restrictions),
        "p_value": p_value,
    }


def run_interaction_model(
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = prepare_data(variant)

    X = build_interaction_design(data)

    model = sm.OLS(
        data["btc"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5},
    )

    ci = model.conf_int(alpha=0.05)

    coefficients = pd.DataFrame(
        {
            "variant": variant,
            "term": model.params.index,
            "coef": model.params.values,
            "se_hac": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci95_low": ci.iloc[:, 0].values,
            "ci95_high": ci.iloc[:, 1].values,
        }
    )

    tests = []

    # Period 2 sensitivities vs pre-2020.
    tests.append(
        _wald_result(
            model=model,
            restrictions=[
                {f"{f}__D2": 1.0}
                for f in CANONICAL_FACTORS
            ],
            label="P2_vs_P1_joint_beta_change",
            variant=variant,
        )
    )

    # ETF era sensitivities vs pre-2020.
    tests.append(
        _wald_result(
            model=model,
            restrictions=[
                {f"{f}__D3": 1.0}
                for f in CANONICAL_FACTORS
            ],
            label="P3_vs_P1_joint_beta_change",
            variant=variant,
        )
    )

    # ETF era vs 2020-2023.
    tests.append(
        _wald_result(
            model=model,
            restrictions=[
                {
                    f"{f}__D3": 1.0,
                    f"{f}__D2": -1.0,
                }
                for f in CANONICAL_FACTORS
            ],
            label="P3_vs_P2_joint_beta_change",
            variant=variant,
        )
    )

    # Individual changes.
    for factor in CANONICAL_FACTORS:
        tests.append(
            _wald_result(
                model=model,
                restrictions=[
                    {f"{factor}__D2": 1.0}
                ],
                label=f"P2_vs_P1__{factor}",
                variant=variant,
            )
        )

        tests.append(
            _wald_result(
                model=model,
                restrictions=[
                    {f"{factor}__D3": 1.0}
                ],
                label=f"P3_vs_P1__{factor}",
                variant=variant,
            )
        )

        tests.append(
            _wald_result(
                model=model,
                restrictions=[
                    {
                        f"{factor}__D3": 1.0,
                        f"{factor}__D2": -1.0,
                    }
                ],
                label=f"P3_vs_P2__{factor}",
                variant=variant,
            )
        )

    tests_df = pd.DataFrame(tests)

    # Reconstruct economically readable period-specific betas.
    rows = []

    params = model.params

    for factor in CANONICAL_FACTORS:
        beta_p1 = float(params[factor])

        beta_p2 = (
            beta_p1
            + float(params[f"{factor}__D2"])
        )

        beta_p3 = (
            beta_p1
            + float(params[f"{factor}__D3"])
        )

        rows.extend(
            [
                {
                    "variant": variant,
                    "factor": factor,
                    "period": "pre_2020",
                    "beta": beta_p1,
                },
                {
                    "variant": variant,
                    "factor": factor,
                    "period": "post_covid_pre_etf",
                    "beta": beta_p2,
                },
                {
                    "variant": variant,
                    "factor": factor,
                    "period": "spot_etf_era",
                    "beta": beta_p3,
                },
            ]
        )

    reconstructed = pd.DataFrame(rows)

    return coefficients, tests_df, reconstructed


def run_all_interactions():
    all_coef = []
    all_tests = []
    all_reconstructed = []

    for variant in ["primary", "strict"]:
        coef, tests, reconstructed = (
            run_interaction_model(variant)
        )

        all_coef.append(coef)
        all_tests.append(tests)
        all_reconstructed.append(reconstructed)

    coef_df = pd.concat(
        all_coef,
        ignore_index=True,
    )

    tests_df = pd.concat(
        all_tests,
        ignore_index=True,
    )

    reconstructed_df = pd.concat(
        all_reconstructed,
        ignore_index=True,
    )

    coef_df.to_csv(
        TABLES_DIR / "interaction_coefficients.csv",
        index=False,
    )

    tests_df.to_csv(
        TABLES_DIR / "interaction_wald_tests.csv",
        index=False,
    )

    reconstructed_df.to_csv(
        TABLES_DIR / "interaction_period_betas.csv",
        index=False,
    )

    return (
        coef_df,
        tests_df,
        reconstructed_df,
    )
