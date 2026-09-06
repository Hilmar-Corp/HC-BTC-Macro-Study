from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from hc_macro_integration.paths import TABLES_DIR
from hc_macro_integration.models.periods import (
    CANONICAL_FACTORS,
    PERIOD_ORDER,
    prepare_data,
)


def ols_r2(
    data: pd.DataFrame,
    factors: tuple[str, ...],
) -> float:
    if len(factors) == 0:
        return 0.0

    X = sm.add_constant(
        data[list(factors)],
        has_constant="add",
    )

    model = sm.OLS(
        data["btc"],
        X,
    ).fit()

    return float(model.rsquared)


def shapley_r2(
    data: pd.DataFrame,
) -> pd.DataFrame:
    factors = CANONICAL_FACTORS

    r2_cache = {}

    for size in range(len(factors) + 1):
        for subset in itertools.combinations(
            factors,
            size,
        ):
            r2_cache[subset] = ols_r2(
                data=data,
                factors=subset,
            )

    K = len(factors)

    rows = []

    full_r2 = r2_cache[tuple(factors)]

    for factor in factors:
        others = [
            f for f in factors
            if f != factor
        ]

        contribution = 0.0

        for size in range(len(others) + 1):
            for subset in itertools.combinations(
                others,
                size,
            ):
                subset = tuple(subset)

                expanded = tuple(
                    f for f in factors
                    if f in set(subset) | {factor}
                )

                base = tuple(
                    f for f in factors
                    if f in set(subset)
                )

                weight = (
                    math.factorial(len(subset))
                    * math.factorial(
                        K - len(subset) - 1
                    )
                    / math.factorial(K)
                )

                marginal = (
                    r2_cache[expanded]
                    - r2_cache[base]
                )

                contribution += (
                    weight * marginal
                )

        rows.append(
            {
                "factor": factor,
                "shapley_r2": contribution,
                "share_of_full_r2": (
                    contribution / full_r2
                    if full_r2 > 0
                    else np.nan
                ),
                "full_r2": full_r2,
            }
        )

    return pd.DataFrame(rows)


def run_shapley():
    rows = []

    for variant in ["primary", "strict"]:
        data = prepare_data(variant)

        for period in PERIOD_ORDER:
            sample = data.loc[
                data["period"] == period
            ].copy()

            result = shapley_r2(sample)

            result.insert(
                0,
                "period",
                period,
            )

            result.insert(
                0,
                "variant",
                variant,
            )

            result["nobs"] = len(sample)

            rows.append(result)

    out = pd.concat(
        rows,
        ignore_index=True,
    )

    out.to_csv(
        TABLES_DIR / "shapley_r2_by_period.csv",
        index=False,
    )

    return out
