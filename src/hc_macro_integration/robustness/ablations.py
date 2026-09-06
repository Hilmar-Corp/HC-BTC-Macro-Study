from __future__ import annotations

from typing import Any, cast


import numpy as np
import pandas as pd
import statsmodels.api as sm

from hc_macro_integration.data.download import download_yahoo_daily
from hc_macro_integration.paths import MARKET_DIR, TABLES_DIR
from hc_macro_integration.models.periods import (
    PERIOD_ORDER,
    prepare_data,
)
from hc_macro_integration.models.breaks_supf import (
    build_prefix_crossproducts,
    candidate_indices,
    full_rss,
    segment_rss,
)


FACTOR_SETS = {
    "NDX_ONLY": [
        "ndx",
    ],
    "NDX_USD": [
        "ndx",
        "usd",
    ],
    "FULL_NDX": [
        "ndx",
        "usd",
        "real_rate",
        "credit",
    ],
    "SPX_ONLY": [
        "spx",
    ],
    "FULL_SPX": [
        "spx",
        "usd",
        "real_rate",
        "credit",
    ],
}


def ensure_sp500() -> pd.DataFrame:
    path = (
        MARKET_DIR
        / "sp500_daily.parquet"
    )

    if not path.exists():
        print(
            "[DOWNLOAD] Yahoo ^GSPC"
        )

        df = download_yahoo_daily(
            ticker="^GSPC",
            start="2017-08-17",
        )

        df.to_parquet(
            path,
            index=False,
        )

        print(
            f"[OK] {path} | "
            f"{len(df):,} rows"
        )

    df = pd.read_parquet(
        path
    ).copy()

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if "adj_close" in df.columns:
        price_col = "adj_close"
    else:
        price_col = "close"

    df = (
        df[
            [
                "date",
                price_col,
            ]
        ]
        .rename(
            columns={
                price_col:
                "spx_close",
            }
        )
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    df["spx"] = np.log(
        df["spx_close"]
        / df["spx_close"].shift(1)
    )

    return df


def prepare_extended_data(
    variant: str,
    spx: pd.DataFrame,
) -> pd.DataFrame:
    data = prepare_data(
        variant
    ).copy()

    data["date"] = (
        pd.to_datetime(data["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    data = data.merge(
        spx[
            [
                "date",
                "spx",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )

    return (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def fit_hac(
    data: pd.DataFrame,
    factors: list[str],
):
    clean = (
        data[
            ["btc"]
            + factors
        ]
        .dropna()
        .copy()
    )

    X = sm.add_constant(
        clean[factors],
        has_constant="add",
    )

    model = sm.OLS(
        clean["btc"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": 5,
        },
    )

    return model, clean


def period_factor_set_models(
    data: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    rows = []

    for model_name, factors in (
        FACTOR_SETS.items()
    ):
        for period in PERIOD_ORDER:
            sample = data.loc[
                data["period"]
                == period
            ].copy()

            model, clean = fit_hac(
                sample,
                factors,
            )

            rows.append(
                {
                    "variant":
                        variant,
                    "factor_set":
                        model_name,
                    "period":
                        period,
                    "nobs":
                        int(
                            model.nobs
                        ),
                    "r2":
                        float(
                            model.rsquared
                        ),
                    "adj_r2":
                        float(
                            model.rsquared_adj
                        ),
                }
            )

    return pd.DataFrame(rows)


def build_interaction_design(
    data: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    X = data[
        factors
    ].copy()

    X["D2"] = (
        data["period"]
        == "post_covid_pre_etf"
    ).astype(float)

    X["D3"] = (
        data["period"]
        == "spot_etf_era"
    ).astype(float)

    for factor in factors:
        X[
            f"{factor}__D2"
        ] = (
            X[factor]
            * X["D2"]
        )

        X[
            f"{factor}__D3"
        ] = (
            X[factor]
            * X["D3"]
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
            "Ablation interaction design must remain a pandas DataFrame."
        )

    return design


def wald_test(
    model,
    restrictions:
        list[
            dict[str, float]
        ],
) -> tuple[float, float]:
    names = list(
        model.params.index
    )

    R = np.zeros(
        (
            len(restrictions),
            len(names),
        ),
        dtype=float,
    )

    for i, restriction in enumerate(
        restrictions
    ):
        for term, weight in (
            restriction.items()
        ):
            R[
                i,
                names.index(term),
            ] = weight

    result = model.wald_test(
        R,
        scalar=True,
    )

    stat = float(
        np.asarray(
            result.statistic
        ).reshape(-1)[0]
    )

    p = float(
        np.asarray(
            result.pvalue
        ).reshape(-1)[0]
    )

    return stat, p


def interaction_tests(
    data: pd.DataFrame,
    factors: list[str],
) -> dict:
    required = (
        ["btc", "period"]
        + factors
    )

    clean = (
        data[
            required
        ]
        .dropna()
        .copy()
    )

    X = build_interaction_design(
        clean,
        factors,
    )

    model = sm.OLS(
        clean["btc"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": 5,
        },
    )

    tests = {}

    restrictions = [
        {
            f"{f}__D2": 1.0
        }
        for f in factors
    ]

    stat, p = wald_test(
        model,
        restrictions,
    )

    tests[
        "P2_VS_P1_JOINT_STAT"
    ] = stat

    tests[
        "P2_VS_P1_JOINT_P"
    ] = p

    restrictions = [
        {
            f"{f}__D3": 1.0
        }
        for f in factors
    ]

    stat, p = wald_test(
        model,
        restrictions,
    )

    tests[
        "P3_VS_P1_JOINT_STAT"
    ] = stat

    tests[
        "P3_VS_P1_JOINT_P"
    ] = p

    restrictions = [
        {
            f"{f}__D3": 1.0,
            f"{f}__D2": -1.0,
        }
        for f in factors
    ]

    stat, p = wald_test(
        model,
        restrictions,
    )

    tests[
        "P3_VS_P2_JOINT_STAT"
    ] = stat

    tests[
        "P3_VS_P2_JOINT_P"
    ] = p

    # Nasdaq-specific tests if present.
    if "ndx" in factors:
        for prefix, restriction in [
            (
                "NDX_P2_VS_P1",
                {
                    "ndx__D2":
                    1.0,
                },
            ),
            (
                "NDX_P3_VS_P1",
                {
                    "ndx__D3":
                    1.0,
                },
            ),
            (
                "NDX_P3_VS_P2",
                {
                    "ndx__D3":
                    1.0,
                    "ndx__D2":
                    -1.0,
                },
            ),
        ]:
            stat, p = wald_test(
                model,
                [restriction],
            )

            tests[
                prefix + "_STAT"
            ] = stat

            tests[
                prefix + "_P"
            ] = p

    return tests


def run_leave_one_year_out(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    years = sorted(
        data["date"]
        .dt.year
        .unique()
        .tolist()
    )

    for year in years:
        sample = data.loc[
            data["date"].dt.year
            != year
        ].copy()

        tests = interaction_tests(
            sample,
            FACTOR_SETS[
                "FULL_NDX"
            ],
        )

        row = {
            "excluded_year":
                int(year),
            "nobs":
                int(len(sample)),
        }

        row.update(tests)

        rows.append(row)

    return pd.DataFrame(rows)


def run_exclusion_windows(
    data: pd.DataFrame,
) -> pd.DataFrame:
    windows = [
        (
            "NO_COVID_CRASH_WINDOW",
            pd.Timestamp(
                "2020-02-15"
            ),
            pd.Timestamp(
                "2020-06-30"
            ),
        ),
        (
            "NO_2020",
            pd.Timestamp(
                "2020-01-01"
            ),
            pd.Timestamp(
                "2020-12-31"
            ),
        ),
        (
            "NO_2021",
            pd.Timestamp(
                "2021-01-01"
            ),
            pd.Timestamp(
                "2021-12-31"
            ),
        ),
        (
            "NO_2022",
            pd.Timestamp(
                "2022-01-01"
            ),
            pd.Timestamp(
                "2022-12-31"
            ),
        ),
        (
            "NO_2023",
            pd.Timestamp(
                "2023-01-01"
            ),
            pd.Timestamp(
                "2023-12-31"
            ),
        ),
        (
            "NO_2024",
            pd.Timestamp(
                "2024-01-01"
            ),
            pd.Timestamp(
                "2024-12-31"
            ),
        ),
    ]

    rows = []

    for (
        label,
        start,
        end,
    ) in windows:
        mask = ~(
            (
                data["date"]
                >= start
            )
            & (
                data["date"]
                <= end
            )
        )

        sample = data.loc[
            mask
        ].copy()

        tests = interaction_tests(
            sample,
            FACTOR_SETS[
                "FULL_NDX"
            ],
        )

        row = {
            "exclusion":
                label,
            "start":
                start,
            "end":
                end,
            "nobs":
                len(sample),
        }

        row.update(tests)

        rows.append(row)

    return pd.DataFrame(rows)


def exact_supf_scan(
    data: pd.DataFrame,
    factors: list[str],
    trim_fraction: float = 0.15,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    clean = (
        data[
            ["date", "btc"]
            + factors
        ]
        .dropna()
        .reset_index(drop=True)
    )

    y = clean[
        "btc"
    ].to_numpy(
        dtype=float
    )

    X_raw = clean[
        factors
    ].to_numpy(
        dtype=float
    )

    X = np.column_stack(
        [
            np.ones(
                len(clean)
            ),
            X_raw,
        ]
    )

    n, p = X.shape

    (
        prefix_xx,
        prefix_xy,
        prefix_yy,
    ) = build_prefix_crossproducts(
        X,
        y,
    )

    restricted_rss = full_rss(
        X,
        y,
    )

    candidates = candidate_indices(
        n=n,
        trim_fraction=
            trim_fraction,
        step=1,
    )

    rows = []

    denominator_df = (
        n - 2 * p
    )

    for split in candidates:
        left_rss = segment_rss(
            prefix_xx,
            prefix_xy,
            prefix_yy,
            0,
            split,
        )

        right_rss = segment_rss(
            prefix_xx,
            prefix_xy,
            prefix_yy,
            split,
            n,
        )

        unrestricted_rss = (
            left_rss
            + right_rss
        )

        numerator = (
            restricted_rss
            - unrestricted_rss
        ) / p

        denominator = (
            unrestricted_rss
            / denominator_df
        )

        statistic = (
            numerator
            / denominator
        )

        rows.append(
            {
                "observation_index":
                    split,
                "date":
                    clean.iloc[
                        split
                    ]["date"],
                "f_stat":
                    max(
                        float(
                            statistic
                        ),
                        0.0,
                    ),
            }
        )

    scan = pd.DataFrame(
        rows
    )

    max_idx = (
        scan["f_stat"]
        .idxmax()
    )

    maximum = float(
        cast(
            Any,
            scan.loc[
                max_idx,
                "f_stat",
            ],
        )
    )

    best_date = pd.Timestamp(
        cast(
            Any,
            scan.loc[
                max_idx,
                "date",
            ],
        )
    )

    plateau = scan.loc[
        scan["f_stat"]
        >= 0.95 * maximum
    ].copy()

    summary = {
        "supf":
            maximum,
        "best_date":
            best_date,
        "plateau_95_start":
            pd.Timestamp(
                plateau[
                    "date"
                ].min()
            ),
        "plateau_95_end":
            pd.Timestamp(
                plateau[
                    "date"
                ].max()
            ),
        "plateau_95_count":
            int(
                len(
                    plateau
                )
            ),
        "nobs":
            int(n),
        "parameters_per_segment":
            int(p),
    }

    return scan, summary


def run_break_diagnostics(
    datasets:
        dict[
            str,
            pd.DataFrame,
        ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    summaries = []
    top_dates = []

    models_to_scan = [
        "NDX_ONLY",
        "FULL_NDX",
        "SPX_ONLY",
        "FULL_SPX",
    ]

    for variant, data in (
        datasets.items()
    ):
        for model_name in (
            models_to_scan
        ):
            factors = FACTOR_SETS[
                model_name
            ]

            print(
                "[BREAK SCAN] "
                f"{variant} "
                f"{model_name}"
            )

            scan, summary = (
                exact_supf_scan(
                    data=data,
                    factors=factors,
                )
            )

            summary.update(
                {
                    "variant":
                        variant,
                    "factor_set":
                        model_name,
                }
            )

            summaries.append(
                summary
            )

            top = (
                scan
                .sort_values(
                    "f_stat",
                    ascending=False,
                )
                .head(10)
                .copy()
            )

            top.insert(
                0,
                "factor_set",
                model_name,
            )

            top.insert(
                0,
                "variant",
                variant,
            )

            top_dates.append(
                top
            )

    return (
        pd.DataFrame(
            summaries
        ),
        pd.concat(
            top_dates,
            ignore_index=True,
        ),
    )


def run_phase5_robustness():
    spx = ensure_sp500()

    datasets = {
        variant:
        prepare_extended_data(
            variant,
            spx,
        )
        for variant in [
            "primary",
            "strict",
        ]
    }

    period_rows = []

    for variant, data in (
        datasets.items()
    ):
        result = (
            period_factor_set_models(
                data=data,
                variant=variant,
            )
        )

        period_rows.append(
            result
        )

    period_models = pd.concat(
        period_rows,
        ignore_index=True,
    )

    period_models.to_csv(
        TABLES_DIR
        / "phase5_period_factor_sets.csv",
        index=False,
    )

    primary = datasets[
        "primary"
    ]

    loyo = (
        run_leave_one_year_out(
            primary
        )
    )

    loyo.to_csv(
        TABLES_DIR
        / "phase5_leave_one_year_out.csv",
        index=False,
    )

    exclusions = (
        run_exclusion_windows(
            primary
        )
    )

    exclusions.to_csv(
        TABLES_DIR
        / "phase5_exclusion_windows.csv",
        index=False,
    )

    (
        break_summary,
        break_top,
    ) = run_break_diagnostics(
        datasets
    )

    break_summary.to_csv(
        TABLES_DIR
        / "phase5_break_robustness.csv",
        index=False,
    )

    break_top.to_csv(
        TABLES_DIR
        / "phase5_break_top10_dates.csv",
        index=False,
    )

    return (
        period_models,
        loyo,
        exclusions,
        break_summary,
        break_top,
    )
