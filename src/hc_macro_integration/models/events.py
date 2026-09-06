from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from hc_macro_integration.data.download import download_yahoo_daily
from hc_macro_integration.paths import MARKET_DIR, TABLES_DIR
from hc_macro_integration.models.periods import (
    PERIOD_ORDER,
    prepare_data,
)


EVENT_HORIZONS = {
    "SAME_DAY": 0,
    "FWD_1": 1,
    "FWD_5": 5,
    "FWD_20": 20,
}


PERIOD_COMPARISONS = [
    (
        "P2_MINUS_P1",
        "post_covid_pre_etf",
        "pre_2020",
    ),
    (
        "P3_MINUS_P1",
        "spot_etf_era",
        "pre_2020",
    ),
    (
        "P3_MINUS_P2",
        "spot_etf_era",
        "post_covid_pre_etf",
    ),
]


def ensure_vix() -> pd.DataFrame:
    path = (
        MARKET_DIR
        / "vix_daily.parquet"
    )

    if not path.exists():
        print(
            "[DOWNLOAD] Yahoo ^VIX"
        )

        df = download_yahoo_daily(
            ticker="^VIX",
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

    price_col = (
        "adj_close"
        if "adj_close" in df.columns
        else "close"
    )

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
                "vix_close",
            }
        )
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    df["vix_log_change"] = np.log(
        df["vix_close"]
        / df["vix_close"].shift(1)
    )

    return df


def zscore(
    series: pd.Series,
) -> pd.Series:
    mean = series.mean()
    std = series.std(
        ddof=0
    )

    if (
        not np.isfinite(std)
        or std <= 0
    ):
        raise ValueError(
            "Cannot standardize series."
        )

    return (
        series - mean
    ) / std


def prepare_event_panel():
    data = prepare_data(
        "primary"
    ).copy()

    data["date"] = (
        pd.to_datetime(data["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    vix = ensure_vix()

    data = data.merge(
        vix,
        on="date",
        how="left",
        validate="one_to_one",
    )

    data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    data["obs_index"] = (
        np.arange(
            len(data)
        )
    )

    # -----------------------------------------
    # Orthogonalise VIX changes to equity moves.
    # -----------------------------------------

    ortho_sample = (
        data[
            [
                "ndx",
                "vix_log_change",
            ]
        ]
        .dropna()
    )

    X = sm.add_constant(
        ortho_sample[
            ["ndx"]
        ],
        has_constant="add",
    )

    model = sm.OLS(
        ortho_sample[
            "vix_log_change"
        ],
        X,
    ).fit()

    data[
        "vix_ortho"
    ] = np.nan

    data.loc[
        ortho_sample.index,
        "vix_ortho",
    ] = model.resid

    ortho_metrics = {
        "nobs":
            int(model.nobs),
        "alpha":
            float(
                model.params[
                    "const"
                ]
            ),
        "ndx_beta":
            float(
                model.params[
                    "ndx"
                ]
            ),
        "r2":
            float(
                model.rsquared
            ),
    }

    # -----------------------------------------
    # Response variables.
    #
    # SAME_DAY = BTC return over same close-to-close
    # interval as the shock.
    #
    # FWD_h = strictly AFTER the shock:
    # t+1 ... t+h.
    # -----------------------------------------

    data[
        "response__SAME_DAY"
    ] = np.expm1(
        data["btc"]
    )

    for horizon in [
        1,
        5,
        20,
    ]:
        future_log = None

        for k in range(
            1,
            horizon + 1,
        ):
            term = (
                data["btc"]
                .shift(-k)
            )

            if future_log is None:
                future_log = term
            else:
                future_log = (
                    future_log
                    + term
                )

        if future_log is None:
            raise RuntimeError(
                "Forward-return accumulator unexpectedly empty."
            )

        data[
            f"response__FWD_{horizon}"
        ] = np.expm1(
            future_log
        )

        # Prevent period-label contamination:
        # an event near a regime boundary must
        # complete its response window inside
        # the same period.
        data[
            f"same_period__FWD_{horizon}"
        ] = (
            data["period"]
            == data[
                "period"
            ].shift(-horizon)
        )

    return (
        data,
        ortho_metrics,
    )


def create_event_catalogue(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    clean = data.dropna(
        subset=[
            "ndx",
            "usd",
            "real_rate",
            "credit",
            "vix_ortho",
        ]
    ).copy()

    thresholds = {
        "ndx_q10":
            clean[
                "ndx"
            ].quantile(
                0.10
            ),

        "ndx_q90":
            clean[
                "ndx"
            ].quantile(
                0.90
            ),

        "usd_q10":
            clean[
                "usd"
            ].quantile(
                0.10
            ),

        "usd_q90":
            clean[
                "usd"
            ].quantile(
                0.90
            ),

        "real_q10":
            clean[
                "real_rate"
            ].quantile(
                0.10
            ),

        "real_q90":
            clean[
                "real_rate"
            ].quantile(
                0.90
            ),

        "credit_q10":
            clean[
                "credit"
            ].quantile(
                0.10
            ),

        "credit_q90":
            clean[
                "credit"
            ].quantile(
                0.90
            ),

        "vix_q10":
            clean[
                "vix_ortho"
            ].quantile(
                0.10
            ),

        "vix_q90":
            clean[
                "vix_ortho"
            ].quantile(
                0.90
            ),
    }

    # Composite uses broader 20/80 tails so that
    # "two out of three" produces enough episodes.
    thresholds.update(
        {
            "ndx_q20":
                clean[
                    "ndx"
                ].quantile(
                    0.20
                ),

            "ndx_q80":
                clean[
                    "ndx"
                ].quantile(
                    0.80
                ),

            "credit_q20":
                clean[
                    "credit"
                ].quantile(
                    0.20
                ),

            "credit_q80":
                clean[
                    "credit"
                ].quantile(
                    0.80
                ),

            "vix_q20":
                clean[
                    "vix_ortho"
                ].quantile(
                    0.20
                ),

            "vix_q80":
                clean[
                    "vix_ortho"
                ].quantile(
                    0.80
                ),
        }
    )

    event_specs = []

    def add_event(
        name: str,
        mask: pd.Series,
        severity: pd.Series,
        shock_value: pd.Series,
    ):
        selected = clean.loc[
            mask,
            [
                "date",
                "period",
                "obs_index",
            ],
        ].copy()

        selected[
            "event"
        ] = name

        selected[
            "severity"
        ] = severity.loc[
            selected.index
        ].to_numpy()

        selected[
            "shock_value"
        ] = shock_value.loc[
            selected.index
        ].to_numpy()

        event_specs.append(
            selected
        )

    # -----------------------------------------
    # Individual tail events.
    # -----------------------------------------

    add_event(
        "NDX_DOWN_10",
        (
            clean["ndx"]
            <= thresholds[
                "ndx_q10"
            ]
        ),
        -clean["ndx"],
        clean["ndx"],
    )

    add_event(
        "NDX_UP_10",
        (
            clean["ndx"]
            >= thresholds[
                "ndx_q90"
            ]
        ),
        clean["ndx"],
        clean["ndx"],
    )

    add_event(
        "VIX_ORTHO_UP_10",
        (
            clean[
                "vix_ortho"
            ]
            >= thresholds[
                "vix_q90"
            ]
        ),
        clean[
            "vix_ortho"
        ],
        clean[
            "vix_ortho"
        ],
    )

    add_event(
        "CREDIT_WIDEN_10",
        (
            clean[
                "credit"
            ]
            >= thresholds[
                "credit_q90"
            ]
        ),
        clean[
            "credit"
        ],
        clean[
            "credit"
        ],
    )

    add_event(
        "CREDIT_TIGHTEN_10",
        (
            clean[
                "credit"
            ]
            <= thresholds[
                "credit_q10"
            ]
        ),
        -clean[
            "credit"
        ],
        clean[
            "credit"
        ],
    )

    add_event(
        "USD_UP_10",
        (
            clean["usd"]
            >= thresholds[
                "usd_q90"
            ]
        ),
        clean["usd"],
        clean["usd"],
    )

    add_event(
        "USD_DOWN_10",
        (
            clean["usd"]
            <= thresholds[
                "usd_q10"
            ]
        ),
        -clean["usd"],
        clean["usd"],
    )

    add_event(
        "REAL_RATE_UP_10",
        (
            clean[
                "real_rate"
            ]
            >= thresholds[
                "real_q90"
            ]
        ),
        clean[
            "real_rate"
        ],
        clean[
            "real_rate"
        ],
    )

    add_event(
        "REAL_RATE_DOWN_10",
        (
            clean[
                "real_rate"
            ]
            <= thresholds[
                "real_q10"
            ]
        ),
        -clean[
            "real_rate"
        ],
        clean[
            "real_rate"
        ],
    )

    # -----------------------------------------
    # Composite risk-off / risk-on.
    # External financial variables only.
    #
    # Risk-off:
    # - bad NDX
    # - positive orthogonal VIX shock
    # - credit widening
    #
    # At least 2 of 3 must be in 20% tails.
    # -----------------------------------------

    risk_off_score = (
        (
            clean["ndx"]
            <= thresholds[
                "ndx_q20"
            ]
        ).astype(int)
        +
        (
            clean[
                "vix_ortho"
            ]
            >= thresholds[
                "vix_q80"
            ]
        ).astype(int)
        +
        (
            clean[
                "credit"
            ]
            >= thresholds[
                "credit_q80"
            ]
        ).astype(int)
    )

    risk_on_score = (
        (
            clean["ndx"]
            >= thresholds[
                "ndx_q80"
            ]
        ).astype(int)
        +
        (
            clean[
                "vix_ortho"
            ]
            <= thresholds[
                "vix_q20"
            ]
        ).astype(int)
        +
        (
            clean[
                "credit"
            ]
            <= thresholds[
                "credit_q20"
            ]
        ).astype(int)
    )

    risk_off_severity = (
        zscore(
            -clean["ndx"]
        )
        +
        zscore(
            clean[
                "vix_ortho"
            ]
        )
        +
        zscore(
            clean[
                "credit"
            ]
        )
    )

    risk_on_severity = (
        zscore(
            clean["ndx"]
        )
        +
        zscore(
            -clean[
                "vix_ortho"
            ]
        )
        +
        zscore(
            -clean[
                "credit"
            ]
        )
    )

    add_event(
        "RISK_OFF_2OF3",
        (
            risk_off_score
            >= 2
        ),
        risk_off_severity,
        risk_off_score,
    )

    add_event(
        "RISK_ON_2OF3",
        (
            risk_on_score
            >= 2
        ),
        risk_on_severity,
        risk_on_score,
    )

    events = pd.concat(
        event_specs,
        ignore_index=True,
    )

    threshold_df = pd.DataFrame(
        [
            {
                "threshold":
                    key,
                "value":
                    value,
            }
            for (
                key,
                value,
            )
            in thresholds.items()
        ]
    )

    return (
        events,
        threshold_df,
    )


def select_non_overlapping(
    events: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    # Strongest event first.
    ranked = (
        events
        .sort_values(
            [
                "severity",
                "date",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    selected_rows = []
    selected_indices = []

    for _, row in (
        ranked.iterrows()
    ):
        idx = int(
            row[
                "obs_index"
            ]
        )

        conflict = any(
            abs(
                idx - existing
            )
            <= horizon
            for existing
            in selected_indices
        )

        if conflict:
            continue

        selected_rows.append(
            row
        )

        selected_indices.append(
            idx
        )

    if not selected_rows:
        return events.iloc[
            0:0
        ].copy()

    selected = pd.DataFrame(
        selected_rows
    )

    return (
        selected
        .sort_values(
            "obs_index"
        )
        .reset_index(drop=True)
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_count: int,
) -> tuple[
    float,
    float,
]:
    n = len(values)

    if n < 2:
        return (
            np.nan,
            np.nan,
        )

    indices = rng.integers(
        0,
        n,
        size=(
            bootstrap_count,
            n,
        ),
    )

    sampled = values[
        indices
    ]

    means = sampled.mean(
        axis=1
    )

    low, high = np.quantile(
        means,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(low),
        float(high),
    )


def permutation_difference(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    permutation_count: int,
) -> tuple[
    float,
    float,
]:
    if (
        len(a) < 5
        or len(b) < 5
    ):
        return (
            np.nan,
            np.nan,
        )

    observed = (
        float(
            np.mean(a)
            - np.mean(b)
        )
    )

    combined = np.concatenate(
        [
            a,
            b,
        ]
    )

    n_a = len(a)

    exceed = 0

    for _ in range(
        permutation_count
    ):
        permuted = rng.permutation(
            combined
        )

        difference = (
            permuted[
                :n_a
            ].mean()
            -
            permuted[
                n_a:
            ].mean()
        )

        if (
            abs(difference)
            >= abs(observed)
        ):
            exceed += 1

    p_value = (
        1 + exceed
    ) / (
        permutation_count
        + 1
    )

    return (
        observed,
        float(
            p_value
        ),
    )


def run_event_study(
    bootstrap_count: int = 1999,
    permutation_count: int = 1999,
    random_seed: int = 20260906,
):
    (
        data,
        ortho_metrics,
    ) = prepare_event_panel()

    (
        events,
        thresholds,
    ) = create_event_catalogue(
        data
    )

    events.to_csv(
        TABLES_DIR
        / "phase6_event_catalogue_raw.csv",
        index=False,
    )

    thresholds.to_csv(
        TABLES_DIR
        / "phase6_event_thresholds.csv",
        index=False,
    )

    rng = np.random.default_rng(
        random_seed
    )

    summary_rows = []
    difference_rows = []
    selected_rows = []

    response_lookup = (
        data
        .set_index(
            "obs_index"
        )
    )

    for event_name in sorted(
        events[
            "event"
        ].unique()
    ):
        event_data = events.loc[
            events[
                "event"
            ]
            == event_name
        ].copy()

        for (
            horizon_label,
            horizon,
        ) in (
            EVENT_HORIZONS.items()
        ):
            selected = (
                select_non_overlapping(
                    event_data,
                    horizon=horizon,
                )
            )

            selected[
                "horizon"
            ] = horizon_label

            selected_rows.append(
                selected
            )

            response_col = (
                "response__"
                + horizon_label
            )

            values_by_period = {}

            for period in (
                PERIOD_ORDER
            ):
                current = (
                    selected.loc[
                        selected[
                            "period"
                        ]
                        == period
                    ]
                    .copy()
                )

                if horizon > 0:
                    valid_period = (
                        response_lookup.loc[
                            current[
                                "obs_index"
                            ],
                            (
                                "same_period__"
                                + horizon_label
                            ),
                        ]
                        .to_numpy(
                            dtype=bool
                        )
                    )

                    current = (
                        current.loc[
                            valid_period
                        ]
                        .copy()
                    )

                if current.empty:
                    values = np.array(
                        [],
                        dtype=float,
                    )
                else:
                    values = (
                        response_lookup.loc[
                            current[
                                "obs_index"
                            ],
                            response_col,
                        ]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    values = values[
                        np.isfinite(
                            values
                        )
                    ]

                values_by_period[
                    period
                ] = values

                if len(values):
                    ci_low, ci_high = (
                        bootstrap_mean_ci(
                            values,
                            rng=rng,
                            bootstrap_count=
                                bootstrap_count,
                        )
                    )

                    mean_return = float(
                        np.mean(
                            values
                        )
                    )

                    median_return = float(
                        np.median(
                            values
                        )
                    )

                    std_return = float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ) if len(values) > 1 else np.nan

                else:
                    mean_return = np.nan
                    median_return = np.nan
                    std_return = np.nan
                    ci_low = np.nan
                    ci_high = np.nan

                summary_rows.append(
                    {
                        "event":
                            event_name,

                        "horizon":
                            horizon_label,

                        "period":
                            period,

                        "n_events":
                            int(
                                len(
                                    values
                                )
                            ),

                        "mean_return":
                            mean_return,

                        "median_return":
                            median_return,

                        "std_return":
                            std_return,

                        "bootstrap_ci95_low":
                            ci_low,

                        "bootstrap_ci95_high":
                            ci_high,
                    }
                )

            for (
                comparison,
                period_a,
                period_b,
            ) in (
                PERIOD_COMPARISONS
            ):
                difference, p_value = (
                    permutation_difference(
                        values_by_period[
                            period_a
                        ],
                        values_by_period[
                            period_b
                        ],
                        rng=rng,
                        permutation_count=
                            permutation_count,
                    )
                )

                difference_rows.append(
                    {
                        "event":
                            event_name,

                        "horizon":
                            horizon_label,

                        "comparison":
                            comparison,

                        "period_a":
                            period_a,

                        "period_b":
                            period_b,

                        "n_a":
                            int(
                                len(
                                    values_by_period[
                                        period_a
                                    ]
                                )
                            ),

                        "n_b":
                            int(
                                len(
                                    values_by_period[
                                        period_b
                                    ]
                                )
                            ),

                        "mean_difference":
                            difference,

                        "permutation_p":
                            p_value,
                    }
                )

    summary = pd.DataFrame(
        summary_rows
    )

    differences = pd.DataFrame(
        difference_rows
    )

    selected_all = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    valid = differences[
        "permutation_p"
    ].notna()

    differences[
        "q_value_bh"
    ] = np.nan

    differences[
        "reject_fdr_5pct"
    ] = False

    if valid.any():
        (
            reject,
            q_values,
            _,
            _,
        ) = multipletests(
            differences.loc[
                valid,
                "permutation_p",
            ].to_numpy(),
            alpha=0.05,
            method="fdr_bh",
        )

        differences.loc[
            valid,
            "q_value_bh",
        ] = q_values

        differences.loc[
            valid,
            "reject_fdr_5pct",
        ] = reject

    summary.to_csv(
        TABLES_DIR
        / "phase6_event_response_summary.csv",
        index=False,
    )

    differences.to_csv(
        TABLES_DIR
        / "phase6_period_difference_tests.csv",
        index=False,
    )

    selected_all.to_csv(
        TABLES_DIR
        / "phase6_selected_nonoverlap_events.csv",
        index=False,
    )

    ortho_df = pd.DataFrame(
        [
            ortho_metrics
        ]
    )

    ortho_df.to_csv(
        TABLES_DIR
        / "phase6_vix_orthogonalization.csv",
        index=False,
    )

    return (
        data,
        events,
        thresholds,
        summary,
        differences,
        ortho_metrics,
    )
