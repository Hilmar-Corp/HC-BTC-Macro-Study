from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from hc_macro_integration.paths import TABLES_DIR
from hc_macro_integration.models.periods import PERIOD_ORDER
from hc_macro_integration.models.events import (
    EVENT_HORIZONS,
    PERIOD_COMPARISONS,
    prepare_event_panel,
    create_event_catalogue,
    select_non_overlapping,
)


def select_nonoverlap_within_period(
    events: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Enforce non-overlap independently inside each research period.

    This prevents an event near a regime boundary from suppressing
    an event belonging to the neighbouring regime.
    """

    frames = []

    for period in PERIOD_ORDER:
        current = events.loc[
            events["period"] == period
        ].copy()

        if current.empty:
            continue

        selected = select_non_overlapping(
            current,
            horizon=horizon,
        )

        frames.append(selected)

    if not frames:
        return events.iloc[0:0].copy()

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            "obs_index"
        )
        .reset_index(drop=True)
    )


def welch_statistic(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    na = len(a)
    nb = len(b)

    if na < 2 or nb < 2:
        return np.nan

    mean_diff = (
        np.mean(a)
        - np.mean(b)
    )

    va = np.var(
        a,
        ddof=1,
    )

    vb = np.var(
        b,
        ddof=1,
    )

    denominator = np.sqrt(
        va / na
        + vb / nb
    )

    if (
        not np.isfinite(denominator)
        or denominator <= 0
    ):
        return np.nan

    return float(
        mean_diff
        / denominator
    )


def studentized_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    permutation_count: int,
) -> tuple[
    float,
    float,
]:
    """
    Two-sided studentized permutation test.

    The studentization reduces sensitivity to unequal variances
    between historical regimes.
    """

    if (
        len(a) < 5
        or len(b) < 5
    ):
        return (
            np.nan,
            np.nan,
        )

    observed_diff = float(
        np.mean(a)
        - np.mean(b)
    )

    observed_t = welch_statistic(
        a,
        b,
    )

    if not np.isfinite(
        observed_t
    ):
        return (
            observed_diff,
            np.nan,
        )

    combined = np.concatenate(
        [
            a,
            b,
        ]
    )

    na = len(a)

    exceed = 0

    for _ in range(
        permutation_count
    ):
        perm = rng.permutation(
            combined
        )

        a_star = perm[:na]
        b_star = perm[na:]

        t_star = welch_statistic(
            a_star,
            b_star,
        )

        if (
            np.isfinite(t_star)
            and abs(t_star)
            >= abs(observed_t)
        ):
            exceed += 1

    p_value = (
        1 + exceed
    ) / (
        permutation_count + 1
    )

    return (
        observed_diff,
        float(p_value),
    )


def bootstrap_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    bootstrap_count: int,
) -> tuple[
    float,
    float,
]:
    if (
        len(a) < 2
        or len(b) < 2
    ):
        return (
            np.nan,
            np.nan,
        )

    ia = rng.integers(
        0,
        len(a),
        size=(
            bootstrap_count,
            len(a),
        ),
    )

    ib = rng.integers(
        0,
        len(b),
        size=(
            bootstrap_count,
            len(b),
        ),
    )

    mean_a = (
        a[ia]
        .mean(axis=1)
    )

    mean_b = (
        b[ib]
        .mean(axis=1)
    )

    differences = (
        mean_a
        - mean_b
    )

    low, high = np.quantile(
        differences,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(low),
        float(high),
    )


def run_phase6b(
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

    response_lookup = (
        data
        .set_index(
            "obs_index"
        )
    )

    rng = np.random.default_rng(
        random_seed
        + 600000
    )

    response_rows = []
    test_rows = []
    selection_rows = []

    event_names = sorted(
        events[
            "event"
        ].unique()
    )

    for event_name in event_names:
        event_data = events.loc[
            events["event"]
            == event_name
        ].copy()

        for (
            horizon_label,
            horizon,
        ) in EVENT_HORIZONS.items():

            selected = (
                select_nonoverlap_within_period(
                    event_data,
                    horizon=horizon,
                )
            )

            selected = selected.copy()

            selected[
                "horizon"
            ] = horizon_label

            selection_rows.append(
                selected
            )

            response_col = (
                "response__"
                + horizon_label
            )

            values_by_period = {}

            for period in PERIOD_ORDER:
                current = selected.loc[
                    selected["period"]
                    == period
                ].copy()

                if horizon > 0:
                    if current.empty:
                        valid = np.array(
                            [],
                            dtype=bool,
                        )
                    else:
                        valid = (
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

                    current = current.loc[
                        valid
                    ].copy()

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
                    mean_return = float(
                        np.mean(values)
                    )

                    median_return = float(
                        np.median(values)
                    )

                else:
                    mean_return = np.nan
                    median_return = np.nan

                response_rows.append(
                    {
                        "event":
                            event_name,

                        "horizon":
                            horizon_label,

                        "period":
                            period,

                        "n_events":
                            int(
                                len(values)
                            ),

                        "mean_return":
                            mean_return,

                        "median_return":
                            median_return,
                    }
                )

            for (
                comparison,
                period_a,
                period_b,
            ) in PERIOD_COMPARISONS:

                a = values_by_period[
                    period_a
                ]

                b = values_by_period[
                    period_b
                ]

                (
                    difference,
                    permutation_p,
                ) = (
                    studentized_permutation_test(
                        a=a,
                        b=b,
                        rng=rng,
                        permutation_count=
                            permutation_count,
                    )
                )

                if (
                    len(a) >= 2
                    and len(b) >= 2
                ):
                    welch = ttest_ind(
                        a,
                        b,
                        equal_var=False,
                        nan_policy="omit",
                    )

                    welch_p = float(
                        welch.pvalue
                    )

                else:
                    welch_p = np.nan

                (
                    ci_low,
                    ci_high,
                ) = (
                    bootstrap_difference_ci(
                        a=a,
                        b=b,
                        rng=rng,
                        bootstrap_count=
                            bootstrap_count,
                    )
                )

                test_rows.append(
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
                            int(len(a)),

                        "n_b":
                            int(len(b)),

                        "mean_difference":
                            difference,

                        "studentized_permutation_p":
                            permutation_p,

                        "welch_p":
                            welch_p,

                        "bootstrap_diff_ci95_low":
                            ci_low,

                        "bootstrap_diff_ci95_high":
                            ci_high,
                    }
                )

    responses = pd.DataFrame(
        response_rows
    )

    tests = pd.DataFrame(
        test_rows
    )

    selections = pd.concat(
        selection_rows,
        ignore_index=True,
    )

    # Preserve the exact same broad secondary-test family
    # used in Phase 6. We do NOT redefine a smaller family
    # after seeing the results.
    valid = tests[
        "studentized_permutation_p"
    ].notna()

    tests[
        "q_value_bh"
    ] = np.nan

    tests[
        "reject_fdr_5pct"
    ] = False

    if valid.any():
        (
            reject,
            q_values,
            _,
            _,
        ) = multipletests(
            tests.loc[
                valid,
                "studentized_permutation_p",
            ].to_numpy(),
            alpha=0.05,
            method="fdr_bh",
        )

        tests.loc[
            valid,
            "q_value_bh",
        ] = q_values

        tests.loc[
            valid,
            "reject_fdr_5pct",
        ] = reject

    responses.to_csv(
        TABLES_DIR
        / "phase6b_corrected_event_responses.csv",
        index=False,
    )

    tests.to_csv(
        TABLES_DIR
        / "phase6b_corrected_difference_tests.csv",
        index=False,
    )

    selections.to_csv(
        TABLES_DIR
        / "phase6b_corrected_nonoverlap_events.csv",
        index=False,
    )

    return (
        responses,
        tests,
        selections,
        ortho_metrics,
    )
