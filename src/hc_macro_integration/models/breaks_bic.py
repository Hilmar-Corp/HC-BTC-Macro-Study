from __future__ import annotations

from typing import Any, cast

import math

import numpy as np
import pandas as pd

from hc_macro_integration.paths import TABLES_DIR
from hc_macro_integration.models.periods import (
    CANONICAL_FACTORS,
    prepare_data,
)


def segment_rss(
    prefix_xx: np.ndarray,
    prefix_xy: np.ndarray,
    prefix_yy: np.ndarray,
    i: int,
    j: int,
) -> float:
    xx = prefix_xx[j] - prefix_xx[i]
    xy = prefix_xy[j] - prefix_xy[i]
    yy = prefix_yy[j] - prefix_yy[i]

    beta = np.linalg.pinv(xx) @ xy

    rss = (
        float(yy)
        - float(xy.T @ beta)
    )

    return max(rss, 1e-15)


def build_prefixes(
    X: np.ndarray,
    y: np.ndarray,
):
    n, p = X.shape

    prefix_xx = np.zeros(
        (n + 1, p, p),
        dtype=float,
    )

    prefix_xy = np.zeros(
        (n + 1, p),
        dtype=float,
    )

    prefix_yy = np.zeros(
        n + 1,
        dtype=float,
    )

    for t in range(n):
        x = X[t]
        yt = y[t]

        prefix_xx[t + 1] = (
            prefix_xx[t]
            + np.outer(x, x)
        )

        prefix_xy[t + 1] = (
            prefix_xy[t]
            + x * yt
        )

        prefix_yy[t + 1] = (
            prefix_yy[t]
            + yt * yt
        )

    return (
        prefix_xx,
        prefix_xy,
        prefix_yy,
    )


def search_breaks(
    data: pd.DataFrame,
    max_breaks: int = 4,
    min_segment_fraction: float = 0.15,
    jump: int = 5,
):
    y = data["btc"].to_numpy(dtype=float)

    X_raw = data[
        CANONICAL_FACTORS
    ].to_numpy(dtype=float)

    X = np.column_stack(
        [
            np.ones(len(data)),
            X_raw,
        ]
    )

    n = len(data)
    p = X.shape[1]

    min_size = int(
        math.ceil(
            min_segment_fraction * n
        )
    )

    (
        prefix_xx,
        prefix_xy,
        prefix_yy,
    ) = build_prefixes(X, y)

    endpoints = sorted(
        set(
            [0]
            + list(range(jump, n, jump))
            + [n]
        )
    )

    m = len(endpoints)

    cost = np.full(
        (m, m),
        np.inf,
        dtype=float,
    )

    for a in range(m):
        i = endpoints[a]

        for b in range(a + 1, m):
            j = endpoints[b]

            if j - i < min_size:
                continue

            cost[a, b] = segment_rss(
                prefix_xx=prefix_xx,
                prefix_xy=prefix_xy,
                prefix_yy=prefix_yy,
                i=i,
                j=j,
            )

    max_segments = max_breaks + 1

    dp = np.full(
        (max_segments + 1, m),
        np.inf,
    )

    prev = np.full(
        (max_segments + 1, m),
        -1,
        dtype=int,
    )

    # One segment.
    for b in range(1, m):
        dp[1, b] = cost[0, b]

    for segments in range(
        2,
        max_segments + 1,
    ):
        for b in range(1, m):
            j = endpoints[b]

            if j < segments * min_size:
                continue

            for a in range(1, b):
                i = endpoints[a]

                if j - i < min_size:
                    continue

                if not np.isfinite(
                    dp[segments - 1, a]
                ):
                    continue

                if not np.isfinite(
                    cost[a, b]
                ):
                    continue

                candidate = (
                    dp[segments - 1, a]
                    + cost[a, b]
                )

                if candidate < dp[segments, b]:
                    dp[segments, b] = candidate
                    prev[segments, b] = a

    end_idx = m - 1

    model_rows = []
    solutions = {}

    for breaks in range(
        0,
        max_breaks + 1,
    ):
        segments = breaks + 1

        rss = dp[segments, end_idx]

        if not np.isfinite(rss):
            continue

        # Segment-specific intercept + 4 betas.
        # Break locations themselves also count as parameters.
        param_count = (
            segments * p
            + breaks
        )

        bic = (
            n * np.log(rss / n)
            + param_count * np.log(n)
        )

        current = end_idx
        break_endpoint_indices = []

        for seg in range(
            segments,
            1,
            -1,
        ):
            previous = prev[seg, current]

            if previous < 0:
                raise RuntimeError(
                    "Break reconstruction failed."
                )

            break_endpoint_indices.append(
                previous
            )

            current = previous

        break_endpoint_indices = sorted(
            break_endpoint_indices
        )

        breaks_rows = [
            endpoints[idx]
            for idx in break_endpoint_indices
        ]

        solutions[breaks] = breaks_rows

        model_rows.append(
            {
                "breaks": breaks,
                "segments": segments,
                "rss": rss,
                "bic": bic,
                "parameter_count": param_count,
                "min_segment_size": min_size,
                "jump": jump,
            }
        )

    models = pd.DataFrame(model_rows)

    selected_row = models.loc[
        models["bic"].idxmin()
    ]

    selected_breaks_count = int(
        cast(
            Any,
            selected_row["breaks"],
        )
    )

    selected_indices = solutions[
        selected_breaks_count
    ]

    selected_dates = []

    for idx in selected_indices:
        # idx is the first observation of the new segment.
        selected_dates.append(
            data.iloc[idx]["date"]
        )

    break_dates_df = pd.DataFrame(
        {
            "break_number": range(
                1,
                len(selected_dates) + 1,
            ),
            "observation_index": selected_indices,
            "date": selected_dates,
        }
    )

    return (
        models,
        break_dates_df,
    )


def run_break_search():
    data = prepare_data("primary")

    models, breaks = search_breaks(
        data=data,
        max_breaks=4,
        min_segment_fraction=0.15,
        jump=5,
    )

    models.to_csv(
        TABLES_DIR / "structural_break_bic_models.csv",
        index=False,
    )

    breaks.to_csv(
        TABLES_DIR / "structural_break_selected_dates.csv",
        index=False,
    )

    return models, breaks
