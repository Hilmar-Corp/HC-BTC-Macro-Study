from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution

from hc_macro_integration.paths import TABLES_DIR
from hc_macro_integration.models.periods import (
    CANONICAL_FACTORS,
    prepare_data,
)


@dataclass
class SupFResult:
    variant: str
    observed_supf: float
    break_index: int
    break_date: pd.Timestamp
    bootstrap_p: float
    bootstrap_count: int
    block_length: int
    trim_fraction: float
    step: int
    scan: pd.DataFrame
    bootstrap_distribution: np.ndarray


def build_design(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    factors = data[
        CANONICAL_FACTORS
    ].to_numpy(dtype=float)

    X = np.column_stack(
        [
            np.ones(len(data)),
            factors,
        ]
    )

    y = data["btc"].to_numpy(dtype=float)

    return X, y


def build_prefix_crossproducts(
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

    for i in range(n):
        x = X[i]
        yi = y[i]

        prefix_xx[i + 1] = (
            prefix_xx[i]
            + np.outer(x, x)
        )

        prefix_xy[i + 1] = (
            prefix_xy[i]
            + x * yi
        )

        prefix_yy[i + 1] = (
            prefix_yy[i]
            + yi * yi
        )

    return (
        prefix_xx,
        prefix_xy,
        prefix_yy,
    )


def rss_from_crossproducts(
    xx: np.ndarray,
    xy: np.ndarray,
    yy: float,
) -> float:
    beta = np.linalg.pinv(xx) @ xy

    rss = float(
        yy - xy.T @ beta
    )

    return max(
        rss,
        1e-15,
    )


def segment_rss(
    prefix_xx: np.ndarray,
    prefix_xy: np.ndarray,
    prefix_yy: np.ndarray,
    start: int,
    end: int,
) -> float:
    xx = (
        prefix_xx[end]
        - prefix_xx[start]
    )

    xy = (
        prefix_xy[end]
        - prefix_xy[start]
    )

    yy = (
        prefix_yy[end]
        - prefix_yy[start]
    )

    return rss_from_crossproducts(
        xx=xx,
        xy=xy,
        yy=yy,
    )


def full_rss(
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    xx = X.T @ X
    xy = X.T @ y
    yy = float(y.T @ y)

    return rss_from_crossproducts(
        xx=xx,
        xy=xy,
        yy=yy,
    )


def candidate_indices(
    n: int,
    trim_fraction: float,
    step: int,
) -> np.ndarray:
    lower = int(
        np.ceil(
            trim_fraction * n
        )
    )

    upper = int(
        np.floor(
            (1.0 - trim_fraction) * n
        )
    )

    candidates = np.arange(
        lower,
        upper + 1,
        step,
        dtype=int,
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "No valid SupF candidate break dates."
        )

    return candidates


def scan_supf(
    X: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
) -> tuple[
    float,
    int,
    np.ndarray,
]:
    n, p = X.shape

    (
        prefix_xx,
        prefix_xy,
        prefix_yy,
    ) = build_prefix_crossproducts(
        X,
        y,
    )

    rss_restricted = full_rss(
        X,
        y,
    )

    statistics = np.full(
        len(candidates),
        np.nan,
        dtype=float,
    )

    for j, split in enumerate(candidates):
        rss_left = segment_rss(
            prefix_xx,
            prefix_xy,
            prefix_yy,
            0,
            split,
        )

        rss_right = segment_rss(
            prefix_xx,
            prefix_xy,
            prefix_yy,
            split,
            n,
        )

        rss_unrestricted = (
            rss_left
            + rss_right
        )

        numerator = (
            rss_restricted
            - rss_unrestricted
        ) / p

        denominator_df = (
            n - 2 * p
        )

        denominator = (
            rss_unrestricted
            / denominator_df
        )

        if denominator <= 0:
            continue

        statistics[j] = max(
            numerator / denominator,
            0.0,
        )

    if not np.isfinite(statistics).any():
        raise RuntimeError(
            "SupF scan produced no finite statistics."
        )

    max_pos = int(
        np.nanargmax(statistics)
    )

    return (
        float(statistics[max_pos]),
        int(candidates[max_pos]),
        statistics,
    )


def fit_null(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    beta = (
        np.linalg.pinv(X.T @ X)
        @ (X.T @ y)
    )

    fitted = X @ beta

    residuals = y - fitted

    residuals = (
        residuals
        - residuals.mean()
    )

    return fitted, residuals


def moving_block_resample(
    residuals: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(residuals)

    blocks_needed = int(
        np.ceil(
            n / block_length
        )
    )

    starts = rng.integers(
        0,
        n,
        size=blocks_needed,
    )

    pieces = []

    for start in starts:
        indices = (
            start
            + np.arange(block_length)
        ) % n

        pieces.append(
            residuals[indices]
        )

    out = np.concatenate(
        pieces
    )[:n]

    return out


def run_supf_variant(
    variant: str,
    bootstrap_count: int = 499,
    block_length: int = 20,
    trim_fraction: float = 0.15,
    step: int = 5,
    random_seed: int = 20260906,
) -> SupFResult:
    data = prepare_data(
        variant
    )

    X, y = build_design(
        data
    )

    n, p = X.shape

    candidates = candidate_indices(
        n=n,
        trim_fraction=trim_fraction,
        step=step,
    )

    (
        observed_supf,
        observed_break,
        observed_statistics,
    ) = scan_supf(
        X=X,
        y=y,
        candidates=candidates,
    )

    fitted, residuals = fit_null(
        X,
        y,
    )

    rng = np.random.default_rng(
        random_seed
        + (
            0
            if variant == "primary"
            else 100000
        )
    )

    bootstrap_supf = np.empty(
        bootstrap_count,
        dtype=float,
    )

    for b in range(bootstrap_count):
        sampled_residuals = (
            moving_block_resample(
                residuals=residuals,
                block_length=block_length,
                rng=rng,
            )
        )

        y_star = (
            fitted
            + sampled_residuals
        )

        statistic, _, _ = scan_supf(
            X=X,
            y=y_star,
            candidates=candidates,
        )

        bootstrap_supf[b] = statistic

        if (
            (b + 1) % 50 == 0
            or b + 1 == bootstrap_count
        ):
            print(
                f"[SUPF BOOTSTRAP] "
                f"{variant} "
                f"{b + 1}/{bootstrap_count}"
            )

    bootstrap_p = (
        1.0
        + np.sum(
            bootstrap_supf
            >= observed_supf
        )
    ) / (
        bootstrap_count
        + 1.0
    )

    scan = pd.DataFrame(
        {
            "observation_index": candidates,
            "date": [
                data.iloc[i]["date"]
                for i in candidates
            ],
            "f_stat": observed_statistics,
        }
    )

    df1 = p
    df2 = n - 2 * p

    scan["local_naive_p"] = (
        f_distribution.sf(
            scan["f_stat"],
            df1,
            df2,
        )
    )

    break_date = pd.Timestamp(
        data.iloc[
            observed_break
        ]["date"]
    )

    return SupFResult(
        variant=variant,
        observed_supf=observed_supf,
        break_index=observed_break,
        break_date=break_date,
        bootstrap_p=float(
            bootstrap_p
        ),
        bootstrap_count=bootstrap_count,
        block_length=block_length,
        trim_fraction=trim_fraction,
        step=step,
        scan=scan,
        bootstrap_distribution=bootstrap_supf,
    )


def run_supf_all(
    bootstrap_count: int = 499,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    summaries = []
    scans = []

    for variant in [
        "primary",
        "strict",
    ]:
        print(
            f"[SUPF] variant={variant}"
        )

        result = run_supf_variant(
            variant=variant,
            bootstrap_count=bootstrap_count,
        )

        summaries.append(
            {
                "variant": variant,
                "nobs": len(
                    prepare_data(
                        variant
                    )
                ),
                "supf": (
                    result.observed_supf
                ),
                "selected_break_index": (
                    result.break_index
                ),
                "selected_break_date": (
                    result.break_date
                ),
                "bootstrap_p": (
                    result.bootstrap_p
                ),
                "bootstrap_count": (
                    result.bootstrap_count
                ),
                "block_length": (
                    result.block_length
                ),
                "trim_fraction": (
                    result.trim_fraction
                ),
                "step": result.step,
            }
        )

        scan = result.scan.copy()

        scan.insert(
            0,
            "variant",
            variant,
        )

        scans.append(scan)

        bootstrap_df = pd.DataFrame(
            {
                "supf_bootstrap": (
                    result.bootstrap_distribution
                )
            }
        )

        bootstrap_df.to_csv(
            TABLES_DIR
            / (
                "supf_bootstrap_"
                f"{variant}.csv"
            ),
            index=False,
        )

    summary_df = pd.DataFrame(
        summaries
    )

    scan_df = pd.concat(
        scans,
        ignore_index=True,
    )

    summary_df.to_csv(
        TABLES_DIR
        / "supf_unknown_break_summary.csv",
        index=False,
    )

    scan_df.to_csv(
        TABLES_DIR
        / "supf_unknown_break_scan.csv",
        index=False,
    )

    return (
        summary_df,
        scan_df,
    )
