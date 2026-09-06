from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


class ResearchContractError(ValueError):
    """Raised when an empirical input violates a locked research invariant."""


def require_unique_monotonic_dates(
    data: pd.DataFrame,
    column: str = "date",
) -> None:
    if column not in data.columns:
        raise ResearchContractError(
            f"Missing date column: {column}"
        )

    dates = pd.to_datetime(
        data[column],
        errors="raise",
    )

    if dates.isna().any():
        raise ResearchContractError(
            "Date column contains missing values."
        )

    if not dates.is_unique:
        raise ResearchContractError(
            "Duplicate dates are forbidden."
        )

    if not dates.is_monotonic_increasing:
        raise ResearchContractError(
            "Dates must be strictly ordered."
        )


def require_finite_columns(
    data: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    for column in columns:
        if column not in data.columns:
            raise ResearchContractError(
                f"Missing required column: {column}"
            )

        values = pd.to_numeric(
            data[column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ResearchContractError(
                f"Non-finite values in: {column}"
            )


def require_positive_columns(
    data: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    for column in columns:
        if column not in data.columns:
            raise ResearchContractError(
                f"Missing required column: {column}"
            )

        values = pd.to_numeric(
            data[column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ResearchContractError(
                f"Non-finite values in: {column}"
            )

        if not (values > 0).all():
            raise ResearchContractError(
                f"Non-positive values in: {column}"
            )


def require_minimum_rows(
    data: pd.DataFrame,
    minimum: int,
) -> None:
    if minimum < 1:
        raise ValueError(
            "minimum must be >= 1"
        )

    if len(data) < minimum:
        raise ResearchContractError(
            f"Insufficient observations: "
            f"{len(data)} < {minimum}"
        )
