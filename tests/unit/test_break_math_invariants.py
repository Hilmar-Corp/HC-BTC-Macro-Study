from __future__ import annotations

import numpy as np

from hc_macro_integration.models.breaks_supf import (
    build_prefix_crossproducts,
    full_rss,
    segment_rss,
)


def test_segment_rss_additivity_is_bounded_by_full_rss():
    rng = np.random.default_rng(
        20260906
    )

    n = 300

    factors = rng.normal(
        size=(n, 4)
    )

    X = np.column_stack(
        [
            np.ones(n),
            factors,
        ]
    )

    beta = np.array(
        [
            0.0,
            0.8,
            -0.2,
            0.05,
            -0.03,
        ]
    )

    y = (
        X @ beta
        + rng.normal(
            scale=0.5,
            size=n,
        )
    )

    (
        prefix_xx,
        prefix_xy,
        prefix_yy,
    ) = build_prefix_crossproducts(
        X,
        y,
    )

    split = 150

    left = segment_rss(
        prefix_xx,
        prefix_xy,
        prefix_yy,
        0,
        split,
    )

    right = segment_rss(
        prefix_xx,
        prefix_xy,
        prefix_yy,
        split,
        n,
    )

    restricted = full_rss(
        X,
        y,
    )

    assert (
        left + right
        <= restricted + 1e-10
    )
