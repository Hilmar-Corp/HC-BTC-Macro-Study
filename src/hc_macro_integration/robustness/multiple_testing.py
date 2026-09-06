from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests

from hc_macro_integration.paths import TABLES_DIR


JOINT_TESTS = {
    "P2_vs_P1_joint_beta_change",
    "P3_vs_P1_joint_beta_change",
    "P3_vs_P2_joint_beta_change",
}


def run_fdr() -> pd.DataFrame:
    path = TABLES_DIR / "interaction_wald_tests.csv"

    tests = pd.read_csv(path)

    secondary = tests.loc[
        ~tests["test"].isin(JOINT_TESTS)
    ].copy()

    outputs = []

    for variant, group in secondary.groupby("variant"):
        group = group.copy()

        reject, q_values, _, _ = multipletests(
            group["p_value"].to_numpy(),
            alpha=0.05,
            method="fdr_bh",
        )

        group["q_value_bh"] = q_values
        group["reject_fdr_5pct"] = reject

        outputs.append(group)

    out = pd.concat(
        outputs,
        ignore_index=True,
    )

    out.to_csv(
        TABLES_DIR / "interaction_secondary_fdr.csv",
        index=False,
    )

    return out
