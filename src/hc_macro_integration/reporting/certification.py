from __future__ import annotations

import json

from hc_macro_integration.paths import (
    DIAGNOSTICS_DIR,
    FIGURES_DIR,
)


REQUIRED_FIGURES = [
    "final_period_adj_r2.png",
    "final_period_ndx_beta.png",
    "final_ndx_vs_spx_adj_r2.png",
    "final_rolling_adj_r2_252.png",
]


def run_phase7_certification(
    phase7_output: dict,
) -> dict:
    results = phase7_output[
        "results"
    ]

    r = results[
        "main_results"
    ]

    checks = {}

    # Main empirical structure.
    checks[
        "PRE2020_ADJ_R2_BELOW_1PCT"
    ] = (
        r[
            "period_r2"
        ][
            "pre_2020"
        ][
            "adj_r2"
        ]
        < 0.01
    )

    checks[
        "P2_ADJ_R2_ABOVE_10PCT"
    ] = (
        r[
            "period_r2"
        ][
            "post_covid_pre_etf"
        ][
            "adj_r2"
        ]
        > 0.10
    )

    checks[
        "P3_ADJ_R2_ABOVE_10PCT"
    ] = (
        r[
            "period_r2"
        ][
            "spot_etf_era"
        ][
            "adj_r2"
        ]
        > 0.10
    )

    # Joint changes.
    checks[
        "P2_VS_P1_JOINT_SIGNIFICANT"
    ] = (
        r[
            "joint_wald"
        ][
            "P2_vs_P1_joint_beta_change"
        ][
            "p_value"
        ]
        < 0.05
    )

    checks[
        "P3_VS_P1_JOINT_SIGNIFICANT"
    ] = (
        r[
            "joint_wald"
        ][
            "P3_vs_P1_joint_beta_change"
        ][
            "p_value"
        ]
        < 0.05
    )

    checks[
        "P3_VS_P2_JOINT_NOT_SIGNIFICANT"
    ] = (
        r[
            "joint_wald"
        ][
            "P3_vs_P2_joint_beta_change"
        ][
            "p_value"
        ]
        >= 0.05
    )

    # NDX secondary changes survive FDR.
    checks[
        "NDX_P2_VS_P1_FDR"
    ] = bool(
        r[
            "factor_change_fdr"
        ][
            "P2_vs_P1__ndx"
        ][
            "reject_5pct"
        ]
    )

    checks[
        "NDX_P3_VS_P1_FDR"
    ] = bool(
        r[
            "factor_change_fdr"
        ][
            "P3_vs_P1__ndx"
        ][
            "reject_5pct"
        ]
    )

    # Event-study family does NOT survive FDR.
    checks[
        "EVENT_STUDY_NO_FDR_DISCOVERY"
    ] = (
        not r[
            "any_phase6b_event_change_survives_fdr"
        ]
    )

    # COVID ablation survives.
    checks[
        "COVID_ABLATION_JOINT_SURVIVES"
    ] = (
        r[
            "covid_ablation"
        ][
            "joint_p2_vs_p1_p"
        ]
        < 0.05
    )

    checks[
        "COVID_ABLATION_NDX_SURVIVES"
    ] = (
        r[
            "covid_ablation"
        ][
            "ndx_p2_vs_p1_p"
        ]
        < 0.05
    )

    for figure in (
        REQUIRED_FIGURES
    ):
        checks[
            "FIGURE_"
            + figure.upper()
            .replace(".", "_")
            .replace("-", "_")
        ] = (
            FIGURES_DIR
            / figure
        ).exists()

    overall = all(
        checks.values()
    )

    output = {
        "certification_status":
            (
                "PASS"
                if overall
                else "FAIL"
            ),
        "checks":
            checks,
        "artifact_count":
            phase7_output[
                "artifact_count"
            ],
    }

    path = (
        DIAGNOSTICS_DIR
        / "phase7_certification.json"
    )

    path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output
