from __future__ import annotations

from typing import Any, cast

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hc_macro_integration.paths import (
    DIAGNOSTICS_DIR,
    FIGURES_DIR,
    PROCESSED_DIR,
    TABLES_DIR,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact missing: {path}"
        )

    return path


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(
        require(
            TABLES_DIR / name
        )
    )


def build_key_results() -> dict:
    period_metrics = load_csv(
        "period_metrics.csv"
    )

    period_betas = load_csv(
        "interaction_period_betas.csv"
    )

    wald = load_csv(
        "interaction_wald_tests.csv"
    )

    fdr = load_csv(
        "interaction_secondary_fdr.csv"
    )

    shapley = load_csv(
        "shapley_r2_by_period.csv"
    )

    phase5 = load_csv(
        "phase5_period_factor_sets.csv"
    )

    exclusions = load_csv(
        "phase5_exclusion_windows.csv"
    )

    break_robustness = load_csv(
        "phase5_break_robustness.csv"
    )

    phase6b = load_csv(
        "phase6b_corrected_difference_tests.csv"
    )

    # --------------------------------------------------
    # Period R²
    # --------------------------------------------------

    pm = period_metrics.loc[
        period_metrics["variant"]
        == "primary"
    ].set_index("period")

    r2_periods = {
        period: {
            "r2": float(
                cast(
                    Any,
                    pm.loc[
                        period,
                        "r2",
                    ],
                )
            ),
            "adj_r2": float(
                cast(
                    Any,
                    pm.loc[
                        period,
                        "adj_r2",
                    ],
                )
            ),
            "nobs": int(
                cast(
                    Any,
                    pm.loc[
                        period,
                        "nobs",
                    ],
                )
            ),
        }
        for period in [
            "pre_2020",
            "post_covid_pre_etf",
            "spot_etf_era",
        ]
    }

    # --------------------------------------------------
    # Period betas
    # --------------------------------------------------

    pb = period_betas.loc[
        period_betas["variant"]
        == "primary"
    ]

    betas = {}

    for factor in [
        "ndx",
        "usd",
        "real_rate",
        "credit",
    ]:
        current = (
            pb.loc[
                pb["factor"]
                == factor
            ]
            .set_index(
                "period"
            )
        )

        betas[factor] = {
            period: float(
                cast(
                    Any,
                    current.loc[
                        period,
                        "beta",
                    ],
                )
            )
            for period in [
                "pre_2020",
                "post_covid_pre_etf",
                "spot_etf_era",
            ]
        }

    # --------------------------------------------------
    # Joint Wald
    # --------------------------------------------------

    primary_wald = wald.loc[
        wald["variant"]
        == "primary"
    ].set_index("test")

    joint_tests = {}

    for test in [
        "P2_vs_P1_joint_beta_change",
        "P3_vs_P1_joint_beta_change",
        "P3_vs_P2_joint_beta_change",
    ]:
        row = primary_wald.loc[
            test
        ]

        joint_tests[test] = {
            "statistic": float(
                cast(
                    Any,
                    row["statistic"],
                )
            ),
            "p_value": float(
                cast(
                    Any,
                    row["p_value"],
                )
            ),
        }

    # --------------------------------------------------
    # FDR factor changes
    # --------------------------------------------------

    primary_fdr = fdr.loc[
        fdr["variant"]
        == "primary"
    ]

    fdr_results = {}

    for _, row in (
        primary_fdr.iterrows()
    ):
        fdr_results[
            str(row["test"])
        ] = {
            "p_value":
                float(
                    row["p_value"]
                ),
            "q_value_bh":
                float(
                    row["q_value_bh"]
                ),
            "reject_5pct":
                bool(
                    row[
                        "reject_fdr_5pct"
                    ]
                ),
        }

    # --------------------------------------------------
    # Shapley
    # --------------------------------------------------

    sh = shapley.loc[
        shapley["variant"]
        == "primary"
    ]

    shapley_results = {}

    for period in [
        "pre_2020",
        "post_covid_pre_etf",
        "spot_etf_era",
    ]:
        current = sh.loc[
            sh["period"]
            == period
        ]

        shapley_results[
            period
        ] = {
            str(row["factor"]): {
                "shapley_r2":
                    float(
                        row[
                            "shapley_r2"
                        ]
                    ),
                "share":
                    float(
                        row[
                            "share_of_full_r2"
                        ]
                    ),
            }
            for _, row
            in current.iterrows()
        }

    # --------------------------------------------------
    # NDX/SPX single-factor comparison
    # --------------------------------------------------

    p5 = phase5.loc[
        phase5["variant"]
        == "primary"
    ]

    equity_factor_models = {}

    for factor_set in [
        "NDX_ONLY",
        "SPX_ONLY",
    ]:
        current = (
            p5.loc[
                p5[
                    "factor_set"
                ]
                == factor_set
            ]
            .set_index(
                "period"
            )
        )

        equity_factor_models[
            factor_set
        ] = {
            period: float(
                cast(
                    Any,
                    current.loc[
                        period,
                        "adj_r2",
                    ],
                )
            )
            for period in [
                "pre_2020",
                "post_covid_pre_etf",
                "spot_etf_era",
            ]
        }

    # --------------------------------------------------
    # COVID ablation
    # --------------------------------------------------

    covid = exclusions.loc[
        exclusions["exclusion"]
        == "NO_COVID_CRASH_WINDOW"
    ].iloc[0]

    covid_ablation = {
        "joint_p2_vs_p1_p":
            float(
                covid[
                    "P2_VS_P1_JOINT_P"
                ]
            ),

        "joint_p3_vs_p1_p":
            float(
                covid[
                    "P3_VS_P1_JOINT_P"
                ]
            ),

        "ndx_p2_vs_p1_p":
            float(
                covid[
                    "NDX_P2_VS_P1_P"
                ]
            ),
    }

    # --------------------------------------------------
    # Break diagnostics
    # --------------------------------------------------

    br = break_robustness.copy()

    break_results = []

    for _, row in br.iterrows():
        break_results.append(
            {
                "variant":
                    str(
                        row[
                            "variant"
                        ]
                    ),

                "factor_set":
                    str(
                        row[
                            "factor_set"
                        ]
                    ),

                "supf":
                    float(
                        row[
                            "supf"
                        ]
                    ),

                "best_date":
                    str(
                        pd.Timestamp(
                            row[
                                "best_date"
                            ]
                        ).date()
                    ),
            }
        )

    # --------------------------------------------------
    # Corrected event study
    # --------------------------------------------------

    core_same_day = phase6b.loc[
        (
            phase6b[
                "horizon"
            ]
            == "SAME_DAY"
        )
        &
        (
            phase6b[
                "event"
            ].isin(
                [
                    "NDX_DOWN_10",
                    "RISK_OFF_2OF3",
                    "RISK_ON_2OF3",
                    "VIX_ORTHO_UP_10",
                ]
            )
        )
    ].copy()

    event_results = []

    for _, row in (
        core_same_day.iterrows()
    ):
        event_results.append(
            {
                "event":
                    str(
                        row["event"]
                    ),

                "comparison":
                    str(
                        row[
                            "comparison"
                        ]
                    ),

                "mean_difference":
                    float(
                        row[
                            "mean_difference"
                        ]
                    ),

                "studentized_permutation_p":
                    float(
                        row[
                            "studentized_permutation_p"
                        ]
                    ),

                "q_value_bh":
                    float(
                        row[
                            "q_value_bh"
                        ]
                    ),

                "fdr_reject_5pct":
                    bool(
                        row[
                            "reject_fdr_5pct"
                        ]
                    ),
            }
        )

    any_event_fdr = bool(
        phase6b[
            "reject_fdr_5pct"
        ].any()
    )

    return {
        "study":
            "HC-BTC-MACRO-INTEGRATION",

        "freeze_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "sample": {
            "start":
                "2017-08-17",
            "last_market_session":
                "2026-09-04",
        },

        "main_results": {
            "period_r2":
                r2_periods,

            "period_betas":
                betas,

            "joint_wald":
                joint_tests,

            "factor_change_fdr":
                fdr_results,

            "shapley":
                shapley_results,

            "equity_single_factor_models":
                equity_factor_models,

            "covid_ablation":
                covid_ablation,

            "break_diagnostics":
                break_results,

            "same_day_event_tests":
                event_results,

            "any_phase6b_event_change_survives_fdr":
                any_event_fdr,
        },

        "interpretation_lock": {
            "supported": [
                (
                    "Bitcoin's contemporaneous statistical "
                    "integration with US equity risk increased "
                    "materially after the pre-2020 period."
                ),
                (
                    "The result is robust to Nasdaq-100 versus "
                    "S&P 500 definitions and to removal of the "
                    "2020 Covid crash window."
                ),
                (
                    "The post-2020 increase is driven primarily "
                    "by the US equity factor."
                ),
                (
                    "The 2024 spot-ETF era does not exhibit a "
                    "statistically established additional joint "
                    "beta break relative to 2020-2023."
                ),
                (
                    "Event-study evidence is directionally "
                    "consistent with stronger contemporaneous "
                    "risk-on/risk-off synchronization."
                ),
            ],

            "not_supported": [
                (
                    "The ETF launch caused Bitcoin's "
                    "macro-financial integration."
                ),
                (
                    "Dollar, real rates and credit all underwent "
                    "equally robust structural changes."
                ),
                (
                    "Macro shocks robustly predict Bitcoin "
                    "returns over the following 1-20 sessions."
                ),
                (
                    "A precise single causal break date can be "
                    "identified."
                ),
            ],
        },
    }


def plot_period_r2(
    results: dict,
) -> None:
    periods = [
        "pre_2020",
        "post_covid_pre_etf",
        "spot_etf_era",
    ]

    labels = [
        "Pré-2020",
        "2020–2023",
        "Depuis 2024",
    ]

    values = [
        100
        * results[
            "main_results"
        ][
            "period_r2"
        ][period][
            "adj_r2"
        ]
        for period in periods
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_ylabel(
        "R² ajusté (%)"
    )

    ax.set_title(
        "Pouvoir explicatif du modèle macrofinancier"
    )

    for i, value in enumerate(
        values
    ):
        ax.text(
            i,
            value,
            f"{value:.1f} %",
            ha="center",
            va=(
                "bottom"
                if value >= 0
                else "top"
            ),
        )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "final_period_adj_r2.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_equity_betas(
    results: dict,
) -> None:
    periods = [
        "pre_2020",
        "post_covid_pre_etf",
        "spot_etf_era",
    ]

    labels = [
        "Pré-2020",
        "2020–2023",
        "Depuis 2024",
    ]

    values = [
        results[
            "main_results"
        ][
            "period_betas"
        ][
            "ndx"
        ][period]
        for period in periods
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_ylabel(
        "Beta Nasdaq-100"
    )

    ax.set_title(
        "Sensibilité contemporaine de Bitcoin au Nasdaq-100"
    )

    for i, value in enumerate(
        values
    ):
        ax.text(
            i,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "final_period_ndx_beta.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_equity_model_comparison(
    results: dict,
) -> None:
    periods = [
        "pre_2020",
        "post_covid_pre_etf",
        "spot_etf_era",
    ]

    labels = [
        "Pré-2020",
        "2020–2023",
        "Depuis 2024",
    ]

    ndx = np.array(
        [
            results[
                "main_results"
            ][
                "equity_single_factor_models"
            ][
                "NDX_ONLY"
            ][period]
            * 100
            for period in periods
        ]
    )

    spx = np.array(
        [
            results[
                "main_results"
            ][
                "equity_single_factor_models"
            ][
                "SPX_ONLY"
            ][period]
            * 100
            for period in periods
        ]
    )

    x = np.arange(
        len(periods)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    ax.bar(
        x - width / 2,
        ndx,
        width,
        label="Nasdaq-100",
    )

    ax.bar(
        x + width / 2,
        spx,
        width,
        label="S&P 500",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "R² ajusté (%)"
    )

    ax.set_title(
        "Le changement ne dépend pas du choix Nasdaq / S&P 500"
    )

    ax.legend()

    ax.axhline(
        0,
        linewidth=1,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "final_ndx_vs_spx_adj_r2.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_rolling_r2() -> None:
    path = require(
        TABLES_DIR
        / "rolling_macro_window_252.csv"
    )

    df = pd.read_csv(
        path
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    ax.plot(
        df["date"],
        100 * df["adj_r2"],
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.axvline(
        cast(
            Any,
            pd.Timestamp(
                "2020-03-09"
            ),
        ),
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        cast(
            Any,
            pd.Timestamp(
                "2024-01-11"
            ),
        ),
        linestyle="--",
        linewidth=1,
    )

    ax.set_ylabel(
        "R² ajusté (%)"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_title(
        "Pouvoir explicatif macrofinancier roulant — 252 séances"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "final_rolling_adj_r2_252.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_markdown_summary(
    results: dict,
) -> Path:
    r = results[
        "main_results"
    ]

    pre_r2 = (
        100
        * r[
            "period_r2"
        ][
            "pre_2020"
        ][
            "adj_r2"
        ]
    )

    p2_r2 = (
        100
        * r[
            "period_r2"
        ][
            "post_covid_pre_etf"
        ][
            "adj_r2"
        ]
    )

    p3_r2 = (
        100
        * r[
            "period_r2"
        ][
            "spot_etf_era"
        ][
            "adj_r2"
        ]
    )

    ndx_pre = r[
        "period_betas"
    ][
        "ndx"
    ][
        "pre_2020"
    ]

    ndx_p2 = r[
        "period_betas"
    ][
        "ndx"
    ][
        "post_covid_pre_etf"
    ]

    ndx_p3 = r[
        "period_betas"
    ][
        "ndx"
    ][
        "spot_etf_era"
    ]

    lines = [
        "# HC-BTC-Macro-Integration-Study — Result Freeze",
        "",
        "## Question",
        "",
        (
            "La dépendance statistique de Bitcoin aux facteurs "
            "macrofinanciers a-t-elle augmenté depuis 2017 ?"
        ),
        "",
        "## Résultat principal",
        "",
        (
            f"R² ajusté du modèle macro : {pre_r2:.2f} % "
            f"avant 2020, {p2_r2:.2f} % sur 2020–2023, "
            f"et {p3_r2:.2f} % depuis janvier 2024."
        ),
        "",
        (
            f"Beta Nasdaq-100 : {ndx_pre:.3f} → "
            f"{ndx_p2:.3f} → {ndx_p3:.3f}."
        ),
        "",
        (
            "Le changement conjoint des sensibilités est "
            "statistiquement établi entre la période pré-2020 "
            "et les deux périodes suivantes, mais pas entre "
            "2020–2023 et l'ère post-ETF."
        ),
        "",
        "## Interprétation",
        "",
        (
            "L'évidence est compatible avec une intégration "
            "beaucoup plus forte de Bitcoin au facteur actions "
            "américain à partir de la zone de transition de 2020."
        ),
        "",
        (
            "Cette transformation ne constitue pas une preuve "
            "que les ETF l'ont causée, ni une preuve que les "
            "chocs macro prédisent les rendements futurs de Bitcoin."
        ),
        "",
        "## Limites verrouillées",
        "",
        "- Associations contemporaines, non causalité.",
        "- Pas de date causale unique de rupture.",
        (
            "- Les changements dollar, taux réels et crédit sont "
            "moins robustes que le changement actions."
        ),
        (
            "- Aucun changement de rendement conditionnel de "
            "l'event study ne survit au contrôle FDR à 5 %."
        ),
        "",
    ]

    path = (
        DIAGNOSTICS_DIR
        / "RESULT_FREEZE.md"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


def build_manifest() -> dict:
    candidate_files = []

    for directory in [
        PROCESSED_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        DIAGNOSTICS_DIR,
    ]:
        for path in sorted(
            directory.glob("*")
        ):
            if (
                path.is_file()
                and path.name
                not in {
                    "artifact_manifest.json",
                }
            ):
                candidate_files.append(
                    path
                )

    return {
        str(
            path.relative_to(
                PROCESSED_DIR.parents[1]
            )
        ): {
            "sha256":
                sha256(path),
            "bytes":
                path.stat().st_size,
        }
        for path in candidate_files
    }


def run_phase7():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = build_key_results()

    json_path = (
        DIAGNOSTICS_DIR
        / "final_results_freeze.json"
    )

    json_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plot_period_r2(
        results
    )

    plot_equity_betas(
        results
    )

    plot_equity_model_comparison(
        results
    )

    plot_rolling_r2()

    md_path = (
        write_markdown_summary(
            results
        )
    )

    manifest = (
        build_manifest()
    )

    manifest_path = (
        DIAGNOSTICS_DIR
        / "artifact_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "results_json":
            json_path,
        "summary_markdown":
            md_path,
        "manifest":
            manifest_path,
        "results":
            results,
        "artifact_count":
            len(manifest),
    }
