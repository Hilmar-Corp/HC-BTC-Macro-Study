from __future__ import annotations

from pathlib import Path

from hc_macro_integration.data.audit import run_factor_audit
from hc_macro_integration.data.certification import run_certification
from hc_macro_integration.data.download import run_downloads
from hc_macro_integration.data.panel import run_build_panel
from hc_macro_integration.data.strict import build_strict_panel
from hc_macro_integration.models.attribution import run_shapley
from hc_macro_integration.models.break_details import run_break_details
from hc_macro_integration.models.breaks_bic import run_break_search
from hc_macro_integration.models.breaks_supf import run_supf_all
from hc_macro_integration.models.interactions import run_all_interactions
from hc_macro_integration.models.linear import run_full_sample_regressions
from hc_macro_integration.models.periods import run_period_models
from hc_macro_integration.models.rolling import run_rolling_analysis
from hc_macro_integration.reporting.artifacts import (
    reset_derived_workspace,
    stabilize_final_freeze,
    write_canonical_artifact_manifest,
)
from hc_macro_integration.reporting.certification import run_phase7_certification
from hc_macro_integration.reporting.freeze import run_phase7
from hc_macro_integration.robustness.ablations import run_phase5_robustness
from hc_macro_integration.robustness.event_study import run_phase6b
from hc_macro_integration.robustness.multiple_testing import run_fdr
from hc_macro_integration.reproducibility import verify_results_freeze


def run_data_pipeline() -> None:
    print("[PIPELINE] data")
    run_downloads()
    run_build_panel()
    run_certification()


def run_core_pipeline() -> None:
    print("[PIPELINE] core")

    run_factor_audit()

    strict_panel = build_strict_panel()

    run_full_sample_regressions(
        strict_panel
    )

    run_rolling_analysis(
        strict_panel
    )

    run_period_models()
    run_all_interactions()
    run_shapley()
    run_break_search()


def run_robustness_pipeline(
    supf_bootstraps: int = 499,
) -> None:
    print(
        "[PIPELINE] robustness "
        f"supf_bootstraps={supf_bootstraps}"
    )

    run_fdr()

    supf, _ = run_supf_all(
        bootstrap_count=supf_bootstraps
    )

    run_break_details(
        supf_summary=supf
    )

    run_phase5_robustness()


def run_event_pipeline(
    bootstraps: int = 1999,
    permutations: int = 1999,
) -> None:
    print(
        "[PIPELINE] events "
        f"bootstraps={bootstraps} "
        f"permutations={permutations}"
    )

    run_phase6b(
        bootstrap_count=bootstraps,
        permutation_count=permutations,
    )


def run_freeze_pipeline() -> dict:
    print("[PIPELINE] freeze")

    output = run_phase7()

    stabilize_final_freeze(
        Path.cwd()
    )

    manifest_path = (
        write_canonical_artifact_manifest(
            Path.cwd()
        )
    )

    print(
        "CANONICAL_ARTIFACT_MANIFEST="
        + str(manifest_path)
    )

    certification = (
        run_phase7_certification(
            output
        )
    )

    status = certification[
        "certification_status"
    ]

    print(
        "FINAL_CERTIFICATION_STATUS="
        + status
    )

    if status != "PASS":
        raise RuntimeError(
            "Final research certification failed."
        )

    return certification


def run_all_pipeline(
    supf_bootstraps: int = 499,
    event_bootstraps: int = 1999,
    event_permutations: int = 1999,
) -> None:
    reset_derived_workspace(
        Path.cwd()
    )

    print(
        "DERIVED_WORKSPACE_RESET=PASS"
    )

    run_build_panel()
    run_certification()
    run_core_pipeline()

    run_robustness_pipeline(
        supf_bootstraps=
            supf_bootstraps
    )

    run_event_pipeline(
        bootstraps=
            event_bootstraps,
        permutations=
            event_permutations,
    )

    run_freeze_pipeline()


def verify_against_baseline() -> None:
    root = Path.cwd()

    verify_results_freeze(
        root / "reproducibility" / "baseline" / "final_results_freeze.json",
        root / "outputs" / "diagnostics" / "final_results_freeze.json",
    )

    print("NUMERICAL_FREEZE_REPRODUCIBILITY=PASS")
