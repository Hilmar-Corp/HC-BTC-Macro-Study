from __future__ import annotations

import argparse
import json

from hc_macro_integration.paths import DIAGNOSTICS_DIR
from .pipeline import (
    run_all_pipeline,
    run_core_pipeline,
    run_data_pipeline,
    run_event_pipeline,
    run_freeze_pipeline,
    run_robustness_pipeline,
    verify_against_baseline,
)


def status_command() -> int:
    certification_path = (
        DIAGNOSTICS_DIR
        / "phase7_certification.json"
    )

    freeze_path = (
        DIAGNOSTICS_DIR
        / "final_results_freeze.json"
    )

    if not certification_path.exists():
        print("RESEARCH_STATUS=NOT_CERTIFIED")
        return 1

    certification = json.loads(
        certification_path.read_text(
            encoding="utf-8"
        )
    )

    print(
        "RESEARCH_STATUS="
        + certification[
            "certification_status"
        ]
    )

    print(
        "ARTIFACT_COUNT="
        + str(
            certification.get(
                "artifact_count",
                "NA",
            )
        )
    )

    if freeze_path.exists():
        freeze = json.loads(
            freeze_path.read_text(
                encoding="utf-8"
            )
        )

        print(
            "STUDY="
            + freeze.get(
                "study",
                "NA",
            )
        )

        print(
            "SAMPLE_START="
            + freeze[
                "sample"
            ][
                "start"
            ]
        )

        print(
            "SAMPLE_END="
            + freeze[
                "sample"
            ][
                "last_market_session"
            ]
        )

    return (
        0
        if certification[
            "certification_status"
        ]
        == "PASS"
        else 1
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hc-macro",
        description=(
            "HilmarCorp — Bitcoin Macro "
            "Integration Study"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "data",
        help=(
            "Download, align and certify "
            "the research panel."
        ),
    )

    subparsers.add_parser(
        "core",
        help=(
            "Run core regressions, rolling "
            "models and structural analysis."
        ),
    )

    robustness = subparsers.add_parser(
        "robustness",
        help=(
            "Run FDR, SupF and structural "
            "robustness suite."
        ),
    )

    robustness.add_argument(
        "--supf-bootstraps",
        type=int,
        default=499,
    )

    events = subparsers.add_parser(
        "events",
        help=(
            "Run corrected conditional "
            "event study."
        ),
    )

    events.add_argument(
        "--bootstraps",
        type=int,
        default=1999,
    )

    events.add_argument(
        "--permutations",
        type=int,
        default=1999,
    )

    subparsers.add_parser(
        "freeze",
        help=(
            "Freeze results and run final "
            "research certification."
        ),
    )

    subparsers.add_parser(
        "verify-freeze",
        help="Verify exact numerical identity against the certified baseline.",
    )

    subparsers.add_parser(
        "status",
        help=(
            "Display frozen research status."
        ),
    )

    all_parser = subparsers.add_parser(
        "all",
        help=(
            "Run the complete research "
            "pipeline."
        ),
    )

    all_parser.add_argument(
        "--supf-bootstraps",
        type=int,
        default=499,
    )

    all_parser.add_argument(
        "--event-bootstraps",
        type=int,
        default=1999,
    )

    all_parser.add_argument(
        "--event-permutations",
        type=int,
        default=1999,
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "data":
        run_data_pipeline()

    elif args.command == "core":
        run_core_pipeline()

    elif args.command == "robustness":
        run_robustness_pipeline(
            supf_bootstraps=args.supf_bootstraps
        )

    elif args.command == "events":
        run_event_pipeline(
            bootstraps=args.bootstraps,
            permutations=args.permutations,
        )

    elif args.command == "freeze":
        run_freeze_pipeline()

    elif args.command == "verify-freeze":
        verify_against_baseline()

    elif args.command == "status":
        return status_command()

    elif args.command == "all":
        run_all_pipeline(
            supf_bootstraps=args.supf_bootstraps,
            event_bootstraps=args.event_bootstraps,
            event_permutations=args.event_permutations,
        )

    else:
        parser.error("Unknown command.")

    return 0
