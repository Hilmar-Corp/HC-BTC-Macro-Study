from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from hc_macro_integration.integrity import (
    sha256_file,
    write_json,
)
from hc_macro_integration.reproducibility import (
    verify_results_freeze,
)


class ArtifactContractError(RuntimeError):
    pass


DERIVED_DIRECTORIES = (
    Path("data/processed"),
    Path("outputs/tables"),
    Path("outputs/figures"),
    Path("outputs/diagnostics"),
)

MANIFEST_RELATIVE_PATH = Path(
    "outputs/diagnostics/artifact_manifest.json"
)

BASELINE_MANIFEST_RELATIVE_PATH = Path(
    "reproducibility/baseline/artifact_manifest.json"
)

BASELINE_FREEZE_RELATIVE_PATH = Path(
    "reproducibility/baseline/final_results_freeze.json"
)

CURRENT_FREEZE_RELATIVE_PATH = Path(
    "outputs/diagnostics/final_results_freeze.json"
)

NON_SCIENTIFIC_FILENAMES = {
    "artifact_manifest.json",
    "phase7_certification.json",
    "pyright_8c2.json",
    "pyright_8c2_summary.json",
    "pyright_8c3.txt",
    "pyright_after_official_stubs.txt",
    "clean_room_artifact_forensics.json",
}


def reset_derived_workspace(
    root: Path,
) -> None:
    for relative in DERIVED_DIRECTORIES:
        path = root / relative

        if path.exists():
            shutil.rmtree(path)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def stabilize_final_freeze(
    root: Path,
) -> None:
    baseline = (
        root
        / BASELINE_FREEZE_RELATIVE_PATH
    )

    current = (
        root
        / CURRENT_FREEZE_RELATIVE_PATH
    )

    if not baseline.exists():
        raise ArtifactContractError(
            "Immutable baseline freeze is missing."
        )

    if not current.exists():
        raise ArtifactContractError(
            "Recomputed final freeze is missing."
        )

    verify_results_freeze(
        baseline,
        current,
    )

    current.write_bytes(
        baseline.read_bytes()
    )


def _candidate_artifacts(
    root: Path,
) -> list[Path]:
    paths: list[Path] = []

    for relative in DERIVED_DIRECTORIES:
        directory = root / relative

        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            if path.name in NON_SCIENTIFIC_FILENAMES:
                continue

            paths.append(path)

    return sorted(
        paths,
        key=lambda path:
            str(path.relative_to(root)),
    )


def _entry(
    path: Path,
) -> dict[str, Any]:
    return {
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def build_canonical_artifact_manifest(
    root: Path,
) -> dict[str, dict[str, Any]]:
    baseline_path = (
        root
        / BASELINE_MANIFEST_RELATIVE_PATH
    )

    if baseline_path.exists():
        baseline = json.loads(
            baseline_path.read_text(
                encoding="utf-8"
            )
        )

        expected_paths = sorted(
            baseline.keys()
        )

        manifest: dict[
            str,
            dict[str, Any],
        ] = {}

        for relative in expected_paths:
            path = root / relative

            if not path.exists():
                raise ArtifactContractError(
                    "Missing canonical artifact: "
                    + relative
                )

            manifest[relative] = _entry(
                path
            )

        return manifest

    manifest = {}

    for path in _candidate_artifacts(
        root
    ):
        relative = str(
            path.relative_to(root)
        )

        manifest[relative] = _entry(
            path
        )

    if not manifest:
        raise ArtifactContractError(
            "No scientific artifacts were produced."
        )

    return manifest


def write_canonical_artifact_manifest(
    root: Path,
) -> Path:
    manifest = (
        build_canonical_artifact_manifest(
            root
        )
    )

    destination = (
        root
        / MANIFEST_RELATIVE_PATH
    )

    write_json(
        destination,
        manifest,
    )

    return destination
