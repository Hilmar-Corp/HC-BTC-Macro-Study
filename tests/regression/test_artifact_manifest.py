from __future__ import annotations

import json

from hc_macro_integration.integrity import (
    sha256_file,
)


def test_every_frozen_artifact_matches_manifest(
    repo_root,
):
    current_path = (
        repo_root
        / "outputs"
        / "diagnostics"
        / "artifact_manifest.json"
    )

    baseline_path = (
        repo_root
        / "reproducibility"
        / "baseline"
        / "artifact_manifest.json"
    )

    current = json.loads(
        current_path.read_text(
            encoding="utf-8"
        )
    )

    baseline = json.loads(
        baseline_path.read_text(
            encoding="utf-8"
        )
    )

    assert current == baseline

    for relative_path, metadata in (
        current.items()
    ):
        path = (
            repo_root
            / relative_path
        )

        assert path.exists(), (
            "Missing frozen artifact: "
            + relative_path
        )

        assert (
            path.stat().st_size
            == metadata["bytes"]
        )

        assert (
            sha256_file(path)
            == metadata["sha256"]
        )
