from __future__ import annotations

import json


def test_final_certification_is_pass(
    final_certification,
    repo_root,
):
    baseline_manifest = json.loads(
        (
            repo_root
            / "reproducibility"
            / "baseline"
            / "artifact_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        final_certification[
            "certification_status"
        ]
        == "PASS"
    )

    assert all(
        final_certification[
            "checks"
        ].values()
    )

    assert (
        final_certification[
            "artifact_count"
        ]
        == len(
            baseline_manifest
        )
    )
