from __future__ import annotations

import json
import subprocess
import sys


def test_module_cli_status(
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

    expected_count = len(
        baseline_manifest
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hc_macro_integration",
            "status",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    assert (
        "RESEARCH_STATUS=PASS"
        in result.stdout
    )

    assert (
        f"ARTIFACT_COUNT={expected_count}"
        in result.stdout
    )
