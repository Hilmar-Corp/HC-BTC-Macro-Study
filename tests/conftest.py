from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import settings


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

settings.register_profile(
    "hc_ci",
    max_examples=100,
    derandomize=True,
    deadline=None,
)

settings.load_profile("hc_ci")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def frozen_results(repo_root: Path) -> dict:
    path = (
        repo_root
        / "outputs"
        / "diagnostics"
        / "final_results_freeze.json"
    )

    assert path.exists()

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="session")
def final_certification(repo_root: Path) -> dict:
    path = (
        repo_root
        / "outputs"
        / "diagnostics"
        / "phase7_certification.json"
    )

    assert path.exists()

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
