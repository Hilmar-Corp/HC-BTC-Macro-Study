from __future__ import annotations

from pathlib import Path


ALLOWED_ROOT_MODULES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "config.py",
    "contracts.py",
    "paths.py",
    "pipeline.py",
    "integrity.py",
    "reproducibility.py",
    "run_manifest.py",
}


def test_no_legacy_phase_modules_remain(
    repo_root: Path,
):
    package = (
        repo_root
        / "src"
        / "hc_macro_integration"
    )

    legacy = sorted(
        path.name
        for path in package.glob(
            "phase*.py"
        )
    )

    assert legacy == []


def test_no_root_run_phase_scripts_remain(
    repo_root: Path,
):
    assert list(
        repo_root.glob(
            "run_phase*.py"
        )
    ) == []


def test_top_level_python_surface_is_minimal(
    repo_root: Path,
):
    package = (
        repo_root
        / "src"
        / "hc_macro_integration"
    )

    actual = {
        path.name
        for path in package.glob(
            "*.py"
        )
    }

    assert actual == (
        ALLOWED_ROOT_MODULES
    )


def test_domain_packages_are_present(
    repo_root: Path,
):
    package = (
        repo_root
        / "src"
        / "hc_macro_integration"
    )

    for domain in [
        "data",
        "models",
        "robustness",
        "reporting",
    ]:
        assert (
            package
            / domain
            / "__init__.py"
        ).exists()
