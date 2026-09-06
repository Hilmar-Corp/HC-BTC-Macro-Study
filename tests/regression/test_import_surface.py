from __future__ import annotations

import importlib
import pkgutil

import hc_macro_integration


def test_every_package_module_imports_cleanly():
    failures = []

    prefix = (
        hc_macro_integration.__name__
        + "."
    )

    for module in pkgutil.walk_packages(
        hc_macro_integration.__path__,
        prefix=prefix,
    ):
        name = module.name

        try:
            importlib.import_module(
                name
            )
        except Exception as exc:
            failures.append(
                (
                    name,
                    type(exc).__name__,
                    str(exc),
                )
            )

    assert failures == []
