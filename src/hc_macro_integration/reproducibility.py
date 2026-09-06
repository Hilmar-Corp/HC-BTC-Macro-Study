from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ReproducibilityError(RuntimeError):
    pass


def canonicalize_freeze(payload: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    value.pop("freeze_timestamp_utc", None)
    return value


def verify_results_freeze(
    baseline_path: Path,
    rebuilt_path: Path,
) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rebuilt = json.loads(rebuilt_path.read_text(encoding="utf-8"))

    if canonicalize_freeze(baseline) != canonicalize_freeze(rebuilt):
        raise ReproducibilityError(
            "Rebuilt numerical freeze differs from certified baseline."
        )
