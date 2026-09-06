import copy
import json

import pytest

from hc_macro_integration.reproducibility import (
    ReproducibilityError,
    canonicalize_freeze,
    verify_results_freeze,
)


def sample():
    return {
        "freeze_timestamp_utc": "2026-09-06T00:00:00+00:00",
        "main_results": {"beta": 0.817706, "r2": 0.167010},
    }


def test_timestamp_is_explicitly_non_empirical():
    a = sample()
    b = copy.deepcopy(a)
    b["freeze_timestamp_utc"] = "2030-01-01T00:00:00+00:00"

    assert canonicalize_freeze(a) == canonicalize_freeze(b)


def test_1e_minus_12_numerical_drift_fails(tmp_path):
    a = sample()
    b = copy.deepcopy(a)
    b["main_results"]["beta"] += 1e-12

    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"

    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")

    with pytest.raises(ReproducibilityError):
        verify_results_freeze(pa, pb)
