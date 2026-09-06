# Architecture

## Canonical repository structure

```text
HC-BTC-Macro-Integration-Study/
├── README.md
├── Makefile
├── pyproject.toml
├── config/
│   └── protocol.yaml
├── docs/
│   ├── ARCHITECTURE.md
│   └── RESEARCH_CONTRACT.md
├── src/
│   └── hc_macro_integration/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── contracts.py
│       ├── paths.py
│       ├── pipeline.py
│       ├── data/
│       │   ├── audit.py
│       │   ├── certification.py
│       │   ├── download.py
│       │   ├── panel.py
│       │   └── strict.py
│       ├── models/
│       │   ├── attribution.py
│       │   ├── break_details.py
│       │   ├── breaks_bic.py
│       │   ├── breaks_supf.py
│       │   ├── events.py
│       │   ├── interactions.py
│       │   ├── linear.py
│       │   ├── periods.py
│       │   └── rolling.py
│       ├── robustness/
│       │   ├── ablations.py
│       │   ├── event_study.py
│       │   └── multiple_testing.py
│       └── reporting/
│           ├── certification.py
│           └── freeze.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── data/
└── outputs/
```

## Design rules

### 1. Domain structure, not historical phase structure

Source modules are organized by scientific responsibility. Historical
`phaseN_*` module names and root-level `run_phaseN.py` scripts are forbidden.

### 2. One execution surface

The supported execution interface is:

```bash
hc-macro data
hc-macro core
hc-macro robustness
hc-macro events
hc-macro freeze
hc-macro status
hc-macro all
```

### 3. Frozen empirical contract

Refactoring may change source layout. It may not change certified empirical
outputs without an explicit new research freeze.

### 4. Absolute internal imports

Domain modules use canonical absolute package imports. Moving a source file
must not silently retarget a relative dependency.

### 5. Three test layers

- `tests/unit`: mathematical and algorithmic invariants.
- `tests/integration`: local frozen-data pipeline contracts.
- `tests/regression`: certified numerical outputs, artifact hashes and
  architecture constraints.

### 6. Network isolation

Pytest runs with sockets disabled. Tests cannot repair missing fixtures by
silently downloading fresh market data.

### 7. Paranoia principle

Silent failure is worse than explicit failure. Duplicated dates, non-finite
inputs, non-positive prices, insufficient samples, artifact drift, import
drift and look-ahead contamination must fail loudly.
