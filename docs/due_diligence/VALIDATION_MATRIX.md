# Validation Matrix

| Control | Mechanism |
|---|---|
| Exact Python | `.python-version` |
| Exact dependencies | `uv.lock` |
| Data provenance | `data/provenance.yaml` |
| Raw identity | SHA-256 manifest |
| Run environment | automatic run manifests |
| No-network pytest | pytest-socket |
| No look-ahead | rolling mutation tests |
| Date integrity | contracts + adversarial tests |
| Factor alignment | strict FRED tests |
| Multiple testing | frozen BH-FDR tests |
| Break stability | frozen Wald/SupF tests |
| Event overlap | property-based tests |
| Seed determinism | adversarial permutation test |
| File corruption | SHA-256 corruption test |
| Numerical identity | exact canonical freeze comparator |
| Clean checkout | clean-room script |
| Static analysis | Ruff + Pyright |
| Dependency CVEs | pip-audit |
| CI | GitHub Actions |
