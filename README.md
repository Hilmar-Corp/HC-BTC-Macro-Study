# HC-BTC-Macro-Integration-Study

Quantitative research repository for:

**Bitcoin est-il devenu un actif macro ?**

## Status

The empirical corpus is frozen and certified.

Frozen sample:

- start: **2017-08-17**
- last traditional-market session: **2026-09-04**

## Reproducibility

Environment:

```bash
uv sync --locked --all-extras
```

Exact dependency resolution:

```text
uv.lock
```

Exact Python patch version:

```text
.python-version
```

Raw-data provenance:

```text
data/provenance.yaml
```

Raw-data cryptographic identity:

```text
reproducibility/manifests/raw_data_manifest.json
```

Certified numerical baseline:

```text
reproducibility/baseline/final_results_freeze.json
```

## Research commands

```bash
hc-macro status
hc-macro data
hc-macro core
hc-macro robustness
hc-macro events
hc-macro freeze
hc-macro verify-freeze
hc-macro all
```

`hc-macro all` is the frozen reproducibility path. It rebuilds from the
existing frozen raw snapshot and verifies exact numerical identity against the
certified baseline.

External data refreshes remain explicit through `hc-macro data`.

## Due-diligence gate

```bash
make due-diligence
```

## Private data bundle

Raw third-party snapshots are excluded from public Git redistribution.

Create the authorized internal reproducibility bundle with:

```bash
make bundle
```

After the first Git commit:

```bash
make clean-room
```

This creates a temporary clean checkout, restores the exact private raw
snapshot, installs the locked environment, runs the test suite, rebuilds the
study and verifies the frozen numerical result.

## GitHub CI

The repository ships with:

- push / PR CI;
- lockfile verification;
- type checking;
- dependency audit;
- manual full empirical clean-room workflow using a private data bundle.

## License

Original source code and documentation: **Apache License 2.0**.

Third-party datasets are not relicensed. See `NOTICE`.
