# Contributing

Before a pull request:

```bash
uv sync --locked --all-extras
make due-diligence
```

A refactor may change architecture. It may not silently change frozen
empirical outputs.

Any change to a coefficient, statistic, sample boundary, transformation,
break date, FDR result or research fingerprint must be explicitly declared as
a new research revision.

Do not publish restricted third-party raw market data without confirming
redistribution rights.
