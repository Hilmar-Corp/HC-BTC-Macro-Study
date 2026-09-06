from __future__ import annotations

from pathlib import Path

import yaml

from hc_macro_integration.paths import CONFIG_DIR


def load_protocol(path: Path | None = None) -> dict:
    if path is None:
        path = CONFIG_DIR / "protocol.yaml"

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Protocol configuration must be a mapping.")

    return config
