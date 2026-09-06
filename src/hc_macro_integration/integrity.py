from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class IntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_raw_manifest(raw_dir: Path) -> dict[str, Any]:
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file())

    if not files:
        raise IntegrityError("No raw files found.")

    result: dict[str, Any] = {
        "schema_version": 1,
        "files": {},
    }

    root = raw_dir.parent.parent

    for path in files:
        relative = str(path.relative_to(root))

        entry: dict[str, Any] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            entry["rows"] = int(len(df))
            entry["columns"] = list(map(str, df.columns))

        result["files"][relative] = entry

    return result


def verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files", {})

    if not files:
        raise IntegrityError("Manifest contains no files.")

    for relative, metadata in files.items():
        path = root / relative

        if not path.exists():
            raise IntegrityError(f"Missing frozen file: {relative}")

        if path.stat().st_size != int(metadata["bytes"]):
            raise IntegrityError(f"Size drift: {relative}")

        if sha256_file(path) != str(metadata["sha256"]):
            raise IntegrityError(f"SHA256 drift: {relative}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
