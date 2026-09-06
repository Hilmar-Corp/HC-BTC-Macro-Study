from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hc_macro_integration.integrity import sha256_file, write_json


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_manifest(
    command: str,
    argv: list[str],
    status: str,
    started: str,
    finished: str,
    error: str | None,
) -> Path:
    root = Path.cwd()

    packages = sorted(
        [
            {
                "name": dist.metadata.get("Name") or "UNKNOWN",
                "version": dist.version,
            }
            for dist in importlib.metadata.distributions()
        ],
        key=lambda x: x["name"].lower(),
    )

    def digest(relative: str):
        path = root / relative
        return sha256_file(path) if path.exists() else None

    payload = {
        "schema_version": 1,
        "command": command,
        "argv": argv,
        "status": status,
        "error": error,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "hashes": {
            "protocol": digest("config/protocol.yaml"),
            "raw_manifest": digest(
                "reproducibility/manifests/raw_data_manifest.json"
            ),
            "uv_lock": digest("uv.lock"),
        },
        "packages": packages,
    }

    directory = root / "outputs" / "run_manifests"
    directory.mkdir(parents=True, exist_ok=True)

    stamp = finished.replace(":", "").replace("-", "").replace("+00:00", "Z")
    path = directory / f"{stamp}__{command}.json"

    write_json(path, payload)
    return path
