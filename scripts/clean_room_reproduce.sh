#!/usr/bin/env bash
set -euo pipefail

BUNDLE="${1:-reproducibility/private/hc_macro_raw_snapshot.tar.gz}"

test -f "$BUNDLE" || {
  echo "ABORT=PRIVATE_REPRO_BUNDLE_MISSING"
  exit 1
}

git rev-parse HEAD >/dev/null 2>&1 || {
  echo "ABORT=GIT_COMMIT_REQUIRED"
  exit 1
}

ROOT="$(pwd)"
TMP="$(mktemp -d)"

trap 'rm -rf "$TMP"' EXIT

echo "================================================================================================"
echo "HILMARCORP — CLEAN-ROOM EMPIRICAL REPRODUCTION"
echo "================================================================================================"
echo "SOURCE_COMMIT=$(git rev-parse HEAD)"
echo "CLEAN_ROOM=$TMP"

echo
echo "[1/8] Exporting committed source tree..."

git archive \
  --format=tar \
  HEAD \
  | tar -xf - -C "$TMP"

echo "SOURCE_EXPORT=PASS"

echo
echo "[2/8] Restoring certified private raw snapshot..."

tar -xzf \
  "$ROOT/$BUNDLE" \
  -C "$TMP"

echo "RAW_SNAPSHOT_RESTORED=PASS"

cd "$TMP"

echo
echo "[3/8] Recreating exact locked environment..."

uv sync \
  --locked \
  --all-extras

echo "LOCKED_ENVIRONMENT=PASS"

echo
echo "[4/8] Verifying raw-data cryptographic identity..."

uv run python \
  scripts/verify_raw_manifest.py

echo
echo "[5/8] Static source gates before empirical execution..."

uv run python -m compileall -q src tests
uv run ruff check --select E9,F src tests
uv run pyright

echo "STATIC_GATES=PASS"

echo
echo "[6/8] Rebuilding complete empirical corpus FROM RAW..."

PYTHONHASHSEED=0 \
uv run hc-macro all

echo "EMPIRICAL_REBUILD=PASS"

echo
echo "[7/8] Testing reconstructed processed data and outputs..."

PYTHONHASHSEED=0 \
uv run pytest -q

echo "RECONSTRUCTED_TEST_SUITE=PASS"

echo
echo "[8/8] Verifying exact certified numerical identity..."

uv run hc-macro verify-freeze

echo
echo "================================================================================================"
echo "CLEAN_ROOM_REPRODUCIBILITY=PASS"
echo "SOURCE_REPRODUCIBILITY=PASS"
echo "ENVIRONMENT_REPRODUCIBILITY=PASS"
echo "RAW_DATA_INTEGRITY=PASS"
echo "EMPIRICAL_REBUILD=PASS"
echo "NUMERICAL_FREEZE_REPRODUCIBILITY=PASS"
echo "================================================================================================"
