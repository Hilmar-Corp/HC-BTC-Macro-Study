#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "GITHUB PRE-PUSH DUE-DILIGENCE"
echo "================================================================================"

for f in \
  LICENSE \
  NOTICE \
  SECURITY.md \
  CONTRIBUTING.md \
  CITATION.cff \
  uv.lock \
  .python-version \
  data/provenance.yaml \
  reproducibility/manifests/raw_data_manifest.json \
  reproducibility/baseline/final_results_freeze.json
do
  test -f "$f" || { echo "ABORT=MISSING:$f"; exit 1; }
done

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git ls-files | grep -E '^data/raw/' >/dev/null 2>&1; then
    echo "ABORT=RAW_THIRD_PARTY_DATA_TRACKED"
    exit 1
  fi

  if git ls-files | grep -E '^reproducibility/private/' \
      | grep -v '.gitkeep' >/dev/null 2>&1; then
    echo "ABORT=PRIVATE_REPRO_BUNDLE_TRACKED"
    exit 1
  fi
fi

uv lock --check
make due-diligence

echo "GITHUB_PREFLIGHT=PASS"
