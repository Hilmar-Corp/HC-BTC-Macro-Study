import json

from hc_macro_integration.integrity import verify_manifest


def test_frozen_raw_manifest_verifies(repo_root):
    path = (
        repo_root
        / "reproducibility"
        / "manifests"
        / "raw_data_manifest.json"
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert len(manifest["files"]) >= 7

    verify_manifest(repo_root, manifest)
