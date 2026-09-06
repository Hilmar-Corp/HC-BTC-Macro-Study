import json
from pathlib import Path
from hc_macro_integration.integrity import verify_manifest

root = Path.cwd()
path = root / "reproducibility" / "manifests" / "raw_data_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))

verify_manifest(root, manifest)

print("RAW_DATA_INTEGRITY=PASS")
print("RAW_FILES_VERIFIED=" + str(len(manifest["files"])))
