from pathlib import Path
from hc_macro_integration.integrity import build_raw_manifest, write_json

root = Path.cwd()
manifest = build_raw_manifest(root / "data" / "raw")
destination = root / "reproducibility" / "manifests" / "raw_data_manifest.json"
write_json(destination, manifest)

print("RAW_MANIFEST_FILES=" + str(len(manifest["files"])))
print("RAW_MANIFEST=" + str(destination))
