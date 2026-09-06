import hashlib
import tarfile
from pathlib import Path

root = Path.cwd()
raw = root / "data" / "raw"
destination = root / "reproducibility" / "private" / "hc_macro_raw_snapshot.tar.gz"

destination.parent.mkdir(parents=True, exist_ok=True)

with tarfile.open(destination, "w:gz") as archive:
    archive.add(raw, arcname="data/raw")

h = hashlib.sha256(destination.read_bytes()).hexdigest()

destination.with_suffix(".tar.gz.sha256").write_text(
    h + "  " + destination.name + "\n",
    encoding="utf-8",
)

print("PRIVATE_REPRO_BUNDLE=" + str(destination))
print("PRIVATE_REPRO_BUNDLE_SHA256=" + h)
print("PRIVATE_REPRO_BUNDLE_BYTES=" + str(destination.stat().st_size))
