import pytest

from hc_macro_integration.integrity import IntegrityError, sha256_file, verify_manifest


def test_one_byte_corruption_is_detected(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"abcdef")

    manifest = {
        "files": {
            "raw.bin": {
                "bytes": 6,
                "sha256": sha256_file(path),
            }
        }
    }

    path.write_bytes(b"abcdeg")

    with pytest.raises(IntegrityError, match="SHA256 drift"):
        verify_manifest(tmp_path, manifest)


def test_missing_file_is_detected(tmp_path):
    manifest = {
        "files": {
            "missing.bin": {
                "bytes": 1,
                "sha256": "0" * 64,
            }
        }
    }

    with pytest.raises(IntegrityError, match="Missing frozen file"):
        verify_manifest(tmp_path, manifest)
