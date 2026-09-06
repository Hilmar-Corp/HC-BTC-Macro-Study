def test_uv_lock_exists(repo_root):
    path = repo_root / "uv.lock"
    assert path.exists()
    assert path.stat().st_size > 1000


def test_python_patch_version_is_pinned(repo_root):
    version = (repo_root / ".python-version").read_text(encoding="utf-8").strip()
    assert version.startswith("3.12.")
    assert version.count(".") == 2


def test_apache_2_license_present(repo_root):
    text = (repo_root / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0" in text
