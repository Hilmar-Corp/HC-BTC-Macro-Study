.PHONY: sync lock compile lint typecheck test audit legacy-scan verify-raw \
        verify-freeze bundle clean-room paranoia due-diligence status tree \
        publication-figures

sync:
	uv sync --locked --all-extras

lock:
	uv lock
	uv export --locked --all-extras --format requirements-txt --output-file requirements.lock.txt

compile:
	uv run python -m compileall -q src tests

lint:
	uv run ruff check --select E9,F src tests

typecheck:
	uv run pyright

test:
	PYTHONHASHSEED=0 uv run pytest -q

audit:
	uv run pip-audit

legacy-scan:
	@test -z "$$(find src/hc_macro_integration -maxdepth 1 -type f -name 'phase*.py' -print)"
	@test -z "$$(find . -maxdepth 1 -type f -name 'run_phase*.py' -print)"

verify-raw:
	uv run python scripts/verify_raw_manifest.py

verify-freeze:
	uv run hc-macro verify-freeze

bundle:
	uv run python scripts/build_private_repro_bundle.py

clean-room:
	bash scripts/clean_room_reproduce.sh

publication-figures:
	PYTHONHASHSEED=0 uv run python -m hc_macro_integration.reporting.publication_figures

paranoia:
	PYTHONHASHSEED=0 uv run python -m compileall -q src tests
	PYTHONHASHSEED=0 uv run ruff check --select E9,F src tests
	$(MAKE) legacy-scan
	PYTHONHASHSEED=0 uv run pytest -q

due-diligence:
	uv lock --check
	$(MAKE) verify-raw
	$(MAKE) compile
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) legacy-scan
	$(MAKE) test
	$(MAKE) verify-freeze

status:
	uv run hc-macro status

tree:
	find src/hc_macro_integration -maxdepth 2 -type f \
		! -path '*/__pycache__/*' \
		| sort

