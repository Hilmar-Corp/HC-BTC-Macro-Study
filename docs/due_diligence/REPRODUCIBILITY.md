# Reproducibility Standard

R1 — source reproducibility:
clean source checkout imports, compiles and passes architecture controls.

R2 — environment reproducibility:
`.python-version` pins the Python patch release and `uv.lock` pins dependency
resolution.

R3 — data reproducibility:
every frozen raw input is identified by path, byte size and SHA-256.

R4 — numerical reproducibility:
the rebuilt `final_results_freeze.json` must match the certified baseline
exactly after removing only the non-empirical run timestamp.

R5 — clean-room reproducibility:
a fresh Git checkout plus the separately supplied authorized raw-data bundle
must reconstruct the certified numerical freeze.

Public GitHub alone intentionally does not redistribute third-party raw market
data where provider redistribution rights have not been established.
