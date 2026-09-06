from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "config"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

COINBASE_DIR = RAW_DIR / "coinbase"
MARKET_DIR = RAW_DIR / "market"
FRED_DIR = RAW_DIR / "fred"

OUTPUT_DIR = ROOT / "outputs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

for path in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    COINBASE_DIR,
    MARKET_DIR,
    FRED_DIR,
    OUTPUT_DIR,
    DIAGNOSTICS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
