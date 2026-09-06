from __future__ import annotations

import io
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

from hc_macro_integration.config import load_protocol
from hc_macro_integration.paths import COINBASE_DIR, FRED_DIR, MARKET_DIR


COINBASE_CANDLES_URL = (
    "https://api.exchange.coinbase.com/products/{product}/candles"
)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def download_coinbase_hourly(
    product: str,
    start: str,
    end: datetime | None = None,
    granularity: int = 3600,
) -> pd.DataFrame:
    if end is None:
        end = datetime.now(timezone.utc)

    cursor = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end = end.astimezone(timezone.utc)

    chunk_hours = 290
    frames: list[pd.DataFrame] = []

    headers = {
        "User-Agent": "HilmarCorp-Research/1.0",
        "Accept": "application/json",
    }

    session = requests.Session()

    while cursor < end:
        chunk_end = min(
            cursor + timedelta(hours=chunk_hours),
            end,
        )

        url = COINBASE_CANDLES_URL.format(product=product)

        params = {
            "start": utc_iso(cursor),
            "end": utc_iso(chunk_end),
            "granularity": granularity,
        }

        response = session.get(
            url,
            params=params,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()

        if payload:
            frame = pd.DataFrame(
                payload,
                columns=[
                    "timestamp",
                    "low",
                    "high",
                    "open",
                    "close",
                    "volume",
                ],
            )

            frames.append(frame)

        cursor = chunk_end
        time.sleep(0.12)

    if not frames:
        raise RuntimeError("No Coinbase candles downloaded.")

    df = pd.concat(frames, ignore_index=True)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    )

    numeric_cols = [
        "low",
        "high",
        "open",
        "close",
        "volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df = (
        df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Coinbase timestamps identify the START of the hourly candle.
    # The close belongs exactly one hour later.
    df["close_time"] = df["timestamp"] + pd.Timedelta(hours=1)

    return df


def download_yahoo_daily(
    ticker: str,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=False,
    )

    if data is None:
        raise RuntimeError(
            f"Yahoo returned no dataframe for {ticker}."
        )

    if data.empty:
        raise RuntimeError(f"No Yahoo data downloaded for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in data.columns
        ]

    data = data.reset_index()

    rename = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }

    data = data.rename(columns=rename)

    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    return data


def download_fred(series_id: str) -> pd.DataFrame:
    url = FRED_CSV_URL.format(series_id=series_id)

    response = requests.get(
        url,
        headers={"User-Agent": "HilmarCorp-Research/1.0"},
        timeout=30,
    )
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    return df


def run_downloads() -> None:
    protocol = load_protocol()

    start = protocol["sample"]["start"]

    product = protocol["bitcoin"]["primary"]["product"]
    granularity = protocol["bitcoin"]["primary"]["granularity_seconds"]

    print(f"[DOWNLOAD] Coinbase {product}")

    btc = download_coinbase_hourly(
        product=product,
        start=start,
        granularity=granularity,
    )

    btc_path = COINBASE_DIR / "btc_usd_hourly.parquet"
    btc.to_parquet(btc_path, index=False)

    print(
        f"[OK] {btc_path} | "
        f"{len(btc):,} rows | "
        f"{btc['timestamp'].min()} -> {btc['timestamp'].max()}"
    )

    ticker = protocol["market_factor"]["primary"]["ticker"]

    print(f"[DOWNLOAD] Yahoo {ticker}")

    market = download_yahoo_daily(
        ticker=ticker,
        start=start,
    )

    market_path = MARKET_DIR / "nasdaq100_daily.parquet"
    market.to_parquet(market_path, index=False)

    print(
        f"[OK] {market_path} | "
        f"{len(market):,} rows | "
        f"{market['date'].min().date()} -> "
        f"{market['date'].max().date()}"
    )

    for key in ["dollar", "real_rate", "credit"]:
        spec = protocol["fred"][key]
        series_id = spec["series_id"]

        print(f"[DOWNLOAD] FRED {series_id}")

        df = download_fred(series_id)

        path = FRED_DIR / f"{series_id}.parquet"
        df.to_parquet(path, index=False)

        print(
            f"[OK] {path} | "
            f"{df['value'].notna().sum():,} non-null observations"
        )


if __name__ == "__main__":
    run_downloads()
