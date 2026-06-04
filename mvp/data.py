"""yfinance loader with local parquet cache. TimeWindow-aware accessor."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache" / "prices"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PriceBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


def fetch_history(
    ticker: str,
    start: date | str,
    end: date | str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV history. Caches to parquet by ticker.

    Returns DataFrame indexed by date with columns: open, high, low, close, volume.
    """
    cache_file = CACHE_DIR / f"{ticker}.parquet"

    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        df.index = pd.to_datetime(df.index).date
        start_d = pd.to_datetime(start).date()
        end_d = pd.to_datetime(end).date()
        # check coverage
        if df.index.min() <= start_d and df.index.max() >= end_d:
            return df.loc[(df.index >= start_d) & (df.index <= end_d)]

    # fetch fresh — fetch wider window for cache
    fetch_start = (pd.to_datetime(start) - pd.Timedelta(days=365)).date()
    fetch_end = (pd.to_datetime(end) + pd.Timedelta(days=1)).date()
    raw = yf.download(
        ticker,
        start=str(fetch_start),
        end=str(fetch_end),
        progress=False,
        auto_adjust=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned empty for {ticker} {fetch_start} → {fetch_end}")

    # flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index).date
    df.index.name = "date"

    df.to_parquet(cache_file)
    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()
    return df.loc[(df.index >= start_d) & (df.index <= end_d)]


def time_window_view(df: pd.DataFrame, decision_date: date, lookback_days: int = 30) -> pd.DataFrame:
    """Return view of df strictly before decision_date, last lookback_days trading days.
    Used so personas only see info available at decision time."""
    earlier = df[df.index < decision_date]
    return earlier.tail(lookback_days)


def trading_days_in_window(df: pd.DataFrame, start: date, end: date) -> list[date]:
    """Return trading days from df in [start, end]."""
    mask = (df.index >= start) & (df.index <= end)
    return [d for d in df[mask].index]
