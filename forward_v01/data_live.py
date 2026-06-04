"""LIVE market data for forward testing — no bundled JSON, no stale cache.

The whole point of forward testing is that "tomorrow" does not exist yet, so we
MUST hit yfinance live every run (analyst_v01/data.py prefers bundled JSONs that
end ~today — useless here). We reuse its pure-pandas indicator math but fetch
fresh prices ourselves.

  context(ticker, date)             -> as-of-`date` view the agents read
  actual_close(ticker, target_date) -> live close on target_date, or None if not
                                        yet available (keeps it pending; NEVER
                                        scored in the same run it was predicted)
  next_trading_day(ticker, date)    -> the next date with a real bar after `date`
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "toy_v06"))  # for news_fetcher

import news_fetcher  # noqa: E402  GDELT news, disk-cached per ticker/date


def _load(name: str, path: Path):
    """Load a module by explicit file path — mvp/data.py and analyst_v01/data.py
    are both named `data`, so a plain import would collide on sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses/typing resolve via sys.modules[__module__]
    spec.loader.exec_module(mod)
    return mod


mvp_data = _load("mvp_data", ROOT / "mvp" / "data.py")          # yfinance loader
_avd = _load("analyst_v01_data", ROOT / "analyst_v01" / "data.py")  # indicators
_indicators = _avd._indicators
_trend = _avd._trend


# ── live price access (fresh every run) ──────────────────────────────────────
_PRICE_MEMO: dict[str, pd.DataFrame] = {}  # per-process: one fetch per ticker/run


def _live_prices(ticker: str, retries: int = 5) -> pd.DataFrame:
    """Full daily OHLCV up to the latest bar. Fetches FRESH from yfinance
    (use_cache=False → newest close), with retry/backoff for yfinance's frequent
    transient failures. Falls back to the parquet cache if the live fetch keeps
    failing — stale data is leakage-SAFE (it only omits the newest bar, never
    adds future data) and beats crashing the daily run."""
    if ticker in _PRICE_MEMO:
        return _PRICE_MEMO[ticker]
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).date()
    start = (pd.Timestamp.today() - pd.Timedelta(days=900)).date()

    def _norm(df):
        df = df.sort_index()
        df.index = pd.to_datetime(df.index).date
        return df

    last_err = None
    for attempt in range(retries):
        try:
            df = mvp_data.fetch_history(ticker, str(start), str(end), use_cache=False)
            if df is not None and not df.empty:
                df = _norm(df)
                _PRICE_MEMO[ticker] = df
                return df
        except Exception as e:
            last_err = e
        time.sleep(2 * (attempt + 1))

    # fallback: read the parquet cache DIRECTLY (mvp.fetch_history's coverage
    # check rejects it when `end` is in the future). Stale is leakage-SAFE — it
    # only omits the newest bar, never adds future data.
    cache_file = mvp_data.CACHE_DIR / f"{ticker}.parquet"
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty:
                df = _norm(df)
                print(f"  ⚠ {ticker}: live fetch failed, using CACHED prices "
                      f"(latest {df.index[-1]}) — {last_err}")
                # Do NOT memoize a stale fallback: a later call in the same run
                # (e.g. Phase 2 after Phase 1) must be free to retry live and get
                # fresh data, instead of being stuck on yesterday's bar.
                return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"no price data for {ticker} (live+cache failed): {last_err}")


def latest_trading_date(ticker: str) -> str:
    return str(_live_prices(ticker).index[-1])


def headlines(ticker: str, date: str, n: int = 8, retries: int = 2) -> list[str]:
    """News titles for `date`, robust to GDELT 429s. news_fetcher caches an
    empty `[]` on any fetch error, which would permanently mask real news — so
    we treat an empty result as suspect: drop the poisoned cache and retry with
    backoff. A genuinely no-news day simply re-checks each run (cheap)."""
    cache = news_fetcher.NEWS_CACHE / f"{ticker}_{date}.json"
    for attempt in range(retries + 1):
        arts = news_fetcher._fetch_day(ticker, date)
        if arts:
            return [a["title"] for a in arts if a.get("title")][:n]
        if cache.exists():
            cache.unlink()  # don't let an error-empty result poison future runs
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    return []


def context(ticker: str, date: str, lookback: int = 60) -> dict:
    """As-of-`date` view: history STRICTLY <= date. No lookahead."""
    d = pd.to_datetime(date).date()
    df = _live_prices(ticker)
    upto = df[df.index <= d]
    if upto.empty:
        raise RuntimeError(f"no live price for {ticker} on/before {date}")
    t0 = upto.index[-1]
    closes = upto["close"]

    head = headlines(ticker, str(t0))

    window = upto.tail(lookback)
    price_window = [{"date": str(i), "close": round(float(c), 2)}
                    for i, c in window["close"].items()]
    return {
        "ticker": ticker,
        "date": str(t0),
        "t0_close": round(float(closes.iloc[-1]), 2),
        "indicators": _indicators(closes),
        "trend": _trend(closes),
        "news_headlines": head,
        "price_window": price_window,
    }


def actual_close(ticker: str, target_date: str) -> float | None:
    """Live close on `target_date`. None if that bar does not exist yet (market
    hasn't reached/closed that day) — the prediction stays pending."""
    d = pd.to_datetime(target_date).date()
    df = _live_prices(ticker)
    if d in set(df.index):
        return round(float(df.loc[d, "close"]), 2)
    return None


def next_trading_day(ticker: str, date: str) -> str | None:
    """First real trading bar strictly after `date`, or None if not yet known."""
    d = pd.to_datetime(date).date()
    fwd = _live_prices(ticker)
    fwd = fwd[fwd.index > d]
    return str(fwd.index[0]) if len(fwd) else None


def nth_trading_close(ticker: str, as_of_date: str, h: int) -> tuple[str | None, float | None]:
    """(date, close) of the h-th trading bar AFTER as_of_date. (None, None) if
    that bar does not exist yet — the prediction stays pending until it does."""
    d = pd.to_datetime(as_of_date).date()
    fwd = _live_prices(ticker)
    fwd = fwd[fwd.index > d]
    if len(fwd) >= h:
        return str(fwd.index[h - 1]), round(float(fwd["close"].iloc[h - 1]), 2)
    return None, None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    args = ap.parse_args()
    ld = latest_trading_date(args.ticker)
    print(f"{args.ticker} latest live trading date: {ld}")
    ctx = context(args.ticker, ld)
    print(f"  t0_close={ctx['t0_close']}  rsi14={ctx['indicators']['rsi14']}  "
          f"ret_5d={ctx['trend']['ret_5d_pct']}%  headlines={len(ctx['news_headlines'])}")
    for h in ctx["news_headlines"][:3]:
        print(f"    • {h[:80]}")
