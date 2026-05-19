"""Fetch AAPL news from GDELT, cache per date. Tier-filtered output for prompts."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

NEWS_CACHE = Path(__file__).parent / "cache" / "news"
NEWS_CACHE.mkdir(parents=True, exist_ok=True)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Source domains by quality / institutional-ness
RETAIL_DOMAINS = {"finance.yahoo.com", "fool.com", "businessinsider.com", "cnbc.com", "marketwatch.com"}
SELLSIDE_DOMAINS = {"morningstar.com", "seekingalpha.com", "barrons.com", "investors.com"}
INSTITUTIONAL_DOMAINS = {"reuters.com", "bloomberg.com", "ft.com", "wsj.com", "ibtimes.com"}
CONTRARIAN_DOMAINS = {"insidermonkey.com", "zerohedge.com"}


def _fetch_day(ticker: str, day_str: str, max_records: int = 25) -> list[dict]:
    """Fetch all financial-English news for one day. Cached to disk."""
    cache_file = NEWS_CACHE / f"{ticker}_{day_str}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    # GDELT expects YYYYMMDDHHMMSS
    start = day_str.replace("-", "") + "000000"
    end = day_str.replace("-", "") + "235959"

    query = f"{ticker} stock sourcelang:eng"
    params = {
        "query": query,
        "format": "JSON",
        "maxrecords": str(max_records),
        "startdatetime": start,
        "enddatetime": end,
    }
    url = f"{GDELT_URL}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        if not text.strip():
            articles = []
        else:
            articles = json.loads(text).get("articles", [])
    except Exception as e:
        print(f"  GDELT fetch error for {day_str}: {e}")
        articles = []

    # Light cleanup
    cleaned = []
    for a in articles:
        cleaned.append({
            "title": (a.get("title") or "").strip(),
            "domain": (a.get("domain") or "").strip(),
            "seendate": (a.get("seendate") or "")[:8],  # YYYYMMDD
            "url": a.get("url", ""),
        })

    with open(cache_file, "w") as f:
        json.dump(cleaned, f)
    time.sleep(0.2)  # be polite to GDELT
    return cleaned


def fetch_window(ticker: str, dates: list[str]) -> dict[str, list[dict]]:
    """Fetch news for all dates. Returns dict[date_str → list of articles]."""
    out = {}
    for i, d in enumerate(dates):
        articles = _fetch_day(ticker, d)
        out[d] = articles
        if (i + 1) % 5 == 0:
            print(f"  ...fetched {i+1}/{len(dates)} days", flush=True)
    return out


def format_news_for_tier(articles: list[dict], tier: int) -> str:
    """Filter + format news depending on agent's info_tier."""
    if not articles:
        return "(no AAPL news today)"

    # Score each article by source quality
    def article_score(a):
        d = a.get("domain", "")
        if d in INSTITUTIONAL_DOMAINS:
            return 4
        if d in SELLSIDE_DOMAINS:
            return 3
        if d in CONTRARIAN_DOMAINS:
            return 2  # institutional/contrarian-only
        if d in RETAIL_DOMAINS:
            return 1
        return 0  # other / unknown

    scored = [(article_score(a), a) for a in articles]
    scored.sort(key=lambda x: (-x[0], x[1].get("seendate", "")))

    # Tier policy
    if tier == 1:
        # Retail: 3 retail-source headlines only
        selected = [a for s, a in scored if s == 1][:3]
        if not selected:
            selected = [a for s, a in scored][:3]
        prefix = "[Tier 1 Retail — top headlines]"
    elif tier == 2:
        # Sell-Side: 5 sell-side + 3 retail
        selected = ([a for s, a in scored if s in (3, 1)])[:8]
        prefix = "[Tier 2 Sell-Side — analyst-grade + retail context]"
    elif tier == 3:
        # Activist: 5 institutional + 5 contrarian (insider tracking)
        selected = ([a for s, a in scored if s in (4, 2)])[:10]
        if not selected:
            selected = [a for s, a in scored][:10]
        prefix = "[Tier 3 Activist — institutional + contrarian / forensic angle]"
    elif tier == 4:
        # Pod PM: top 12 by score (institutional first)
        selected = [a for s, a in scored][:12]
        prefix = "[Tier 4 Hedge Fund — full institutional feed]"
    else:  # tier 5
        # Fed watcher: all up to 15
        selected = [a for s, a in scored][:15]
        prefix = "[Tier 5 Macro — comprehensive feed + Fed channel context]"

    lines = [prefix]
    for a in selected:
        lines.append(f"  • [{a['domain']}] {a['title'][:140]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    # Quick test
    test_day = "2026-04-21"
    arts = _fetch_day("AAPL", test_day)
    print(f"Fetched {len(arts)} articles for {test_day}")
    for tier in [1, 3, 5]:
        print(f"\n--- Tier {tier} view ---")
        print(format_news_for_tier(arts, tier))
