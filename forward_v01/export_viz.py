"""Bundle results into a single viz_data.json for the static explorer page.

Packs: the population (personas + signed influence edges), and for every
(date, ticker) we have predicted: the aggregate forecast, the non-LLM baseline,
the scored outcome (if matured), a recent price path, and every agent's view
(from results/views/, when captured). The static explore.html reads only this
file — no server, no Python at view time.

    python -m forward_v01.export_viz [--date-stamp 2026-06-04]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from forward_v01 import scoreboard as sb  # noqa: E402
from forward_v01 import data_live          # noqa: E402
from forward_v01 import market             # noqa: E402

RESULTS = HERE / "results"
VIEWS = RESULTS / "views"
OUT = RESULTS / "viz_data.json"

PERSONA_FIELDS = ("pid", "archetype", "name", "capital", "time_horizon_days",
                  "info_tier", "influence_in", "influence_out", "contrarian", "temperament")


def _read_jsonl(name):
    p = RESULTS / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _price_window(ticker, date, n=60):
    """Recent closes up to `date` (live/cached prices, no LLM/news/fundamentals)."""
    try:
        df = data_live._live_prices(ticker)
        d = pd.to_datetime(date).date()
        upto = df[df.index <= d].tail(n)
        return [{"date": str(i), "close": round(float(c), 2)} for i, c in upto["close"].items()]
    except Exception:
        return []


def _views(date, ticker):
    f = VIEWS / f"{date}_{ticker}.json"
    if not f.exists():
        return None
    try:
        return json.load(open(f))
    except Exception:
        return None


def build(date_stamp: str | None) -> dict:
    pop = json.load(open(RESULTS / "population.json"))
    personas = [{k: p.get(k) for k in PERSONA_FIELDS} for p in pop["personas"]]
    by_pid = {p["pid"]: p for p in pop["personas"]}

    scored = _read_jsonl("scored.jsonl")
    pending = _read_jsonl("pending.jsonl")
    scored_idx = {(r["as_of_date"], r["ticker"]): r for r in scored}

    records = scored + [r for r in pending
                        if (r["as_of_date"], r["ticker"]) not in scored_idx]

    cells, dates, tickers = {}, set(), set()
    for r in records:
        d, t = r["as_of_date"], r["ticker"]
        dates.add(d); tickers.add(t)
        sc = scored_idx.get((d, t))
        vws = _views(d, t)
        # recompute the call-auction clearing (with supply/demand curve) for the
        # explorer; stored record only keeps the summary (no curve).
        mkt = None
        if vws:
            mkt = market.clearing(vws, by_pid, r["t0_close"], r["forecast"].get("daily_vol_pct"))
        cells.setdefault(d, {})[t] = {
            "t0_close": r["t0_close"],
            "forecast": r["forecast"],
            "baseline": r.get("baseline"),
            "market": mkt,
            "price_window": _price_window(t, d),
            "views": vws,
            "scored": None if not sc else {
                "target_date": sc.get("target_date"),
                "actual_close": sc.get("actual_close"),
                "actual_return_pct": sc.get("actual_return_pct"),
                "direction_correct": sc.get("direction_correct"),
                "in_ci": sc.get("in_ci"),
                "baseline_dir_correct": sc.get("baseline_dir_correct"),
                "market_dir_correct": sc.get("market_dir_correct"),
            },
        }

    return {
        "generated": date_stamp,
        "population": {"n": pop["n"], "personas": personas, "edges": pop["edges"]},
        "dates": sorted(dates),
        "tickers": sorted(tickers),
        "cells": cells,
        "scoreboard": sb.build(scored) if scored else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-stamp", default=None, help="label only (e.g. today's date)")
    args = ap.parse_args()
    data = build(args.date_stamp)
    blob = json.dumps(data, default=str, separators=(",", ":"))
    OUT.write_text(blob)
    # viz_data.js: explore.html loads this via <script src> so it works on
    # file:// too (a fetch() of local JSON is CORS-blocked in Chrome).
    (RESULTS / "viz_data.js").write_text("window.VIZ_DATA=" + blob + ";")
    nv = sum(1 for d in data["cells"].values() for c in d.values() if c["views"])
    print(f"✓ {OUT}  ({OUT.stat().st_size//1024} KB)")
    print(f"  dates={data['dates']} tickers={data['tickers']} "
          f"cells={sum(len(v) for v in data['cells'].values())} with-views={nv}")


if __name__ == "__main__":
    main()
