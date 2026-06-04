"""The one daily command: SCORE yesterday, then PREDICT today.

    ANALYST_LLM=claude python -m forward_v01.daily [--agents N] [--tickers ...]

Phase 1 SCORE — for every pending prediction whose h-th trading day after its
as-of date now has a LIVE close, compute the outcome and move it to scored.jsonl.
A prediction is NEVER scored in the same run it was made (its target bar does not
exist yet) — that structural gap is what makes this leakage-free.

Phase 2 PREDICT — for each ticker, gather today's live context, run the 300-agent
population, aggregate to a next-day forecast, and append to pending.jsonl.

Idempotent: re-running the same day re-uses the LLM cache and only refills agents
that previously failed/abstained, replacing the day's pending entry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from forward_v01 import aggregate, baseline, data_live, population
from forward_v01.agent_think import think_population, MODEL

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
PENDING = RESULTS / "pending.jsonl"
SCORED = RESULTS / "scored.jsonl"

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
DEADBAND = 0.10  # % move below which we call it "flat" for direction scoring
MIN_VOTE_FRACTION = 0.40  # skip a ticker's prediction if too few agents got through
                          # (quota exhaustion) — a near-empty vote is not a forecast


# ── jsonl helpers ─────────────────────────────────────────────────────────────
def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def _append(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _dir(x: float) -> int:
    return 1 if x > DEADBAND else -1 if x < -DEADBAND else 0


# ── Phase 1: score matured predictions ───────────────────────────────────────
def score_pending() -> tuple[int, int]:
    pending = _read(PENDING)
    if not pending:
        print("  (no pending predictions to score)")
        return 0, 0
    still_pending, scored_now = [], 0
    for rec in pending:
        tkr, as_of, h = rec["ticker"], rec["as_of_date"], rec["horizon_days"]
        target_date, actual_close = data_live.nth_trading_close(tkr, as_of, h)
        if actual_close is None:
            still_pending.append(rec)         # target bar not out yet → keep
            continue
        t0 = rec["t0_close"]
        actual_pct = (actual_close / t0 - 1) * 100
        fc = rec["forecast"]
        exp = fc["expected_return_pct"]
        rec.update({
            "target_date": target_date,
            "actual_close": actual_close,
            "actual_return_pct": round(actual_pct, 3),
            "pred_dir": _dir(exp),
            "actual_dir": _dir(actual_pct),
            "direction_correct": _dir(exp) == _dir(actual_pct),
            "in_ci": fc["ci_low_pct"] <= actual_pct <= fc["ci_high_pct"],
            "abs_error_pct": round(abs(actual_pct - exp), 3),
        })
        bl = rec.get("baseline")
        if bl:
            rec["baseline_dir_correct"] = _dir(bl["mean_pct"]) == _dir(actual_pct)
            rec["baseline_abs_error_pct"] = round(abs(actual_pct - bl["mean_pct"]), 3)
        _append(SCORED, rec)
        scored_now += 1
        print(f"  scored {tkr} {as_of}→{target_date}: pred {exp:+.2f}% "
              f"act {actual_pct:+.2f}%  {'✓' if rec['direction_correct'] else '✗'}"
              f"  {'in-CI' if rec['in_ci'] else 'out-CI'}")
    _write(PENDING, still_pending)
    return scored_now, len(still_pending)


# ── Phase 2: predict today ────────────────────────────────────────────────────
def predict_today(date: str | None, tickers: list[str], agents: int,
                  model: str, horizon: int) -> list[dict]:
    pop = population.load_population(agents if agents else 300)
    personas = pop["personas"][:agents] if agents else pop["personas"]
    edges = pop["edges"]

    pending = _read(PENDING)
    out = []
    for tkr in tickers:
        as_of = date or data_live.latest_trading_date(tkr)
        ctx = data_live.context(tkr, as_of)
        as_of = ctx["date"]  # snap to the real bar date

        # Non-LLM time-series baseline: shown ONLY to institutional agents
        # (tier>=4, via ctx["tsfm"]) and scored as a benchmark vs the crowd.
        closes = [p["close"] for p in ctx.get("price_window", [])]
        bl = baseline.forecast(closes, horizon) if len(closes) >= 6 else None
        if bl:
            ctx["tsfm"] = bl

        print(f"\n  {tkr} @ {as_of} (close ${ctx['t0_close']}, "
              f"{len(ctx['news_headlines'])} headlines) — {len(personas)} agents"
              + (f" · baseline[{bl['model']}] {bl['mean_pct']:+.2f}%" if bl else ""))
        views = think_population(personas, ctx, model=model)
        daily_vol = (ctx["indicators"].get("ann_vol_pct") or 0) / (252 ** 0.5)
        fc = aggregate.aggregate(views, personas, edges, target_horizon=horizon,
                                 daily_vol_pct=daily_vol)

        # Quota guard: if too many agents abstained (claude window exhausted),
        # this isn't a real forecast — skip it (don't pollute pending/scoreboard).
        # Re-running later fills it from cache + fresh quota.
        if fc["n_votes"] < max(10, int(len(personas) * MIN_VOTE_FRACTION)):
            print(f"    => SKIP {tkr}: only {fc['n_votes']}/{len(personas)} votes "
                  f"({fc['n_abstain']} abstained — likely quota). Re-run later to fill.")
            continue

        rec = {
            "as_of_date": as_of, "ticker": tkr, "horizon_days": horizon,
            "t0_close": ctx["t0_close"], "model": model, "n_agents": len(personas),
            "n_headlines": len(ctx["news_headlines"]),
            "forecast": fc,
            "baseline": bl,  # non-LLM benchmark forecast for this same day
            "sample_narratives": [v["narrative"] for v in views[:5] if v.get("narrative")],
        }
        # dedupe: replace any existing pending entry for same (ticker, as_of)
        pending = [p for p in pending if not (p["ticker"] == tkr and p["as_of_date"] == as_of)]
        pending.append(rec)
        out.append(rec)
        arrow = "↑" if fc["consensus_lean"] == "long" else "↓" if fc["consensus_lean"] == "short" else "→"
        print(f"    => T+{horizon} {arrow} {fc['expected_return_pct']:+.2f}% "
              f"[{fc['ci_low_pct']:+.2f}, {fc['ci_high_pct']:+.2f}]  "
              f"L/S/N={fc['n_long']}/{fc['n_short']}/{fc['n_neutral']} "
              f"abstain={fc['n_abstain']}")
    _write(PENDING, pending)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="as-of date YYYY-MM-DD (default: latest live bar)")
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--agents", type=int, default=300)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--no-predict", action="store_true")
    args = ap.parse_args()

    provider = os.environ.get("ANALYST_LLM", "claude")
    print(f"=== forward_v01 daily === provider={provider} model={args.model} "
          f"tickers={args.tickers} agents={args.agents}")

    if not args.no_score:
        print("\n[Phase 1] SCORE matured predictions")
        n, rem = score_pending()
        print(f"  scored {n}, still pending {rem}")

    if not args.no_predict:
        print("\n[Phase 2] PREDICT today")
        predict_today(args.date, args.tickers, args.agents, args.model, args.horizon)

    print("\n✓ done.  See: forward_v01/results/  ·  scoreboard: "
          "python -m forward_v01.scoreboard")


if __name__ == "__main__":
    main()
