"""Running scorecard over forward, leakage-free predictions.

Reads results/scored.jsonl and reports — overall and per-ticker —
  • direction hit-rate (on directional calls) with a Wilson 95% CI, and whether
    that CI still straddles 0.50 (i.e. not yet distinguishable from a coin flip),
  • CI calibration (share of actuals inside the 95% interval; target ≈ 0.95),
  • mean absolute error of the expected return.

    python -m forward_v01.scoreboard            # print
    python -m forward_v01.scoreboard --json     # also dump scoreboard.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SCORED = RESULTS / "scored.jsonl"
SNAP = RESULTS / "scoreboard.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return max(0.0, center - half), min(1.0, center + half)


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    directional = [r for r in rows if r.get("pred_dir", 0) != 0]
    nd = len(directional)
    hits = sum(1 for r in directional if r["direction_correct"])
    lo, hi = wilson(hits, nd) if nd else (0.0, 1.0)
    in_ci = sum(1 for r in rows if r.get("in_ci"))
    mae = sum(r["abs_error_pct"] for r in rows) / n if n else 0.0
    return {
        "n": n, "n_directional": nd,
        "hits": hits,
        "hit_rate": round(hits / nd, 4) if nd else None,
        "hit_ci95": [round(lo, 4), round(hi, 4)],
        "coinflip_excluded": bool(nd and (lo > 0.5 or hi < 0.5)),
        "ci_calibration": round(in_ci / n, 4) if n else None,
        "mae_pct": round(mae, 4),
    }


def _baseline_stats(rows: list[dict]) -> dict | None:
    """Direction hit-rate of the non-LLM time-series baseline, on the same days."""
    bl = [r for r in rows if "baseline_dir_correct" in r]
    if not bl:
        return None
    n = len(bl)
    hits = sum(1 for r in bl if r["baseline_dir_correct"])
    lo, hi = wilson(hits, n)
    mae = sum(r.get("baseline_abs_error_pct", 0) for r in bl) / n
    return {"n": n, "hits": hits, "hit_rate": round(hits / n, 4),
            "hit_ci95": [round(lo, 4), round(hi, 4)], "mae_pct": round(mae, 4),
            "model": next((r["baseline"]["model"] for r in bl if r.get("baseline")), "?")}


def _model_stats(rows: list[dict], flag: str, err: str, label: str) -> dict | None:
    sub = [r for r in rows if flag in r]
    if not sub:
        return None
    n = len(sub)
    hits = sum(1 for r in sub if r[flag])
    lo, hi = wilson(hits, n)
    mae = sum(r.get(err, 0) for r in sub) / n
    return {"n": n, "hits": hits, "hit_rate": round(hits / n, 4),
            "hit_ci95": [round(lo, 4), round(hi, 4)], "mae_pct": round(mae, 4), "label": label}


def build(rows: list[dict]) -> dict:
    overall = _stats(rows)
    per_ticker = {}
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["ticker"]].append(r)
    for t, rs in sorted(by_t.items()):
        per_ticker[t] = _stats(rs)
    return {
        "overall": overall, "per_ticker": per_ticker,
        "baseline": _baseline_stats(rows),
        "market": _model_stats(rows, "market_dir_correct", "market_abs_error_pct", "call-auction"),
    }


def _fmt(s: dict) -> str:
    if s["n_directional"] == 0:
        return f"n={s['n']:<4} (no directional calls yet)"
    hr = s["hit_rate"] * 100
    lo, hi = s["hit_ci95"]
    sig = "SIGNIFICANT (excludes 50%)" if s["coinflip_excluded"] else "not yet vs coin-flip"
    cal = f"{s['ci_calibration']*100:.0f}%" if s["ci_calibration"] is not None else "—"
    return (f"n={s['n']:<4} dir={s['n_directional']:<4} "
            f"hit={hr:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]  {sig:<26} "
            f"CI-cal={cal:<5} MAE={s['mae_pct']:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="also write scoreboard.json")
    args = ap.parse_args()

    if not SCORED.exists():
        print("No scored predictions yet. Run a few days of `daily`, then score.")
        return
    rows = [json.loads(ln) for ln in SCORED.read_text().splitlines() if ln.strip()]
    sb = build(rows)

    print("=== forward_v01 scoreboard (leakage-free, walk-forward) ===")
    print(f"CROWD     {_fmt(sb['overall'])}  ← weighted opinion consensus")
    mk = sb.get("market")
    if mk:
        lo, hi = mk["hit_ci95"]
        print(f"MARKET    n={mk['n']:<4} dir={mk['n']:<4} "
              f"hit={mk['hit_rate']*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]  "
              f"{'(call-auction clearing)':<26} {'':5} MAE={mk['mae_pct']:.2f}%  ← #3 emergent price")
    bl = sb.get("baseline")
    if bl:
        lo, hi = bl["hit_ci95"]
        print(f"BASELINE  n={bl['n']:<4} dir={bl['n']:<4} "
              f"hit={bl['hit_rate']*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]  "
              f"{'model='+bl['model']:<26} {'':5} MAE={bl['mae_pct']:.2f}%  ← non-LLM benchmark")
    print("per ticker:")
    for t, s in sb["per_ticker"].items():
        print(f"  {t:<6} {_fmt(s)}")
    o = sb["overall"]
    if o["n_directional"] < 30:
        print(f"\n⚠ Only {o['n_directional']} directional obs — far from significant. "
              "Forward testing needs dozens of trading days to separate edge from noise.")

    if args.json:
        SNAP.write_text(json.dumps(sb, indent=2))
        print(f"\n✓ wrote {SNAP}")


if __name__ == "__main__":
    main()
