"""Per-persona LLM reasoning.

Each persona reads TODAY's real market context and reasons IN LANGUAGE to its own
view (lean + conviction + a one-line thesis). We reuse analyst_v01.llm.ask_json
(disk-cached by prompt hash, claude-CLI backend by default). Because every
persona's genome makes its prompt unique, cache keys never collide — and a
re-run the same day only re-issues the calls that previously failed.

`think_population` runs at LOW concurrency with quota-aware backoff: on a claude
rate-limit/usage-limit error it backs off (exp, capped) and retries; if it still
can't get through, that persona ABSTAINS (neutral, conviction 0) rather than
crashing the run. Re-running daily.py later fills the gaps from cache + retries.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analyst_v01"))
import llm  # noqa: E402  analyst_v01/llm.py — ask_json(system, user, model)

MODEL = os.environ.get("FORWARD_MODEL", "sonnet")
CONCURRENCY = int(os.environ.get("FORWARD_CONCURRENCY", "4"))
MAX_BACKOFF = float(os.environ.get("FORWARD_MAX_BACKOFF", "300"))  # cap per-call sleep
QUOTA_RETRIES = int(os.environ.get("FORWARD_QUOTA_RETRIES", "5"))

_RATE_SIGNATURES = ("rate limit", "rate_limit", "quota", "usage limit", "429",
                    "overloaded", "too many requests", "resource_exhausted")

VALID_LEAN = {"long", "short", "neutral"}


def _is_quota_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(sig in s for sig in _RATE_SIGNATURES)


# ── prompt construction (genome → a distinct individual) ─────────────────────
_ROLE_FLAVOR = {
    "super_influencer": "a high-profile fund manager whose public words move markets",
    "pod_pm": "a multi-manager pod PM on a tight risk leash (a bad week gets you cut)",
    "activist_short": "an activist short-seller who publishes adversarial theses",
    "sell_side": "a sell-side analyst defending a 12-month price target",
    "cta_forced": "a systematic CTA that trades the trend mechanically, ignoring narrative",
    "retail_fomo": "a retail trader who chases momentum and headlines with small size",
    "permabull": "a structurally bullish growth investor",
    "day_trader": "an active day-trader reading the tape and intraday flow",
    "economist_macro": "a macro economist judging the name through rates/liquidity",
    "economist_political": "a political economist weighing regulation/policy risk",
    "economist_trader": "a markets economist who trades his own macro views",
}


def persona_system(p: dict) -> str:
    cap = p["capital"]
    cap_str = (f"${cap/1e9:.1f}B" if cap >= 1e9 else
               f"${cap/1e6:.0f}M" if cap >= 1e6 else
               f"${cap/1e3:.0f}k" if cap > 0 else "no trading book")
    contr = ("strongly contrarian — you fade the crowd" if p["contrarian"] > 0.6 else
             "somewhat contrarian" if p["contrarian"] > 0.4 else
             "a trend-follower who moves with the crowd")
    return (
        f"You are {p['name']}, {_ROLE_FLAVOR.get(p['archetype'], 'a market participant')}.\n"
        f"Profile: capital {cap_str}; preferred holding horizon ~{p['time_horizon_days']} trading days; "
        f"information tier {p['info_tier']}/5; career-risk tolerance {p['career_risk']:.2f}; "
        f"susceptibility to peers {p['influence_in']:.2f}; you are {contr}; "
        f"temperament: {p['temperament']}.\n"
        "Reason from YOUR vantage point and constraints — not as a neutral analyst. "
        "Give your honest directional view on this stock over YOUR horizon."
    )


def context_user(ctx: dict) -> str:
    ind, tr = ctx["indicators"], ctx["trend"]
    news = ctx["news_headlines"]
    news_block = ("\n".join(f"- {h}" for h in news) if news
                  else "(no fresh headlines today)")
    return (
        f"As of {ctx['date']}, {ctx['ticker']} closed at ${ctx['t0_close']}.\n"
        f"Technicals: RSI14 {ind['rsi14']}, MACD-hist {ind['macd_hist']}, "
        f"vs SMA20 {ind['pct_vs_sma20']:+.1f}%, 5d momentum {ind['mom_5d_pct']:+.1f}%, "
        f"ann.vol {ind['ann_vol_pct']}%, {ind['pct_in_52w_range']:.0f}% of 52w range "
        f"(Bollinger [{ind['bb_lower']}, {ind['bb_upper']}]).\n"
        f"Trailing returns: 5d {tr['ret_5d_pct']}%, 20d {tr['ret_20d_pct']}%, 60d {tr['ret_60d_pct']}%.\n"
        f"Today's news:\n{news_block}\n\n"
        "Predict the NEXT TRADING DAY's move. Respond as JSON ONLY:\n"
        '{"lean": "long|short|neutral", "conviction": 0.0-1.0, '
        '"narrative": "one sentence, your actual reasoning"}'
    )


# ── single-persona reasoning ─────────────────────────────────────────────────
def think(persona: dict, ctx: dict, model: str = MODEL, use_cache: bool = True) -> dict:
    system = persona_system(persona)
    user = context_user(ctx)
    out, _cost = llm.ask_json(system, user, model=model, use_cache=use_cache)
    return _parse(out, persona)


def _parse(out: dict, persona: dict) -> dict:
    lean = str(out.get("lean", "neutral")).lower().strip()
    if lean not in VALID_LEAN:
        lean = "neutral"
    try:
        conv = float(out.get("conviction", 0.5))
    except (TypeError, ValueError):
        conv = 0.5
    conv = max(0.0, min(1.0, conv))
    return {
        "pid": persona["pid"],
        "archetype": persona["archetype"],
        "lean": lean,
        "conviction": round(conv, 3),
        "narrative": str(out.get("narrative", ""))[:300],
        "abstained": bool(out.get("_error")),
    }


def _abstain(persona: dict, reason: str) -> dict:
    return {"pid": persona["pid"], "archetype": persona["archetype"],
            "lean": "neutral", "conviction": 0.0, "narrative": f"[abstain: {reason}]",
            "abstained": True}


# ── population reasoning (low concurrency + quota backoff) ────────────────────
def think_population(personas: list[dict], ctx: dict, model: str = MODEL,
                     concurrency: int = CONCURRENCY, progress: bool = True) -> list[dict]:
    done = {"n": 0, "abstain": 0}
    lock = threading.Lock()
    total = len(personas)

    def worker(p):
        for attempt in range(QUOTA_RETRIES + 1):
            try:
                v = think(p, ctx, model=model)
                with lock:
                    done["n"] += 1
                    if v["abstained"]:
                        done["abstain"] += 1
                    if progress and done["n"] % 25 == 0:
                        print(f"    {ctx['ticker']}: {done['n']}/{total} "
                              f"({done['abstain']} abstain)", flush=True)
                return v
            except Exception as e:
                if _is_quota_error(e) and attempt < QUOTA_RETRIES:
                    time.sleep(min(MAX_BACKOFF, 5 * (2 ** attempt)))
                    continue
                with lock:
                    done["n"] += 1
                    done["abstain"] += 1
                return _abstain(p, "quota" if _is_quota_error(e) else str(e)[:60])
        return _abstain(p, "quota")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(worker, p) for p in personas]
        for f in as_completed(futs):
            results.append(f.result())
    if progress:
        print(f"    {ctx['ticker']}: {len(results)} views "
              f"({done['abstain']} abstained)", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    from forward_v01 import data_live, population
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    pop = population.load_population(300)
    personas = pop["personas"][:args.agents]
    ctx = data_live.context(args.ticker, data_live.latest_trading_date(args.ticker))
    print(f"Thinking: {len(personas)} personas on {args.ticker} @ {ctx['date']} "
          f"(model={args.model})")
    views = think_population(personas, ctx, model=args.model)
    for v in views:
        print(f"  {v['pid']:<24} {v['lean']:<8} {v['conviction']:.2f}  {v['narrative'][:70]}")
