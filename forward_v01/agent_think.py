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

import hashlib
import os
import random
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


# ── information heterogeneity: each agent sees a DIFFERENT slice ──────────────
# Gated by info_tier (1=retail … 5=institution), plus per-agent variation.
# Retail sees price + a few headlines; mid adds technicals; institutions add
# fundamentals + the recent price path + (if present) the TimesFM prior. The
# per-agent RNG uses a STABLE hash (not Python's per-process hash) so each
# agent's slice is identical across runs — otherwise prompts would change every
# run and blow the LLM cache.
def _agent_rng(pid: str) -> "random.Random":
    seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def _tech_block(ind: dict, tr: dict) -> str:
    return (f"Technicals: RSI14 {ind['rsi14']}, MACD-hist {ind['macd_hist']}, "
            f"vs SMA20 {ind['pct_vs_sma20']:+.1f}%, 5d momentum {ind['mom_5d_pct']:+.1f}%, "
            f"ann.vol {ind['ann_vol_pct']}%, {ind['pct_in_52w_range']:.0f}% of 52w range "
            f"(Bollinger [{ind['bb_lower']}, {ind['bb_upper']}]).\n"
            f"Trailing returns: 5d {tr['ret_5d_pct']}%, 20d {tr['ret_20d_pct']}%, 60d {tr['ret_60d_pct']}%.")


def _fund_block(f: dict) -> str:
    if not f:
        return ""
    p = []
    if f.get("trailing_pe"): p.append(f"P/E {f['trailing_pe']:.1f}")
    if f.get("forward_pe"): p.append(f"fwd P/E {f['forward_pe']:.1f}")
    if f.get("revenue_growth_pct") is not None: p.append(f"rev growth {f['revenue_growth_pct']}%")
    if f.get("earnings_growth_pct") is not None: p.append(f"EPS growth {f['earnings_growth_pct']}%")
    if f.get("profit_margin_pct") is not None: p.append(f"net margin {f['profit_margin_pct']}%")
    if f.get("beta"): p.append(f"beta {f['beta']:.2f}")
    return ("Fundamentals: " + ", ".join(p) + ".") if p else ""


def _path_block(pw: list) -> str:
    if not pw or len(pw) < 10:
        return ""
    last = pw[-10:]
    return "Recent closes (last 10d): " + ", ".join(f"{p['close']:.1f}" for p in last) + "."


def _tsfm_block(ts: dict) -> str:
    if not ts:
        return ""
    return (f"Quant time-series model ({ts.get('model', 'baseline')}) next-day forecast: "
            f"mean {ts['mean_pct']:+.2f}%, 80% range [{ts['q10_pct']:+.2f}%, {ts['q90_pct']:+.2f}%].")


def _market_block(ctx: dict) -> str:
    """Broad-market context — shown to ALL agents because daily stock moves are
    mostly market beta, which per-stock technicals can't see."""
    m = ctx.get("market") or {}
    parts = []
    if m.get("spy_1d") is not None: parts.append(f"S&P500(SPY) {m['spy_1d']:+.1f}% today, 5d {m.get('spy_5d')}%")
    if m.get("qqq_1d") is not None: parts.append(f"Nasdaq(QQQ) {m['qqq_1d']:+.1f}% today")
    if m.get("vix") is not None: parts.append(f"VIX {m['vix']} ({m.get('vix_chg', 0):+.1f})")
    line = ("Market backdrop: " + ", ".join(parts) + ".") if parts else ""
    ed = ctx.get("earnings_in_days")
    if ed is not None and 0 <= ed <= 4:
        line += f"  ⚠ {ctx['ticker']} reports EARNINGS in ~{ed}d (expect an outsized move)."
    return line


def _news_block(headlines: list, rng, k: int) -> str:
    if not headlines:
        return "(no fresh headlines today)"
    hs = headlines[:]
    rng.shuffle(hs)               # each agent reads a different subset/order
    pick = hs[:max(1, min(k, len(hs)))]
    return "\n".join(f"- {h}" for h in pick)


def build_user(persona: dict, ctx: dict) -> str:
    tier = int(persona.get("info_tier", 1))
    rng = _agent_rng(persona["pid"])
    # asymmetric within-tier variation: a few retail/mid agents occasionally see
    # one tier up (savvy/lucky); institutions (tier>=4) always see everything.
    eff = tier
    if tier <= 2 and rng.random() < 0.15:
        eff = 3

    lines = [f"As of {ctx['date']}, {ctx['ticker']} closed at ${ctx['t0_close']}."]
    mb = _market_block(ctx)        # market backdrop + earnings — every tier sees it
    if mb:
        lines.append(mb)
    if eff >= 3:
        lines.append(_tech_block(ctx["indicators"], ctx["trend"]))
    if eff >= 4:
        for blk in (_fund_block(ctx.get("fundamentals")),
                    _path_block(ctx.get("price_window")),
                    _tsfm_block(ctx.get("tsfm"))):
            if blk:
                lines.append(blk)
    k = 8 if eff >= 4 else 5 if eff == 3 else 3
    lines.append("Today's news:\n" + _news_block(ctx["news_headlines"], rng, k))
    lines.append('\nPredict the NEXT TRADING DAY\'s move. Respond as JSON ONLY:\n'
                 '{"lean": "long|short|neutral", "conviction": 0.0-1.0, '
                 '"narrative": "one sentence, your actual reasoning"}')
    return "\n".join(lines)


# ── single-persona reasoning ─────────────────────────────────────────────────
def think(persona: dict, ctx: dict, model: str = MODEL, use_cache: bool = True) -> dict:
    system = persona_system(persona)
    user = build_user(persona, ctx)
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
