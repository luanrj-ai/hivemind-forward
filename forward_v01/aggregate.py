"""Aggregate ~300 real LLM views into a next-day forecast distribution.

Two stages:
  1. (optional) SOCIAL PROPAGATION — herding. Each persona's belief is nudged
     toward the signed sum of the peers it listens to, over the population's
     SIGNED influence network. This models "being swayed by the room" applied to
     opinions that are ALREADY genuine LLM outputs — it does not fabricate views.
  2. WEIGHTED AGGREGATION — each persona votes with weight =
     horizon-fit × (base + capital + influence). Capital/influence tilt the
     result without letting a few whales drown the crowd. Weighted signed-lean →
     expected return; weighted dispersion → 95% CI.

Calibration (RETURN_PER_SIGNAL_PER_DAY) inherited from predict_v01/aggregate.py.
"""
from __future__ import annotations

import math
import os

LEAN_SIGN = {"long": 1.0, "short": -1.0, "neutral": 0.0}
RETURN_PER_SIGNAL_PER_DAY = 0.008  # 0.8%/day per unit signal (from predict_v01)

SOCIAL_ROUNDS = int(os.environ.get("FORWARD_SOCIAL_ROUNDS", "1"))
SOCIAL_ALPHA = float(os.environ.get("FORWARD_SOCIAL_ALPHA", "0.7"))  # inertia
SOCIAL_BETA = float(os.environ.get("FORWARD_SOCIAL_BETA", "0.3"))    # peer pull


def horizon_fit_weight(agent_horizon: int, target_horizon: int) -> float:
    width = max(5.0, target_horizon * 0.6)
    d = agent_horizon - target_horizon
    return math.exp(-(d * d) / (2 * width * width))


def _cap_norm(capital: float, max_log: float) -> float:
    return math.log10(1 + capital) / max_log if max_log > 0 else 0.0


def persona_weight(p: dict, target_horizon: int, max_log: float) -> float:
    cap = _cap_norm(p["capital"], max_log)
    infl = p.get("influence_out", 0.0)
    base = 0.40 + 0.40 * cap + 0.20 * infl       # headcount + capital + influence
    return horizon_fit_weight(p["time_horizon_days"], target_horizon) * base


# ── social propagation (herding over real LLM beliefs) ───────────────────────
def social_propagate(views: list[dict], personas: list[dict], edges: dict,
                     rounds: int, alpha: float, beta: float) -> dict[str, float]:
    """Return pid -> signed belief in [-1,1] after `rounds` of opinion dynamics."""
    belief = {v["pid"]: LEAN_SIGN[v["lean"]] * v["conviction"] for v in views}
    idx_to_pid = [p["pid"] for p in personas]
    for _ in range(max(0, rounds)):
        nxt = {}
        for pid, b in belief.items():
            es = edges.get(pid, [])
            if es:
                peer = 0.0
                wsum = 0.0
                for src_idx, w in es:
                    if src_idx < len(idx_to_pid):
                        peer += w * belief.get(idx_to_pid[src_idx], 0.0)
                        wsum += abs(w)
                peer = peer / wsum if wsum else 0.0
            else:
                peer = 0.0
            nxt[pid] = max(-1.0, min(1.0, alpha * b + beta * peer))
        belief = nxt
    return belief


# ── aggregation ──────────────────────────────────────────────────────────────
def aggregate(views: list[dict], personas: list[dict], edges: dict,
              target_horizon: int = 1, social_rounds: int | None = None,
              daily_vol_pct: float | None = None) -> dict:
    by_pid = {p["pid"]: p for p in personas}
    active = [v for v in views if not v.get("abstained")]
    n_abstain = len(views) - len(active)

    rounds = SOCIAL_ROUNDS if social_rounds is None else social_rounds
    if rounds and edges:
        belief = social_propagate(views, personas, edges, rounds, SOCIAL_ALPHA, SOCIAL_BETA)
    else:
        belief = {v["pid"]: LEAN_SIGN[v["lean"]] * v["conviction"] for v in views}

    max_log = max((math.log10(1 + by_pid[v["pid"]]["capital"]) for v in active
                   if v["pid"] in by_pid), default=1.0) or 1.0

    rows = []          # (pid, weight, signed)
    total_w = 0.0
    for v in active:
        p = by_pid.get(v["pid"])
        if not p:
            continue
        w = persona_weight(p, target_horizon, max_log)
        rows.append((v["pid"], w, belief.get(v["pid"], 0.0)))
        total_w += w

    counts = {"long": 0, "short": 0, "neutral": 0}
    for v in active:
        counts[v["lean"]] += 1

    if total_w == 0 or not rows:
        return _empty(target_horizon, counts, n_abstain)

    signal = sum(w * s for _, w, s in rows) / total_w
    var = sum((w / total_w) * (s - signal) ** 2 for _, w, s in rows)
    dispersion = math.sqrt(var)

    exp_pct = signal * RETURN_PER_SIGNAL_PER_DAY * target_horizon * 100

    # 95% CI for the next-h-day move. The dominant source of uncertainty is the
    # stock's own realized volatility, NOT how much the agents disagree — so when
    # we know daily vol, set the band from it (scales with sqrt(time)). Opinion
    # dispersion only matters as a small fallback when vol is unavailable.
    if daily_vol_pct and daily_vol_pct > 0:
        ci_half = 1.96 * daily_vol_pct * math.sqrt(target_horizon)
    else:
        ci_half = max(1.96 * dispersion * RETURN_PER_SIGNAL_PER_DAY * target_horizon * 100,
                      target_horizon * 0.5)

    lean = "long" if signal > 0.08 else "short" if signal < -0.08 else "neutral"

    # top contributors by |weight × signed|
    top = sorted(rows, key=lambda r: -abs(r[1] * r[2]))[:8]
    contributors = [{"pid": pid, "weight": round(w / total_w, 4), "signed": round(s, 3)}
                    for pid, w, s in top]

    return {
        "horizon_days": target_horizon,
        "expected_return_pct": round(exp_pct, 3),
        "ci_low_pct": round(exp_pct - ci_half, 3),
        "ci_high_pct": round(exp_pct + ci_half, 3),
        "consensus_lean": lean,
        "consensus_strength": round(min(1.0, abs(signal)), 3),
        "dispersion": round(dispersion, 4),
        "daily_vol_pct": round(daily_vol_pct, 3) if daily_vol_pct else None,
        "signal": round(signal, 4),
        "n_votes": len(active),
        "n_long": counts["long"], "n_short": counts["short"],
        "n_neutral": counts["neutral"], "n_abstain": n_abstain,
        "social_rounds": rounds,
        "top_contributors": contributors,
    }


def _empty(h, counts, n_abstain):
    return {
        "horizon_days": h, "expected_return_pct": 0.0,
        "ci_low_pct": -h * 0.5, "ci_high_pct": h * 0.5,
        "consensus_lean": "neutral", "consensus_strength": 0.0, "dispersion": 0.0,
        "signal": 0.0, "n_votes": 0,
        "n_long": counts["long"], "n_short": counts["short"],
        "n_neutral": counts["neutral"], "n_abstain": n_abstain,
        "social_rounds": 0, "top_contributors": [],
    }
