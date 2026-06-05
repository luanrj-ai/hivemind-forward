"""#3 — price formation from order flow (single-shot call auction).

Turns each agent's view into a LIMIT ORDER, then clears one uniform-price call
auction to find the price where supply meets demand. The next-day price is no
longer an average of opinions — it EMERGES from a capital-weighted order book.

Each trading agent (capital > 0; commentators like economists only voice, they
don't trade) posts:
  - a reservation return  rho = sign(lean) * conviction * Z * daily_vol
  - a limit price         t0 * (1 + rho)
  - a size                ~ sqrt(capital) * conviction   (institutions dominate
                            price impact without absolutely crushing the crowd)
Bulls are buyers (pay up to their limit); bears are sellers (sell down to it).
A band of passive maker/index liquidity around t0 gives the book depth and damps
degenerate clears (e.g. all-buy days). The clearing price maximizes matched
volume; emergent_return = clear/t0 - 1.

Pure post-processing on stored views — no LLM calls. Tunables via env:
  FORWARD_MKT_Z (reservation aggressiveness, default 2.0)
  FORWARD_MKT_ALPHA (capital exponent, default 0.5)
  FORWARD_MKT_LIQ (passive depth as a fraction of agent notional, default 0.6)
"""
from __future__ import annotations

import math
import os

LEAN_SIGN = {"long": 1.0, "short": -1.0, "neutral": 0.0}
Z = float(os.environ.get("FORWARD_MKT_Z", "2.0"))
ALPHA = float(os.environ.get("FORWARD_MKT_ALPHA", "0.5"))
LIQ_FRAC = float(os.environ.get("FORWARD_MKT_LIQ", "0.6"))
BAND = 0.05  # passive liquidity spread ±5% around t0


def _orders(views: list[dict], by_pid: dict, t0: float, dvol: float):
    """Return (buys, sells): lists of (limit_price, size)."""
    buys, sells, agent_notional = [], [], 0.0
    for v in views:
        if v.get("abstained"):
            continue
        p = by_pid.get(v["pid"])
        if not p or (p.get("capital") or 0) <= 0:   # commentators don't trade
            continue
        sign = LEAN_SIGN.get(v["lean"], 0.0)
        if sign == 0:
            continue
        conv = float(v.get("conviction", 0.5))
        rho = sign * conv * Z * dvol / 100.0          # dvol is in %, rho is fraction
        limit = t0 * (1 + rho)
        size = (p["capital"] ** ALPHA) * conv
        agent_notional += size
        (buys if sign > 0 else sells).append((limit, size))
    return buys, sells, agent_notional


def _passive(t0: float, depth: float, n: int = 24):
    """Maker/index liquidity: resting buys below t0, sells above, total `depth`
    per side spread over the ±BAND band."""
    buys, sells, each = [], [], depth / max(1, n)
    for i in range(1, n + 1):
        k = BAND * i / n
        buys.append((t0 * (1 - k), each))    # willing to buy that low
        sells.append((t0 * (1 + k), each))   # willing to sell that high
    return buys, sells


def clearing(views: list[dict], by_pid: dict, t0: float, daily_vol_pct: float | None):
    dvol = daily_vol_pct if daily_vol_pct and daily_vol_pct > 0 else 1.0
    a_buys, a_sells, notional = _orders(views, by_pid, t0, dvol)
    p_buys, p_sells = _passive(t0, LIQ_FRAC * max(notional, 1.0))
    buys = a_buys + p_buys
    sells = a_sells + p_sells

    # candidate clearing prices = every distinct limit, scanned for max matched vol
    cand = sorted({lp for lp, _ in buys} | {lp for lp, _ in sells})
    if not cand:
        return _empty(t0)
    best_p, best_vol = t0, -1.0
    for p in cand:
        demand = sum(s for lp, s in buys if lp >= p)    # buyers willing to pay >= p
        supply = sum(s for lp, s in sells if lp <= p)   # sellers willing to take <= p
        vol = min(demand, supply)
        if vol > best_vol or (vol == best_vol and abs(p - t0) < abs(best_p - t0)):
            best_vol, best_p = vol, p

    ret = (best_p / t0 - 1) * 100
    return {
        "clear_price": round(best_p, 2),
        "return_pct": round(ret, 3),
        "matched_notional": round(best_vol, 1),
        "n_buyers": len(a_buys), "n_sellers": len(a_sells),
        "curve": _curve(buys, sells, t0),   # sampled supply/demand for the viz
    }


def _curve(buys, sells, t0, pts: int = 41):
    """Sampled (price, demand, supply) across ±BAND for the explorer chart."""
    out = []
    for i in range(pts):
        p = t0 * (1 - BAND + 2 * BAND * i / (pts - 1))
        d = sum(s for lp, s in buys if lp >= p)
        s = sum(sz for lp, sz in sells if lp <= p)
        out.append([round((p / t0 - 1) * 100, 2), round(d, 1), round(s, 1)])
    return out


def _empty(t0):
    return {"clear_price": round(t0, 2), "return_pct": 0.0, "matched_notional": 0.0,
            "n_buyers": 0, "n_sellers": 0, "curve": []}


if __name__ == "__main__":
    # smoke: 3 bulls vs 2 bears, institution bull is big
    vs = [{"pid": "a", "lean": "long", "conviction": .8},
          {"pid": "b", "lean": "long", "conviction": .5},
          {"pid": "c", "lean": "short", "conviction": .9},
          {"pid": "d", "lean": "short", "conviction": .6}]
    bp = {"a": {"capital": 5e9}, "b": {"capital": 12000}, "c": {"capital": 8e7}, "d": {"capital": 85000}}
    print(clearing(vs, bp, 100.0, 1.5))
