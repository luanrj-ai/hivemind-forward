"""Toy v0.6 simulation: 5 days, 6 agents, cascade via Influence Graph."""

import sys
from pathlib import Path

# Reuse mvp/data.py for yfinance loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mvp"))
import data as mvp_data  # type: ignore  # noqa: E402

from agents import ALL_AGENTS, CTA_FORCED, init_portfolios, StructuralAgent
from influence_graph import INFLUENCE_GRAPH, INFLUENCE_THRESHOLD, get_inbound_influencers
from info_tiers import filter_data_by_tier, make_other_posts_section
from llm_call import FourLayerState, agent_decide


ACTION_SIZE_MULT = {
    # v0.7.5: removed fixed multipliers. size_pct now means LITERAL fraction (0.0-1.0).
    # Agents decide their own sizing including YOLO 100% if they want.
    "buy_strong": 1.0, "buy_lite": 1.0, "hold": 0.0,
    "sell_lite": 1.0, "sell_strong": 1.0,
}


# v0.7.6: per-role short-sale leverage. Multiple of `initial capital` available as
# notional short capacity. 0 = cannot short (sell on zero shares is a no-op).
SHORT_LEVERAGE = {
    "activist_short": 2.0,   # core strategy is shorting
    "pod_pm": 1.0,           # pair trades / hedges
    "cta_forced": 1.0,       # trend follower must go short on downtrend
    "day_trader": 0.5,       # short via puts equivalent
    "super_influencer": 0.0, # talks long book up; doesn't short publicly
    "retail_fomo": 0.0,
    "permabull": 0.0,
    "sell_side": 0.0,
    "economist_macro": 0.0,
    "economist_political": 0.0,
    "economist_trader": 0.0,
}


def _max_short_shares(agent: "StructuralAgent", fill_price: float) -> int:
    """Headroom in shares for new short exposure given current position + leverage cap."""
    lev = SHORT_LEVERAGE.get(agent.role, 0.0)
    if lev <= 0 or fill_price <= 0 or agent.capital <= 0:
        return 0
    max_short_notional = agent.capital * lev
    current_short_notional = -agent.shares * fill_price if agent.shares < 0 else 0
    headroom = max(0.0, max_short_notional - current_short_notional)
    return int(headroom / fill_price)


def cta_deterministic_action(price_history) -> dict:
    """CTA mechanical rule: 20-day SMA trend follower."""
    if len(price_history) < 20:
        return {"action_type": "hold", "size_pct": 0.0, "rationale_internal": "insufficient SMA window"}
    sma_20 = price_history['close'].tail(20).mean()
    last_close = price_history.iloc[-1]['close']
    if last_close > sma_20 * 1.01:
        return {"action_type": "buy_lite", "size_pct": 0.30, "rationale_internal": f"close ${last_close:.2f} > 20SMA ${sma_20:.2f}, trend up"}
    elif last_close < sma_20 * 0.99:
        return {"action_type": "sell_lite", "size_pct": 0.30, "rationale_internal": f"close ${last_close:.2f} < 20SMA ${sma_20:.2f}, trend down"}
    else:
        return {"action_type": "hold", "size_pct": 0.0, "rationale_internal": "in 1% band of SMA, no signal"}


def execute_action(agent: StructuralAgent, action: dict, fill_price: float) -> dict:
    """Apply trade to agent's portfolio. Returns trade record.

    v0.7.5: size_pct is LITERAL fraction of cash (buy) or position (sell).
    v0.7.6: shorting allowed per-role via SHORT_LEVERAGE.
      - buy_*: covers short first, then goes long with remaining size
      - sell_*: closes long first, then opens/grows short up to leverage cap

    Returns trade dict with:
      side: "buy" (long-add or cover) | "sell" (long-trim or short-open) | "hold"
      detail: "buy_long" | "cover" | "flip_to_long" | "sell_long" | "short" | "flip_to_short" | "hold"
      shares: absolute shares traded (always >= 0)
      value: notional USD
      shares_before / shares_after: signed position
    """
    a_type = action.get("action_type", "hold") or "hold"
    size = action.get("size_pct", 0.0) or 0.0
    size = max(0.0, min(1.0, size))
    shares_before = agent.shares

    if a_type == "hold" or size <= 0 or fill_price <= 0:
        return {"side": "hold", "detail": "hold", "shares": 0, "value": 0,
                "shares_before": shares_before, "shares_after": shares_before}

    # ---- BUY: spend a cash budget to move long-ward (cover shorts first, then go long).
    # buy_lite/strong with size_pct=X spends X% of cash budget. buy_strong with size=1.0
    # additionally tries to fully cover any remaining short past the budget.
    if a_type.startswith("buy"):
        cash_budget = agent.cash * size
        cover_shares = 0
        cover_value = 0.0
        long_shares = 0
        long_value = 0.0

        # Step 1: cover existing short, capped by cash_budget (or full cover for buy_strong)
        if agent.shares < 0:
            full_cover_cost = (-agent.shares) * fill_price
            target_cover_value = full_cover_cost if a_type == "buy_strong" else min(cash_budget, full_cover_cost)
            cover_shares = int(target_cover_value / fill_price)
            cover_shares = min(cover_shares, -agent.shares, int(agent.cash / fill_price))
            cover_value = cover_shares * fill_price
            agent.cash -= cover_value
            agent.shares += cover_shares

        # Step 2: with remaining cash budget, go long
        remaining_budget = max(0.0, cash_budget - cover_value)
        # buy_strong with size=1.0 → deploy ALL remaining cash beyond the cover
        if a_type == "buy_strong" and size >= 1.0:
            remaining_budget = agent.cash
        long_shares = int(remaining_budget / fill_price) if fill_price > 0 else 0
        long_shares = min(long_shares, int(agent.cash / fill_price))
        if long_shares > 0:
            long_value = long_shares * fill_price
            agent.cash -= long_value
            agent.shares += long_shares

        total_shares = cover_shares + long_shares
        total_value = cover_value + long_value
        if total_shares == 0:
            return {"side": "hold", "detail": "buy_noop", "shares": 0, "value": 0,
                    "shares_before": shares_before, "shares_after": agent.shares}
        if cover_shares > 0 and long_shares > 0:
            detail = "flip_to_long"
        elif cover_shares > 0:
            detail = "cover"
        else:
            detail = "buy_long"
        return {"side": "buy", "detail": detail, "shares": total_shares, "value": total_value,
                "shares_before": shares_before, "shares_after": agent.shares}

    # ---- SELL: move short-ward. If long, trim long first. If flat/short, open or grow short.
    # sell_lite/strong with size_pct=X trims X% of long (or all if strong); once flat,
    # opens X% of remaining short capacity (full capacity if strong).
    if a_type.startswith("sell"):
        trim_shares = 0
        trim_value = 0.0
        short_shares = 0
        short_value = 0.0

        # Step 1: trim long
        if agent.shares > 0:
            trim_shares = agent.shares if a_type == "sell_strong" else max(1, int(agent.shares * size))
            trim_shares = min(trim_shares, agent.shares)
            trim_value = trim_shares * fill_price
            agent.cash += trim_value
            agent.shares -= trim_shares

        # Step 2: open/grow short ONLY if no long was trimmed this call
        # (so size_pct doesn't double-act); also sell_strong always tops up the short cap
        if trim_shares == 0 or a_type == "sell_strong":
            max_new_short = _max_short_shares(agent, fill_price)
            if max_new_short > 0:
                short_shares = max_new_short if a_type == "sell_strong" else max(1, int(max_new_short * size))
                short_shares = min(short_shares, max_new_short)
                short_value = short_shares * fill_price
                agent.cash += short_value
                agent.shares -= short_shares

        total_shares = trim_shares + short_shares
        total_value = trim_value + short_value
        if total_shares == 0:
            return {"side": "hold", "detail": "sell_noop_no_capacity", "shares": 0, "value": 0,
                    "shares_before": shares_before, "shares_after": agent.shares}
        if trim_shares > 0 and short_shares > 0:
            detail = "flip_to_short"
        elif short_shares > 0:
            detail = "short"
        else:
            detail = "sell_long"
        return {"side": "sell", "detail": detail, "shares": total_shares, "value": total_value,
                "shares_before": shares_before, "shares_after": agent.shares}

    return {"side": "hold", "detail": "unknown_action", "shares": 0, "value": 0,
            "shares_before": shares_before, "shares_after": shares_before}


def run_simulation(
    ticker: str = "AAPL",
    start_date: str = "2026-04-21",
    end_date: str = "2026-04-25",
    use_llm_cache: bool = True,
    on_step=None,
) -> dict:
    """Run 5-day toy v0.6 simulation with cascade via Influence Graph."""
    import pandas as pd
    # Load price data
    df = mvp_data.fetch_history(ticker, "2026-01-01", "2026-05-31")
    df = df.sort_index()

    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date).date()
    trading_days = [d for d in df.index if start_d <= d <= end_d]

    if not trading_days:
        raise RuntimeError(f"No trading days in {start_date}..{end_date}")

    # Init portfolios
    init_portfolios(ALL_AGENTS)

    # Output structure
    all_days = []
    total_cost = 0.0
    total_steps = len(trading_days) * len(ALL_AGENTS)  # super first, then 4 thinkers, then CTA = 6
    step_idx = 0

    for day in trading_days:
        day_str = str(day)
        real_open = float(df.loc[day, "open"])
        real_close = float(df.loc[day, "close"])
        price_hist = mvp_data.time_window_view(df, day, lookback_days=30)

        day_record = {
            "date": day_str,
            "real_open": real_open,
            "real_close": real_close,
            "agent_outputs": {},  # agent_id → 4-layer dict
            "trades": {},
        }

        # ===== Phase 1: Super-Influencer posts first (Tier 5) =====
        super_agent = ALL_AGENTS[0]
        super_info = filter_data_by_tier(price_hist, super_agent.info_tier, ticker)
        super_inf_posts = "(no prior posts today; you go first)"
        super_portfolio = f"Cash: ${super_agent.cash:,.0f} | Shares: {super_agent.shares} | Total: ${super_agent.total_value(real_open):,.0f}"
        super_state, super_cost = agent_decide(
            agent_id=super_agent.id,
            system_prompt=super_agent.system_prompt,
            info_view=super_info,
            influencer_posts=super_inf_posts,
            portfolio_state=super_portfolio,
            day=day_str,
            use_cache=use_llm_cache,
        )
        total_cost += super_cost
        step_idx += 1
        if on_step:
            on_step(step_idx, total_steps, f"{day_str} {super_agent.name}")

        # Execute super's trade
        super_trade = execute_action(super_agent, super_state.personal_action, real_open)
        day_record["agent_outputs"][super_agent.id] = {
            "name": super_agent.name,
            "role": super_agent.role,
            "state": super_state.to_dict(),
        }
        day_record["trades"][super_agent.id] = super_trade

        # Make super's public post available to others
        public_posts_so_far = [{
            "agent_id": super_agent.id,
            "agent_name": super_agent.name,
            "public_statement": super_state.public_statement,
        }]

        # ===== Phase 2: Other thinkers (4) read super + own tier info, post =====
        for idx in [1, 2, 3, 5]:  # skip CTA (4) which is deterministic
            agent = ALL_AGENTS[idx]
            # What influencers does this agent see?
            inbound = get_inbound_influencers(idx)
            influence_weights = {ALL_AGENTS[i].id: w for i, w in inbound}
            # Filter public_posts_so_far by which ones are above threshold for this agent
            posts_received = [p for p in public_posts_so_far if p["agent_id"] in influence_weights]
            inf_text = make_other_posts_section(posts_received, influence_weights)

            info_view = filter_data_by_tier(price_hist, agent.info_tier, ticker)
            portfolio = f"Cash: ${agent.cash:,.0f} | Shares: {agent.shares} | Total: ${agent.total_value(real_open):,.0f}"

            state, cost = agent_decide(
                agent_id=agent.id,
                system_prompt=agent.system_prompt,
                info_view=info_view,
                influencer_posts=inf_text,
                portfolio_state=portfolio,
                day=day_str,
                use_cache=use_llm_cache,
            )
            total_cost += cost
            step_idx += 1
            if on_step:
                on_step(step_idx, total_steps, f"{day_str} {agent.name}")

            trade = execute_action(agent, state.personal_action, real_open)
            day_record["agent_outputs"][agent.id] = {
                "name": agent.name,
                "role": agent.role,
                "state": state.to_dict(),
            }
            day_record["trades"][agent.id] = trade

            # Add this agent's public statement to the pool for any LATER agent (in current toy: simultaneous, no extra sub-rounds)
            public_posts_so_far.append({
                "agent_id": agent.id,
                "agent_name": agent.name,
                "public_statement": state.public_statement,
            })

        # ===== Phase 3: CTA mechanical action =====
        cta_action = cta_deterministic_action(price_hist)
        cta_trade = execute_action(CTA_FORCED, cta_action, real_open)
        day_record["agent_outputs"][CTA_FORCED.id] = {
            "name": CTA_FORCED.name,
            "role": "cta_forced",
            "state": {
                "private_belief": {"lean": "deterministic", "conviction": 0.0, "actual_thesis": cta_action["rationale_internal"]},
                "public_statement": {"stated_lean": "n/a", "stated_conviction": 0.0, "narrative": "[CTA does not speak]"},
                "desired_market_reaction": "n/a",
                "personal_action": cta_action,
            },
        }
        day_record["trades"][CTA_FORCED.id] = cta_trade
        step_idx += 1
        if on_step:
            on_step(step_idx, total_steps, f"{day_str} CTA mechanical")

        # End-of-day snapshots
        day_record["portfolios_close"] = {
            a.id: {
                "cash": a.cash,
                "shares": a.shares,
                "total": a.total_value(real_close),
                "pnl_pct": (a.total_value(real_close) - a.capital) / a.capital * 100 if a.capital > 0 else 0,
            } for a in ALL_AGENTS
        }

        all_days.append(day_record)

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "total_cost_usd": total_cost,
        "days": all_days,
        "final_portfolios": {
            a.id: {
                "name": a.name,
                "role": a.role,
                "initial_capital": a.capital,
                "final_total": a.total_value(float(df.loc[trading_days[-1], "close"])),
                "final_pnl_pct": (a.total_value(float(df.loc[trading_days[-1], "close"])) - a.capital) / a.capital * 100 if a.capital > 0 else 0,
                "final_shares": a.shares,
                "final_cash": a.cash,
            } for a in ALL_AGENTS
        },
    }
