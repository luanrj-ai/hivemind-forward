"""v0.7 1-hour demo: 6 agents × 30 days × memory + reputation + β-mode virtual price."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd

# import mvp data loader (append so toy_v06 stays first)
sys.path.append(str(Path(__file__).resolve().parent.parent / "mvp"))
import data as mvp_data  # type: ignore  # noqa: E402

from agents import ALL_AGENTS, CTA_FORCED, init_portfolios
from influence_graph import INFLUENCE_GRAPH, INFLUENCE_THRESHOLD, get_inbound_influencers
from info_tiers import filter_data_by_tier, make_other_posts_section
from llm_call import agent_decide
from memory import FactStore, Fact, auto_invalidate_predictions
from news_fetcher import fetch_window, format_news_for_tier
from price_engine_v07 import compute_beta_virtual_price, compute_net_pressure
from reputation import ReputationTracker, PostRecord
from simulation import cta_deterministic_action, execute_action


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def format_price_history(df: pd.DataFrame) -> str:
    buf = StringIO()
    df_out = df[["open", "high", "low", "close", "volume"]].copy()
    df_out.index.name = "date"
    df_out.to_csv(buf, float_format="%.2f")
    return buf.getvalue()


def build_memory_context(store: FactStore, ticker: str, as_of: str, max_facts: int = 10) -> str:
    """Pull the most relevant recent facts for prompting."""
    recent = store.recall(ticker=ticker, max_results=max_facts, as_of_date=as_of, only_active=False)
    # Show both active AND recently-invalidated (the invalidations teach the agent who to trust less)
    if not recent:
        return "(no relevant memory yet)"
    return store.format_for_prompt(recent)


def main(
    ticker: str = "AAPL",
    start_date: str = "2026-03-30",
    end_date: str = "2026-05-13",
    sensitivity: float = 0.3,
    max_concurrency: int = 10,
    reputation_window: int = 30,
):
    start_time = time.time()

    print("=" * 70, flush=True)
    print("TOY v0.7 1-HOUR DEMO — Memory + Reputation + β-mode virtual price", flush=True)
    print("=" * 70, flush=True)
    print(f"Ticker: {ticker} | Window: {start_date} → {end_date}", flush=True)
    print(f"Agents: 6 (5 LLM thinker + 1 CTA mechanical)", flush=True)
    print(f"Sensitivity (β): {sensitivity}", flush=True)
    print(f"Reputation window: {reputation_window} days", flush=True)
    print(f"Memory: in-memory FactStore (per-agent + shared)", flush=True)
    print("", flush=True)

    # === Setup ===
    df = mvp_data.fetch_history(ticker, "2026-01-01", "2026-05-31")
    df = df.sort_index()

    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date).date()
    trading_days = [d for d in df.index if start_d <= d <= end_d]
    print(f"Trading days in window: {len(trading_days)}", flush=True)
    if len(trading_days) == 0:
        print("ERROR: no trading days in window", flush=True)
        return

    # === Pre-fetch news for all days ===
    print(f"Pre-fetching news from GDELT (will cache to cache/news/)...", flush=True)
    news_window = fetch_window(ticker, [str(d) for d in trading_days])
    n_with_news = sum(1 for v in news_window.values() if v)
    print(f"  → {n_with_news}/{len(trading_days)} days have news articles", flush=True)

    init_portfolios(ALL_AGENTS)

    # Memory stores
    private_stores: dict[str, FactStore] = {a.id: FactStore(a.id) for a in ALL_AGENTS}
    shared_store = FactStore("SHARED_MARKET")

    # Reputation
    rep_tracker = ReputationTracker(window_days=reputation_window)
    influence_multipliers: dict[str, float] = {a.id: 1.0 for a in ALL_AGENTS}

    # v0.7.5: per-agent stops/targets (single-ticker AAPL toy)
    agent_cost_basis: dict[str, float] = {a.id: 0.0 for a in ALL_AGENTS}
    agent_stops: dict[str, float | None] = {a.id: None for a in ALL_AGENTS}
    agent_targets: dict[str, float | None] = {a.id: None for a in ALL_AGENTS}

    # Price data for invalidation lookups
    price_data: dict[str, dict[str, float]] = {ticker: {}}
    for d in df.index:
        if pd.notna(df.loc[d, "close"]):
            price_data[ticker][str(d)] = float(df.loc[d, "close"])

    # Output
    days_records = []
    total_cost = 0.0
    prev_real_close = float(df.loc[trading_days[0], "open"])
    virtual_price = prev_real_close  # init at day 1 open

    for d_idx, d in enumerate(trading_days):
        day_str = str(d)
        real_open = float(df.loc[d, "open"])
        real_close = float(df.loc[d, "close"])
        price_hist = mvp_data.time_window_view(df, d, lookback_days=30)
        price_csv = format_price_history(price_hist)

        # v0.7.5: Check stops/targets BEFORE today's decisions → force-close
        # NOTE: stops/targets only apply to long positions (shares > 0). Shorts are not
        # auto-managed by this loop — agents must explicitly cover them.
        forced_closes_today = []
        for a in ALL_AGENTS:
            if not a.has_portfolio or a.shares <= 0:
                continue
            stop = agent_stops.get(a.id)
            target = agent_targets.get(a.id)
            if stop is not None and real_open <= stop:
                # force-close at open
                proceeds = a.shares * real_open
                a.cash += proceeds
                forced_closes_today.append({"agent": a.id, "name": a.name, "reason": "stop_loss",
                                            "shares": a.shares, "price": real_open, "trigger": stop,
                                            "cost_basis": agent_cost_basis.get(a.id, 0)})
                a.shares = 0
                agent_stops[a.id] = None
                agent_targets[a.id] = None
                agent_cost_basis[a.id] = 0.0
            elif target is not None and real_open >= target:
                proceeds = a.shares * real_open
                a.cash += proceeds
                forced_closes_today.append({"agent": a.id, "name": a.name, "reason": "take_profit",
                                            "shares": a.shares, "price": real_open, "trigger": target,
                                            "cost_basis": agent_cost_basis.get(a.id, 0)})
                a.shares = 0
                agent_stops[a.id] = None
                agent_targets[a.id] = None
                agent_cost_basis[a.id] = 0.0

        # Auto-invalidate predictions whose target_date passed and were wrong
        # First compute past 3-day price changes per fact in each store
        invalidation_log = []
        for a in ALL_AGENTS:
            store = private_stores[a.id]
            # For each prediction in store, if target_date has passed compute actual change
            for fact in list(store.facts):
                if fact.fact_type != "prediction" or fact.invalid_at:
                    continue
                if not fact.target_date:
                    continue
                if fact.target_date > day_str:
                    continue
                # find close at valid_at vs at target_date
                close_at_pred = price_data[ticker].get(fact.valid_at)
                close_at_target = None
                for d_key in sorted(price_data[ticker].keys()):
                    if d_key >= fact.target_date:
                        close_at_target = price_data[ticker][d_key]
                        break
                if close_at_pred is None or close_at_target is None:
                    continue
                change = (close_at_target - close_at_pred) / close_at_pred
                correct = False
                if fact.stated_lean == "long" and change > 0.005:
                    correct = True
                elif fact.stated_lean == "short" and change < -0.005:
                    correct = True
                elif fact.stated_lean == "neutral" and abs(change) < 0.01:
                    correct = True
                if not correct:
                    store.invalidate(fact.id, when=day_str)
                    invalidation_log.append({
                        "owner": a.id,
                        "fact_id": fact.id,
                        "source": fact.source,
                        "content": fact.content[:80],
                        "stated_lean": fact.stated_lean,
                        "actual_change_pct": change * 100,
                    })

        # === Phase 1: 2-round cascade ===
        today_news = news_window.get(day_str, [])
        public_posts_so_far = []
        thinker_results = {}

        def _portfolio_state_str(a):
            if a.has_portfolio:
                return f"Cash: ${a.cash:,.0f} | Shares: {a.shares} | Total: ${a.total_value(real_open):,.0f}"
            return "(no portfolio — you are a commentator)"

        def _decide_one(idx_local, prior_posts):
            a = ALL_AGENTS[idx_local]
            inbound = get_inbound_influencers(idx_local)
            inf_weights = {ALL_AGENTS[i].id: w * influence_multipliers.get(ALL_AGENTS[i].id, 1.0) for i, w in inbound}
            posts_received = [p for p in prior_posts if p["agent_id"] in inf_weights]
            inf_text = make_other_posts_section(posts_received, inf_weights) if posts_received else "(no prior posts you read this round)"
            news_text = format_news_for_tier(today_news, a.info_tier)
            info_view = filter_data_by_tier(price_hist, a.info_tier, ticker, news_text=news_text)
            mem_ctx = build_memory_context(private_stores[a.id], ticker, day_str)
            return idx_local, agent_decide(
                agent_id=a.id,
                system_prompt=a.system_prompt,
                info_view=info_view,
                influencer_posts=inf_text,
                portfolio_state=_portfolio_state_str(a),
                day=day_str,
                memory_context=mem_ctx,
            )

        # Round 1: Super (0) + 3 Economists (8, 9, 10) — senior voices set tone
        ROUND_1_IDX = [0, 8, 9, 10]
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = [pool.submit(_decide_one, i, []) for i in ROUND_1_IDX]
            for fut in as_completed(futures):
                i_local, (state, cost) = fut.result()
                a = ALL_AGENTS[i_local]
                thinker_results[a.id] = state
                public_posts_so_far.append({
                    "agent_id": a.id,
                    "agent_name": a.name,
                    "public_statement": state.public_statement,
                })
                total_cost += cost

        # Round 2: 6 other thinkers read Round 1 posts
        ROUND_2_IDX = [1, 2, 3, 5, 6, 7]  # Pod, Activist, Sell-Side, Retail, Permabull, Day Trader
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = [pool.submit(_decide_one, i, public_posts_so_far) for i in ROUND_2_IDX]
            for fut in as_completed(futures):
                i_local, (state, cost) = fut.result()
                a = ALL_AGENTS[i_local]
                thinker_results[a.id] = state
                public_posts_so_far.append({
                    "agent_id": a.id,
                    "agent_name": a.name,
                    "public_statement": state.public_statement,
                })
                total_cost += cost

        # === Phase 3: CTA mechanical ===
        cta_action = cta_deterministic_action(price_hist)

        # === Phase 4: Execute trades with cascading linear price impact ===
        # v0.7.6: running fill price moves intraday based on cumulative order flow.
        # Each agent fills at the *current* running price; buyers lift it, sellers depress.
        # impact_coef × (order_notional / daily_dollar_volume) shifts the next fill in
        # the direction of the trade. Big-money agents no longer fill $2B at the open
        # for free.
        buy_usd = 0.0
        sell_usd = 0.0
        total_capital = sum(a.total_value(real_open) for a in ALL_AGENTS if a.has_portfolio)
        trades = {}
        daily_share_vol = float(df.loc[d, "volume"])
        daily_dollar_vol = max(daily_share_vol * real_open, 1.0)  # guard div-by-zero
        IMPACT_COEF = 0.08
        running_fill = real_open
        for a in ALL_AGENTS:
            if not a.has_portfolio:
                trades[a.id] = {"side": "no_portfolio", "shares": 0, "value": 0}
                continue
            shares_before = a.shares
            if a.role == "cta_forced":
                action = cta_action
            else:
                state = thinker_results[a.id]
                action = state.personal_action
            trade = execute_action(a, action, running_fill)
            trades[a.id] = trade
            # Apply price impact for next agent in queue
            if trade.get("value", 0) > 0:
                direction = +1 if trade["side"] == "buy" else (-1 if trade["side"] == "sell" else 0)
                if direction != 0:
                    running_fill = running_fill * (1.0 + IMPACT_COEF * direction * trade["value"] / daily_dollar_vol)
                    running_fill = max(0.01, running_fill)  # safety floor

            # v0.7.6: cost basis + stops support shorting.
            # buy → cover/long; sell → trim-long/open-short. Stops only set when ending net-long.
            if trade.get("side") == "buy":
                buy_usd += trade["value"]
            elif trade.get("side") == "sell":
                sell_usd += trade["value"]

            if trade.get("side") in ("buy", "sell") and trade.get("shares", 0) > 0:
                shares_after = a.shares
                old_basis = agent_cost_basis.get(a.id, 0.0) or 0.0
                # signed-position cost basis: long avg if shares > 0, short avg if < 0
                if shares_after == 0:
                    new_basis = 0.0
                elif (shares_before > 0) != (shares_after > 0) and shares_before * shares_after <= 0:
                    # crossed zero (cover-to-long, sell-to-short, or full flip) → fresh basis
                    new_basis = real_open
                elif abs(shares_after) > abs(shares_before):
                    added = abs(shares_after) - abs(shares_before)
                    new_basis = (abs(shares_before) * old_basis + added * real_open) / abs(shares_after)
                else:
                    new_basis = old_basis  # reducing same-side position; keep basis
                agent_cost_basis[a.id] = new_basis

                # Stops/targets: only apply to long positions for now (shorts not auto-managed)
                if shares_after > 0 and trade.get("detail") in ("buy_long", "flip_to_long"):
                    stop_pct = action.get("stop_loss_pct")
                    target_pct = action.get("take_profit_pct")
                    if stop_pct is not None and stop_pct < 0:
                        agent_stops[a.id] = new_basis * (1 + stop_pct)
                    if target_pct is not None and target_pct > 0:
                        agent_targets[a.id] = new_basis * (1 + target_pct)
                else:
                    # flat or short → clear long-style stops/targets
                    agent_stops[a.id] = None
                    agent_targets[a.id] = None
                    if shares_after == 0:
                        agent_cost_basis[a.id] = 0.0

        net_pressure = compute_net_pressure(buy_usd, sell_usd, total_capital)

        # === Phase 5: Compute β virtual price ===
        new_virtual = compute_beta_virtual_price(
            prev_virtual=virtual_price,
            prev_real=prev_real_close,
            cur_real=real_close,
            net_pressure=net_pressure,
            sensitivity=sensitivity,
        )

        # === Phase 6: Update memory ===
        # Each thinker's prediction becomes a fact in their own store + shared (public)
        # target_date = day_str + agent.time_horizon_days
        target_date_per_agent = {}
        for aid, state in thinker_results.items():
            agent = next(a for a in ALL_AGENTS if a.id == aid)
            horizon = agent.time_horizon_days
            # find trading day ~ horizon days later
            target_d_actual = None
            for d_future in trading_days:
                if d_future > d and (d_future - d).days >= horizon:
                    target_d_actual = str(d_future)
                    break
            target_d_actual = target_d_actual or trading_days[-1].isoformat()
            target_date_per_agent[aid] = target_d_actual

            # Add own fact in private store
            stated_lean = state.public_statement.get("stated_lean", "neutral")
            private_stores[aid].add(
                content=f"I said {stated_lean.upper()} on {ticker}: {state.public_statement.get('narrative', '')[:100]}",
                source="self_post",
                valid_at=day_str,
                ticker=ticker,
                fact_type="prediction",
                stated_lean=stated_lean,
                target_date=target_d_actual,
            )
            # Add to other agents' stores as social observation
            for other_aid, other_store in private_stores.items():
                if other_aid == aid:
                    continue
                # only add if they would have received it via influence
                idx_self = next(i for i, ag in enumerate(ALL_AGENTS) if ag.id == aid)
                idx_other = next(i for i, ag in enumerate(ALL_AGENTS) if ag.id == other_aid)
                edge_weight = INFLUENCE_GRAPH.get((idx_self, idx_other), 0.0)
                if edge_weight >= INFLUENCE_THRESHOLD:
                    other_store.add(
                        content=f"{ALL_AGENTS[idx_self].name} said {stated_lean.upper()} on {ticker}: {state.public_statement.get('narrative', '')[:80]}",
                        source=aid,
                        valid_at=day_str,
                        ticker=ticker,
                        fact_type="prediction",
                        stated_lean=stated_lean,
                        target_date=target_d_actual,
                    )
            # Also record for reputation tracking
            rep_tracker.record_post(PostRecord(
                date=day_str,
                agent_id=aid,
                ticker=ticker,
                stated_lean=stated_lean,
                private_lean=state.private_belief.get("lean", "neutral"),
                action_type=state.personal_action.get("action_type", "hold"),
                target_date=target_d_actual,
            ))

        # Add market event (price close) to shared store
        shared_store.add(
            content=f"{ticker} closed at ${real_close:.2f} (change {(real_close-prev_real_close)/prev_real_close*100:+.2f}%)",
            source="market",
            valid_at=day_str,
            ticker=ticker,
            fact_type="price_event",
        )

        # === Phase 7: Daily reputation update (v0.7.6: was weekly, now daily) ===
        # Cold-start days where no posts have matured yet fall back to acc=0.5 / mult=1.0
        # via the default branch in ReputationTracker.update.
        rep_update_today = True
        reputation_snapshots = None
        if rep_update_today:
            reputation_snapshots = rep_tracker.update(
                today=day_str,
                price_data=price_data,
                agents=ALL_AGENTS,
            )
            # Update influence multipliers
            for aid, snap in reputation_snapshots.items():
                influence_multipliers[aid] = snap.influence_multiplier

        # === Record day ===
        agent_outputs = {}
        for aid, state in thinker_results.items():
            agent = next(a for a in ALL_AGENTS if a.id == aid)
            agent_outputs[aid] = {
                "name": agent.name,
                "role": agent.role,
                "state": {
                    "private_belief": state.private_belief,
                    "public_statement": state.public_statement,
                    "desired_market_reaction": state.desired_market_reaction,
                    "personal_action": state.personal_action,
                },
            }
        agent_outputs[CTA_FORCED.id] = {
            "name": CTA_FORCED.name,
            "role": "cta_forced",
            "state": {
                "private_belief": {"lean": "deterministic", "conviction": 0.0, "actual_thesis": cta_action["rationale_internal"]},
                "public_statement": {"stated_lean": "n/a", "stated_conviction": 0.0, "narrative": "[CTA does not speak]"},
                "desired_market_reaction": "n/a",
                "personal_action": cta_action,
            },
        }

        days_records.append({
            "date": day_str,
            "real_open": real_open,
            "real_close": real_close,
            "virtual_close": new_virtual,
            "net_pressure": net_pressure,
            "buy_volume_usd": buy_usd,
            "sell_volume_usd": sell_usd,
            "forced_closes": forced_closes_today,
            "agent_stops": dict(agent_stops),
            "agent_targets": dict(agent_targets),
            "agent_cost_basis": dict(agent_cost_basis),
            "agent_outputs": agent_outputs,
            "trades": trades,
            "invalidations_today": invalidation_log,
            "reputation_update": rep_update_today,
            "reputation_snapshots": {
                aid: {"accuracy": s.accuracy, "n_posts": s.n_posts_evaluated, "mult": s.influence_multiplier}
                for aid, s in (reputation_snapshots or {}).items()
            },
            "influence_multipliers_after": dict(influence_multipliers),
            "portfolios_close": {
                a.id: {
                    "cash": a.cash,
                    "shares": a.shares,
                    "total": a.total_value(real_close),
                    "pnl_pct": (a.total_value(real_close) - a.capital) / a.capital * 100 if a.capital > 0 else 0,
                } for a in ALL_AGENTS
            },
        })

        elapsed = time.time() - start_time
        n_inv = len(invalidation_log)
        rep_marker = "📊REP" if rep_update_today else ""
        print(f"[Day {d_idx+1:2d}/{len(trading_days)}] {day_str} | real=${real_close:.2f} virtual=${new_virtual:.2f} "
              f"({(new_virtual-real_close)/real_close*100:+.2f}%) | pressure={net_pressure:+.4f} | "
              f"invalidations={n_inv} {rep_marker} | elapsed={elapsed:.0f}s | cost=${total_cost:.3f}", flush=True)

        prev_real_close = real_close
        virtual_price = new_virtual

    # Final output
    result = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "sensitivity": sensitivity,
        "reputation_window_days": reputation_window,
        "total_cost_usd": total_cost,
        "days": days_records,
        "final_portfolios": {
            a.id: {
                "name": a.name,
                "role": a.role,
                "initial_capital": a.capital,
                "final_total": a.total_value(float(df.loc[trading_days[-1], "close"])),
                "final_pnl_pct": (a.total_value(float(df.loc[trading_days[-1], "close"])) - a.capital) / a.capital * 100 if a.capital > 0 else 0,
                "final_shares": a.shares,
                "final_cash": a.cash,
                "memory_facts_total": private_stores[a.id].count(only_active=False),
                "memory_facts_active": private_stores[a.id].count(only_active=True),
            } for a in ALL_AGENTS
        },
        "reputation_history": rep_tracker.to_dict(),
        "private_memory_snapshot": {aid: s.to_dict() for aid, s in private_stores.items()},
        "shared_memory_snapshot": shared_store.to_dict(),
    }

    out_path = RESULTS_DIR / "v07_demo_latest.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"DONE: {len(days_records)} days, {elapsed/60:.1f} min, ${total_cost:.2f}", flush=True)
    print(f"Saved: {out_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
