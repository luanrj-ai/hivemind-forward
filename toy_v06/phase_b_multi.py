"""Phase B: Multi-ticker (AAPL/NVDA/MSFT) + stop-loss/take-profit + 35-day window.

Self-contained — doesn't modify v0.7 toy. Outputs to results/phase_b_latest.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "mvp"))
import data as mvp_data  # type: ignore  # noqa: E402

from agents import ALL_AGENTS, get_agent
from influence_graph import INFLUENCE_GRAPH, INFLUENCE_THRESHOLD, get_inbound_influencers
from info_tiers import filter_data_by_tier, make_other_posts_section
from memory import FactStore
from news_fetcher import format_news_for_tier
from reputation import ReputationTracker, PostRecord


# ============================================================================
TICKERS = ["AAPL", "NVDA", "MSFT"]
START_DATE = "2026-03-12"
END_DATE = "2026-05-13"
SENSITIVITY = 0.3
MAX_CONCURRENCY = 10
REP_WINDOW = 30
CACHE_LLM = Path(__file__).parent / "cache" / "llm_v08"
CACHE_LLM.mkdir(parents=True, exist_ok=True)
NEWS_CACHE_DIR = Path(__file__).parent / "cache" / "news"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MULTI-TICKER PORTFOLIO
# ============================================================================
@dataclass
class MultiPortfolio:
    agent_id: str
    initial_capital: float
    cash: float
    positions: dict[str, int] = field(default_factory=dict)     # ticker → shares
    stops: dict[str, float] = field(default_factory=dict)       # ticker → stop_loss price
    targets: dict[str, float] = field(default_factory=dict)     # ticker → take_profit price
    cost_basis: dict[str, float] = field(default_factory=dict)  # ticker → avg cost
    transactions: list = field(default_factory=list)

    def total_value(self, prices: dict[str, float]) -> float:
        v = self.cash
        for t, n in self.positions.items():
            v += n * prices.get(t, 0.0)
        return v

    def pnl_pct(self, prices: dict[str, float]) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self.total_value(prices) - self.initial_capital) / self.initial_capital * 100

    def buy(self, ticker: str, cash_to_use: float, price: float, day: str, source: str) -> int:
        if price <= 0:
            return 0
        shares = int(cash_to_use / price)
        cost = shares * price
        if shares == 0 or cost > self.cash:
            return 0
        old_shares = self.positions.get(ticker, 0)
        old_basis = self.cost_basis.get(ticker, 0)
        new_total = old_shares + shares
        new_basis = (old_shares * old_basis + cost) / new_total if new_total > 0 else 0
        self.cash -= cost
        self.positions[ticker] = new_total
        self.cost_basis[ticker] = new_basis
        self.transactions.append({"date": day, "side": "buy", "ticker": ticker,
                                  "shares": shares, "price": price, "source": source})
        return shares

    def sell(self, ticker: str, shares_to_sell: int, price: float, day: str, source: str) -> int:
        held = self.positions.get(ticker, 0)
        if held <= 0:
            return 0
        actual = min(held, shares_to_sell)
        proceeds = actual * price
        self.cash += proceeds
        self.positions[ticker] = held - actual
        if self.positions[ticker] == 0:
            self.cost_basis.pop(ticker, None)
            self.stops.pop(ticker, None)
            self.targets.pop(ticker, None)
        self.transactions.append({"date": day, "side": "sell", "ticker": ticker,
                                  "shares": actual, "price": price, "source": source})
        return actual


ACTION_MULT = {"buy_strong": 0.50, "buy_lite": 0.20, "hold": 0.0,
               "sell_lite": 0.30, "sell_strong": 1.00}


def apply_action(portfolio: MultiPortfolio, ticker: str, action: dict, price: float, day: str):
    """Apply one ticker's action. Returns trade summary."""
    a_type = action.get("action_type", "hold") or "hold"
    size_pct = action.get("size_pct", 0.0) or 0.0
    stop_pct = action.get("stop_loss_pct")  # negative number like -0.08
    target_pct = action.get("take_profit_pct")  # positive like +0.20

    if a_type == "hold" or size_pct <= 0:
        # Still update stops/targets if specified
        if stop_pct is not None and portfolio.positions.get(ticker, 0) > 0:
            basis = portfolio.cost_basis.get(ticker, price)
            portfolio.stops[ticker] = basis * (1 + stop_pct)
        if target_pct is not None and portfolio.positions.get(ticker, 0) > 0:
            basis = portfolio.cost_basis.get(ticker, price)
            portfolio.targets[ticker] = basis * (1 + target_pct)
        return {"side": "hold", "shares": 0}

    if a_type.startswith("buy"):
        mult = ACTION_MULT[a_type]
        cash_to_use = portfolio.cash * mult * size_pct
        shares = portfolio.buy(ticker, cash_to_use, price, day, source="action")
        if shares > 0 and stop_pct is not None:
            portfolio.stops[ticker] = price * (1 + stop_pct)
        if shares > 0 and target_pct is not None:
            portfolio.targets[ticker] = price * (1 + target_pct)
        return {"side": "buy", "shares": shares, "value": shares * price}

    elif a_type.startswith("sell"):
        held = portfolio.positions.get(ticker, 0)
        if held == 0:
            return {"side": "skip_no_shares", "shares": 0}
        mult = ACTION_MULT[a_type]
        shares_to_sell = held if a_type == "sell_strong" else max(1, int(held * mult * size_pct))
        actual = portfolio.sell(ticker, shares_to_sell, price, day, source="action")
        return {"side": "sell", "shares": actual, "value": actual * price}

    return {"side": "noop", "shares": 0}


def check_stops_and_targets(portfolio: MultiPortfolio, prices_open: dict[str, float], day: str) -> list:
    """Force-close positions hitting stop or target. Run BEFORE today's action decision."""
    closed = []
    for ticker in list(portfolio.positions.keys()):
        if portfolio.positions[ticker] == 0:
            continue
        price = prices_open.get(ticker)
        if price is None:
            continue
        stop = portfolio.stops.get(ticker)
        target = portfolio.targets.get(ticker)
        if stop is not None and price <= stop:
            shares = portfolio.positions[ticker]
            portfolio.sell(ticker, shares, price, day, source="STOP_LOSS")
            closed.append({"ticker": ticker, "reason": "stop_loss", "shares": shares, "price": price, "trigger": stop})
        elif target is not None and price >= target:
            shares = portfolio.positions[ticker]
            portfolio.sell(ticker, shares, price, day, source="TAKE_PROFIT")
            closed.append({"ticker": ticker, "reason": "take_profit", "shares": shares, "price": price, "trigger": target})
    return closed


# ============================================================================
# LLM CALL (multi-ticker prompt)
# ============================================================================
@dataclass
class MultiState:
    private_belief: dict
    public_statement: dict
    desired_market_reaction: str
    actions: dict  # ticker → action dict
    raw_response: str = ""

    @classmethod
    def fallback(cls, reason="parse_err"):
        return cls(
            private_belief={"leans": {t: "neutral" for t in TICKERS}, "conviction": 0.0, "actual_thesis": f"[fallback: {reason}]"},
            public_statement={"leans": {t: "neutral" for t in TICKERS}, "narrative": "[silent]"},
            desired_market_reaction=f"[fallback]",
            actions={t: {"action_type": "hold", "size_pct": 0.0} for t in TICKERS},
        )

    def to_dict(self):
        return {
            "private_belief": self.private_belief,
            "public_statement": self.public_statement,
            "desired_market_reaction": self.desired_market_reaction,
            "actions": self.actions,
        }


def _call_claude(system_prompt: str, user_msg: str) -> tuple[str, float]:
    cmd = ["claude", "-p", "--model", "haiku", "--no-session-persistence",
           "--output-format", "json", "--max-budget-usd", "0.15",
           "--system-prompt", system_prompt, user_msg]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:200]}")
    data = json.loads(result.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"api err: {data.get('result','')[:200]}")
    return data.get("result", ""), float(data.get("total_cost_usd", 0.0))


def _parse_json(raw: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            return None


def agent_decide_multi(
    agent,
    day: str,
    info_views: dict[str, str],         # ticker → info_view string
    influencer_posts: str,
    portfolio: MultiPortfolio,
    memory_context: str,
    prices_open: dict[str, float],
) -> tuple[MultiState, float]:
    """Get one agent's multi-ticker decision."""
    info_hash = hashlib.sha256(json.dumps(info_views, sort_keys=True).encode()).hexdigest()[:8]
    inf_hash = hashlib.sha256(influencer_posts.encode()).hexdigest()[:8]
    mem_hash = hashlib.sha256(memory_context.encode()).hexdigest()[:8] if memory_context else ""
    pf_state = json.dumps({"cash": portfolio.cash, "positions": portfolio.positions}, sort_keys=True)
    pf_hash = hashlib.sha256(pf_state.encode()).hexdigest()[:8]
    key = hashlib.sha256(f"v08|{agent.id}|{day}|{info_hash}|{inf_hash}|{mem_hash}|{pf_hash}".encode()).hexdigest()[:16]
    cache_file = CACHE_LLM / f"{key}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        return MultiState(**{k: v for k, v in data.items() if k != "cost"}), 0.0

    # Build prompt
    portfolio_str = f"Cash: ${portfolio.cash:,.0f}\nPositions:\n"
    for t in TICKERS:
        held = portfolio.positions.get(t, 0)
        basis = portfolio.cost_basis.get(t, 0)
        cur = prices_open.get(t, 0)
        if held > 0:
            mtm_pnl_pct = (cur - basis) / basis * 100 if basis > 0 else 0
            stop = portfolio.stops.get(t)
            target = portfolio.targets.get(t)
            stop_str = f" (stop=${stop:.2f})" if stop else ""
            target_str = f" (target=${target:.2f})" if target else ""
            portfolio_str += f"  {t}: {held} shares @ basis ${basis:.2f}, MTM ${cur:.2f} ({mtm_pnl_pct:+.1f}%){stop_str}{target_str}\n"
        else:
            portfolio_str += f"  {t}: 0 shares\n"
    portfolio_str += f"Total: ${portfolio.total_value(prices_open):,.0f} (initial ${portfolio.initial_capital:,.0f}, PnL {portfolio.pnl_pct(prices_open):+.2f}%)"

    info_block = ""
    for t in TICKERS:
        info_block += f"\n========== {t} ==========\n{info_views.get(t, 'no data')}\n"

    memory_block = f"\nYOUR MEMORY RECALL:\n{memory_context}\n" if memory_context else ""

    user_msg = f"""Day: {day}

YOUR MULTI-TICKER PORTFOLIO:
{portfolio_str}

INFLUENCER POSTS YOU RECEIVED:
{influencer_posts}
{memory_block}
INFORMATION VIEW (filtered by your tier {agent.info_tier}):
{info_block}

You manage a portfolio across {len(TICKERS)} tickers: {', '.join(TICKERS)}.

For EACH ticker, decide independently:
- action_type: buy_strong/buy_lite/hold/sell_lite/sell_strong
- size_pct: 0.0-1.0 (fraction of cash for buy, fraction of position for sell)
- stop_loss_pct: optional, negative number like -0.08 (force sell if -8% from cost)
- take_profit_pct: optional, positive like +0.20 (force sell if +20%)

Output STRICT JSON (no prose):
{{
  "private_belief": {{
    "leans": {{"AAPL": "long|neutral|short", "NVDA": "...", "MSFT": "..."}},
    "conviction": 0.0-1.0,
    "actual_thesis": "1-2 sentences honest cross-ticker view"
  }},
  "public_statement": {{
    "leans": {{"AAPL": "long|neutral|short", "NVDA": "...", "MSFT": "..."}},
    "narrative": "40-100 words public framing"
  }},
  "desired_market_reaction": "what you want to happen",
  "actions": {{
    "AAPL": {{"action_type": "...", "size_pct": ..., "stop_loss_pct": -0.X or null, "take_profit_pct": 0.X or null}},
    "NVDA": {{...}},
    "MSFT": {{...}}
  }}
}}"""

    for attempt in range(3):
        try:
            raw, cost = _call_claude(agent.system_prompt + "\n\nIMPORTANT: You are now managing MULTIPLE tickers. Use the per-ticker output schema in user message.", user_msg)
            parsed = _parse_json(raw)
            if parsed and "actions" in parsed:
                state = MultiState(
                    private_belief=parsed.get("private_belief", {"leans": {}, "actual_thesis": ""}),
                    public_statement=parsed.get("public_statement", {"leans": {}, "narrative": ""}),
                    desired_market_reaction=str(parsed.get("desired_market_reaction", ""))[:300],
                    actions=parsed.get("actions", {}),
                    raw_response=raw,
                )
                # Normalize: ensure each ticker has an action
                for t in TICKERS:
                    if t not in state.actions:
                        state.actions[t] = {"action_type": "hold", "size_pct": 0.0}
                    if state.actions[t].get("action_type") not in {"buy_strong", "buy_lite", "hold", "sell_lite", "sell_strong"}:
                        state.actions[t]["action_type"] = "hold"
                with open(cache_file, "w") as f:
                    json.dump({**state.to_dict(), "cost": cost}, f)
                return state, cost
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return MultiState.fallback(f"{e}"), 0.0
        time.sleep(1)
    return MultiState.fallback("max_retry"), 0.0


# ============================================================================
# CTA mechanical action (per ticker, simple SMA)
# ============================================================================
def cta_action_multi(price_hist_per_ticker: dict[str, pd.DataFrame]) -> dict:
    """Mechanical 20-day SMA per ticker."""
    out = {}
    for t, hist in price_hist_per_ticker.items():
        if len(hist) < 20:
            out[t] = {"action_type": "hold", "size_pct": 0.0}
            continue
        sma_20 = hist['close'].tail(20).mean()
        last = hist.iloc[-1]['close']
        if last > sma_20 * 1.01:
            out[t] = {"action_type": "buy_lite", "size_pct": 0.25, "stop_loss_pct": -0.08}
        elif last < sma_20 * 0.99:
            out[t] = {"action_type": "sell_lite", "size_pct": 0.30}
        else:
            out[t] = {"action_type": "hold", "size_pct": 0.0}
    return out


# ============================================================================
# DATA LOADERS
# ============================================================================
def format_price_history(df: pd.DataFrame) -> str:
    buf = StringIO()
    df_out = df[["open", "high", "low", "close", "volume"]].copy()
    df_out.index.name = "date"
    df_out.to_csv(buf, float_format="%.2f")
    return buf.getvalue()


def load_news_cache(ticker: str, day: str) -> list:
    cache_file = NEWS_CACHE_DIR / f"{ticker}_{day}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def build_memory_context(store: FactStore, as_of: str, max_facts: int = 15) -> str:
    recent = []
    for t in TICKERS:
        recent.extend(store.recall(ticker=t, max_results=5, as_of_date=as_of, only_active=False))
    if not recent:
        return "(no relevant memory yet)"
    return store.format_for_prompt(recent[:max_facts])


# ============================================================================
# MAIN
# ============================================================================
def main():
    start_time = time.time()

    print("=" * 75, flush=True)
    print(f"PHASE B — Multi-ticker ({', '.join(TICKERS)}) + stop-loss/take-profit", flush=True)
    print("=" * 75, flush=True)
    print(f"Window: {START_DATE} → {END_DATE}", flush=True)
    print(f"Agents: {len(ALL_AGENTS)} ({sum(1 for a in ALL_AGENTS if a.has_portfolio)} with portfolio)", flush=True)
    print(f"Concurrency: {MAX_CONCURRENCY}, β sensitivity: {SENSITIVITY}", flush=True)
    print("", flush=True)

    # Load prices per ticker
    print(f"Loading price data...", flush=True)
    prices_df = {t: mvp_data.fetch_history(t, "2026-01-01", "2026-05-31").sort_index() for t in TICKERS}

    start_d = pd.to_datetime(START_DATE).date()
    end_d = pd.to_datetime(END_DATE).date()
    trading_days = [d for d in prices_df[TICKERS[0]].index if start_d <= d <= end_d]
    print(f"  Trading days: {len(trading_days)}", flush=True)

    # Init portfolios per agent (skip no-portfolio agents)
    portfolios: dict[str, MultiPortfolio] = {}
    for a in ALL_AGENTS:
        if not a.has_portfolio:
            continue
        portfolios[a.id] = MultiPortfolio(
            agent_id=a.id,
            initial_capital=a.capital,
            cash=a.capital,
        )

    # Memory stores per agent
    private_stores = {a.id: FactStore(a.id) for a in ALL_AGENTS}

    # Reputation tracker
    rep_tracker = ReputationTracker(window_days=REP_WINDOW)
    influence_mult = {a.id: 1.0 for a in ALL_AGENTS}

    # Build price_data dict for reputation
    price_data_for_rep = {t: {} for t in TICKERS}
    for t in TICKERS:
        for d in prices_df[t].index:
            if pd.notna(prices_df[t].loc[d, "close"]):
                price_data_for_rep[t][str(d)] = float(prices_df[t].loc[d, "close"])

    days_records = []
    total_cost = 0.0

    for d_idx, d in enumerate(trading_days):
        day_str = str(d)
        prices_open = {t: float(prices_df[t].loc[d, "open"]) for t in TICKERS}
        prices_close = {t: float(prices_df[t].loc[d, "close"]) for t in TICKERS}

        # 1. Check stops/targets BEFORE today's decisions
        forced_closes = {}
        for aid, pf in portfolios.items():
            closed = check_stops_and_targets(pf, prices_open, day_str)
            if closed:
                forced_closes[aid] = closed

        # 2. Invalidate stale predictions
        invalidations_today = []
        for aid, store in private_stores.items():
            for fact in list(store.facts):
                if fact.fact_type != "prediction" or fact.invalid_at or not fact.target_date or not fact.ticker:
                    continue
                if fact.target_date > day_str:
                    continue
                px_at_pred = price_data_for_rep.get(fact.ticker, {}).get(fact.valid_at)
                px_target = None
                for d_key in sorted(price_data_for_rep.get(fact.ticker, {}).keys()):
                    if d_key >= fact.target_date:
                        px_target = price_data_for_rep[fact.ticker][d_key]
                        break
                if px_at_pred is None or px_target is None:
                    continue
                change = (px_target - px_at_pred) / px_at_pred
                correct = ((fact.stated_lean == "long" and change > 0.005) or
                          (fact.stated_lean == "short" and change < -0.005) or
                          (fact.stated_lean == "neutral" and abs(change) < 0.01))
                if not correct:
                    store.invalidate(fact.id, when=day_str)
                    invalidations_today.append({"owner": aid, "fact_id": fact.id, "ticker": fact.ticker,
                                                "stated_lean": fact.stated_lean, "change_pct": change * 100})

        # 3. Build info_views per ticker
        info_views_global = {}
        for t in TICKERS:
            hist = prices_df[t][prices_df[t].index < d].tail(30)
            news = load_news_cache(t, day_str)
            info_views_global[t] = {"history": hist, "news": news}

        def get_info_view_for_agent(agent):
            views = {}
            for t in TICKERS:
                hist = info_views_global[t]["history"]
                news = info_views_global[t]["news"]
                news_text = format_news_for_tier(news, agent.info_tier)
                views[t] = filter_data_by_tier(hist, agent.info_tier, t, news_text=news_text)
            return views

        # 4. Cascade: Round 1 (Super + 3 economists) then Round 2 (6 thinkers)
        thinker_results: dict[str, MultiState] = {}
        public_posts_so_far = []

        def _decide_one(idx_local, prior_posts):
            a = ALL_AGENTS[idx_local]
            inbound = get_inbound_influencers(idx_local)
            inf_weights = {ALL_AGENTS[i].id: w * influence_mult.get(ALL_AGENTS[i].id, 1.0) for i, w in inbound}
            posts_received = [p for p in prior_posts if p["agent_id"] in inf_weights]
            inf_text = make_other_posts_section(posts_received, inf_weights) if posts_received else "(none)"
            info_views = get_info_view_for_agent(a)
            mem_ctx = build_memory_context(private_stores[a.id], day_str)
            # For no-portfolio agents, give a placeholder portfolio
            pf = portfolios.get(a.id, MultiPortfolio(agent_id=a.id, initial_capital=0, cash=0))
            return idx_local, agent_decide_multi(a, day_str, info_views, inf_text, pf, mem_ctx, prices_open)

        ROUND_1_IDX = [0, 8, 9, 10]
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
            futures = [pool.submit(_decide_one, i, []) for i in ROUND_1_IDX]
            for fut in as_completed(futures):
                i, (state, cost) = fut.result()
                thinker_results[ALL_AGENTS[i].id] = state
                public_posts_so_far.append({
                    "agent_id": ALL_AGENTS[i].id,
                    "agent_name": ALL_AGENTS[i].name,
                    "public_statement": {"narrative": state.public_statement.get("narrative", ""),
                                         "leans": state.public_statement.get("leans", {})},
                })
                total_cost += cost

        ROUND_2_IDX = [1, 2, 3, 5, 6, 7]
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
            futures = [pool.submit(_decide_one, i, public_posts_so_far) for i in ROUND_2_IDX]
            for fut in as_completed(futures):
                i, (state, cost) = fut.result()
                thinker_results[ALL_AGENTS[i].id] = state
                public_posts_so_far.append({
                    "agent_id": ALL_AGENTS[i].id,
                    "agent_name": ALL_AGENTS[i].name,
                    "public_statement": {"narrative": state.public_statement.get("narrative", ""),
                                         "leans": state.public_statement.get("leans", {})},
                })
                total_cost += cost

        # 5. CTA mechanical
        cta_idx = 4
        cta_actions = cta_action_multi({t: info_views_global[t]["history"] for t in TICKERS})

        # 6. Execute actions per ticker
        all_trades = {}  # agent_id → ticker → trade
        for a in ALL_AGENTS:
            if not a.has_portfolio:
                continue
            pf = portfolios[a.id]
            if a.role == "cta_forced":
                actions = cta_actions
            else:
                state = thinker_results.get(a.id)
                if state is None:
                    continue
                actions = state.actions
            agent_trades = {}
            for t in TICKERS:
                trade = apply_action(pf, t, actions.get(t, {"action_type": "hold", "size_pct": 0.0}),
                                     prices_open[t], day_str)
                agent_trades[t] = trade
            all_trades[a.id] = agent_trades

        # 7. Update memory + reputation tracking
        for aid, state in thinker_results.items():
            agent = next(a for a in ALL_AGENTS if a.id == aid)
            horizon = agent.time_horizon_days
            target_d_actual = None
            for d_future in trading_days:
                if d_future > d and (d_future - d).days >= horizon:
                    target_d_actual = str(d_future)
                    break
            target_d_actual = target_d_actual or str(trading_days[-1])

            leans = state.public_statement.get("leans", {})
            for t in TICKERS:
                stated = leans.get(t, "neutral")
                private_stores[aid].add(
                    content=f"I said {stated.upper()} on {t}: {state.public_statement.get('narrative','')[:80]}",
                    source="self_post", valid_at=day_str, ticker=t,
                    fact_type="prediction", stated_lean=stated, target_date=target_d_actual,
                )
                # propagate to influenced agents
                for other_aid, store in private_stores.items():
                    if other_aid == aid:
                        continue
                    idx_self = next(i for i, ag in enumerate(ALL_AGENTS) if ag.id == aid)
                    idx_other = next(i for i, ag in enumerate(ALL_AGENTS) if ag.id == other_aid)
                    if INFLUENCE_GRAPH.get((idx_self, idx_other), 0.0) >= INFLUENCE_THRESHOLD:
                        store.add(
                            content=f"{agent.name} said {stated.upper()} on {t}",
                            source=aid, valid_at=day_str, ticker=t,
                            fact_type="prediction", stated_lean=stated, target_date=target_d_actual,
                        )
                # Record for reputation
                rep_tracker.record_post(PostRecord(
                    date=day_str, agent_id=aid, ticker=t,
                    stated_lean=stated, private_lean=state.private_belief.get("leans", {}).get(t, "neutral"),
                    action_type=state.actions.get(t, {}).get("action_type", "hold"),
                    target_date=target_d_actual,
                ))

        # 8. Weekly reputation
        rep_update = (d_idx + 1) % 7 == 0
        rep_snapshots = None
        if rep_update:
            rep_snapshots = rep_tracker.update(today=day_str, price_data=price_data_for_rep, agents=ALL_AGENTS)
            for aid, snap in rep_snapshots.items():
                influence_mult[aid] = snap.influence_multiplier

        # 9. Record day
        agent_outputs = {}
        for aid in thinker_results:
            agent = next(a for a in ALL_AGENTS if a.id == aid)
            agent_outputs[aid] = {
                "name": agent.name, "role": agent.role,
                "state": thinker_results[aid].to_dict(),
            }
        agent_outputs[ALL_AGENTS[cta_idx].id] = {
            "name": ALL_AGENTS[cta_idx].name, "role": "cta_forced",
            "state": {"private_belief": {"leans": {t: "deterministic" for t in TICKERS}, "actual_thesis": "SMA"},
                      "public_statement": {"leans": {t: "n/a" for t in TICKERS}, "narrative": "[silent]"},
                      "desired_market_reaction": "n/a", "actions": cta_actions},
        }

        portfolios_snapshot = {}
        for aid, pf in portfolios.items():
            portfolios_snapshot[aid] = {
                "cash": pf.cash,
                "positions": dict(pf.positions),
                "stops": dict(pf.stops),
                "targets": dict(pf.targets),
                "total": pf.total_value(prices_close),
                "pnl_pct": pf.pnl_pct(prices_close),
            }

        days_records.append({
            "date": day_str,
            "prices_open": prices_open,
            "prices_close": prices_close,
            "agent_outputs": agent_outputs,
            "trades": all_trades,
            "forced_closes": forced_closes,
            "invalidations_today": invalidations_today,
            "reputation_update": rep_update,
            "reputation_snapshots": {aid: {"accuracy": s.accuracy, "n_posts": s.n_posts_evaluated, "mult": s.influence_multiplier}
                                     for aid, s in (rep_snapshots or {}).items()},
            "influence_multipliers_after": dict(influence_mult),
            "portfolios_close": portfolios_snapshot,
        })

        elapsed = time.time() - start_time
        n_forced = sum(len(v) for v in forced_closes.values())
        rep_marker = "📊REP" if rep_update else ""
        prices_str = " | ".join(f"{t}:${prices_close[t]:.0f}" for t in TICKERS)
        print(f"[Day {d_idx+1:2d}/{len(trading_days)}] {day_str} | {prices_str} | "
              f"stops/targets fired={n_forced} | inval={len(invalidations_today)} {rep_marker} | "
              f"elapsed={elapsed:.0f}s cost=${total_cost:.2f}", flush=True)

    # Final
    last_d = trading_days[-1]
    last_prices = {t: float(prices_df[t].loc[last_d, "close"]) for t in TICKERS}
    final_portfolios = {}
    for a in ALL_AGENTS:
        if not a.has_portfolio:
            final_portfolios[a.id] = {"name": a.name, "role": a.role, "initial_capital": 0,
                                       "final_total": 0, "final_pnl_pct": 0,
                                       "positions": {}, "cash": 0,
                                       "memory_facts_total": private_stores[a.id].count(only_active=False),
                                       "memory_facts_active": private_stores[a.id].count(only_active=True)}
            continue
        pf = portfolios[a.id]
        final_portfolios[a.id] = {
            "name": a.name, "role": a.role,
            "initial_capital": a.capital,
            "final_total": pf.total_value(last_prices),
            "final_pnl_pct": pf.pnl_pct(last_prices),
            "positions": dict(pf.positions),
            "cash": pf.cash,
            "transactions": pf.transactions,
            "memory_facts_total": private_stores[a.id].count(only_active=False),
            "memory_facts_active": private_stores[a.id].count(only_active=True),
        }

    # Multi-ticker buy-and-hold benchmark (equal weight)
    first_prices = {t: float(prices_df[t].loc[trading_days[0], "open"]) for t in TICKERS}
    bh_per_ticker = {t: (last_prices[t] - first_prices[t]) / first_prices[t] * 100 for t in TICKERS}
    bh_equal_weight = sum(bh_per_ticker.values()) / len(TICKERS)

    result = {
        "tickers": TICKERS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "sensitivity": SENSITIVITY,
        "total_cost_usd": total_cost,
        "days": days_records,
        "final_portfolios": final_portfolios,
        "benchmark_buy_hold_per_ticker": bh_per_ticker,
        "benchmark_buy_hold_equal_weight": bh_equal_weight,
        "reputation_history": rep_tracker.to_dict(),
        "private_memory_snapshot": {aid: s.to_dict() for aid, s in private_stores.items()},
    }

    out_path = RESULTS_DIR / "phase_b_latest.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print("", flush=True)
    print("=" * 75, flush=True)
    print(f"DONE: {len(days_records)} days, {elapsed/60:.1f} min, ${total_cost:.2f}", flush=True)
    print(f"AAPL B&H: {bh_per_ticker['AAPL']:+.2f}% | NVDA: {bh_per_ticker['NVDA']:+.2f}% | MSFT: {bh_per_ticker['MSFT']:+.2f}%", flush=True)
    print(f"Equal-weight B&H: {bh_equal_weight:+.2f}%", flush=True)
    print("", flush=True)
    print("Final PnL by agent:", flush=True)
    for aid, p in sorted(final_portfolios.items(), key=lambda x: -x[1]["final_pnl_pct"]):
        print(f"  {p['name']:30s} {p['role']:20s} {p['final_pnl_pct']:+6.2f}%", flush=True)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
