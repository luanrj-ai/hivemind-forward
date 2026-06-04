"""Simulation runner: replays historical days, calls all personas per day per mode, tracks 3 parallel worlds."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Literal

import pandas as pd

from data import fetch_history, time_window_view
from llm import Backend, PersonaAction, persona_decide
from personas import ALL_PERSONAS, PersonaDef
from price_engine import (
    Mode,
    PriceFormationInput,
    compute_intended_order,
    compute_net_pressure,
    form_price,
)


RESULTS_DIR = Path(__file__).parent / "cache" / "simulation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PersonaPortfolio:
    persona_id: str
    initial_capital: float
    cash: float
    shares: int = 0
    transactions: list[dict] = field(default_factory=list)

    def total_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def pnl_pct(self, price: float) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self.total_value(price) - self.initial_capital) / self.initial_capital * 100


@dataclass
class DayRecord:
    date: str
    mode: Mode
    real_open: float
    real_close: float
    virtual_open: float
    virtual_close: float
    buy_volume_usd: float
    sell_volume_usd: float
    net_pressure: float
    actions: dict  # persona_id → action dict + post-trade portfolio
    sensitivity: float


@dataclass
class SimulationResult:
    ticker: str
    start_date: str
    end_date: str
    sensitivity: float
    modes: list[Mode]
    days: list[DayRecord]
    final_portfolios: dict  # mode → persona_id → final portfolio


def _format_price_history(df: pd.DataFrame) -> str:
    """Compact CSV of last N days."""
    buf = StringIO()
    df_out = df[["open", "high", "low", "close", "volume"]].copy()
    df_out.index.name = "date"
    df_out.to_csv(buf, float_format="%.2f")
    return buf.getvalue()


def run_simulation(
    ticker: str = "AAPL",
    start_date: str = "2026-04-14",
    end_date: str = "2026-05-02",
    sensitivity: float = 0.3,
    modes: tuple[Mode, ...] = ("alpha", "beta", "gamma"),
    use_llm_cache: bool = True,
    mock: bool = False,
    backend: Backend = "anthropic_sdk",
    max_concurrency: int = 5,
    on_step=None,  # callback(idx, total, message)
) -> SimulationResult:
    """Run the simulation. Returns full SimulationResult with all 3 modes in parallel."""
    # fetch data
    df = fetch_history(ticker, "2026-01-01", "2026-05-31")
    df = df.sort_index()

    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date).date()
    trading_days = [d for d in df.index if start_d <= d <= end_d]

    if not trading_days:
        raise RuntimeError(f"No trading days in {start_date}..{end_date} for {ticker}")

    # Initialize one portfolio per (mode, persona)
    portfolios: dict[Mode, dict[str, PersonaPortfolio]] = {
        m: {p.id: PersonaPortfolio(persona_id=p.id, initial_capital=p.initial_capital, cash=p.initial_capital)
            for p in ALL_PERSONAS}
        for m in modes
    }

    # Initialize virtual prices = first day's open
    first_day = trading_days[0]
    virtual_prices: dict[Mode, float] = {m: float(df.loc[first_day, "open"]) for m in modes}

    days_records: list[DayRecord] = []
    total_steps = len(trading_days) * len(modes)
    step_idx = 0

    prev_real_close: float = float(df.loc[first_day, "open"])  # Day 1 starts here

    for di, d in enumerate(trading_days):
        real_open = float(df.loc[d, "open"])
        real_close = float(df.loc[d, "close"])

        # For each mode, run persona decisions independently
        for m in modes:
            # Price history view: last 30 trading days before d (TimeWindow-sliced)
            hist = time_window_view(df, d, lookback_days=30)
            price_csv = _format_price_history(hist)

            prev_virtual = virtual_prices[m]

            # Collect actions for all personas
            actions: dict[str, dict] = {}
            buy_usd = 0.0
            sell_usd = 0.0
            total_capital = sum(p.total_value(prev_virtual if m != "gamma" else real_open)
                                for p in portfolios[m].values())

            ref_price = prev_virtual if m != "gamma" else real_open

            # Parallel persona calls within this (day, mode)
            def _decide_one(persona: PersonaDef) -> tuple[str, PersonaAction]:
                portfolio = portfolios[m][persona.id]
                action = persona_decide(
                    persona=persona,
                    ticker=ticker,
                    price_history_csv=price_csv,
                    cash=portfolio.cash,
                    shares=portfolio.shares,
                    total_value=portfolio.total_value(ref_price),
                    decision_date=str(d),
                    mode=m,
                    use_cache=use_llm_cache,
                    mock=mock,
                    backend=backend,
                )
                return persona.id, action

            with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
                futures = [pool.submit(_decide_one, p) for p in ALL_PERSONAS]
                persona_actions: dict[str, PersonaAction] = {}
                for fut in as_completed(futures):
                    pid, action = fut.result()
                    persona_actions[pid] = action

            for persona in ALL_PERSONAS:
                portfolio = portfolios[m][persona.id]
                action = persona_actions[persona.id]

                shares_signed, cash_signed = compute_intended_order(
                    action_type=action.action_type,
                    size_pct=action.action_size_pct,
                    cash=portfolio.cash,
                    shares=portfolio.shares,
                    price=ref_price,
                )
                if shares_signed > 0:
                    buy_usd += cash_signed
                elif shares_signed < 0:
                    sell_usd += abs(cash_signed)

                actions[persona.id] = {
                    "action": asdict(action),
                    "intended_shares": shares_signed,
                    "intended_cash": cash_signed,
                    "pre_cash": portfolio.cash,
                    "pre_shares": portfolio.shares,
                    "pre_total": portfolio.total_value(ref_price),
                }

            net_pressure = compute_net_pressure(buy_usd, sell_usd, max(total_capital, 1.0))

            # Compute virtual price evolution + fill price
            fp_inp = PriceFormationInput(
                prev_virtual_price=prev_virtual,
                prev_real_price=prev_real_close,
                cur_real_price=real_close,
                net_pressure=net_pressure,
                sensitivity=sensitivity,
            )
            fp_out = form_price(fp_inp, m)
            fill_price = fp_out.persona_fill_price

            # Execute trades at fill_price
            for persona in ALL_PERSONAS:
                portfolio = portfolios[m][persona.id]
                shares_signed, _ = compute_intended_order(
                    action_type=actions[persona.id]["action"]["action_type"],
                    size_pct=actions[persona.id]["action"]["action_size_pct"],
                    cash=portfolio.cash,
                    shares=portfolio.shares,
                    price=fill_price,
                )
                if shares_signed > 0:
                    cost = shares_signed * fill_price
                    if cost <= portfolio.cash:
                        portfolio.cash -= cost
                        portfolio.shares += shares_signed
                        portfolio.transactions.append({
                            "date": str(d), "side": "buy", "shares": shares_signed,
                            "price": fill_price, "mode": m,
                        })
                elif shares_signed < 0:
                    sell_shares = abs(shares_signed)
                    proceeds = sell_shares * fill_price
                    portfolio.cash += proceeds
                    portfolio.shares -= sell_shares
                    portfolio.transactions.append({
                        "date": str(d), "side": "sell", "shares": sell_shares,
                        "price": fill_price, "mode": m,
                    })

                # Append post-trade snapshot
                actions[persona.id]["post_cash"] = portfolio.cash
                actions[persona.id]["post_shares"] = portfolio.shares
                # post total at virtual close (for non-gamma) or real close (for gamma)
                post_ref = fp_out.next_virtual_price if m != "gamma" else real_close
                actions[persona.id]["post_total"] = portfolio.total_value(post_ref)

            virtual_prices[m] = fp_out.next_virtual_price

            days_records.append(DayRecord(
                date=str(d),
                mode=m,
                real_open=real_open,
                real_close=real_close,
                virtual_open=prev_virtual,
                virtual_close=fp_out.next_virtual_price,
                buy_volume_usd=buy_usd,
                sell_volume_usd=sell_usd,
                net_pressure=net_pressure,
                actions=actions,
                sensitivity=sensitivity,
            ))

            step_idx += 1
            if on_step:
                on_step(step_idx, total_steps, f"{d} {m}")

        prev_real_close = real_close

    # Final portfolios
    final_p: dict = {}
    last_day = trading_days[-1]
    last_real_close = float(df.loc[last_day, "close"])
    for m in modes:
        final_p[m] = {}
        last_virtual = virtual_prices[m]
        ref = last_virtual if m != "gamma" else last_real_close
        for pid, pf in portfolios[m].items():
            final_p[m][pid] = {
                "cash": pf.cash,
                "shares": pf.shares,
                "total_value": pf.total_value(ref),
                "initial_capital": pf.initial_capital,
                "pnl_pct": pf.pnl_pct(ref),
                "transactions": pf.transactions,
            }

    return SimulationResult(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        sensitivity=sensitivity,
        modes=list(modes),
        days=days_records,
        final_portfolios=final_p,
    )


def save_result(result: SimulationResult, name: str = "latest") -> Path:
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump({
            "ticker": result.ticker,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "sensitivity": result.sensitivity,
            "modes": result.modes,
            "days": [asdict(d) for d in result.days],
            "final_portfolios": result.final_portfolios,
        }, f, indent=2, default=str)
    return path


def load_result(name: str = "latest") -> SimulationResult | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    days = [DayRecord(**d) for d in data["days"]]
    return SimulationResult(
        ticker=data["ticker"],
        start_date=data["start_date"],
        end_date=data["end_date"],
        sensitivity=data["sensitivity"],
        modes=data["modes"],
        days=days,
        final_portfolios=data["final_portfolios"],
    )
