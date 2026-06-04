"""3 price formation modes: alpha (closed), beta (anchored hybrid), gamma (real + auxiliary)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Mode = Literal["alpha", "beta", "gamma"]


@dataclass
class PriceFormationInput:
    prev_virtual_price: float
    prev_real_price: float
    cur_real_price: float
    net_pressure: float  # in [-1, 1] approximately
    sensitivity: float   # in [0, 1]


@dataclass
class PriceFormationOutput:
    mode: Mode
    next_virtual_price: float
    real_drift_pct: float          # (cur_real - prev_real) / prev_real
    pressure_contribution_pct: float
    persona_fill_price: float       # which price personas trade at for THIS mode


def compute_net_pressure(buy_volume: float, sell_volume: float, total_capital: float) -> float:
    """Net pressure in (-1, 1). Positive = net buying."""
    if total_capital <= 0:
        return 0.0
    raw = (buy_volume - sell_volume) / total_capital
    # squash to (-1, 1)
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return raw


def form_price(inp: PriceFormationInput, mode: Mode) -> PriceFormationOutput:
    """Apply the chosen mode to compute next virtual price + persona fill price."""
    real_drift = (inp.cur_real_price - inp.prev_real_price) / inp.prev_real_price if inp.prev_real_price > 0 else 0.0

    if mode == "alpha":
        # Closed: virtual evolves ONLY by agent pressure, ignores real
        next_virtual = inp.prev_virtual_price * (1.0 + inp.sensitivity * inp.net_pressure)
        fill_price = inp.prev_virtual_price  # personas fill at virtual open (= yesterday's close)
        pressure_pct = inp.sensitivity * inp.net_pressure

    elif mode == "beta":
        # Anchored hybrid: virtual = real drift * (1 + pressure)
        # mathematically equivalent to: prev_virtual × (1 + real_drift) × (1 + sensitivity × net_pressure)
        next_virtual = inp.prev_virtual_price * (1.0 + real_drift) * (1.0 + inp.sensitivity * inp.net_pressure)
        fill_price = inp.prev_virtual_price
        # decompose: total pct change is approximately real_drift + sensitivity * net_pressure
        pressure_pct = inp.sensitivity * inp.net_pressure

    elif mode == "gamma":
        # Dual layer: personas fill at REAL prices; virtual is auxiliary signal
        # virtual moves purely by sensitivity * pressure but anchored to real
        next_virtual = inp.cur_real_price * (1.0 + inp.sensitivity * inp.net_pressure)
        fill_price = inp.cur_real_price  # personas trade at real
        pressure_pct = inp.sensitivity * inp.net_pressure

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return PriceFormationOutput(
        mode=mode,
        next_virtual_price=next_virtual,
        real_drift_pct=real_drift,
        pressure_contribution_pct=pressure_pct,
        persona_fill_price=fill_price,
    )


# ---- Action → buy/sell volume conversion ----

ACTION_SIZE_MULTIPLIER = {
    "buy_strong": 0.50,   # use 50% of cash × size_pct
    "buy_lite": 0.20,
    "hold": 0.0,
    "sell_lite": 0.30,
    "sell_strong": 1.00,
}


def compute_intended_order(
    action_type: str,
    size_pct: float,
    cash: float,
    shares: int,
    price: float,
) -> tuple[int, float]:
    """Return (shares_to_trade, signed_cash_value). Positive = buy."""
    if action_type == "hold":
        return 0, 0.0

    mult = ACTION_SIZE_MULTIPLIER.get(action_type, 0.0)

    if action_type.startswith("buy"):
        cash_to_use = cash * mult * size_pct
        shares_to_buy = int(cash_to_use / price) if price > 0 else 0
        return shares_to_buy, shares_to_buy * price
    else:  # sell
        if shares == 0:
            return 0, 0.0
        shares_to_sell = shares if action_type == "sell_strong" else max(1, int(shares * mult * size_pct))
        shares_to_sell = min(shares_to_sell, shares)
        return -shares_to_sell, -shares_to_sell * price
