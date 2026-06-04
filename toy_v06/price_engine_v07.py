"""β-mode virtual price formation: real drift + agent net-pressure adjustment.

Used in the 1-hour demo to show "simulated vs real" daily prices.
"""


def compute_net_pressure(buy_volume_usd: float, sell_volume_usd: float, total_capital: float) -> float:
    """Net pressure in (-1, 1). Positive = net buying."""
    if total_capital <= 0:
        return 0.0
    raw = (buy_volume_usd - sell_volume_usd) / total_capital
    return max(-1.0, min(1.0, raw))


def compute_beta_virtual_price(
    prev_virtual: float,
    prev_real: float,
    cur_real: float,
    net_pressure: float,
    sensitivity: float = 0.3,
    anchor_lambda: float = 0.07,
) -> float:
    """Mode β: virtual = prev_virtual × (1 + real_drift) × (1 + sensitivity × pressure)
    then pulled toward cur_real with strength anchor_lambda.

    The anchor prevents the gap (virt/real) from being permanently locked-in once flow
    decays to zero. anchor_lambda=0 disables (legacy behavior); 0.07 ≈ 7% of the
    deviation closes per day.
    """
    if prev_real <= 0:
        return cur_real
    real_drift = (cur_real - prev_real) / prev_real
    base = prev_virtual * (1.0 + real_drift) * (1.0 + sensitivity * net_pressure)
    if anchor_lambda > 0 and cur_real > 0:
        deviation = base / cur_real - 1.0
        base = base * (1.0 - anchor_lambda * deviation)
    return base
