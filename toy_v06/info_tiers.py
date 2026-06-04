"""Information Hierarchy: filter market data per agent's info_tier.

Tier 1 (Retail): headlines + price chart only
Tier 2 (Sell-Side): + analyst-grade fundamentals + technicals
Tier 3 (Activist Short): + deep filings + skew + insider trades
Tier 4 (Pod PM): + options flow + 13F lag + dealer positioning
Tier 5 (Fed/CB watcher): + macro nowcast + rate path probabilities + Fed draft leaks
"""

import pandas as pd
import numpy as np


def filter_data_by_tier(
    price_history: pd.DataFrame,  # last 30 days OHLCV
    tier: int,
    ticker: str = "AAPL",
    news_text: str = "",  # v0.7 NEW: pre-formatted news for this tier
) -> str:
    """Return formatted string of what this tier sees."""
    if price_history.empty:
        return "No data available"

    last = price_history.iloc[-1]
    prev = price_history.iloc[-2] if len(price_history) > 1 else last
    pct_1d = (last['close'] - prev['close']) / prev['close'] * 100 if prev['close'] > 0 else 0

    # === ALL TIERS see basic price chart ===
    base = [
        f"Ticker: {ticker}",
        f"Current price: ${last['close']:.2f}",
        f"Today change: {pct_1d:+.2f}%",
        f"30-day range: ${price_history['close'].min():.2f} - ${price_history['close'].max():.2f}",
        f"Last 5 closes: " + ", ".join(f"${p:.2f}" for p in price_history['close'].tail(5)),
    ]

    if tier >= 1:  # Retail
        sma_20 = price_history['close'].tail(20).mean()
        mom_5d = ((last['close'] / price_history['close'].iloc[-6] - 1) * 100) if len(price_history) >= 6 else 0
        base.append(f"\n[TIER 1 — Retail headlines + chart]")
        base.append(f"20-day SMA: ${sma_20:.2f} (price is {'above' if last['close'] > sma_20 else 'below'} SMA)")
        base.append(f"5-day momentum: {mom_5d:+.2f}%")
        if news_text:
            base.append(f"\n{news_text}")

    if tier >= 2:  # Sell-Side
        # Realized vol
        returns = price_history['close'].pct_change().dropna()
        vol = returns.std() * (252 ** 0.5) * 100
        # RSI(14) — Wilder
        rsi_window = price_history['close'].tail(14)
        gains = rsi_window.diff().clip(lower=0).mean()
        losses = -rsi_window.diff().clip(upper=0).mean()
        rsi = 100 - (100 / (1 + (gains / losses))) if losses > 0 else 70.0
        # MACD(12,26,9)
        if len(price_history) >= 26:
            ema_12 = price_history['close'].ewm(span=12, adjust=False).mean()
            ema_26 = price_history['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_now = float(macd_line.iloc[-1])
            sig_now = float(signal_line.iloc[-1])
            hist_now = macd_now - sig_now
            hist_prev = float((macd_line - signal_line).iloc[-2]) if len(macd_line) > 1 else hist_now
            cross = ""
            if hist_prev <= 0 < hist_now:
                cross = " [bullish-cross today]"
            elif hist_prev >= 0 > hist_now:
                cross = " [bearish-cross today]"
            macd_str = f"MACD(12,26,9): line={macd_now:+.2f}, signal={sig_now:+.2f}, hist={hist_now:+.2f}{cross}"
        else:
            macd_str = "MACD(12,26,9): insufficient history"
        # Bollinger Bands(20, 2σ)
        if len(price_history) >= 20:
            bb_mid = float(price_history['close'].tail(20).mean())
            bb_std = float(price_history['close'].tail(20).std())
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_width_pct = (bb_upper - bb_lower) / bb_mid * 100 if bb_mid > 0 else 0
            bb_pos = (last['close'] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            bb_tag = " [touching upper]" if bb_pos > 0.95 else (" [touching lower]" if bb_pos < 0.05 else "")
            bb_str = f"BB(20,2σ): upper=${bb_upper:.2f} mid=${bb_mid:.2f} lower=${bb_lower:.2f} width={bb_width_pct:.1f}% pos={bb_pos:.2f}{bb_tag}"
        else:
            bb_str = "BB(20,2σ): insufficient history"
        base.append(f"\n[TIER 2 — Sell-Side analyst]")
        base.append(f"Annualized vol: {vol:.1f}%")
        base.append(f"RSI(14): {rsi:.1f}")
        base.append(macd_str)
        base.append(bb_str)
        base.append(f"PT consensus: $290 (currently below; +10% upside)")
        base.append(f"Earnings in: 18 days")

    if tier >= 3:  # Activist Short
        # Volume anomalies (dynamic — was correct already, kept; clarified guard)
        vol_30d = float(price_history['volume'].tail(30).mean())
        vol_today = float(last['volume'])
        vol_ratio = vol_today / vol_30d if vol_30d > 0 else 1.0
        # ATR(14) — realized intra-day range
        if len(price_history) >= 15:
            tail15 = price_history.tail(15).copy()
            tail15['prev_close'] = tail15['close'].shift(1)
            tail15 = tail15.dropna()
            tr = (tail15[['high', 'prev_close']].max(axis=1) - tail15[['low', 'prev_close']].min(axis=1))
            atr = float(tr.mean())
            atr_pct = atr / last['close'] * 100 if last['close'] > 0 else 0
            atr_str = f"ATR(14): ${atr:.2f} ({atr_pct:.2f}% of price, realized daily range)"
        else:
            atr_str = "ATR(14): insufficient history"
        # 20-day momentum (factor proxy)
        if len(price_history) >= 21:
            mom_20 = (last['close'] / price_history['close'].iloc[-21] - 1) * 100
        else:
            mom_20 = 0
        base.append(f"\n[TIER 3 — Activist research]")
        base.append(f"Volume today: {vol_today/1e6:.1f}M vs 30-day avg {vol_30d/1e6:.1f}M (ratio {vol_ratio:.2f}x)")
        base.append(atr_str)
        base.append(f"20-day momentum: {mom_20:+.2f}%")
        base.append(f"Recent 10-Q line item: services growth +12% YoY, but tablet -8%")
        base.append(f"Insider Form 4 in past 30d: 3 sells, 0 buys (executives)")
        base.append(f"Options skew (3M put/call IV ratio): 1.18 (bearish skew rising)")

    if tier >= 4:  # Pod PM
        base.append(f"\n[TIER 4 — Hedge Fund flow data]")
        base.append(f"Net options flow last 5d: -$280M (puts outweighing calls)")
        base.append(f"Dealer gamma exposure: -$1.2B (dealers short gamma, vol amplifies moves)")
        base.append(f"13F lag (most recent +45d): Top holders net sold 0.3% of float")
        base.append(f"Sec lending utilization: 4.2% (moderate short interest)")
        base.append(f"Sector momentum vs SPY (XLK): -1.4% past week (rotation out of tech)")

    if tier >= 5:  # Fed/CB watcher
        base.append(f"\n[TIER 5 — Macro / Fed channel]")
        base.append(f"Fed minutes draft (received 24h before public): 'careful hold' wording")
        base.append(f"PCE nowcast (Goldman / JPM models): 2.4% (in-line)")
        base.append(f"Rate path probabilities (CME + private models): -25bp in 60d at 65% prob")
        base.append(f"China A50 overnight: -0.8% (modest risk-off)")
        base.append(f"USD/JPY: 152.4 (BoJ intervention watch)")

    return "\n".join(base)


def make_other_posts_section(
    other_posts: list[dict],
    influence_weights: dict[str, float],
) -> str:
    """Format influencer posts that THIS agent receives, weighted by G(i,j)."""
    if not other_posts:
        return "No external voices reached you today."

    lines = ["\n=== INFLUENCER POSTS YOU SAW TODAY ==="]
    for post in other_posts:
        agent_id = post['agent_id']
        weight = influence_weights.get(agent_id, 0)
        lines.append(f"\n[{post.get('agent_name', agent_id)}] (influence weight on you: {weight:.2f})")
        lines.append(f"  Stated lean: {post['public_statement']['stated_lean']} (conviction {post['public_statement']['stated_conviction']:.2f})")
        lines.append(f"  Says: \"{post['public_statement']['narrative']}\"")
    return "\n".join(lines)
