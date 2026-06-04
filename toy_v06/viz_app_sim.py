"""Stock Sim — minimal toC trading UI with indicators + history + range zoom."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Import mvp_data for full price history
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mvp"))
import data as mvp_data  # noqa: E402

from agents import ALL_AGENTS  # noqa: E402

# ============================================================================
DATA_PATH = Path(__file__).parent / "results" / "v07_demo_latest.json"
NEWS_CACHE = Path(__file__).parent / "cache" / "news"
ANCHOR_LAMBDA = 0.07

AGENT_NAMES = {
    "super_influencer_001":      "Catherine",
    "pod_pm_001":                "David",
    "activist_short_001":        "Sarah",
    "sell_side_001":             "Michael",
    "cta_forced_001":            "CTA",
    "retail_fomo_001":           "Alex",
    "permabull_001":             "Thomas",
    "day_trader_001":            "Devon",
    "economist_macro_001":       "Bernanke",
    "economist_political_001":   "Krugman",
    "economist_trader_001":      "Druck",
}

AGENT_ROLES = {
    "super_influencer_001":      "Super Influencer ($5B fund)",
    "pod_pm_001":                "Hedge Fund Pod PM",
    "activist_short_001":        "Activist Short",
    "sell_side_001":             "Sell-Side Analyst",
    "cta_forced_001":            "Quant CTA (trend)",
    "retail_fomo_001":           "Retail (WSB)",
    "permabull_001":             "Permabull PM",
    "day_trader_001":            "Day Trader",
    "economist_macro_001":       "Macro Economist",
    "economist_political_001":   "Political Economist",
    "economist_trader_001":      "Retired Macro Trader",
}

# What-if scenarios — rule-based agent reactions derived from each agent's personality
SCENARIOS = {
    "War breaks out": {"shock": -0.10, "dir": "crash", "blurb": "Risk-off, flight to safety, vol spikes"},
    "Fed surprise hike +50bps": {"shock": -0.04, "dir": "crash", "blurb": "Multiple compression, growth re-rates"},
    "Apple earnings blowout": {"shock": +0.06, "dir": "rally", "blurb": "Margin beat, guidance raise"},
    "Product recall scandal": {"shock": -0.07, "dir": "crash", "blurb": "Brand hit, near-term cash impact"},
    "AI bubble bursts": {"shock": -0.08, "dir": "crash", "blurb": "Tech selloff, rotation to value"},
    "Massive buyback announcement": {"shock": +0.05, "dir": "rally", "blurb": "EPS accretion, signaling"},
}

# Each agent's tactical style (used by live scenario reaction logic)
AGENT_PERSONALITY = {
    "super_influencer_001":      "contrarian",     # Cathie — buys big on crashes
    "pod_pm_001":                "risk_cut",       # Pod — trims on adverse moves
    "activist_short_001":        "short_focused",  # already short
    "cta_forced_001":            "momentum",       # follow direction
    "retail_fomo_001":           "panic_fomo",     # amplifier of direction
    "permabull_001":             "buy_dip",        # only ever buys
    "day_trader_001":            "momentum",       # follow direction
    # speakers (sell_side, economists) have no portfolio → excluded
}

ACT_LABEL = {"buy_strong": "Buy max", "buy_lite": "Buy", "hold": "Hold",
             "sell_lite": "Sell", "sell_strong": "Sell all"}

GREEN = "#16a34a"
RED = "#dc2626"
INK = "#0a0a0a"
MUTED = "#737373"
FAINT = "#a3a3a3"
HISTORY_TINT = "#cbd5e1"


@st.cache_data
def load_run():
    with open(DATA_PATH) as f:
        return json.load(f)


@st.cache_data
def load_full_history(ticker: str = "AAPL") -> pd.DataFrame:
    """Up to 5 years of real OHLCV for chart context."""
    df = mvp_data.fetch_history(ticker, "2021-05-01", "2026-05-31").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_news_for(date_str: str, ticker: str = "AAPL") -> list[str]:
    f = NEWS_CACHE / f"{ticker}_{date_str}.json"
    if not f.exists():
        return []
    try:
        items = json.load(open(f))
        if isinstance(items, list):
            return [it.get("title", "").strip()[:140] for it in items[:5] if it.get("title")]
    except Exception:
        pass
    return []


def _react(personality: str, direction: str, priv_lean: str, conv: float):
    """Compute an agent's (action, size, reason) given personality + LIVE private belief + scenario.

    The 'live' part: today's private_belief.lean and conviction (pulled from JSON each day)
    scale the response size. Same scenario on different days produces different reactions.
    """
    sign_dir = -1 if direction == "crash" else +1
    if personality == "contrarian":
        if direction == "crash":
            sz = min(0.85, 0.35 + 0.55 * conv)
            return ("buy_lite", sz, f"contrarian buy — conviction {conv:.0%}")
        else:
            return ("hold", 0, "already long, no need to chase")
    if personality == "panic_fomo":
        return (("sell_strong", 1.0, "panic sell — retail capitulation")
                if direction == "crash" else ("buy_strong", 1.0, "FOMO into rally"))
    if personality == "buy_dip":
        sz = min(0.75, 0.30 + 0.4 * conv) if direction == "crash" else min(0.30, 0.10 + 0.2 * conv)
        return ("buy_lite", sz, f"buy {'dip' if direction == 'crash' else 'breakout'}")
    if personality == "momentum":
        sz = 0.30 + 0.20 * conv
        if direction == "crash":
            return ("sell_lite", sz, "follow downside momentum")
        else:
            return ("buy_lite", sz, "follow upside momentum")
    if personality == "risk_cut":
        # private lean clashes with direction → cut harder
        belief_sign = {"long": +1, "short": -1}.get(priv_lean, 0)
        adverse = belief_sign * sign_dir < 0  # holding the wrong side
        if direction == "crash":
            if belief_sign > 0:  # long during crash — cut
                sz = 0.25 + 0.30 * conv
                return ("sell_lite", sz, "risk-managed trim (long into crash)")
            return ("hold", 0, "no long exposure to cut")
        else:
            if belief_sign < 0:  # short during rally — cover
                sz = 0.25 + 0.30 * conv
                return ("buy_lite", sz, "cover short into rally")
            return ("hold", 0, "no short exposure to cover")
    if personality == "short_focused":
        if direction == "crash":
            return ("hold", 0, "short position profiting — let it run")
        else:
            sz = 0.20 + 0.25 * conv
            return ("sell_lite", sz, "double down on short into rally")
    return ("hold", 0, "no thesis on this scenario")


def project_scenario(scenario_key: str, start_day_idx: int, days_list: list, full_df: pd.DataFrame,
                     sensitivity: float, n_days: int = 5):
    """Live projection: read each agent's TODAY private_belief from JSON, run personality model.

    Returns (dates, prices, reactions_list) where reactions_list is
        [{aid, name, action, size, reason, priv_lean, conv}, ...]
    """
    s = SCENARIOS[scenario_key]
    direction = s["dir"]
    start_p = days_list[start_day_idx]["real_open"]
    start_date = pd.to_datetime(days_list[start_day_idx]["date"])
    future_dates = list(full_df[full_df.index > start_date].index[:n_days])
    if not future_dates:
        return [], [], []

    # LIVE: pull today's private belief for each agent and compute reaction
    today_outputs = days_list[start_day_idx].get("agent_outputs", {})
    reactions = []
    total_cap = sum(a.capital for a in ALL_AGENTS if a.has_portfolio and a.capital > 0) or 1.0
    buy_usd = 0.0
    sell_usd = 0.0
    for aid, personality in AGENT_PERSONALITY.items():
        agent = next((a for a in ALL_AGENTS if a.id == aid), None)
        if not agent or not agent.has_portfolio:
            continue
        out = today_outputs.get(aid, {})
        pb = (out.get("state", {}) or {}).get("private_belief", {}) or {}
        priv_lean = pb.get("lean", "neutral")
        conv = float(pb.get("conviction", 0.5) or 0.5)
        act, sz, reason = _react(personality, direction, priv_lean, conv)
        reactions.append({
            "aid": aid, "name": AGENT_NAMES.get(aid, aid),
            "action": act, "size": sz, "reason": reason,
            "priv_lean": priv_lean, "conv": conv,
        })
        notional = agent.capital * sz
        if act.startswith("buy"): buy_usd += notional
        elif act.startswith("sell"): sell_usd += notional
    pressure_1 = max(-1.0, min(1.0, (buy_usd - sell_usd) / total_cap))

    prices = []
    p1 = start_p * (1 + s["shock"]) * (1 + sensitivity * pressure_1)
    prices.append(p1)
    for d in range(1, len(future_dates)):
        decay = 0.5 ** d
        p_next = prices[-1] * (1 + sensitivity * pressure_1 * decay)
        prices.append(p_next)
    return future_dates, prices, reactions


def compute_indicators(closes: pd.Series) -> dict:
    """Compute SMA20 / RSI14 / MACD / BB / momentum on a close series ending at `today`."""
    n = len(closes)
    out = {"sma20": None, "rsi": None, "macd_hist": None, "macd_dir": "—",
           "bb_upper": None, "bb_lower": None, "mom5": None, "vol_ann": None}
    if n < 6:
        return out
    out["mom5"] = (closes.iloc[-1] / closes.iloc[-6] - 1) * 100
    if n >= 14:
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = -delta.clip(upper=0).rolling(14).mean().iloc[-1]
        if loss and not np.isnan(loss):
            rs = (gain / loss) if loss > 0 else 0
            out["rsi"] = 100 - 100 / (1 + rs) if loss > 0 else 70.0
        else:
            out["rsi"] = 70.0
    if n >= 20:
        sma = closes.tail(20)
        out["sma20"] = float(sma.mean())
        out["bb_upper"] = float(sma.mean() + 2 * sma.std())
        out["bb_lower"] = float(sma.mean() - 2 * sma.std())
        ret = closes.pct_change().dropna()
        out["vol_ann"] = float(ret.std() * (252 ** 0.5) * 100) if len(ret) > 0 else None
    if n >= 26:
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        hist = (macd_line - signal)
        out["macd_hist"] = float(hist.iloc[-1])
        prev = float(hist.iloc[-2]) if len(hist) > 1 else 0
        if prev <= 0 < out["macd_hist"]:
            out["macd_dir"] = "▲ bull cross"
        elif prev >= 0 > out["macd_hist"]:
            out["macd_dir"] = "▼ bear cross"
        else:
            out["macd_dir"] = "▲" if out["macd_hist"] > 0 else "▼"
    return out


# ============================================================================
st.set_page_config(page_title="AAPL Sim", layout="wide", page_icon="📈",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
* { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", "PingFang SC", sans-serif !important; }
.block-container { padding: 1rem 1.5rem 1rem 1.5rem !important; max-width: 1180px; }
[data-testid="stHeader"], [data-testid="stToolbar"], footer, #MainMenu { display: none; }

[data-testid="stMetric"] { background: transparent !important; padding: 0 !important; border: none !important; }
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important; font-weight: 700 !important;
    font-variant-numeric: tabular-nums; color: #0a0a0a !important; line-height: 1.1 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important; font-weight: 500 !important; color: #737373 !important;
    text-transform: uppercase; letter-spacing: 0.06em !important;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; font-weight: 500; }

h1 { font-size: 1.4rem !important; font-weight: 700 !important; margin: 0 0 0.1rem 0 !important; }
h2, h3 { font-weight: 600 !important; color: #0a0a0a !important; margin: 0 !important; }
.section-label {
    font-size: 0.7rem; font-weight: 600; color: #737373;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 0 0 0.5rem 0;
}
hr { background: #f1f5f9 !important; height: 1px !important; border: none !important; margin: 0.8rem 0 !important; }

.stButton > button {
    background: white; border: 1px solid #e5e5e5; color: #0a0a0a;
    font-weight: 500; border-radius: 8px; padding: 0.55rem 0;
    transition: all 0.12s; font-size: 0.9rem;
}
.stButton > button:hover { border-color: #0a0a0a; }
.stButton > button[kind="primary"] {
    background: #0a0a0a !important; color: white !important; border-color: #0a0a0a !important;
}

.intro-card {
    background: linear-gradient(180deg, #fafafa 0%, #ffffff 100%);
    border: 1px solid #e5e5e5; border-left: 3px solid #0a0a0a;
    border-radius: 6px; padding: 1rem 1.3rem; margin-bottom: 1rem;
}
.intro-title { font-weight: 700; font-size: 1.02rem; color: #0a0a0a; margin-bottom: 0.4rem; }
.intro-body { font-size: 0.9rem; color: #404040; line-height: 1.6; }

.news-row { font-size: 0.95rem; line-height: 1.5; color: #262626; padding: 0.4rem 0; }
.news-row + .news-row { border-top: 1px solid #fafafa; }

.voice-block { padding: 0.6rem 0; border-top: 1px solid #f5f5f5; }
.voice-block:first-child { border-top: none; }
.voice-name { font-size: 0.95rem; font-weight: 600; color: #0a0a0a; }
.voice-meta { font-size: 0.78rem; margin-left: 0.5rem; }
.voice-quote { font-size: 0.85rem; color: #525252; line-height: 1.5; margin-top: 0.25rem; }

.indicator-pill {
    display: inline-flex; align-items: baseline; gap: 0.35rem;
    padding: 0.25rem 0.6rem; margin-right: 0.5rem;
    background: #fafafa; border: 1px solid #f1f5f9; border-radius: 6px;
    font-size: 0.82rem;
}
.indicator-pill .lbl { color: #737373; font-weight: 500; }
.indicator-pill .val { color: #0a0a0a; font-weight: 600; font-variant-numeric: tabular-nums; }

.standings-row {
    display: flex; align-items: baseline; gap: 0.5rem; padding: 0.28rem 0; font-size: 0.92rem;
}
.standings-rank { color: #a3a3a3; font-variant-numeric: tabular-nums; width: 1.5rem; }
.standings-who { color: #0a0a0a; font-weight: 500; min-width: 110px; }
.standings-pnl { font-weight: 600; font-variant-numeric: tabular-nums; }
.standings-you {
    background: linear-gradient(90deg, #fef9c3 0%, transparent 100%);
    border-radius: 4px; padding-left: 0.5rem; margin-left: -0.5rem;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
data = load_run()
days = data["days"]
ticker = data["ticker"]
sensitivity = data["sensitivity"]
TOTAL_DAYS = len(days)
full_df = load_full_history(ticker)


def reset_state(initial_cash: float):
    st.session_state.day_idx = 0
    st.session_state.user_cash = initial_cash
    st.session_state.user_shares = 0
    st.session_state.user_initial = initial_cash
    st.session_state.user_history = []
    st.session_state.show_virtual = False
    st.session_state.show_sma = False
    st.session_state.show_bb = False
    st.session_state.show_indicators_panel = False
    st.session_state.show_rsi_panel = True       # default ON
    st.session_state.show_macd_panel = False
    st.session_state.show_volume_panel = True    # default ON
    st.session_state.pending_action = "hold"
    st.session_state.pending_size = 0.3
    st.session_state.tutorial_dismissed = False
    st.session_state.range_choice = "3M"          # default 3 months
    # Interactivity
    st.session_state.peek_revealed = {}      # day_str → list of aids revealed THAT DAY (resets daily)
    st.session_state.active_scenario = None  # currently-projected what-if scenario key
    st.session_state.shocks = []             # legacy (always empty, kept for compat)

PEEKS_PER_DAY = 3


if "day_idx" not in st.session_state:
    reset_state(100_000_000.0)

day_idx = st.session_state.day_idx
is_done = day_idx >= TOTAL_DAYS
cur_day = days[min(day_idx, TOTAL_DAYS - 1)]
cur_date = pd.to_datetime(cur_day["date"])
first_open = days[0]["real_open"]

# Cumulative shock multiplier applied to user-perspective prices
def shock_mult_through(date_str: str) -> float:
    mult = 1.0
    for s in st.session_state.shocks:
        if s["day"] <= date_str:
            mult *= (1 + s["pct"])
    return mult

user_price_mult = shock_mult_through(cur_day["date"])
raw_cur_price = cur_day["real_open"] if not is_done else days[-1]["real_close"]
cur_price = raw_cur_price * user_price_mult  # user-perspective mark price (post-shock)

user_total = st.session_state.user_cash + st.session_state.user_shares * cur_price
pnl_pct = (user_total - st.session_state.user_initial) / st.session_state.user_initial * 100
bh_shares = int(st.session_state.user_initial / first_open)
bh_cash_left = st.session_state.user_initial - bh_shares * first_open
bh_value = bh_shares * cur_price + bh_cash_left
bh_pnl = (bh_value - st.session_state.user_initial) / st.session_state.user_initial * 100
alpha = pnl_pct - bh_pnl

# Chart shows up to YESTERDAY's close. Today's open is a separate marker.
# This matches the trading-game mental model: it's the morning of day t, history
# shows up to day t-1's close, and we trade at day t's open.
visible_history = full_df[full_df.index < cur_date]


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Settings**")
    init_cap_choice = st.select_slider(
        "Starting capital",
        options=[100_000, 1_000_000, 10_000_000, 100_000_000, 500_000_000, 2_000_000_000],
        value=st.session_state.user_initial,
        format_func=lambda v: f"${v/1e6:.0f}M" if v >= 1e6 else f"${v/1e3:.0f}k",
    )
    if st.button("Reset simulation", use_container_width=True):
        reset_state(float(init_cap_choice))
        st.rerun()

    st.markdown("---")
    st.caption("Chart overlays & indicator panels are above the chart on the main screen.")
    st.session_state.show_indicators_panel = st.toggle(
        "Show indicator pills under chart",
        value=st.session_state.show_indicators_panel,
    )
    st.caption("v0.7.6 · β-anchored · 32 trading days")


# ----------------------------------------------------------------------------
# WELCOME SCREEN — full-page until Start is clicked
# ----------------------------------------------------------------------------
if not st.session_state.tutorial_dismissed:
    st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:0.78rem;color:{MUTED};text-transform:uppercase;"
            f"letter-spacing:0.14em;font-weight:600;margin-bottom:0.8rem'>Welcome</div>"
            f"<div style='font-size:2.4rem;font-weight:800;color:{INK};line-height:1.1;margin-bottom:0.6rem'>📈 AAPL Trading Sim</div>"
            f"<div style='font-size:0.95rem;color:{MUTED};margin-bottom:2rem'>32 trading days · 11 AI agents · You vs Buy &amp; Hold</div>"
            f"</div>"
            f"<div style='font-size:1rem;color:#404040;line-height:1.75;max-width:560px;margin:0 auto 1.5rem auto'>"
            f"You're a fund manager trading <strong>AAPL</strong> for 32 days, alongside 11 AI agents — "
            f"a hedge-fund PM, an activist short seller, a permabull, a CTA quant, retail FOMO, "
            f"three macro economists, a sell-side analyst, and a Cathie-Wood-style influencer.<br><br>"
            f"<strong>Each day</strong>, read today's headlines, see what the agents are saying publicly, "
            f"then choose <em>Buy / Hold / Sell</em> and click <strong>Next day →</strong>.<br><br>"
            f"<strong>Watch out</strong> — some agents say one thing in public and do another in private. "
            f"You have 3 daily peeks to reveal what they really think."
            f"</div>",
            unsafe_allow_html=True,
        )
        bcols = st.columns([1, 1.2, 1])
        with bcols[1]:
            if st.button("Start →", type="primary", use_container_width=True, key="welcome_start"):
                st.session_state.tutorial_dismissed = True
                st.rerun()
        st.markdown(
            f"<div style='text-align:center;font-size:0.78rem;color:{FAINT};margin-top:1.2rem'>"
            f"Tip: settings (capital, reset, chart options) live in the sidebar — click the chevron at the top-left."
            f"</div>",
            unsafe_allow_html=True,
        )
    st.stop()


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
hdr = st.columns([1.3, 1, 1, 1, 1, 1])

with hdr[0]:
    title = "AAPL"
    if is_done:
        sub = f"Final · {cur_day['date']}"
    else:
        sub = f"Day {day_idx+1} of {TOTAL_DAYS} · {cur_day['date']}"
    st.markdown(f"# {title}")
    st.markdown(f"<div style='color:{MUTED};font-size:0.85rem;font-weight:500'>{sub}</div>", unsafe_allow_html=True)

with hdr[1]:
    today_pct = (cur_day['real_close'] - cur_day['real_open']) / cur_day['real_open'] * 100
    st.metric("Open", f"${cur_day['real_open']:.2f}", f"{today_pct:+.2f}%")
with hdr[2]:
    st.metric("Portfolio", f"${user_total/1e6:.2f}M", f"{pnl_pct:+.2f}%")
with hdr[3]:
    st.metric("Cash", f"${st.session_state.user_cash/1e6:.2f}M")
with hdr[4]:
    pos = "LONG" if st.session_state.user_shares > 0 else ("SHORT" if st.session_state.user_shares < 0 else "—")
    st.metric("Position", pos, f"{st.session_state.user_shares:+,} sh" if st.session_state.user_shares != 0 else None)
with hdr[5]:
    st.metric("Alpha", f"{alpha:+.2f}%", "vs B&H")

st.markdown("---")


# ----------------------------------------------------------------------------
# CHART + RIGHT PANEL
# ----------------------------------------------------------------------------
main_l, main_r = st.columns([1.55, 1], gap="large")

with main_l:
    # Compute indicator overlays
    closes = visible_history["close"]
    inds = compute_indicators(closes)

    # Range selector + chart overlay toggles (inline above chart for discoverability)
    rng_col, ov_col = st.columns([1.4, 2.6])
    with rng_col:
        range_options = ["1M", "3M", "1Y", "5Y", "Sim"]
        if st.session_state.range_choice not in range_options:
            st.session_state.range_choice = "Sim"
        range_choice = st.radio(
            "Time window", options=range_options,
            index=range_options.index(st.session_state.range_choice),
            horizontal=True, label_visibility="collapsed",
        )
        st.session_state.range_choice = range_choice
    with ov_col:
        overlay_opts = ["SMA20", "BB", "RSI", "MACD", "Volume"]
        default_sel = []
        if st.session_state.show_sma: default_sel.append("SMA20")
        if st.session_state.show_bb: default_sel.append("BB")
        if st.session_state.show_rsi_panel: default_sel.append("RSI")
        if st.session_state.show_macd_panel: default_sel.append("MACD")
        if st.session_state.show_volume_panel: default_sel.append("Volume")
        try:
            chosen = st.segmented_control(
                "Overlays", options=overlay_opts, default=default_sel,
                selection_mode="multi", label_visibility="collapsed",
                key=f"overlay_seg_{day_idx}",
            )
        except AttributeError:
            chosen = st.multiselect("Overlays", options=overlay_opts, default=default_sel,
                                    label_visibility="collapsed")
        chosen = chosen or []
        st.session_state.show_sma = "SMA20" in chosen
        st.session_state.show_bb = "BB" in chosen
        st.session_state.show_virtual = False  # β line removed per UX feedback
        st.session_state.show_rsi_panel = "RSI" in chosen
        st.session_state.show_macd_panel = "MACD" in chosen
        st.session_state.show_volume_panel = "Volume" in chosen

    sim_start = pd.to_datetime(days[0]["date"])
    sim_end = pd.to_datetime(days[-1]["date"])
    if range_choice == "1M":
        x_from = cur_date - pd.Timedelta(days=30)
        x_to = cur_date + pd.Timedelta(days=2)
    elif range_choice == "3M":
        x_from = cur_date - pd.Timedelta(days=90)
        x_to = cur_date + pd.Timedelta(days=2)
    elif range_choice == "1Y":
        x_from = cur_date - pd.Timedelta(days=365)
        x_to = cur_date + pd.Timedelta(days=2)
    elif range_choice == "5Y":
        x_from = max(full_df.index.min(), cur_date - pd.Timedelta(days=365 * 5))
        x_to = cur_date + pd.Timedelta(days=2)
    else:  # Sim
        x_from = sim_start - pd.Timedelta(days=2)
        x_to = sim_end + pd.Timedelta(days=2)

    # Build subplots based on which indicator panels are active
    panels = ["price"]
    if st.session_state.show_volume_panel:
        panels.append("volume")
    if st.session_state.show_rsi_panel:
        panels.append("rsi")
    if st.session_state.show_macd_panel:
        panels.append("macd")

    n_rows = len(panels)
    if n_rows == 1:
        row_heights = [1.0]
    elif n_rows == 2:
        row_heights = [0.75, 0.25]
    elif n_rows == 3:
        row_heights = [0.62, 0.19, 0.19]
    else:
        row_heights = [0.55, 0.15, 0.15, 0.15]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.04,
    )

    # ---- Row 1: Price + overlays ----
    pre_sim = visible_history[visible_history.index < sim_start]
    in_sim = visible_history[visible_history.index >= sim_start]
    if len(pre_sim) > 0:
        fig.add_trace(go.Scatter(
            x=pre_sim.index, y=pre_sim["close"], name="History",
            mode="lines", line=dict(color=HISTORY_TINT, width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
    if len(in_sim) > 0:
        fig.add_trace(go.Scatter(
            x=in_sim.index, y=in_sim["close"], name="AAPL",
            mode="lines", line=dict(color=INK, width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # SMA overlay
    if st.session_state.show_sma and len(closes) >= 20:
        sma_series = closes.rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=sma_series.index, y=sma_series, name="SMA20",
            mode="lines", line=dict(color="#f59e0b", width=1.4),
            hovertemplate="SMA20 $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # BB overlay
    if st.session_state.show_bb and len(closes) >= 20:
        bb_mid = closes.rolling(20).mean()
        bb_std = closes.rolling(20).std()
        bb_up = bb_mid + 2 * bb_std
        bb_lo = bb_mid - 2 * bb_std
        fig.add_trace(go.Scatter(
            x=bb_up.index, y=bb_up, name="BB upper",
            mode="lines", line=dict(color="#94a3b8", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bb_lo.index, y=bb_lo, name="BB",
            mode="lines", line=dict(color="#94a3b8", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(148,163,184,0.08)",
            hovertemplate="BB lower $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # Virtual price (sim window only)
    if st.session_state.show_virtual:
        user_buy_by_day = {h["day"]: h["user_buy_usd"] for h in st.session_state.user_history}
        user_sell_by_day = {h["day"]: h["user_sell_usd"] for h in st.session_state.user_history}
        revealed = days[: day_idx + 1] if not is_done else days
        vx, vy = [], []
        prev_v = days[0]["real_open"]
        prev_r = days[0]["real_open"]
        for dd in revealed:
            real_drift = (dd["real_close"] - prev_r) / prev_r if prev_r > 0 else 0
            ub = user_buy_by_day.get(dd["date"], 0)
            us = user_sell_by_day.get(dd["date"], 0)
            buy_usd = dd["buy_volume_usd"] + ub
            sell_usd = dd["sell_volume_usd"] + us
            total_cap = sum(p["total"] for p in dd["portfolios_close"].values() if p.get("total", 0) > 0) or 1e9
            pressure = max(-1.0, min(1.0, (buy_usd - sell_usd) / total_cap))
            v = prev_v * (1 + real_drift) * (1 + sensitivity * pressure)
            if dd["real_close"] > 0:
                v *= 1 - ANCHOR_LAMBDA * (v / dd["real_close"] - 1)
            vx.append(dd["date"])
            vy.append(v)
            prev_v = v
            prev_r = dd["real_close"]
        fig.add_trace(go.Scatter(
            x=vx, y=vy, name="β virtual",
            mode="lines", line=dict(color="#3b82f6", width=1.5, dash="dot"),
            hovertemplate="β $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # TODAY'S OPEN marker — visually anchors the chart "as of now"
    if not is_done:
        fig.add_trace(go.Scatter(
            x=[cur_day["date"]], y=[cur_day["real_open"]],
            mode="markers", name="Today's open",
            marker=dict(size=13, symbol="diamond", color="#3b82f6",
                        line=dict(width=2, color="white")),
            hovertemplate=f"<b>{cur_day['date']} OPEN</b><br>$%{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    # User trade markers (on price row)
    buys = [(h["day"], h["fill_price"]) for h in st.session_state.user_history if h["user_buy_usd"] > 0]
    sells = [(h["day"], h["fill_price"]) for h in st.session_state.user_history if h["user_sell_usd"] > 0]
    if buys:
        fig.add_trace(go.Scatter(
            x=[b[0] for b in buys], y=[b[1] for b in buys],
            mode="markers", name="Buy",
            marker=dict(size=11, symbol="triangle-up", color=GREEN, line=dict(width=1.5, color="white")),
            hovertemplate="Buy %{x}<br>$%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if sells:
        fig.add_trace(go.Scatter(
            x=[s[0] for s in sells], y=[s[1] for s in sells],
            mode="markers", name="Sell",
            marker=dict(size=11, symbol="triangle-down", color=RED, line=dict(width=1.5, color="white")),
            hovertemplate="Sell %{x}<br>$%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ---- Sub-rows: Volume, RSI, MACD ----
    def _row_for(name):
        return panels.index(name) + 1 if name in panels else None

    if "volume" in panels:
        rv = _row_for("volume")
        vol = visible_history["volume"]
        # color volume bars by up/down day
        close_diff = visible_history["close"].diff().fillna(0)
        colors = [GREEN if d >= 0 else RED for d in close_diff]
        fig.add_trace(go.Bar(
            x=vol.index, y=vol, name="Volume",
            marker=dict(color=colors), opacity=0.55,
            hovertemplate="Volume %{y:,.0f}<extra></extra>",
            showlegend=False,
        ), row=rv, col=1)

    if "rsi" in panels and len(closes) >= 15:
        rr = _row_for("rsi")
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rsi_series = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
        fig.add_trace(go.Scatter(
            x=rsi_series.index, y=rsi_series, name="RSI(14)",
            mode="lines", line=dict(color="#7c3aed", width=1.6),
            hovertemplate="RSI %{y:.1f}<extra></extra>",
            showlegend=False,
        ), row=rr, col=1)
        # 30 / 70 zones
        fig.add_hline(y=70, line=dict(color="#fca5a5", width=1, dash="dot"), row=rr, col=1)
        fig.add_hline(y=30, line=dict(color="#86efac", width=1, dash="dot"), row=rr, col=1)
        fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=rr, col=1)

    if "macd" in panels and len(closes) >= 26:
        rm = _row_for("macd")
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal
        # Histogram bars
        hist_colors = [GREEN if v >= 0 else RED for v in hist]
        fig.add_trace(go.Bar(
            x=hist.index, y=hist, name="MACD hist",
            marker=dict(color=hist_colors), opacity=0.55,
            hovertemplate="hist %{y:+.2f}<extra></extra>",
            showlegend=False,
        ), row=rm, col=1)
        # MACD + signal lines
        fig.add_trace(go.Scatter(
            x=macd_line.index, y=macd_line, name="MACD",
            mode="lines", line=dict(color="#0a0a0a", width=1.4),
            hovertemplate="MACD %{y:+.2f}<extra></extra>",
            showlegend=False,
        ), row=rm, col=1)
        fig.add_trace(go.Scatter(
            x=signal.index, y=signal, name="Signal",
            mode="lines", line=dict(color="#f59e0b", width=1.2, dash="dot"),
            hovertemplate="Signal %{y:+.2f}<extra></extra>",
            showlegend=False,
        ), row=rm, col=1)

    # Sim-start marker — vertical shape that spans all rows
    fig.add_shape(
        type="line",
        x0=sim_start.strftime("%Y-%m-%d"), x1=sim_start.strftime("%Y-%m-%d"),
        y0=0, y1=1, yref="paper",
        line=dict(color="#cbd5e1", width=1, dash="dash"),
    )

    # What-if scenario projection line + agents' projected reactions
    if st.session_state.active_scenario and not is_done:
        proj_dates, proj_prices, proj_reactions = project_scenario(
            st.session_state.active_scenario, day_idx, days, full_df, sensitivity, n_days=5,
        )
        if proj_dates:
            scen = SCENARIOS[st.session_state.active_scenario]
            color = "#ea580c" if scen["dir"] == "crash" else "#0891b2"
            # Connect from today's price to first projection point
            connect_dates = [pd.to_datetime(cur_day["date"])] + list(proj_dates)
            connect_prices = [cur_day["real_open"]] + list(proj_prices)
            fig.add_trace(go.Scatter(
                x=connect_dates, y=connect_prices,
                name=f"If {st.session_state.active_scenario}",
                mode="lines+markers",
                line=dict(color=color, width=2, dash="dashdot"),
                marker=dict(size=7, symbol="circle-open"),
                hovertemplate="What-if %{x|%b %d}<br>$%{y:.2f}<extra></extra>",
            ), row=1, col=1)
            # Shock marker at projection start
            fig.add_annotation(
                x=proj_dates[0].strftime("%Y-%m-%d"), y=proj_prices[0],
                text=f"⚡ {st.session_state.active_scenario}<br>{scen['shock']*100:+.0f}% shock",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=color, font=dict(size=10, color=color),
                xanchor="left", ax=20, ay=-30,
                bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderpad=4,
            )

    # X axis range applied to all rows
    fig.update_xaxes(
        range=[x_from.strftime("%Y-%m-%d"), x_to.strftime("%Y-%m-%d")],
        showgrid=False, showline=False, zeroline=False,
        tickfont=dict(size=10, color=MUTED),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#fafafa", showline=False, zeroline=False,
        tickfont=dict(size=10, color=MUTED), title=None,
    )

    # AUTO-FIT y range for price row using only values visible in current x window
    y_vals_in_view: list[float] = []
    mask = (visible_history.index >= x_from) & (visible_history.index <= x_to)
    y_vals_in_view.extend(visible_history.loc[mask, "close"].tolist())
    # Include BB bands if shown so they don't clip
    if st.session_state.show_bb and len(closes) >= 20:
        bb_mid_s = closes.rolling(20).mean()
        bb_std_s = closes.rolling(20).std()
        bb_up_s = (bb_mid_s + 2 * bb_std_s)
        bb_lo_s = (bb_mid_s - 2 * bb_std_s)
        bb_mask = (bb_up_s.index >= x_from) & (bb_up_s.index <= x_to)
        y_vals_in_view.extend(bb_up_s.loc[bb_mask].dropna().tolist())
        y_vals_in_view.extend(bb_lo_s.loc[bb_mask].dropna().tolist())
    # Include scenario projection prices if active
    if st.session_state.active_scenario and not is_done:
        _proj_dates, _proj_prices, _ = project_scenario(
            st.session_state.active_scenario, day_idx, days, full_df, sensitivity, n_days=5,
        )
        for d_, p_ in zip(_proj_dates, _proj_prices):
            if x_from <= d_ <= x_to:
                y_vals_in_view.append(p_)
    if y_vals_in_view:
        y_min = min(y_vals_in_view)
        y_max = max(y_vals_in_view)
        pad = max(0.5, (y_max - y_min) * 0.08)
        fig.update_yaxes(range=[y_min - pad, y_max + pad], row=1, col=1)

    # Per-row Y-axis labels for sub-rows
    for panel_name in panels[1:]:
        rr = _row_for(panel_name)
        label = {"volume": "Vol", "rsi": "RSI", "macd": "MACD"}[panel_name]
        fig.update_yaxes(title_text=label, title_font=dict(size=10, color=MUTED), row=rr, col=1)

    chart_height = 360 + 110 * (n_rows - 1)
    fig.update_layout(
        height=chart_height, margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        bargap=0.1,
        hoverlabel=dict(bgcolor="white", bordercolor="#e5e5e5", font=dict(size=12)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Indicator strip (compact pills below the chart)
    if st.session_state.show_indicators_panel and inds["sma20"] is not None:
        macd_color = GREEN if (inds["macd_hist"] or 0) > 0 else RED
        rsi_color = (RED if inds["rsi"] and inds["rsi"] > 70 else
                     (GREEN if inds["rsi"] and inds["rsi"] < 30 else INK))
        bb_pos = ""
        if inds["bb_upper"] and inds["bb_lower"] and inds["bb_upper"] != inds["bb_lower"]:
            pos = (cur_price - inds["bb_lower"]) / (inds["bb_upper"] - inds["bb_lower"])
            bb_pos = f"{pos*100:.0f}% band"
        pills_html = (
            f"<div style='margin-top:0.4rem'>"
            f"<span class='indicator-pill'><span class='lbl'>RSI(14)</span> <span class='val' style='color:{rsi_color}'>{inds['rsi']:.0f}</span></span>"
            f"<span class='indicator-pill'><span class='lbl'>MACD hist</span> <span class='val' style='color:{macd_color}'>{inds['macd_hist']:+.2f}</span> <span style='color:{MUTED};font-size:0.75rem'>{inds['macd_dir']}</span></span>"
            f"<span class='indicator-pill'><span class='lbl'>SMA20</span> <span class='val'>${inds['sma20']:.2f}</span></span>"
            f"<span class='indicator-pill'><span class='lbl'>5d mom</span> <span class='val' style='color:{GREEN if (inds['mom5'] or 0)>0 else RED}'>{inds['mom5']:+.2f}%</span></span>"
            f"<span class='indicator-pill'><span class='lbl'>Ann vol</span> <span class='val'>{inds['vol_ann']:.1f}%</span></span>"
            + (f"<span class='indicator-pill'><span class='lbl'>BB</span> <span class='val'>{bb_pos}</span></span>" if bb_pos else "")
            + f"</div>"
        )
        st.markdown(pills_html, unsafe_allow_html=True)


with main_r:
    st.markdown("<div class='section-label'>Today's headlines</div>", unsafe_allow_html=True)
    news = load_news_for(cur_day["date"], ticker) if not is_done else []
    if news:
        for h in news[:4]:
            st.markdown(f"<div class='news-row'>{h}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='news-row' style='color:{FAINT}'>Quiet trading day</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Voices today · all 11 agents</div>", unsafe_allow_html=True)
    if not is_done:
        outputs = cur_day["agent_outputs"]
        ranked = sorted(
            outputs.items(),
            key=lambda kv: -(kv[1]["state"]["public_statement"].get("stated_conviction", 0) or 0),
        )
        revealed_today = st.session_state.peek_revealed.get(cur_day["date"], [])
        voice_html_parts = []
        for aid, out in ranked:
            name = AGENT_NAMES.get(aid, aid)
            role = AGENT_ROLES.get(aid, "")
            ps = out["state"]["public_statement"]
            pb = out["state"].get("private_belief", {})
            pa = out["state"].get("personal_action", {})
            lean = ps.get("stated_lean", "—")
            conv = ps.get("stated_conviction", 0)
            lean_color = {"long": GREEN, "short": RED}.get(lean, MUTED)
            narrative = (ps.get("narrative", "") or "")[:240]
            act = pa.get("action_type", "hold")
            act_label = {"buy_strong": "Buy max", "buy_lite": "Buy",
                         "hold": "Hold", "sell_lite": "Sell", "sell_strong": "Sell all"}.get(act, act)
            act_color = (GREEN if "buy" in act else (RED if "sell" in act else MUTED))
            priv_lean = pb.get("lean", "—")
            hypocrisy = "🎭" if priv_lean != lean and lean != "—" and priv_lean != "—" else ""

            block = (
                f"<div class='voice-block'>"
                f"<div style='display:flex;align-items:baseline;gap:0.5rem;flex-wrap:wrap'>"
                f"<span class='voice-name'>{name}</span>"
                f"<span style='font-size:0.72rem;color:{FAINT};font-weight:500'>{role}</span>"
                f"<span>{hypocrisy}</span>"
                f"<span class='voice-meta' style='color:{lean_color}'>{lean} · {conv:.0%}</span>"
                f"<span class='voice-meta' style='color:{act_color};margin-left:auto'>→ {act_label}</span>"
                f"</div>"
                f"<div class='voice-quote'>{narrative}</div>"
            )
            if aid in revealed_today:
                priv_color = {"long": GREEN, "short": RED}.get(priv_lean, MUTED)
                thesis = (pb.get("actual_thesis", "") or "")[:280]
                block += (
                    f"<div style='background:#fef3c7;border-left:3px solid #d97706;"
                    f"padding:0.5rem 0.7rem;margin:0.4rem 0 0 0;border-radius:4px'>"
                    f"<div style='font-size:0.7rem;font-weight:600;color:#92400e;"
                    f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem'>"
                    f"👁 Private thoughts revealed</div>"
                    f"<div style='font-size:0.85rem;color:#0a0a0a;line-height:1.5'>"
                    f"<strong style='color:{priv_color}'>{priv_lean}</strong> "
                    f"<span style='color:{MUTED}'>(conviction {pb.get('conviction', 0):.0%})</span> "
                    f"— <em>{thesis}</em>"
                    f"</div></div>"
                )
            block += "</div>"
            voice_html_parts.append(block)

        # Voices scroll container — height syncs to chart_height (chart_height set inside main_l)
        v_height = max(350, chart_height - 60)
        scroll_html = (
            f"<div style='height:{v_height}px;overflow-y:auto;padding-right:0.5rem'>"
            + "".join(voice_html_parts)
            + "</div>"
        )
        st.markdown(scroll_html, unsafe_allow_html=True)

# Agent reactions banner under chart when scenario active
if st.session_state.active_scenario and not is_done:
    scen = SCENARIOS[st.session_state.active_scenario]
    color = "#ea580c" if scen["dir"] == "crash" else "#0891b2"
    bg = "#fff7ed" if scen["dir"] == "crash" else "#ecfeff"
    _, _, proj_reactions = project_scenario(
        st.session_state.active_scenario, day_idx, days, full_df, sensitivity, n_days=5,
    )
    rows_html = []
    for r in proj_reactions:
        act = r["action"]; sz = r["size"]
        lbl = ACT_LABEL.get(act, act)
        act_color = GREEN if "buy" in act else (RED if "sell" in act else MUTED)
        sz_str = f" ({sz*100:.0f}%)" if act not in ("hold",) and "strong" not in act else (" (max)" if "strong" in act else "")
        belief_color = {"long": GREEN, "short": RED}.get(r["priv_lean"], MUTED)
        role = AGENT_ROLES.get(r["aid"], "")
        rows_html.append(
            f"<div style='display:flex;align-items:baseline;gap:0.6rem;padding:0.3rem 0;font-size:0.86rem;border-top:1px solid #f1f5f9'>"
            f"<div style='min-width:170px'>"
            f"<div style='font-weight:600;color:{INK}'>{r['name']}</div>"
            f"<div style='font-size:0.72rem;color:{FAINT};font-weight:500'>{role}</div>"
            f"</div>"
            f"<span style='font-size:0.74rem;color:{MUTED}'>today: <span style='color:{belief_color};font-weight:600'>{r['priv_lean']}</span> {r['conv']:.0%}</span>"
            f"<span style='color:{act_color};font-weight:600;margin-left:auto'>{lbl}{sz_str}</span>"
            f"<span style='font-size:0.78rem;color:{MUTED};font-style:italic;min-width:200px;max-width:280px;text-align:right'>{r['reason']}</span>"
            f"</div>"
        )
    st.markdown(
        f"<div style='background:{bg};border:1px solid {color};border-left:3px solid {color};"
        f"border-radius:6px;padding:0.8rem 1.1rem;margin:0.4rem 0 0.6rem 0'>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.5rem'>"
        f"<div style='font-size:0.78rem;font-weight:700;color:{color};"
        f"text-transform:uppercase;letter-spacing:0.06em'>"
        f"🌐 If {st.session_state.active_scenario}</div>"
        f"<div style='font-size:0.78rem;color:{MUTED}'>live · uses each agent's belief today</div>"
        f"</div>"
        f"<div style='font-size:0.8rem;color:{MUTED};margin-bottom:0.4rem'>{scen['blurb']} · initial shock {scen['shock']*100:+.0f}%</div>"
        f"<div>{''.join(rows_html)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ----------------------------------------------------------------------------
# POWER TOOLS ROW — peek / what-if scenario / skip
# ----------------------------------------------------------------------------
if not is_done:
    tool_l, tool_m, tool_r = st.columns([2, 2.2, 1.4], gap="medium")
    today_str = cur_day["date"]
    revealed_today = st.session_state.peek_revealed.get(today_str, [])
    peeks_left_today = PEEKS_PER_DAY - len(revealed_today)

    # PEEK ----
    with tool_l:
        not_yet = [aid for aid in cur_day["agent_outputs"] if aid not in revealed_today]
        peek_cols = st.columns([1.6, 1])
        with peek_cols[0]:
            if not_yet and peeks_left_today > 0:
                # Key changes after each reveal so Streamlit refreshes selectbox state
                # (prevents the "selected one, peeked another" stale-state bug).
                peek_target = st.selectbox(
                    f"👁 Peek thoughts ({peeks_left_today}/{PEEKS_PER_DAY})",
                    options=not_yet,
                    format_func=lambda x: f"{AGENT_NAMES.get(x, x)} · {AGENT_ROLES.get(x, '')}",
                    key=f"peek_sel_{day_idx}_{len(revealed_today)}",
                )
            else:
                peek_target = None
                msg = "All revealed today" if not not_yet else "No peeks left today"
                st.markdown(
                    f"<div style='font-size:0.72rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.3rem'>👁 Peek ({peeks_left_today}/{PEEKS_PER_DAY})</div>"
                    f"<div style='font-size:0.85rem;color:{FAINT};padding:0.4rem 0'>{msg}</div>",
                    unsafe_allow_html=True,
                )
        with peek_cols[1]:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            if st.button("Reveal", use_container_width=True,
                         disabled=(peek_target is None),
                         key=f"peek_btn_{day_idx}"):
                st.session_state.peek_revealed.setdefault(today_str, []).append(peek_target)
                st.rerun()

    # WHAT-IF SCENARIO ----
    with tool_m:
        scen_cols = st.columns([1.7, 1])
        with scen_cols[0]:
            scenario_key = st.selectbox(
                f"🌐 What-if scenario · agents react live",
                options=list(SCENARIOS.keys()),
                key=f"scen_sel_{day_idx}",
            )
        with scen_cols[1]:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            btn_cols = st.columns(2)
            with btn_cols[0]:
                if st.button("Project", use_container_width=True, key=f"scen_proj_{day_idx}"):
                    st.session_state.active_scenario = scenario_key
                    st.rerun()
            with btn_cols[1]:
                if st.button("Clear", use_container_width=True, key=f"scen_clr_{day_idx}",
                             disabled=(st.session_state.active_scenario is None)):
                    st.session_state.active_scenario = None
                    st.rerun()

    # SKIP ----
    with tool_r:
        skip_cols = st.columns([1, 1.2])
        with skip_cols[0]:
            skip_n = st.selectbox("⏩ Skip", options=[3, 5, 10],
                                  format_func=lambda v: f"{v} days",
                                  key=f"skip_sel_{day_idx}")
        with skip_cols[1]:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            if st.button("Skip", use_container_width=True, key=f"skip_btn_{day_idx}"):
                for _ in range(min(skip_n, TOTAL_DAYS - st.session_state.day_idx)):
                    skip_day = days[st.session_state.day_idx]
                    st.session_state.user_history.append({
                        "day": skip_day["date"],
                        "action": "hold",
                        "size": 0,
                        "shares_traded": 0,
                        "fill_price": skip_day["real_open"],
                        "user_buy_usd": 0,
                        "user_sell_usd": 0,
                        "cash_after": st.session_state.user_cash,
                        "shares_after": st.session_state.user_shares,
                    })
                    st.session_state.day_idx += 1
                st.session_state.pending_action = "hold"
                st.session_state.active_scenario = None
                st.rerun()

    st.markdown("---")


# ----------------------------------------------------------------------------
# ACTION AREA — 3 clear actions + live preview
# ----------------------------------------------------------------------------
if not is_done:
    pending_action = st.session_state.get("pending_action", "hold")
    pending_size = st.session_state.get("pending_size", 0.3)
    fill = cur_day["real_open"] * user_price_mult
    user_cash = st.session_state.user_cash
    user_shares = st.session_state.user_shares
    max_buy_shares = int(user_cash / fill) if fill > 0 else 0
    max_buy_value = max_buy_shares * fill

    # Section header with strong visual anchor
    st.markdown(
        f"<div style='background:linear-gradient(180deg,#fafafa 0%,#ffffff 100%);"
        f"border:1px solid #e5e5e5;border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.6rem'>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.8rem'>"
        f"<div style='font-size:0.85rem;font-weight:700;color:{INK};"
        f"text-transform:uppercase;letter-spacing:0.08em'>Your move · day {day_idx+1}</div>"
        f"<div style='font-size:0.8rem;color:{MUTED}'>Trading at open: <strong style='color:{INK}'>${fill:.2f}</strong></div>"
        f"</div>"
        f"<div style='display:flex;gap:1.6rem;font-size:0.85rem;color:{MUTED}'>"
        f"<span>Cash: <strong style='color:{INK}'>${user_cash/1e6:,.2f}M</strong></span>"
        f"<span>Shares: <strong style='color:{INK}'>{user_shares:+,}</strong></span>"
        f"<span>Max buy: <strong style='color:{INK}'>{max_buy_shares:,} sh</strong> (${max_buy_value/1e6:,.2f}M)</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Three big action buttons
    a_cols = st.columns(3)
    for i, (act_key, label) in enumerate([("buy_lite", "🟢  BUY"), ("hold", "⏸  HOLD"), ("sell_lite", "🔴  SELL")]):
        with a_cols[i]:
            if st.button(label,
                         type=("primary" if pending_action == act_key else "secondary"),
                         use_container_width=True, key=f"act_{act_key}"):
                st.session_state.pending_action = act_key
                st.rerun()

    # Size + preview + confirm
    if pending_action != "hold":
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        sl_col, qk_col = st.columns([3, 2])
        with sl_col:
            new_size = st.slider(
                "Size (% of available)",
                0.0, 1.0, pending_size, 0.05,
                format="%.0f%%",
                key=f"sz_{day_idx}",
            )
            st.session_state.pending_size = new_size
            pending_size = new_size
        with qk_col:
            st.markdown("<div style='font-size:0.78rem;color:#737373;margin-bottom:0.3rem'>Quick</div>", unsafe_allow_html=True)
            q_cols = st.columns(4)
            for j, q in enumerate([0.25, 0.5, 0.75, 1.0]):
                with q_cols[j]:
                    if st.button(f"{int(q*100)}%", use_container_width=True, key=f"qk_{q}_{day_idx}"):
                        st.session_state.pending_size = q
                        st.rerun()

        # Live preview
        if pending_action == "buy_lite":
            if user_cash <= 0:
                preview_html = f"<div style='color:{RED}'>You have no cash to deploy.</div>"
                preview_color = MUTED
            else:
                will_deploy = user_cash * pending_size
                will_buy = int(will_deploy / fill) if fill > 0 else 0
                actual_cost = will_buy * fill
                cash_after = user_cash - actual_cost
                shares_after = user_shares + will_buy
                preview_html = (
                    f"<div style='font-size:0.95rem;line-height:1.6;color:{INK}'>"
                    f"Buy <strong style='color:{GREEN}'>{will_buy:,} shares</strong> at ${fill:.2f} · "
                    f"<strong style='color:{INK}'>${actual_cost/1e6:,.2f}M</strong> deployed"
                    f"</div>"
                    f"<div style='font-size:0.8rem;color:{MUTED};margin-top:0.25rem'>"
                    f"Cash after: ${cash_after/1e6:,.2f}M · Shares after: {shares_after:+,}"
                    f"</div>"
                )
                preview_color = GREEN
        else:  # sell_lite
            if user_shares <= 0:
                preview_html = (
                    f"<div style='color:{RED};font-size:0.95rem'>"
                    f"You have no long position to sell. (Short selling not enabled in this sim.)"
                    f"</div>"
                )
                preview_color = MUTED
            else:
                will_sell = user_shares if pending_size >= 0.99 else max(1, int(user_shares * pending_size))
                will_sell = min(will_sell, user_shares)
                proceeds = will_sell * fill
                cash_after = user_cash + proceeds
                shares_after = user_shares - will_sell
                preview_html = (
                    f"<div style='font-size:0.95rem;line-height:1.6;color:{INK}'>"
                    f"Sell <strong style='color:{RED}'>{will_sell:,} shares</strong> at ${fill:.2f} · "
                    f"receive <strong style='color:{INK}'>${proceeds/1e6:,.2f}M</strong>"
                    f"</div>"
                    f"<div style='font-size:0.8rem;color:{MUTED};margin-top:0.25rem'>"
                    f"Cash after: ${cash_after/1e6:,.2f}M · Shares after: {shares_after:+,}"
                    f"</div>"
                )
                preview_color = RED
        st.markdown(
            f"<div style='background:#fafafa;border-left:3px solid {preview_color};"
            f"padding:0.7rem 1rem;margin:0.8rem 0 0.6rem 0;border-radius:4px'>"
            f"<div style='font-size:0.68rem;font-weight:600;color:{MUTED};"
            f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem'>Trade preview</div>"
            f"{preview_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:#fafafa;border-left:3px solid {MUTED};"
            f"padding:0.7rem 1rem;margin:0.6rem 0;border-radius:4px;color:{MUTED};font-size:0.9rem'>"
            f"You'll hold this day — no trade executes."
            f"</div>",
            unsafe_allow_html=True,
        )

    # Confirm button — big and obvious
    cc_cols = st.columns([1, 2, 1])
    with cc_cols[1]:
        next_clicked = st.button("Confirm  &  advance to next day  →",
                                  type="primary", use_container_width=True, key="confirm_advance")

    if next_clicked:
        action = pending_action
        user_buy_usd = 0.0
        user_sell_usd = 0.0
        shares_traded = 0
        effective_size = pending_size if action in ("buy_lite", "sell_lite") else 0.0

        if action == "buy_lite" and st.session_state.user_cash > 0 and effective_size > 0:
            cash_to_use = st.session_state.user_cash * effective_size
            shares_to_buy = int(cash_to_use / fill) if fill > 0 else 0
            if shares_to_buy > 0:
                cost = shares_to_buy * fill
                st.session_state.user_cash -= cost
                st.session_state.user_shares += shares_to_buy
                user_buy_usd = cost
                shares_traded = shares_to_buy
        elif action == "sell_lite" and st.session_state.user_shares > 0 and effective_size > 0:
            shares_to_sell = (st.session_state.user_shares if effective_size >= 0.99
                              else max(1, int(st.session_state.user_shares * effective_size)))
            shares_to_sell = min(shares_to_sell, st.session_state.user_shares)
            proceeds = shares_to_sell * fill
            st.session_state.user_cash += proceeds
            st.session_state.user_shares -= shares_to_sell
            user_sell_usd = proceeds
            shares_traded = shares_to_sell

        st.session_state.user_history.append({
            "day": cur_day["date"],
            "action": action,
            "size": effective_size,
            "shares_traded": shares_traded,
            "fill_price": fill,
            "user_buy_usd": user_buy_usd,
            "user_sell_usd": user_sell_usd,
            "cash_after": st.session_state.user_cash,
            "shares_after": st.session_state.user_shares,
        })
        st.session_state.pending_action = "hold"
        st.session_state.pending_size = 0.3
        st.session_state.active_scenario = None
        st.session_state.day_idx += 1
        st.rerun()
else:
    pass  # finale rendered below

st.markdown("---")


# ----------------------------------------------------------------------------
# STANDINGS / FINALE
# ----------------------------------------------------------------------------
cur_day_idx = min(day_idx, TOTAL_DAYS - 1)

# Build rows once
rows = []
for a in ALL_AGENTS:
    if a.capital <= 0:
        continue
    p = days[cur_day_idx]["portfolios_close"].get(a.id, {})
    total = p.get("total", 0)
    pnl = (total - a.capital) / a.capital * 100 if a.capital > 0 else 0
    name = AGENT_NAMES.get(a.id, a.id)
    rows.append({"name": name, "pnl": pnl, "is_user": False})
rows.append({"name": "You", "pnl": pnl_pct, "is_user": True})
rows.append({"name": "Buy & Hold", "pnl": bh_pnl, "is_user": False})
rows.sort(key=lambda r: -r["pnl"])

if is_done:
    # === BIG FINALE ===
    agents_beaten = sum(1 for r in rows if not r["is_user"] and r["name"] != "Buy & Hold" and r["pnl"] < pnl_pct)
    total_agents = sum(1 for r in rows if not r["is_user"] and r["name"] != "Buy & Hold")
    user_rank = next(i for i, r in enumerate(rows) if r["is_user"]) + 1

    pnl_color = GREEN if pnl_pct > 0 else RED
    headline = "🏆 You crushed it." if user_rank <= 3 else \
               ("👏 Solid run." if user_rank <= total_agents // 2 + 1 else \
                "📉 Tough market.")

    st.markdown(
        f"<div style='text-align:center;padding:1.5rem 0 0.5rem 0'>"
        f"<div style='font-size:0.78rem;font-weight:600;color:{MUTED};"
        f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>🏁 Final results</div>"
        f"<div style='font-size:1.5rem;font-weight:600;color:#0a0a0a;margin-bottom:0.6rem'>{headline}</div>"
        f"<div style='font-size:3.2rem;font-weight:800;color:{pnl_color};line-height:1;font-variant-numeric:tabular-nums'>{pnl_pct:+.2f}%</div>"
        f"<div style='font-size:0.95rem;color:{MUTED};margin-top:0.3rem'>Final portfolio: ${user_total/1e6:,.2f}M · started with ${st.session_state.user_initial/1e6:,.2f}M</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    big_stats = st.columns(4)
    with big_stats[0]:
        st.metric("Beat", f"{agents_beaten} / {total_agents}", "AI agents")
    with big_stats[1]:
        st.metric("Rank", f"#{user_rank} / {len(rows)}", "incl. B&H")
    with big_stats[2]:
        st.metric("vs Buy & Hold", f"{alpha:+.2f}%", "alpha")
    with big_stats[3]:
        n_trades = sum(1 for h in st.session_state.user_history if h["shares_traded"] > 0)
        st.metric("Trades", f"{n_trades}", f"{32-n_trades} hold days")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Horizontal bar chart — all agents + user + B&H
    bar_rows = list(rows)  # already sorted desc
    names = [r["name"] for r in bar_rows]
    pnls = [r["pnl"] for r in bar_rows]
    colors = [
        ("#f59e0b" if r["is_user"] else
         ("#94a3b8" if r["name"] == "Buy & Hold" else
          (GREEN if r["pnl"] > 0 else RED)))
        for r in bar_rows
    ]
    fig_bar = go.Figure(go.Bar(
        y=names, x=pnls, orientation="h",
        marker=dict(color=colors),
        text=[f"{p:+.2f}%" for p in pnls],
        textposition="outside",
        textfont=dict(size=11, color=INK),
        hovertemplate="%{y}<br>%{x:+.2f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        height=40 + 32 * len(bar_rows),
        margin=dict(l=10, r=80, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color=INK)),
        xaxis=dict(showgrid=True, gridcolor="#fafafa", zeroline=True,
                   zerolinecolor="#e5e5e5", zerolinewidth=1,
                   tickfont=dict(size=10, color=MUTED), title=None,
                   ticksuffix="%"),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Peek + scenario usage stats
    days_with_peeks = sum(1 for d in st.session_state.peek_revealed.values() if d)
    total_peeks = sum(len(v) for v in st.session_state.peek_revealed.values())
    st.markdown(
        f"<div style='text-align:center;color:{MUTED};font-size:0.85rem;padding:0.5rem 0'>"
        f"You used <strong>{total_peeks} peeks</strong> across <strong>{days_with_peeks} days</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Replay button
    replay_cols = st.columns([1, 1, 1])
    with replay_cols[1]:
        if st.button("Play again ↩", type="primary", use_container_width=True):
            reset_state(st.session_state.user_initial)
            st.rerun()

else:
    # === LIVE STANDINGS (compact list) ===
    st.markdown("<div class='section-label'>Standings</div>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)
    mid = (len(rows) + 1) // 2
    for col, slice_ in [(left_col, rows[:mid]), (right_col, rows[mid:])]:
        with col:
            for r in slice_:
                i = rows.index(r)
                color = GREEN if r["pnl"] > 0 else (RED if r["pnl"] < 0 else MUTED)
                cls = "standings-row standings-you" if r["is_user"] else "standings-row"
                st.markdown(
                    f"<div class='{cls}'>"
                    f"<span class='standings-rank'>#{i+1}</span>"
                    f"<span class='standings-who'>{r['name']}</span>"
                    f"<span class='standings-pnl' style='color:{color}'>{r['pnl']:+.2f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
