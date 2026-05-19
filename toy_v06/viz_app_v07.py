"""v0.7 Streamlit visualization — 30 days × 6 agents × memory + reputation + β virtual price."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents import ALL_AGENTS, get_agent

# ============================================================================
DATA_PATH = Path(__file__).parent / "results" / "v07_demo_latest.json"

AGENT_VIZ = {
    "super_influencer_001":      {"color": "#9b59b6", "emoji": "🎙️", "short": "Super"},
    "pod_pm_001":                {"color": "#3498db", "emoji": "💼", "short": "Pod PM"},
    "activist_short_001":        {"color": "#e74c3c", "emoji": "🎯", "short": "Activist"},
    "sell_side_001":             {"color": "#f39c12", "emoji": "📊", "short": "Sell-Side"},
    "cta_forced_001":            {"color": "#7f8c8d", "emoji": "🤖", "short": "CTA"},
    "retail_fomo_001":           {"color": "#2ecc71", "emoji": "🦍", "short": "Retail"},
    # NEW 5 agents
    "permabull_001":             {"color": "#16a085", "emoji": "🐂", "short": "Permabull"},
    "day_trader_001":            {"color": "#1abc9c", "emoji": "⚡", "short": "Day Trader"},
    "economist_macro_001":       {"color": "#34495e", "emoji": "🏛️", "short": "Bernanke"},
    "economist_political_001":   {"color": "#8e44ad", "emoji": "📰", "short": "Krugman"},
    "economist_trader_001":      {"color": "#2c3e50", "emoji": "🎓", "short": "Druck"},
}

LEAN_COLOR = {"long": "#26c281", "short": "#e74c3c", "neutral": "#95a5a6", "flat": "#95a5a6",
              "deterministic": "#7f8c8d", "n/a": "#7f8c8d"}

ACTION_NUM = {"buy_strong": 2, "buy_lite": 1, "hold": 0, "sell_lite": -1, "sell_strong": -2}
ACTION_EMOJI = {"buy_strong": "🟢🟢", "buy_lite": "🟢", "hold": "⚪", "sell_lite": "🔴", "sell_strong": "🔴🔴"}


def short_label(aid):
    v = AGENT_VIZ.get(aid, {"emoji": "?", "short": aid})
    return f"{v['emoji']} {v['short']}"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    with open(DATA_PATH) as f:
        return json.load(f)


# ============================================================================
st.set_page_config(page_title="Toy v0.7 — Memory + Reputation", layout="wide", page_icon="🧠")
st.markdown("""<style>.block-container { padding-top: 1.5rem; }</style>""", unsafe_allow_html=True)

data = load_data()
if data is None:
    st.title("⏳ Waiting for simulation...")
    st.info(f"Run `python run_v07_demo.py` to generate `{DATA_PATH.name}`. This page auto-loads when ready.")
    st.button("🔄 Reload", on_click=lambda: st.cache_data.clear())
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("🧠 Toy v0.7")
    st.caption(f"{len(data['days'])} days · {data['ticker']} · ${data['total_cost_usd']:.2f} LLM spent")

    day_idx = st.slider("📅 Day", 0, len(data["days"]) - 1, len(data["days"]) - 1)
    day = data["days"][day_idx]

    st.metric("Date", day["date"])
    st.metric(f"Real {data['ticker']}",
              f"${day['real_close']:.2f}",
              f"{((day['real_close']-day['real_open'])/day['real_open']*100):+.2f}%")
    st.metric("Sim Virtual",
              f"${day['virtual_close']:.2f}",
              f"{((day['virtual_close']-day['real_close'])/day['real_close']*100):+.2f}% vs real")

    st.metric("Net Pressure", f"{day['net_pressure']:+.4f}")
    if day.get("reputation_update"):
        st.success("📊 Reputation updated today")

    st.divider()

    # Today's invalidations
    invals = day.get("invalidations_today", [])
    if invals:
        st.markdown(f"### ⚰️ Today: {len(invals)} predictions invalidated")
        for inv in invals[:5]:
            st.caption(f"❌ **{AGENT_VIZ.get(inv['owner'],{'short':inv['owner']})['short']}**'s belief about *{inv['source']}*'s {inv['stated_lean']} call (actual {inv['actual_change_pct']:+.1f}%)")
    else:
        st.caption("No invalidations today.")

    st.divider()
    st.markdown("### 🔍 Agent Inspector")
    selected_aid = st.selectbox(
        "Pick agent →",
        options=[a.id for a in ALL_AGENTS],
        format_func=short_label,
    )

# ============================================================================
# HERO
# ============================================================================
hero = st.columns([2.5, 1, 1, 1, 1])
hero[0].markdown("## 🧠 Toy v0.7 — Memory + Dynamic Reputation + β virtual price")
hero[0].caption(f"Day {day_idx+1}/{len(data['days'])} · Sensitivity β={data['sensitivity']}")

# Cumulative real return vs spy approx
first_real = data["days"][0]["real_close"]
real_cumret = (day["real_close"] - first_real) / first_real * 100
sim_cumret = (day["virtual_close"] - first_real) / first_real * 100

hero[1].metric("Real cum %", f"{real_cumret:+.2f}%")
hero[2].metric("Sim cum %", f"{sim_cumret:+.2f}%", f"{sim_cumret-real_cumret:+.2f} vs real")

n_inval = sum(len(d.get("invalidations_today", [])) for d in data["days"][:day_idx + 1])
hero[3].metric("⚰️ Invalidations cum", n_inval)

# Reputation multipliers spread
mults = day.get("influence_multipliers_after", {})
mult_spread = max(mults.values()) - min(mults.values()) if mults else 0
hero[4].metric("Influence Δ", f"{mult_spread:.2f}",
               help="Max - min reputation multiplier today")

st.divider()

# ============================================================================
# TABS
# ============================================================================
t_price, t_ternus, t_timeline, t_deception, t_asymmetry, t_reputation, t_memory, t_pnl, t_detail = st.tabs([
    "🔮 Real vs Sim Price",
    "📍 Case Study: Ternus Event",
    "📅 Daily Action Timeline",
    "🎭 Deception Detection",
    "🔓 Info Asymmetry",
    "📈 Reputation Evolution",
    "🧠 Memory Inspector",
    "💰 Portfolios",
    "🔍 Agent Detail",
])

# ----------------------- Tab 1: Real vs Sim Price (3 modes) -----------------------
with t_price:
    dates = [d["date"] for d in data["days"]]
    real = [d["real_close"] for d in data["days"]]
    beta_sim = [d["virtual_close"] for d in data["days"]]  # β = stored
    pressure = [d["net_pressure"] for d in data["days"]]
    first_real = real[0]
    sensitivity = data["sensitivity"]

    # Compute α (closed/pure agent-driven) — ignores real entirely
    # α[0] = first_real; α[t+1] = α[t] × (1 + sens × pressure[t+1])
    alpha_sim = [first_real]
    for i, p in enumerate(pressure):
        if i == 0:
            continue
        alpha_sim.append(alpha_sim[-1] * (1 + sensitivity * p))
    # Ensure same length
    while len(alpha_sim) < len(dates):
        alpha_sim.append(alpha_sim[-1])

    # Compute γ (dual layer / real-anchored auxiliary) — recomputed each day from real
    gamma_sim = []
    for i, d in enumerate(data["days"]):
        gamma_sim.append(d["real_close"] * (1 + sensitivity * pressure[i]))

    # Sensitivity slider for live recompute (β & α only — γ uses any sensitivity per-day)
    st.markdown("#### Sensitivity (β, α): how much agents drive vs real follows")
    sens_override = st.slider(
        "Live recompute with this sensitivity (does not change stored β):",
        0.0, 1.0, sensitivity, 0.05,
        help="Stored β uses {:.2f}. Slide to recompute α & β live for comparison.".format(sensitivity),
    )
    if abs(sens_override - sensitivity) > 0.01:
        alpha_sim = [first_real]
        for i, p in enumerate(pressure):
            if i == 0:
                continue
            alpha_sim.append(alpha_sim[-1] * (1 + sens_override * p))
        while len(alpha_sim) < len(dates):
            alpha_sim.append(alpha_sim[-1])

        beta_sim = [first_real]
        prev_real = first_real
        for i, p in enumerate(pressure):
            if i == 0:
                continue
            real_drift = (real[i] - prev_real) / prev_real if prev_real > 0 else 0
            beta_sim.append(beta_sim[-1] * (1 + real_drift) * (1 + sens_override * p))
            prev_real = real[i]
        while len(beta_sim) < len(dates):
            beta_sim.append(beta_sim[-1])

        gamma_sim = [d["real_close"] * (1 + sens_override * pressure[i]) for i, d in enumerate(data["days"])]

    fig = go.Figure()
    # Real (anchor)
    fig.add_trace(go.Scatter(x=dates, y=real, name="📈 Real AAPL close",
                             mode="lines+markers",
                             line=dict(color="#000000", width=4),
                             marker=dict(size=7)))
    # α — closed virtual (pure agent, no real anchor) — MADE PROMINENT
    fig.add_trace(go.Scatter(x=dates, y=alpha_sim, name=f"🔴 α PURE AGENT (sens={sens_override:.2f}) — no real calibration ever",
                             mode="lines+markers",
                             line=dict(color="#e74c3c", width=4),
                             marker=dict(size=6, symbol="diamond")))
    # β — anchored hybrid
    fig.add_trace(go.Scatter(x=dates, y=beta_sim, name=f"β Anchored hybrid (sens={sens_override:.2f})",
                             mode="lines",
                             line=dict(color="#3498db", width=2, dash="dot")))
    # γ — dual layer auxiliary
    fig.add_trace(go.Scatter(x=dates, y=gamma_sim, name=f"γ Dual layer",
                             mode="lines",
                             line=dict(color="#27ae60", width=1.5, dash="dashdot")))

    final_real = real[-1]
    final_alpha = alpha_sim[-1]
    final_beta = beta_sim[-1]
    final_gamma = gamma_sim[-1]
    fig.update_layout(
        title=f"<b>3 Price Formation Modes vs Real {data['ticker']}</b><br>"
              f"<sub>α drift {(final_alpha-final_real)/final_real*100:+.1f}% | "
              f"β drift {(final_beta-final_real)/final_real*100:+.1f}% | "
              f"γ drift {(final_gamma-final_real)/final_real*100:+.1f}% from real</sub>",
        xaxis_title="Date", yaxis_title="Price ($)",
        height=550, hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===== PURE AGENT WORLD CHART (α at multiple sensitivities) =====
    st.divider()
    st.markdown("### 🔴 Pure Agent World — α at multiple sensitivities")
    st.caption("If the 11 agents WERE the entire market with no other forces, here's where price would go. **No real-market calibration anywhere** (only Day 1 anchored to real opening price as starting point).")

    fig_pure = go.Figure()
    # Real line (gray, for reference only)
    fig_pure.add_trace(go.Scatter(x=dates, y=real, name="(Real — reference only)",
                                  mode="lines",
                                  line=dict(color="#999", width=1.5, dash="dot")))

    # α at multiple sensitivities — show as fan
    SENS_LEVELS = [0.1, 0.3, 0.5, 0.7, 1.0]
    SHADES = ["#fadbd8", "#f5b7b1", "#f1948a", "#ec7063", "#c0392b"]  # light → dark red

    for sens_val, color in zip(SENS_LEVELS, SHADES):
        alpha_at_sens = [first_real]
        for i, p in enumerate(pressure):
            if i == 0:
                continue
            alpha_at_sens.append(alpha_at_sens[-1] * (1 + sens_val * p))
        while len(alpha_at_sens) < len(dates):
            alpha_at_sens.append(alpha_at_sens[-1])
        final = alpha_at_sens[-1]
        drift = (final - first_real) / first_real * 100
        is_main = abs(sens_val - 1.0) < 0.01
        fig_pure.add_trace(go.Scatter(
            x=dates, y=alpha_at_sens,
            name=f"α @ sens={sens_val:.1f} → {drift:+.1f}%" + (" 🔥" if is_main else ""),
            mode="lines+markers" if is_main else "lines",
            line=dict(color=color, width=4 if is_main else 2),
            marker=dict(size=5) if is_main else None,
        ))

    real_drift_pct = (real[-1] - first_real) / first_real * 100
    fig_pure.update_layout(
        title=f"<b>Pure Agent Simulation at Multiple Sensitivities</b><br>"
              f"<sub>Real market drifted +{real_drift_pct:.1f}% — agents alone produce a different shape entirely</sub>",
        xaxis_title="Date", yaxis_title="Price ($) — agent-only world",
        height=500, hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_pure, use_container_width=True)

    # Compare drifts table
    drift_rows = []
    for sens_val in SENS_LEVELS:
        a_p = [first_real]
        for i, p in enumerate(pressure):
            if i == 0:
                continue
            a_p.append(a_p[-1] * (1 + sens_val * p))
        while len(a_p) < len(dates):
            a_p.append(a_p[-1])
        final_drift = (a_p[-1] - first_real) / first_real * 100
        vs_real = final_drift - real_drift_pct
        drift_rows.append({
            "α sensitivity": f"{sens_val:.1f}",
            "α final 32-day return": f"{final_drift:+.2f}%",
            "vs Real (+{:.2f}%)".format(real_drift_pct): f"{vs_real:+.2f} pp",
            "Interpretation": (
                "Agents barely move price" if sens_val <= 0.1 else
                "Agents nudge but real-dominant" if sens_val <= 0.3 else
                "Equal agent vs real" if sens_val <= 0.5 else
                "Agents drive most" if sens_val <= 0.7 else
                "Agents = whole market"
            ),
        })
    st.dataframe(pd.DataFrame(drift_rows), hide_index=True, use_container_width=True)

    st.warning(f"""
**Bottom line on Pure Agent World:**

The 11 agents collectively produce **mildly positive net buying pressure** (mean ~+{(sum(pressure)/len(pressure))*100:.2f}%/day).
Even at maximum sensitivity (sens=1.0), they compound to only **~{(((1+sum(pressure)/len(pressure))**len(pressure)-1)*100):.1f}%** over 32 days.

This means: **our agents collectively are NOT as bullish as the real market was**. The real market gained +{real_drift_pct:.1f}% partly from forces our agents don't see/model:
- Passive ETF flows (Vanguard, BlackRock daily buybacks)
- Corporate buyback execution
- Foreign sovereign / pension flows
- HFT / market-maker liquidity provision
- Cross-asset hedging (vol skew, gamma, etc.)

To capture the full +21%, agents would need to be ~3-4x more aggressive OR we'd need additional Forced Flow agents (this is what v0.6 §4.4 already calls out — see Forced Flow Layer 2B).
""")

    # Legend explainer
    with st.expander("📖 What are these 3 modes?"):
        st.markdown("""
**α Closed (red dashed)** — `next = prev × (1 + sens × pressure)`
- Pure agent-driven, ignores real entirely
- **What would happen if these 11 agents WERE the entire market**
- Diverges from real over time

**β Anchored hybrid (blue dotted)** — `next = prev × (1 + real_drift) × (1 + sens × pressure)` — *stored mode*
- Real drift + agent pressure stacked
- Bounded drift but still accumulates

**γ Dual layer (green dashdot)** — `next = real × (1 + sens × pressure)`
- Each day re-anchored to real, agent pressure as overlay
- Tracks real but shows daily agent influence
- Best for "next-day prediction" use case
""")

    # Pressure chart below
    fig2 = go.Figure()
    colors_p = ["#26c281" if p > 0 else "#e74c3c" for p in pressure]
    fig2.add_trace(go.Bar(x=dates, y=pressure, marker_color=colors_p, name="Net pressure"))
    fig2.update_layout(
        title="<b>Daily Net Buying Pressure (agent flow)</b>",
        height=250, xaxis_title="", yaxis_title="Net pressure",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation: today's virtual-real gap vs tomorrow's real change
    rows = []
    for i in range(len(data["days"]) - 1):
        d_i = data["days"][i]
        d_n = data["days"][i + 1]
        gap_today = (d_i["virtual_close"] - d_i["real_close"]) / d_i["real_close"]
        change_tomorrow = (d_n["real_close"] - d_i["real_close"]) / d_i["real_close"]
        rows.append({"day": d_i["date"], "gap_today_pct": gap_today * 100,
                     "tomorrow_change_pct": change_tomorrow * 100})
    if rows:
        pred_df = pd.DataFrame(rows)
        fig3 = px.scatter(pred_df, x="gap_today_pct", y="tomorrow_change_pct",
                          trendline="ols", title="Does today's sim-real gap predict tomorrow's real?",
                          labels={"gap_today_pct": "Sim − Real today (%)",
                                  "tomorrow_change_pct": "Real change tomorrow (%)"},
                          height=400)
        corr = pred_df["gap_today_pct"].corr(pred_df["tomorrow_change_pct"])
        fig3.update_layout(title=f"<b>Predictive correlation</b>  Pearson r = {corr:.3f}")
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------- Tab: Case Study — Ternus Event -----------------------
with t_ternus:
    st.markdown("## 📍 Case Study: The John Ternus CEO Succession Event")
    st.caption(
        "April 20–28, 2026: news of Tim Cook stepping down breaks. "
        "Each of the 11 agents reads the SAME news but reacts differently — "
        "this is the heart of the project: heterogeneous belief formation under shared information."
    )

    # Window: days 15-22 (inclusive) corresponds to 4-20 .. 4-29
    ternus_window = [(i, d) for i, d in enumerate(data["days"]) if "2026-04-20" <= d["date"] <= "2026-04-29"]
    if not ternus_window:
        st.warning("No data in Ternus window.")
    else:
        # ----- Section 1: news timeline -----
        st.markdown("### 📰 News timeline")
        from pathlib import Path as _Path
        news_dir = _Path(__file__).parent / "cache" / "news"
        for i, day in ternus_window:
            f = news_dir / f"AAPL_{day['date']}.json"
            if not f.exists():
                continue
            try:
                items = json.load(open(f))
            except Exception:
                continue
            ternus_headlines = [it.get("title", "") for it in items if "ernus" in it.get("title", "") or "Cook" in it.get("title", "")]
            if ternus_headlines:
                st.markdown(f"**{day['date']}** _(real close ${day['real_close']:.2f}, virtual ${day['virtual_close']:.2f})_")
                for h in ternus_headlines[:4]:
                    st.caption(f"  • {h.strip()}")

        # ----- Section 2: side-by-side agent reactions on Day 21 (4-28) -----
        st.divider()
        st.markdown("### 🧠 Day 21 (Apr 28) — same news, 11 different minds")
        st.caption("This is the day Super-influencer deployed $2.93B alone. What was everyone else thinking?")

        day21 = next((dd for ii, dd in ternus_window if dd["date"] == "2026-04-28"), None)
        if day21:
            outputs = day21["agent_outputs"]
            trades = day21["trades"]
            # Build a comparison table
            agents_to_show = [
                "super_influencer_001", "permabull_001", "pod_pm_001",
                "activist_short_001", "retail_fomo_001", "day_trader_001",
                "sell_side_001", "economist_macro_001", "economist_political_001",
                "economist_trader_001",
            ]
            for aid in agents_to_show:
                out = outputs.get(aid)
                if not out:
                    continue
                viz = AGENT_VIZ.get(aid, {"emoji": "?", "short": aid})
                state = out["state"]
                pb = state.get("private_belief", {})
                ps = state.get("public_statement", {})
                pa = state.get("personal_action", {})
                tr = trades.get(aid, {})

                priv_lean = pb.get("lean", "?")
                pub_lean = ps.get("stated_lean", "?")
                priv_color = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(priv_lean, "⚪")
                pub_color = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(pub_lean, "⚪")
                hypocrisy = "🎭" if priv_lean != pub_lean else "  "

                with st.container():
                    cols = st.columns([2, 3, 3, 2])
                    with cols[0]:
                        st.markdown(f"### {viz['emoji']} {viz['short']}")
                        st.caption(f"private: {priv_color} **{priv_lean}** ({pb.get('conviction',0):.0%})")
                        st.caption(f"public:  {pub_color} **{pub_lean}** ({ps.get('stated_conviction',0):.0%}) {hypocrisy}")
                    with cols[1]:
                        st.markdown(f"**Private thesis:**")
                        st.caption(pb.get("actual_thesis", "")[:300])
                    with cols[2]:
                        st.markdown(f"**Public statement:**")
                        st.caption(f"_{ps.get('narrative', '')[:300]}_")
                    with cols[3]:
                        act = pa.get("action_type", "?")
                        size = pa.get("size_pct", 0)
                        trade_value = tr.get("value", 0)
                        emoji = ACTION_EMOJI.get(act, "?")
                        st.markdown(f"**Action:** {emoji} {act}")
                        st.caption(f"size: {size:.0%}")
                        if trade_value > 0:
                            st.markdown(f"**${trade_value/1e6:,.1f}M** {tr.get('detail','')}")
                        else:
                            st.caption("(no trade)")
                st.divider()

        # ----- Section 3: who was right? -----
        st.markdown("### 🎯 Who was right in retrospect?")
        last_real = data["days"][-1]["real_close"]
        day21_real = day21["real_close"] if day21 else None
        if day21_real:
            move = (last_real - day21_real) / day21_real * 100
            st.metric(
                f"AAPL move from Apr 28 close to May 13 close",
                f"${day21_real:.2f} → ${last_real:.2f}",
                f"{move:+.2f}% over 15 trading days",
            )
            st.markdown(
                f"""
                **Outcome:** AAPL went **up {move:.1f}%** after Day 21.
                - **Super-influencer** (deployed $2.93B long): **CORRECT** — high-conviction contrarian buy paid off
                - **Permabull / Sell-side** (long lean, light or no action): **CORRECT direction, missed size**
                - **Pod PM / Retail / Day trader** (short / sold on Day 21): **WRONG** — sold near a local bottom
                - **Activist short** (already short): **WRONG** — continued losing in the rally
                - **Economists**: speech-only, no skin in the game

                **Reputation impact:** this single 7-day window dropped Activist's accuracy
                from 50% to ~5% by end of run, while Super-influencer's reputation crashed
                because she was caught NEUTRAL in public but LONG in private —
                the system tracks PUBLIC accuracy.
                """
            )


# ----------------------- Tab 2: Daily Action Timeline -----------------------
with t_timeline:
    st.markdown("### Each agent's daily action — full 30 days")
    st.caption("X = trading day · Y = agent · cell color = action intensity (red=sell, green=buy) · size = action size_pct · click cell → details")

    # Build heatmap data
    agent_order = [a.id for a in ALL_AGENTS]
    z = []
    text = []
    hovertext = []
    for aid in agent_order:
        row_z = []
        row_t = []
        row_h = []
        for d in data["days"]:
            info = d["agent_outputs"].get(aid, {})
            state = info.get("state", {})
            act = state.get("personal_action", {})
            atype = act.get("action_type", "hold") or "hold"
            size = act.get("size_pct", 0.0) or 0.0
            val = ACTION_NUM.get(atype, 0) * (0.5 + size)
            row_z.append(val)
            row_t.append(ACTION_EMOJI.get(atype, ""))
            priv = state.get("private_belief", {}).get("lean", "neutral")
            pub = state.get("public_statement", {}).get("stated_lean", "neutral")
            row_h.append(
                f"<b>{info.get('name', aid)}</b><br>"
                f"{d['date']}<br>"
                f"action: <b>{atype}</b> size {size:.2f}<br>"
                f"private: {priv} · public: {pub}<br>"
                f"<i>{state.get('private_belief', {}).get('actual_thesis','')[:120]}</i>"
            )
        z.append(row_z)
        text.append(row_t)
        hovertext.append(row_h)

    fig = go.Figure(go.Heatmap(
        z=z, text=text, hovertext=hovertext,
        x=[d["date"] for d in data["days"]],
        y=[short_label(aid) for aid in agent_order],
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale=[[0, "#c0392b"], [0.5, "#ecf0f1"], [1, "#27ae60"]],
        zmin=-3, zmax=3,
        hoverinfo="text",
        showscale=False,
    ))
    fig.update_layout(
        height=380,
        title="<b>Daily Action Heatmap — sized by intensity, colored by direction</b>",
        xaxis_side="bottom", xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------- Tab 3: Deception Detection -----------------------
with t_deception:
    st.markdown("### 🎭 Strategic Signaling Detection")
    st.caption("Three-track output: PRIVATE belief vs PUBLIC statement vs ACTUAL action. Mismatches = strategic signaling / deception.")

    # Compute deception stats per agent
    agents_no_cta = [a for a in ALL_AGENTS if a.role != "cta_forced"]

    dec_rows = []
    daily_events = []  # for the timeline

    for agent in agents_no_cta:
        n_priv_ne_pub = 0       # private != public (textbook deception)
        n_pub_ne_act = 0        # public != action (talking different from doing)
        n_priv_ne_act = 0       # private != action (acting against own belief)
        n_total = 0
        examples = []           # most dramatic ones

        for d_idx, d in enumerate(data["days"]):
            state = d["agent_outputs"].get(agent.id, {}).get("state", {})
            priv = state.get("private_belief", {}).get("lean", "neutral")
            pub = state.get("public_statement", {}).get("stated_lean", "neutral")
            action_type = state.get("personal_action", {}).get("action_type", "hold")

            # Infer lean from action
            if action_type.startswith("buy"):
                act_lean = "long"
            elif action_type.startswith("sell"):
                act_lean = "short"
            else:
                act_lean = "neutral"

            priv_ne_pub = priv != pub and priv not in ("deterministic", "n/a")
            pub_ne_act = pub != act_lean and pub not in ("neutral", "deterministic", "n/a") and act_lean != "neutral"
            priv_ne_act = priv != act_lean and priv not in ("deterministic", "n/a") and act_lean != "neutral"

            if priv_ne_pub: n_priv_ne_pub += 1
            if pub_ne_act: n_pub_ne_act += 1
            if priv_ne_act: n_priv_ne_act += 1

            if priv_ne_pub or pub_ne_act:
                examples.append({
                    "date": d["date"],
                    "priv": priv, "pub": pub, "action": action_type, "act_lean": act_lean,
                    "priv_ne_pub": priv_ne_pub,
                    "pub_ne_act": pub_ne_act,
                    "thesis": state.get("private_belief", {}).get("actual_thesis", "")[:200],
                    "narrative": state.get("public_statement", {}).get("narrative", "")[:200],
                })
                daily_events.append({
                    "date": d["date"], "agent": agent.id,
                    "agent_short": short_label(agent.id),
                    "type": "priv≠pub" if priv_ne_pub else "pub≠act",
                    "priv": priv, "pub": pub, "act_lean": act_lean,
                })
            n_total += 1

        dec_rows.append({
            "Agent": short_label(agent.id),
            "Days with priv ≠ pub": n_priv_ne_pub,
            "Days with pub ≠ act": n_pub_ne_act,
            "Days with priv ≠ act": n_priv_ne_act,
            "Total days": n_total,
            "Deception rate (priv≠pub)": n_priv_ne_pub / n_total if n_total else 0,
            "Signaling intent (param)": agent.signaling_incentive,
            "_examples": examples,
        })

    # Summary table
    st.markdown("#### Per-Agent Deception Stats")
    df_dec = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in dec_rows])
    df_dec_display = df_dec.sort_values("Deception rate (priv≠pub)", ascending=False)
    st.dataframe(
        df_dec_display, hide_index=True, use_container_width=True,
        column_config={
            "Deception rate (priv≠pub)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1),
            "Signaling intent (param)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        },
    )

    # Daily deception heatmap: agent × day
    st.markdown("#### Deception Timeline — When did each agent lie?")
    st.caption("🟥 = private ≠ public (textbook deception). 🟧 = public ≠ action (talking ≠ doing). White = aligned.")

    z = []
    text = []
    hovertext = []
    for agent in agents_no_cta:
        row_z = []
        row_t = []
        row_h = []
        for d in data["days"]:
            state = d["agent_outputs"].get(agent.id, {}).get("state", {})
            priv = state.get("private_belief", {}).get("lean", "neutral")
            pub = state.get("public_statement", {}).get("stated_lean", "neutral")
            action_type = state.get("personal_action", {}).get("action_type", "hold")
            act_lean = "long" if action_type.startswith("buy") else "short" if action_type.startswith("sell") else "neutral"

            priv_ne_pub = priv != pub
            pub_ne_act = pub != act_lean and pub != "neutral" and act_lean != "neutral"

            # Score: 0=aligned, 1=pub_ne_act only, 2=priv_ne_pub (worst)
            if priv_ne_pub:
                val = 2
                emoji = "🔴"
            elif pub_ne_act:
                val = 1
                emoji = "🟠"
            else:
                val = 0
                emoji = ""
            row_z.append(val)
            row_t.append(emoji)
            row_h.append(
                f"<b>{agent.name}</b> · {d['date']}<br>"
                f"Private: {priv}<br>"
                f"Public: {pub}<br>"
                f"Action lean: {act_lean} ({action_type})"
            )
        z.append(row_z)
        text.append(row_t)
        hovertext.append(row_h)

    fig = go.Figure(go.Heatmap(
        z=z, text=text, hovertext=hovertext,
        x=[d["date"] for d in data["days"]],
        y=[short_label(a.id) for a in agents_no_cta],
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale=[[0, "#ffffff"], [0.5, "#f39c12"], [1, "#c0392b"]],
        zmin=0, zmax=2,
        hoverinfo="text",
        showscale=False,
    ))
    fig.update_layout(height=320, xaxis_tickangle=-45,
                      title="<b>Deception Heatmap</b>  red = private≠public · orange = public≠action")
    st.plotly_chart(fig, use_container_width=True)

    # Most dramatic examples
    st.markdown("#### 🎯 Most Dramatic Deception Examples")
    st.caption("Top 6 cases where private belief was OPPOSITE direction from public statement.")
    opposite_cases = []
    for row in dec_rows:
        for ex in row["_examples"]:
            # Score = full opposite (long vs short)
            priv, pub = ex["priv"], ex["pub"]
            if {priv, pub} == {"long", "short"}:
                opposite_cases.append((row["Agent"], ex))
    opposite_cases.sort(key=lambda x: x[1]["date"], reverse=True)

    if not opposite_cases:
        st.info("No fully-opposite (long↔short) deception detected. Most signaling is private-bear / public-neutral subtle deception.")
    else:
        for agent_lbl, ex in opposite_cases[:6]:
            color = "#c0392b" if ex["priv_ne_pub"] else "#f39c12"
            st.markdown(f"""<div style='border-left:4px solid {color};padding:8px 12px;margin:6px 0;background:#fafafa'>
<b>{agent_lbl}</b> · {ex['date']}<br>
🧠 <b>Private:</b> <span style='color:{LEAN_COLOR.get(ex['priv'])}'>{ex['priv'].upper()}</span> — <i>"{ex['thesis']}"</i><br>
📢 <b>Public:</b> <span style='color:{LEAN_COLOR.get(ex['pub'])}'>{ex['pub'].upper()}</span> — <i>"{ex['narrative']}"</i><br>
🎯 <b>Actual action:</b> {ex['action']} ({ex['act_lean']})
</div>""", unsafe_allow_html=True)

    # Today's deception status
    st.divider()
    st.markdown(f"#### Today's Deceivers — {day['date']}")
    today_deceivers = [e for e in daily_events if e["date"] == day["date"]]
    if today_deceivers:
        for e in today_deceivers:
            st.markdown(f"- **{e['agent_short']}** ({e['type']}): private **{e['priv']}** / public **{e['pub']}** / action **{e['act_lean']}**")
    else:
        st.info("No deception detected today.")


# ----------------------- Tab 4 (NEW): Info Asymmetry -----------------------
with t_asymmetry:
    st.markdown("### 🔓 Information Asymmetry — Who Sees What")
    st.caption("Real markets aren't level playing fields. Different agents access different data tiers, listen to different people, and accumulate different memories.")

    # Tier table
    st.markdown("#### Tier-Based Information Access")
    tier_descriptions = {
        1: "📰 **Tier 1 — Retail**: Headlines + price chart + 20-day SMA only",
        2: "📊 **Tier 2 — Sell-Side**: + analyst notes + technicals (RSI, MACD, BB) + earnings calendar",
        3: "🔍 **Tier 3 — Activist/Forensic**: + insider Form 4 + options skew + volume anomalies + 10-Q line items",
        4: "💼 **Tier 4 — Hedge Fund**: + dealer gamma + options flow + 13F filings + sector rotation",
        5: "🏛️ **Tier 5 — Macro/Fed Channel**: + Fed minutes drafts (24h early) + rate path probabilities + macro nowcasts",
    }

    agents_no_cta = [a for a in ALL_AGENTS]

    tier_table_rows = []
    for agent in agents_no_cta:
        viz = AGENT_VIZ.get(agent.id, {})
        inbound = []
        try:
            from influence_graph import INFLUENCE_GRAPH, INFLUENCE_THRESHOLD
            idx = next(i for i, a in enumerate(ALL_AGENTS) if a.id == agent.id)
            for (src, tgt), w in INFLUENCE_GRAPH.items():
                if tgt == idx and w >= INFLUENCE_THRESHOLD:
                    inbound.append((ALL_AGENTS[src].name.split()[0], w))
            inbound.sort(key=lambda x: -x[1])
        except Exception:
            pass
        mem_count = data.get("private_memory_snapshot", {}).get(agent.id, {}).get("active_count", 0)
        tier_table_rows.append({
            "Agent": short_label(agent.id),
            "Tier": agent.info_tier,
            "Info access": tier_descriptions.get(agent.info_tier, "?").split(":")[0],
            "Inbound voices (G ≥ 0.4)": ", ".join(f"{n} ({w:.2f})" for n, w in inbound[:5]) or "(none — independent)",
            "Active memory facts": mem_count,
            "Has portfolio": "✅" if agent.has_portfolio else "❌ commentator",
        })
    df_tier = pd.DataFrame(tier_table_rows).sort_values(["Tier", "Agent"], ascending=[False, True])
    st.dataframe(df_tier, hide_index=True, use_container_width=True)

    # Visualize as bar chart: info access tier per agent
    st.markdown("#### Information Tier Pyramid")
    tier_counts = {t: 0 for t in range(1, 6)}
    tier_agents = {t: [] for t in range(1, 6)}
    for agent in agents_no_cta:
        tier_counts[agent.info_tier] += 1
        tier_agents[agent.info_tier].append(short_label(agent.id))

    for tier in [5, 4, 3, 2, 1]:
        names = tier_agents[tier]
        bar = "█" * tier_counts[tier] if tier_counts[tier] else "·"
        st.markdown(f"**T{tier}** {bar} {tier_descriptions.get(tier, '')}")
        st.caption(f"    → Agents: {', '.join(names) if names else '(none)'}")

    # Per-day asymmetry detail
    st.markdown(f"#### What did each agent see on {day['date']}?")
    st.caption("Click expander to see exactly what info_view + influencer posts + memory recall this agent received today.")

    today_news_available = "(news may have been available — depends on cache for this date)"
    for agent in agents_no_cta:
        info = day["agent_outputs"].get(agent.id, {})
        state = info.get("state", {})
        with st.expander(f"{short_label(agent.id)} — Tier {agent.info_tier} · {len(tier_descriptions.get(agent.info_tier,'').split(':')[1].split(','))} data slices"):
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**🔍 Their private thesis today:**")
                st.markdown(f"_{state.get('private_belief', {}).get('actual_thesis', '')[:300]}_")
            with cols[1]:
                st.markdown(f"**📢 Their public take:**")
                st.markdown(f"_{state.get('public_statement', {}).get('narrative', '')[:300]}_")
            # Inbound influencer list
            try:
                idx = next(i for i, a in enumerate(ALL_AGENTS) if a.id == agent.id)
                inbound = [(ALL_AGENTS[src].name, w) for (src, tgt), w in INFLUENCE_GRAPH.items()
                          if tgt == idx and w >= INFLUENCE_THRESHOLD]
                if inbound:
                    st.markdown(f"**👁️ Voices they hear today** (above G=0.4 threshold):")
                    for name, w in sorted(inbound, key=lambda x: -x[1]):
                        st.caption(f"  • {name} (influence weight: {w:.2f})")
                else:
                    st.markdown("**👁️ Voices they hear:** (none — totally independent thinker)")
            except Exception:
                pass

    st.divider()
    st.info("""**Key insight**: This is NOT a level playing field.
- 🦍 Retail Alex only sees headlines + chart, listens to Super-Influencer (0.92), Activist (0.78), Sell-Side (0.62) + Krugman (0.45)
- 💼 Pod PM David sees flow data + 13F + dealer gamma, listens to Druck (0.75), Bernanke (0.55), Activist (0.45) + Sell-Side (0.40)
- 🏛️ Bernanke sees EVERYTHING (Tier 5) but listens to NOBODY (he's the lone academic voice)
- 🤖 CTA sees only price (Tier 1) and hears nobody — pure mechanical rule

This asymmetry is **structural** and **deliberate** — it's how real markets work.""")


# ----------------------- Tab 5: Reputation Evolution -----------------------
with t_reputation:
    st.markdown("### Reputation history (30-day rolling accuracy)")
    st.caption("Every 7 days, evaluate predictions made N days ago vs actual. Multiplier = 0.5 + accuracy (bounded 0.05-1.5).")

    rep_history = data.get("reputation_history", {}).get("history", {})
    if not rep_history:
        st.info("Reputation update happens every 7 days — wait until day 7+ to see data.")
    else:
        fig = go.Figure()
        for aid in [a.id for a in ALL_AGENTS]:
            snaps = rep_history.get(aid, [])
            if not snaps:
                continue
            dates = [s["date"] for s in snaps]
            accs = [s["accuracy"] for s in snaps]
            fig.add_trace(go.Scatter(
                x=dates, y=accs, name=short_label(aid),
                mode="lines+markers",
                line=dict(color=AGENT_VIZ[aid]["color"], width=3),
                marker=dict(size=10),
            ))
        fig.update_layout(
            title="<b>Per-Agent Accuracy Over Time</b>",
            xaxis_title="Update date", yaxis_title="30-day rolling accuracy",
            height=480, hovermode="x unified",
            yaxis=dict(range=[0, 1]),
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="random chance")
        st.plotly_chart(fig, use_container_width=True)

        # Final multipliers
        st.markdown("### Current Influence Multipliers (used to dynamically reweight Influence Graph)")
        latest_mults = day.get("influence_multipliers_after", {})
        df_mults = pd.DataFrame([
            {"Agent": short_label(aid), "Multiplier": mult,
             "Δ vs 1.0": mult - 1.0}
            for aid, mult in latest_mults.items()
        ]).sort_values("Multiplier", ascending=False)
        st.dataframe(df_mults, hide_index=True, use_container_width=True)

# ----------------------- Tab 4: Memory Inspector -----------------------
with t_memory:
    st.markdown(f"### Memory facts per agent — as of {day['date']}")
    private_mem = data.get("private_memory_snapshot", {})
    for aid in [a.id for a in ALL_AGENTS]:
        snap = private_mem.get(aid, {})
        with st.expander(f"{short_label(aid)} — {snap.get('active_count', 0)} active / {snap.get('fact_count', 0)} total"):
            facts = snap.get("facts", [])
            # Filter to facts up to current day
            relevant = [f for f in facts if f.get("valid_at", "") <= day["date"]]
            if not relevant:
                st.caption("No facts yet.")
                continue
            # Show last 20
            for f in relevant[-20:][::-1]:
                invalid = f.get("invalid_at")
                marker = "❌ " if invalid else "✓ "
                color = "#c0392b" if invalid else "#27ae60"
                st.markdown(
                    f"<div style='border-left:3px solid {color};padding:6px 10px;margin:4px 0;background:#fafafa;border-radius:4px'>"
                    f"<small>[{f.get('valid_at')}] ({f.get('source')}) {marker}"
                    f"{f.get('content','')[:200]}{' · invalidated ' + invalid if invalid else ''}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ----------------------- Tab: Portfolios (filtered) -----------------------
with t_pnl:
    # AAPL B&H benchmark line
    first_real = data["days"][0]["real_close"]
    bh = [(d["real_close"] - first_real) / first_real * 100 for d in data["days"]]
    dates = [d["date"] for d in data["days"]]

    # Filter to agents WITH portfolio only (skip 0-capital economists)
    portfolio_agents = [a for a in ALL_AGENTS if a.has_portfolio]

    fig = go.Figure()
    # Add B&H benchmark as thick black line
    fig.add_trace(go.Scatter(
        x=dates, y=bh, name="AAPL Buy-and-Hold",
        mode="lines",
        line=dict(color="#000000", width=4),
    ))

    for agent in portfolio_agents:
        aid = agent.id
        if aid not in data["days"][0]["portfolios_close"]:
            continue
        pnls = [d["portfolios_close"][aid]["pnl_pct"] for d in data["days"]]
        fig.add_trace(go.Scatter(
            x=dates, y=pnls, name=short_label(aid), mode="lines+markers",
            line=dict(color=AGENT_VIZ.get(aid, {}).get("color", "#888"), width=2.5),
            marker=dict(size=7),
        ))
    fig.update_layout(
        title=f"<b>Portfolio PnL % vs AAPL Buy-and-Hold (+{bh[-1]:.2f}%)</b>",
        xaxis_title="Date", yaxis_title="PnL %",
        height=520, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Black line = AAPL Buy-and-Hold benchmark (+{bh[-1]:.2f}%). All other lines = agent strategies. Economists (no portfolio) excluded.")

    # Forced closes summary
    total_forced = 0
    for d in data["days"]:
        total_forced += len(d.get("forced_closes", []))
    if total_forced > 0:
        st.markdown(f"#### ⚡ Stop-loss / Take-profit triggers: {total_forced} total")
        forced_rows = []
        for d in data["days"]:
            for fc in d.get("forced_closes", []):
                forced_rows.append({
                    "Date": d["date"],
                    "Agent": fc.get("name", fc.get("agent", "?")),
                    "Trigger": fc.get("reason", "?"),
                    "Shares": fc.get("shares", 0),
                    "Price": f"${fc.get('price', 0):.2f}",
                    "Stop/Target was": f"${fc.get('trigger', 0):.2f}",
                    "Cost basis was": f"${fc.get('cost_basis', 0):.2f}",
                })
        if forced_rows:
            st.dataframe(pd.DataFrame(forced_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No stop-loss / take-profit triggers in this run (agents may not have specified them, or prices stayed within bands).")

    # Final table
    st.markdown("### Final position")
    fp = data["final_portfolios"]
    df = pd.DataFrame([
        {"Agent": short_label(aid),
         "Initial $": p["initial_capital"],
         "Final $": p["final_total"],
         "PnL %": p["final_pnl_pct"],
         "Memory facts (active)": p.get("memory_facts_active", 0),
         "Memory total": p.get("memory_facts_total", 0)}
        for aid, p in fp.items()
    ]).sort_values("PnL %", ascending=False)
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={
                     "Initial $": st.column_config.NumberColumn(format="$%.0f"),
                     "Final $": st.column_config.NumberColumn(format="$%.0f"),
                     "PnL %": st.column_config.NumberColumn(format="%+.2f%%"),
                 })

# ----------------------- Tab 6: Agent Detail -----------------------
with t_detail:
    agent = get_agent(selected_aid)
    viz = AGENT_VIZ[selected_aid]
    st.markdown(f"<h2 style='color:{viz['color']}'>{viz['emoji']} {agent.name}</h2>", unsafe_allow_html=True)
    st.caption(f"{agent.role} · Capital ${agent.capital:,.0f} · Info tier {agent.info_tier}")

    info = day["agent_outputs"].get(selected_aid, {})
    state = info.get("state", {})
    priv = state.get("private_belief", {})
    pub = state.get("public_statement", {})
    desired = state.get("desired_market_reaction", "")
    act = state.get("personal_action", {})

    st.markdown(f"### State on {day['date']}")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**🧠 Private Belief**  \nLean: **{priv.get('lean','?')}** · conv {priv.get('conviction',0):.2f}  \n_{priv.get('actual_thesis','')[:280]}_")
    with cols[1]:
        st.markdown(f"**📢 Public Statement**  \nLean: **{pub.get('stated_lean','?')}** · conv {pub.get('stated_conviction',0):.2f}  \n_{pub.get('narrative','')[:280]}_")

    cols = st.columns(2)
    cols[0].markdown(f"**🎭 Desired Reaction:** _{desired[:280]}_")
    cols[1].markdown(f"**🎯 Action:** **{act.get('action_type','?')}** size {act.get('size_pct',0):.2f}  \n_{act.get('rationale_internal','')[:240]}_")

    # Reputation history for this agent
    rep_h = data.get("reputation_history", {}).get("history", {}).get(selected_aid, [])
    if rep_h:
        st.markdown("### Reputation history")
        df_rh = pd.DataFrame(rep_h)
        st.line_chart(df_rh.set_index("date")[["accuracy", "influence_multiplier"]])

# Footer
st.divider()
st.caption(f"v0.7 demo · {data['ticker']} · {data['start_date']} → {data['end_date']} · sensitivity={data['sensitivity']} · "
           f"Reputation window {data['reputation_window_days']}d · cost ${data['total_cost_usd']:.2f}")
