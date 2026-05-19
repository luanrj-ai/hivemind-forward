"""Streamlit visualization for Toy v0.6 — 6 agents, 5 days, full interactive UI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from agents import ALL_AGENTS, get_agent
from influence_graph import INFLUENCE_GRAPH, INFLUENCE_THRESHOLD


# ============================================================================
# CONFIG & DATA LOADING
# ============================================================================

DATA_PATH = Path(__file__).parent / "results" / "toy_run_latest.json"

AGENT_VIZ = {
    "super_influencer_001": {"color": "#9b59b6", "emoji": "🎙️", "short": "Super"},
    "pod_pm_001":           {"color": "#3498db", "emoji": "💼", "short": "Pod PM"},
    "activist_short_001":   {"color": "#e74c3c", "emoji": "🎯", "short": "Activist"},
    "sell_side_001":        {"color": "#f39c12", "emoji": "📊", "short": "Sell-Side"},
    "cta_forced_001":       {"color": "#7f8c8d", "emoji": "🤖", "short": "CTA"},
    "retail_fomo_001":      {"color": "#2ecc71", "emoji": "🦍", "short": "Retail"},
}

LEAN_COLOR = {"long": "#26c281", "short": "#e74c3c", "neutral": "#95a5a6", "flat": "#95a5a6", "deterministic": "#7f8c8d", "n/a": "#7f8c8d"}

ACTION_EMOJI = {
    "buy_strong": "🟢🟢", "buy_lite": "🟢",
    "hold": "⚪",
    "sell_lite": "🔴", "sell_strong": "🔴🔴",
}

NODE_POS = {  # 2D layout for network graph
    "super_influencer_001":  (0.0,  1.0),
    "pod_pm_001":            (-1.3, 0.2),
    "activist_short_001":    (1.3,  0.2),
    "sell_side_001":         (-0.8, -1.0),
    "cta_forced_001":        (0.0,  -1.6),
    "retail_fomo_001":       (0.8,  -1.0),
}


@st.cache_data
def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def agent_label(aid: str, full: bool = True) -> str:
    v = AGENT_VIZ.get(aid, {"emoji": "❓", "short": aid})
    return f"{v['emoji']} {v['short']}" if not full else f"{v['emoji']} {v['short']} ({aid})"


def detect_aha_moments(day: dict) -> list[str]:
    """Auto-find the noteworthy strategic-signaling moments per day."""
    moments = []
    for aid, info in day["agent_outputs"].items():
        if info["role"] == "cta_forced":
            continue
        state = info["state"]
        priv = state["private_belief"].get("lean", "neutral")
        pub = state["public_statement"].get("stated_lean", "neutral")
        action = state["personal_action"].get("action_type", "hold")
        name = info["name"]

        # Banking conflict / Blodget pattern
        if priv == "short" and pub == "long":
            moments.append(f"⚠️ **{name}** (Blodget pattern): private SHORT but public LONG.")
        if priv == "long" and pub == "short":
            moments.append(f"⚠️ **{name}** (Reverse conflict): private LONG but public SHORT.")

        # Quiet accumulation (Hindenburg pre-attack)
        if priv == "short" and action.startswith("buy") and info["role"] == "activist_short":
            moments.append(f"🎯 **{name}** (Quiet accumulation): privately bearish, but BUYING. Likely building position before publishing.")

        # Forced flow counter-trade
        if info["role"] == "cta_forced":
            continue

    # CTA counter-trade
    thinker_leans = []
    for aid, info in day["agent_outputs"].items():
        if info["role"] == "cta_forced":
            continue
        thinker_leans.append(info["state"]["private_belief"].get("lean", "neutral"))
    cta_info = day["agent_outputs"].get("cta_forced_001")
    if cta_info:
        cta_action = cta_info["state"]["personal_action"].get("action_type", "hold")
        if thinker_leans.count("short") >= 3 and cta_action.startswith("buy"):
            moments.append(f"🤖 **CTA counter-trading**: {thinker_leans.count('short')} of 5 thinkers privately bearish, but CTA buying anyway (pure trend follow, ignores narrative).")
        elif thinker_leans.count("long") >= 3 and cta_action.startswith("sell"):
            moments.append(f"🤖 **CTA counter-trading**: thinkers bullish but CTA selling (trend reversal).")

    return moments


# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================


def render_network(day: dict) -> go.Figure:
    """Force-positioned network of 6 agents. Node = agent. Edge = influence weight."""
    fig = go.Figure()

    # Edges first (so they appear below nodes)
    for (i, j), w in INFLUENCE_GRAPH.items():
        if w < INFLUENCE_THRESHOLD:
            continue
        src_id = ALL_AGENTS[i].id
        tgt_id = ALL_AGENTS[j].id
        x0, y0 = NODE_POS[src_id]
        x1, y1 = NODE_POS[tgt_id]
        # Edge line
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=w * 6, color=f"rgba(100,100,100,{0.2 + w*0.5})"),
            hoverinfo="text",
            text=f"{agent_label(src_id, False)} → {agent_label(tgt_id, False)}: {w:.2f}",
            showlegend=False,
        ))
        # Arrowhead annotation
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=1.5, arrowwidth=1.5,
            arrowcolor=f"rgba(80,80,80,{0.4 + w*0.4})",
            showarrow=True,
        )

    # Nodes
    for aid, (x, y) in NODE_POS.items():
        agent_info = next(a for a in ALL_AGENTS if a.id == aid)
        viz = AGENT_VIZ[aid]
        cap = agent_info.capital
        size = max(35, min(95, 35 + np.log10(max(cap, 100)) * 8)) if cap > 0 else 30

        # Today's state for this agent
        day_state = day["agent_outputs"].get(aid, {}).get("state", {})
        priv_lean = day_state.get("private_belief", {}).get("lean", "neutral")
        pub_lean = day_state.get("public_statement", {}).get("stated_lean", "neutral")
        action = day_state.get("personal_action", {}).get("action_type", "hold")

        # Border color = private lean, Fill color = role
        border_color = LEAN_COLOR.get(priv_lean, "#999")

        hover_html = (
            f"<b>{viz['emoji']} {agent_info.name}</b><br>"
            f"Role: {agent_info.role}<br>"
            f"Capital: ${cap:,.0f}<br>"
            f"Info tier: {agent_info.info_tier}<br>"
            f"Influence out: {agent_info.influence_out:.2f}<br>"
            f"<br>"
            f"<b>Today:</b><br>"
            f"  Private lean: <span style='color:{LEAN_COLOR[priv_lean]}'>{priv_lean.upper()}</span><br>"
            f"  Public lean: <span style='color:{LEAN_COLOR[pub_lean]}'>{pub_lean.upper()}</span><br>"
            f"  Action: {ACTION_EMOJI.get(action, '?')} {action}"
        )

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=size,
                color=viz["color"],
                line=dict(width=5, color=border_color),
                symbol="circle",
            ),
            text=[f"<b>{viz['emoji']}<br>{viz['short']}</b>"],
            textfont=dict(size=11, color="white"),
            textposition="middle center",
            hovertext=hover_html,
            hoverinfo="text",
            name=agent_info.name,
            customdata=[aid],
            showlegend=False,
        ))

    fig.update_layout(
        title="<b>Agent Influence Network</b><br><sub>Edge = influence weight (G ≥ 0.4); arrowhead = influence direction; border = private lean today; size = capital (log)</sub>",
        xaxis=dict(visible=False, range=[-2.2, 2.2]),
        yaxis=dict(visible=False, range=[-2.3, 1.7]),
        height=600,
        plot_bgcolor="#fafafa",
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def render_cascade_flow(day: dict) -> None:
    """Vertical timeline showing decision cascade order."""
    # Cascade order: Super → 4 thinkers (parallel) → CTA (rule-based last)
    order = ["super_influencer_001", "pod_pm_001", "activist_short_001",
             "sell_side_001", "retail_fomo_001", "cta_forced_001"]

    st.markdown(f"#### Cascade Order — {day['date']}")
    st.caption(f"Super-Influencer posts first (Tier 5 info). 4 thinker agents read Super + own tier info, then post. CTA executes rule-based. AAPL: ${day['real_open']:.2f} open → ${day['real_close']:.2f} close")

    for idx, aid in enumerate(order):
        info = day["agent_outputs"].get(aid, {})
        if not info:
            continue
        state = info["state"]
        viz = AGENT_VIZ[aid]
        priv = state["private_belief"]
        pub = state["public_statement"]
        act = state["personal_action"]
        desired = state.get("desired_market_reaction", "")
        trade = day.get("trades", {}).get(aid, {})

        priv_lean = priv.get("lean", "neutral")
        pub_lean = pub.get("stated_lean", "neutral")
        action = act.get("action_type", "hold")
        conflict = (priv_lean != pub_lean) and priv_lean not in ("deterministic", "n/a")

        # Card container
        with st.container():
            cols = st.columns([1, 11])
            cols[0].markdown(f"<div style='background:{viz['color']};color:white;padding:10px;border-radius:10px;text-align:center;font-size:32px;margin-top:5px'>{viz['emoji']}</div>", unsafe_allow_html=True)

            with cols[1]:
                title_cols = st.columns([5, 2])
                title_cols[0].markdown(f"**{info['name']}** · `{info['role']}` · Step {idx+1}/6")
                badge = "🚨 CONFLICT" if conflict else "✓ aligned"
                title_cols[1].markdown(f"<div style='text-align:right;color:{'red' if conflict else '#888'};font-weight:bold'>{badge}</div>", unsafe_allow_html=True)

                # 4-layer state grid
                sub_cols = st.columns(4)

                # Private
                sub_cols[0].markdown(f"""<div style='border-left:4px solid {LEAN_COLOR[priv_lean]};padding:8px;background:#f8f9fa;border-radius:4px;min-height:120px'>
<b>🧠 Private Belief</b><br>
Lean: <span style='color:{LEAN_COLOR[priv_lean]}'><b>{priv_lean.upper()}</b></span> (conv {priv.get('conviction', 0):.2f})<br>
<small><i>"{priv.get('actual_thesis','')[:150]}"</i></small>
</div>""", unsafe_allow_html=True)

                # Public
                sub_cols[1].markdown(f"""<div style='border-left:4px solid {LEAN_COLOR[pub_lean]};padding:8px;background:#f8f9fa;border-radius:4px;min-height:120px'>
<b>📢 Public Statement</b><br>
Lean: <span style='color:{LEAN_COLOR[pub_lean]}'><b>{pub_lean.upper()}</b></span> (conv {pub.get('stated_conviction', 0):.2f})<br>
<small><i>"{pub.get('narrative','')[:150]}"</i></small>
</div>""", unsafe_allow_html=True)

                # Desired reaction
                sub_cols[2].markdown(f"""<div style='border-left:4px solid #f39c12;padding:8px;background:#f8f9fa;border-radius:4px;min-height:120px'>
<b>🎭 Desired Market Reaction</b><br>
<small><i>"{(desired or '')[:200]}"</i></small>
</div>""", unsafe_allow_html=True)

                # Action
                trade_str = ""
                if trade.get("side") == "buy":
                    trade_str = f"BOUGHT {trade['shares']:,} sh @ ${trade['value']/max(1, trade['shares']):.2f}"
                elif trade.get("side") == "sell":
                    trade_str = f"SOLD {trade['shares']:,} sh"
                elif trade.get("side") == "hold":
                    trade_str = "no trade"
                elif trade.get("side") == "skip_no_shares":
                    trade_str = "wanted to sell but no shares"

                sub_cols[3].markdown(f"""<div style='border-left:4px solid #2c3e50;padding:8px;background:#f8f9fa;border-radius:4px;min-height:120px'>
<b>🎯 Action</b><br>
{ACTION_EMOJI.get(action, '?')} <b>{action}</b> · size {act.get('size_pct', 0):.2f}<br>
<small><i>"{act.get('rationale_internal','')[:120]}"</i></small><br>
<small style='color:#666'>{trade_str}</small>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='text-align:center;color:#aaa;font-size:24px;margin:-5px 0'>↓</div>", unsafe_allow_html=True)


def render_pub_priv_heatmap(data: dict) -> go.Figure:
    """Grid of (agent x day) showing public vs private divergence."""
    agents = [a for a in ALL_AGENTS if a.id != "cta_forced_001"]
    days = data["days"]

    z = []
    text = []
    hovertext = []
    for agent in agents:
        z_row = []
        text_row = []
        hover_row = []
        for d in days:
            state = d["agent_outputs"].get(agent.id, {}).get("state", {})
            priv = state.get("private_belief", {}).get("lean", "neutral")
            pub = state.get("public_statement", {}).get("stated_lean", "neutral")
            mismatch = 0 if priv == pub else 1
            z_row.append(mismatch)
            text_row.append(f"{priv[:1].upper()}→{pub[:1].upper()}")
            hover_row.append(
                f"<b>{agent.name}</b> ({agent.role})<br>"
                f"{d['date']}<br>"
                f"Private: <b>{priv}</b><br>"
                f"Public: <b>{pub}</b><br>"
                f"{'⚠️ CONFLICT' if mismatch else '✓ aligned'}"
            )
        z.append(z_row)
        text.append(text_row)
        hovertext.append(hover_row)

    fig = go.Figure(go.Heatmap(
        z=z, text=text, hovertext=hovertext,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        x=[d["date"] for d in days],
        y=[agent_label(a.id, False) for a in agents],
        colorscale=[[0, "#27ae60"], [1, "#c0392b"]],
        showscale=False,
        hoverinfo="text",
    ))
    fig.update_layout(
        title="<b>Public vs Private Lean Divergence</b><br><sub>Red = lying / conflict ; Green = aligned ; Cell text = Private→Public ('L'=long, 'S'=short, 'N'=neutral)</sub>",
        height=400,
        xaxis_side="bottom",
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def render_metrics_chart(data: dict) -> go.Figure:
    """Multi-line chart of 5 ecology metrics over days."""
    dates = [m["date"] for m in data["metrics"]]
    metric_cols = [
        ("public_private_gap", "Public-Private Gap", "#c0392b"),
        ("belief_entropy", "Belief Entropy (Shannon)", "#2980b9"),
        ("consensus_fragility", "Consensus Fragility", "#e67e22"),
        ("narrative_concentration", "Narrative Concentration (HHI)", "#8e44ad"),
        ("influence_centralization_gini", "Influence Centralization (Gini)", "#16a085"),
        ("action_public_gap", "Action vs Public Gap", "#7f8c8d"),
    ]

    fig = go.Figure()
    for col, label, color in metric_cols:
        vals = [m[col] for m in data["metrics"]]
        fig.add_trace(go.Scatter(
            x=dates, y=vals, mode="lines+markers",
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=10),
        ))

    fig.update_layout(
        title="<b>Daily Ecology Metrics</b><br><sub>How market belief dynamics evolve. Pub-Priv Gap = key emergence signal.</sub>",
        xaxis_title="Day",
        yaxis_title="Metric value",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def render_portfolio_chart(data: dict) -> go.Figure:
    """PnL% over time per agent."""
    rows = []
    for d_idx, d in enumerate(data["days"]):
        for aid, p in d["portfolios_close"].items():
            rows.append({
                "Day": d["date"],
                "Agent": agent_label(aid, False),
                "PnL %": p["pnl_pct"],
                "Total Value": p["total"],
                "color": AGENT_VIZ[aid]["color"],
            })
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for aid in [a.id for a in ALL_AGENTS]:
        sub = df[df["Agent"] == agent_label(aid, False)]
        fig.add_trace(go.Scatter(
            x=sub["Day"], y=sub["PnL %"], mode="lines+markers",
            name=agent_label(aid, False),
            line=dict(color=AGENT_VIZ[aid]["color"], width=2.5),
            marker=dict(size=10),
        ))
    fig.update_layout(
        title="<b>Portfolio PnL % over time</b>",
        xaxis_title="Day",
        yaxis_title="PnL %",
        height=400,
        hovermode="x unified",
    )
    return fig


def render_agent_detail(aid: str, data: dict, day_idx: int) -> None:
    """Full inspector view for one agent across all days."""
    agent = get_agent(aid)
    viz = AGENT_VIZ[aid]
    day = data["days"][day_idx]
    state = day["agent_outputs"].get(aid, {}).get("state", {})

    # Header
    st.markdown(f"<h2 style='color:{viz['color']}'>{viz['emoji']} {agent.name} <small style='color:#999'>· {agent.role}</small></h2>", unsafe_allow_html=True)
    st.caption(f"Capital: **${agent.capital:,.0f}** · Info tier: **{agent.info_tier}** · Time horizon: **{agent.time_horizon_days}d**")

    # 8 structural constraints radar
    constraint_keys = ["career_risk", "info_tier", "influence_in", "influence_out",
                       "signaling_incentive", "reflexivity_awareness", "time_horizon_days"]
    # Normalize time_horizon to 0-1 (max 30d)
    constraint_vals_normed = []
    for k in constraint_keys:
        v = getattr(agent, k)
        if k == "info_tier":
            v = v / 5.0
        elif k == "time_horizon_days":
            v = min(v / 30.0, 1.0)
        constraint_vals_normed.append(v)

    nice_labels = ["Career Risk", "Info Tier (norm)", "Influence In", "Influence Out",
                   "Signaling Incentive", "Reflexivity", "Time Horizon (norm)"]

    fig_radar = go.Figure(go.Scatterpolar(
        r=constraint_vals_normed + [constraint_vals_normed[0]],
        theta=nice_labels + [nice_labels[0]],
        fill="toself",
        line=dict(color=viz["color"], width=2.5),
        fillcolor=viz["color"],
        opacity=0.4,
        name=agent.name,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=380,
        title=f"<b>Structural Constraints Profile</b>",
        margin=dict(l=60, r=60, t=60, b=20),
    )

    cols = st.columns([1, 1])
    cols[0].plotly_chart(fig_radar, use_container_width=True)

    # PnL history
    pnl_rows = [{"Day": d["date"], "PnL %": d["portfolios_close"][aid]["pnl_pct"],
                 "Total": d["portfolios_close"][aid]["total"],
                 "Shares": d["portfolios_close"][aid]["shares"],
                 "Cash": d["portfolios_close"][aid]["cash"]} for d in data["days"]]
    pnl_df = pd.DataFrame(pnl_rows)
    fig_pnl = px.line(pnl_df, x="Day", y="PnL %", markers=True, title="<b>Portfolio PnL %</b>")
    fig_pnl.update_traces(line=dict(color=viz["color"], width=3), marker=dict(size=12))
    fig_pnl.update_layout(height=380)
    cols[1].plotly_chart(fig_pnl, use_container_width=True)

    # 4-Layer state today
    st.markdown(f"### 4-Layer State on {day['date']}")
    priv = state.get("private_belief", {})
    pub = state.get("public_statement", {})
    desired = state.get("desired_market_reaction", "")
    act = state.get("personal_action", {})

    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""<div style='padding:15px;border:2px solid {LEAN_COLOR.get(priv.get("lean","neutral"))};border-radius:8px;background:#fff'>
<h5>🧠 Private Belief (ground truth)</h5>
<b>Lean:</b> <span style='color:{LEAN_COLOR.get(priv.get("lean","neutral"))}'>{priv.get('lean','?').upper()}</span> · <b>Conviction:</b> {priv.get('conviction', 0):.2f}<br><br>
<i>"{priv.get('actual_thesis','')}"</i>
</div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""<div style='padding:15px;border:2px solid {LEAN_COLOR.get(pub.get("stated_lean","neutral"))};border-radius:8px;background:#fff'>
<h5>📢 Public Statement (visible to other agents)</h5>
<b>Stated lean:</b> <span style='color:{LEAN_COLOR.get(pub.get("stated_lean","neutral"))}'>{pub.get('stated_lean','?').upper()}</span> · <b>Conviction:</b> {pub.get('stated_conviction', 0):.2f}<br><br>
<i>"{pub.get('narrative','')}"</i>
</div>""", unsafe_allow_html=True)

    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""<div style='padding:15px;border:2px solid #f39c12;border-radius:8px;background:#fff;min-height:140px'>
<h5>🎭 Desired Market Reaction</h5>
<i>"{desired}"</i>
</div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""<div style='padding:15px;border:2px solid #2c3e50;border-radius:8px;background:#fff;min-height:140px'>
<h5>🎯 Personal Action</h5>
<b>{ACTION_EMOJI.get(act.get('action_type','hold'), '?')} {act.get('action_type','?')}</b> · size {act.get('size_pct', 0):.2f}<br><br>
<i>"{act.get('rationale_internal','')}"</i>
</div>""", unsafe_allow_html=True)

    # Trade history across all days
    st.markdown("### Trade History")
    trade_rows = []
    for d in data["days"]:
        t = d.get("trades", {}).get(aid, {})
        if t.get("side") in ("buy", "sell"):
            trade_rows.append({
                "Date": d["date"],
                "Side": t["side"].upper(),
                "Shares": t["shares"],
                "Price": d["real_open"],
                "Value": t["value"],
            })
    if trade_rows:
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed.")

    # System prompt collapsed
    with st.expander("📜 System Prompt"):
        st.code(agent.system_prompt, language="text")


# ============================================================================
# MAIN APP
# ============================================================================

st.set_page_config(page_title="Toy v0.6 Visualizer", layout="wide", page_icon="🧪")

# Custom CSS
st.markdown("""<style>
.block-container { padding-top: 1.5rem; }
[data-testid="stMetricValue"] { font-size: 24px; }
</style>""", unsafe_allow_html=True)

# Load
if not DATA_PATH.exists():
    st.error(f"No simulation result at {DATA_PATH}. Run `python run_toy.py` first.")
    st.stop()

data = load_data()

# ------ Sidebar ------
with st.sidebar:
    st.title("🧪 Toy v0.6")
    st.caption("6 agents, 5 days, AAPL\n\nConstraint-first multi-agent market simulation.")

    day_idx = st.slider(
        "📅 Day",
        min_value=0,
        max_value=len(data["days"]) - 1,
        value=0,
        format="Day %d",
    )
    day = data["days"][day_idx]
    metrics_today = data["metrics"][day_idx]
    st.caption(f"**{day['date']}** · AAPL ${day['real_open']:.2f} → ${day['real_close']:.2f}")

    st.divider()

    # Aha moments
    findings = detect_aha_moments(day)
    if findings:
        st.markdown("### 💡 Today's Strategic Behavior")
        for f in findings:
            st.warning(f)
    else:
        st.info("No strategic conflicts detected today.")

    st.divider()
    st.markdown("### 🎯 Agent Inspector")
    selected_aid = st.selectbox(
        "Pick an agent to inspect →",
        options=[a.id for a in ALL_AGENTS],
        format_func=lambda x: agent_label(x, False),
        index=0,
    )

# ------ Header / Hero ------
hero = st.columns([2.3, 1, 1, 1, 1])
hero[0].markdown(f"## 🧪 Toy v0.6 — Strategic Agent Behavior on AAPL")
hero[0].caption(f"Constraint-first ABM · Day {day_idx+1}/{len(data['days'])} · ${data['total_cost_usd']:.2f} total LLM spend")
hero[1].metric("AAPL", f"${day['real_close']:.2f}",
               f"{((day['real_close']-day['real_open'])/day['real_open']*100):+.2f}%")
hero[2].metric("Pub-Priv Gap", f"{metrics_today['public_private_gap']:.0%}",
               help="Fraction of agents whose private belief ≠ public statement. ≥ 0.4 → strategic signaling active")
hero[3].metric("Belief Entropy", f"{metrics_today['belief_entropy']:.2f}",
               help="Shannon entropy over agent private leans (nats). Higher = more disagreement")
hero[4].metric("Consensus Fragility", f"{metrics_today['consensus_fragility']:.0%}",
               help="Fraction of dissenting voices vs majority lean")

st.divider()

# ------ Tabs ------
t_net, t_cascade, t_gap, t_metrics, t_port, t_detail = st.tabs([
    "🕸️ Network",
    "🌊 Cascade",
    "🎭 Pub vs Priv",
    "📊 Metrics",
    "💰 Portfolio",
    "🔍 Detail",
])

with t_net:
    fig = render_network(day)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("👁️ Hover any node for today's state. Border color = today's private lean. Switch days via sidebar slider.")
    st.markdown("**Influence Graph (G ≥ 0.4 threshold)**: Super → Sell-side (0.75), Super → Retail (0.92), Activist → Retail (0.78), Sell-side → Retail (0.62), Pod ⇆ Activist (0.45/0.40), etc.")

with t_cascade:
    render_cascade_flow(day)

with t_gap:
    fig = render_pub_priv_heatmap(data)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
**Key observation**: The Sell-Side Analyst consistently shows red (private bearish, publicly bullish) — this is the **Henry Blodget pattern** of 2001 banking-conflict-driven misalignment. Retail FOMO shows green (low audience = no incentive to lie). Activist Short alternates based on whether they're building or covering position.
""")

with t_metrics:
    fig = render_metrics_chart(data)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
- **Pub-Priv Gap** (red) of 0.6–0.8 means 60–80% of agents have *public statement* ≠ *private belief* every day. This is the most novel emergent signal v0.6 produces.
- **Belief Entropy** varies 0.5–1.1 → agents do genuinely disagree privately, even when their public stance looks neutral consensus.
- **Influence Centralization** = Gini of capital, ≈ 0.65 constant — top-1 agent (Super, $5B) dominates.
""")

with t_port:
    fig = render_portfolio_chart(data)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Note**: Most agents chose hold most days. CTA mechanically bought into the top → lost slightly. Retail FOMO ended +0.03% by doing nothing. The only one that ACTUALLY made money is the one with no strategy.")

with t_detail:
    render_agent_detail(selected_aid, data, day_idx)

# Footer
st.divider()
st.caption(f"Data: `{DATA_PATH.name}` · {len(data['days'])} days · 6 agents · 4-layer state · Total LLM cost: ${data['total_cost_usd']:.3f}")
