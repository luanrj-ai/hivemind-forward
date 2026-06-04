"""Streamlit dashboard for MVP simulation."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from personas import ALL_PERSONAS
from simulation import load_result, run_simulation, save_result


st.set_page_config(page_title="MVP: Virtual Market Price Formation", layout="wide")

st.title("🧪 MVP — Virtual Market Price Formation Test")
st.caption("Compare 3 price-formation modes (α/β/γ) on 5 LLM personas trading AAPL.")

# ---- Sidebar Controls ----
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Ticker", value="AAPL")
start_date = st.sidebar.text_input("Start (YYYY-MM-DD)", value="2026-04-14")
end_date = st.sidebar.text_input("End (YYYY-MM-DD)", value="2026-05-02")
sensitivity = st.sidebar.slider("Sensitivity (β & γ only)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Modes")
run_alpha = st.sidebar.checkbox("α — Closed", value=True)
run_beta = st.sidebar.checkbox("β — Anchored hybrid", value=True)
run_gamma = st.sidebar.checkbox("γ — Dual layer", value=True)

st.sidebar.markdown("---")
backend_label = st.sidebar.radio(
    "LLM Backend",
    ["Mock (free, deterministic)", "Claude CLI (uses your subscription)", "Anthropic SDK (needs API key)"],
    index=0,
    help="Mock = no API. Claude CLI = subprocess `claude -p`, uses Claude Code auth. SDK = requires ANTHROPIC_API_KEY env/.env.",
)
backend_map = {
    "Mock (free, deterministic)": "mock",
    "Claude CLI (uses your subscription)": "claude_cli",
    "Anthropic SDK (needs API key)": "anthropic_sdk",
}
backend = backend_map[backend_label]
mock_mode = backend == "mock"

max_concurrency = st.sidebar.slider("Parallelism (LLM only)", 1, 10, 5)

if backend == "anthropic_sdk":
    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key_set:
        st.sidebar.warning("⚠️ ANTHROPIC_API_KEY not set. Export it or add to mvp/.env.")
elif backend == "claude_cli":
    st.sidebar.info("ℹ️ Uses `claude` CLI subprocess. ~$0.03/call. Will subprocess your local Claude Code auth.")

st.sidebar.markdown("---")
col_a, col_b = st.sidebar.columns(2)
run_button = col_a.button("▶ Run", type="primary", use_container_width=True)
load_cached = col_b.button("Load last", use_container_width=True)

# ---- Run / Load ----
if "result" not in st.session_state:
    st.session_state.result = None

if run_button:
    modes_tup = tuple([m for m in ("alpha", "beta", "gamma") if locals()[f"run_{m}"]])
    if not modes_tup:
        st.error("Pick at least one mode.")
    else:
        progress = st.progress(0.0, text="Starting simulation…")

        def on_step(idx, total, msg):
            progress.progress(idx / total, text=f"{idx}/{total} • {msg}")

        try:
            result = run_simulation(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                sensitivity=sensitivity,
                modes=modes_tup,
                use_llm_cache=True,
                mock=mock_mode,
                backend=backend,
                max_concurrency=max_concurrency,
                on_step=on_step,
            )
            save_result(result, "latest")
            st.session_state.result = result
            progress.empty()
            st.success(f"Done. {len(result.days)} day-mode records.")
        except Exception as e:
            progress.empty()
            st.error(f"Simulation failed: {e}")
            raise

if load_cached:
    r = load_result("latest")
    if r is None:
        st.error("No cached result found.")
    else:
        st.session_state.result = r
        st.success(f"Loaded cached result: {r.ticker} {r.start_date}→{r.end_date} sens={r.sensitivity}")

result = st.session_state.result

if result is None:
    st.info("👉 Click ▶ Run on the left to start a simulation, or Load last to view cached results.")
    st.markdown("### What this tests")
    st.markdown("""
    Three different ways for **virtual market price** to evolve given **persona buying / selling**:

    - **α (Closed)**: virtual price evolves PURELY by net agent pressure.
      `next = prev × (1 + sens × pressure)` — ignores real market entirely.
    - **β (Anchored hybrid)**: virtual price tracks real drift + agent pressure additively.
      `next = prev × (1 + real_drift) × (1 + sens × pressure)`
    - **γ (Dual layer)**: personas FILL at real prices; virtual price is auxiliary signal.
      `virtual = real × (1 + sens × pressure)`, but persona PnL uses real.

    **Sensitivity** (β & γ) controls how much agent pressure moves virtual relative to real.
    """)
    st.stop()

# ---- Build DataFrames for visualization ----
days_data = []
for d in result.days:
    days_data.append({
        "date": d.date,
        "mode": d.mode,
        "real_open": d.real_open,
        "real_close": d.real_close,
        "virtual_open": d.virtual_open,
        "virtual_close": d.virtual_close,
        "buy_volume_usd": d.buy_volume_usd,
        "sell_volume_usd": d.sell_volume_usd,
        "net_pressure": d.net_pressure,
    })
days_df = pd.DataFrame(days_data)
days_df["date"] = pd.to_datetime(days_df["date"])

# ---- View 1: Price Time Series ----
st.header("📈 Real vs Virtual Price")
st.caption("Per-mode virtual price evolution next to the actual market price.")

fig1 = go.Figure()

# Real price (shared across modes; just plot once)
real_df = days_df[days_df["mode"] == result.modes[0]].sort_values("date")
fig1.add_trace(go.Scatter(
    x=real_df["date"], y=real_df["real_close"],
    name="Real (close)", mode="lines+markers",
    line=dict(color="#222", width=3, dash="solid"),
))

mode_colors = {"alpha": "#e74c3c", "beta": "#3498db", "gamma": "#27ae60"}
mode_names = {"alpha": "α Closed", "beta": "β Anchored", "gamma": "γ Dual"}

for m in result.modes:
    sub = days_df[days_df["mode"] == m].sort_values("date")
    fig1.add_trace(go.Scatter(
        x=sub["date"], y=sub["virtual_close"],
        name=f"{mode_names[m]} virtual",
        mode="lines+markers",
        line=dict(color=mode_colors[m], dash="dot"),
    ))

fig1.update_layout(height=450, hovermode="x unified",
                   xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig1, use_container_width=True)

# ---- View 2: Persona PnL by Mode ----
st.header("💰 Persona PnL by Mode (final)")
st.caption("How much each persona made (or lost) in each parallel world.")

pnl_rows = []
for m in result.modes:
    for pid, pdat in result.final_portfolios[m].items():
        persona = next(p for p in ALL_PERSONAS if p.id == pid)
        pnl_rows.append({
            "mode": mode_names[m],
            "persona": pid.replace(f"{persona.family}_", ""),
            "family": persona.family,
            "archetype": persona.archetype,
            "pnl_pct": pdat["pnl_pct"],
            "total_value": pdat["total_value"],
            "initial_capital": pdat["initial_capital"],
            "transactions": len(pdat["transactions"]),
        })
pnl_df = pd.DataFrame(pnl_rows)

fig2 = px.bar(
    pnl_df,
    x="persona", y="pnl_pct", color="mode",
    barmode="group",
    title="Final PnL % by Persona × Mode",
    color_discrete_map={mode_names[m]: mode_colors[m] for m in result.modes},
    height=450,
)
fig2.update_layout(xaxis_title="", yaxis_title="PnL %", xaxis={'categoryorder': 'array',
                                                                'categoryarray': [r["persona"] for r in pnl_rows[:len(ALL_PERSONAS)]]})
st.plotly_chart(fig2, use_container_width=True)

# ---- View 3: Predictive accuracy (virtual today vs real tomorrow) ----
st.header("🎯 Predictive Accuracy")
st.caption("Each dot = (today's virtual-vs-real divergence) vs (tomorrow's real price change). If the virtual signal predicts real moves, dots should align with the diagonal.")

# Build: for each (date, mode), compute virtual_drift_today = (virtual_close - real_close) / real_close
# Then real_drift_tomorrow = (real_close_tomorrow - real_close_today) / real_close_today
predict_rows = []
for m in result.modes:
    sub = days_df[days_df["mode"] == m].sort_values("date").reset_index(drop=True)
    for i in range(len(sub) - 1):
        v = sub.loc[i, "virtual_close"]
        r = sub.loc[i, "real_close"]
        r_next = sub.loc[i + 1, "real_close"]
        if r > 0:
            predict_rows.append({
                "mode": mode_names[m],
                "date": sub.loc[i, "date"],
                "virtual_minus_real_pct": (v - r) / r * 100,
                "tomorrow_real_change_pct": (r_next - r) / r * 100,
            })

if predict_rows:
    predict_df = pd.DataFrame(predict_rows)
    fig3 = px.scatter(
        predict_df,
        x="virtual_minus_real_pct",
        y="tomorrow_real_change_pct",
        color="mode",
        trendline="ols",
        color_discrete_map={mode_names[m]: mode_colors[m] for m in result.modes},
        labels={
            "virtual_minus_real_pct": "Virtual − Real (% today)",
            "tomorrow_real_change_pct": "Real change (% tomorrow)",
        },
        title="Does virtual signal lead real price?",
        height=500,
    )
    fig3.add_hline(y=0, line_dash="dot", line_color="gray")
    fig3.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig3, use_container_width=True)

    # Compute correlation per mode
    st.subheader("Correlation: virtual today × real tomorrow")
    corr_data = []
    for m in result.modes:
        sub = predict_df[predict_df["mode"] == mode_names[m]]
        if len(sub) >= 3:
            c = sub["virtual_minus_real_pct"].corr(sub["tomorrow_real_change_pct"])
            corr_data.append({"mode": mode_names[m], "n": len(sub), "correlation": c})
    if corr_data:
        st.dataframe(pd.DataFrame(corr_data), hide_index=True, use_container_width=True)

# ---- View 4: Per-persona detail table ----
st.header("🧑 Per-Persona Detail")
st.dataframe(
    pnl_df[["mode", "family", "archetype", "persona", "initial_capital", "total_value", "pnl_pct", "transactions"]]
        .sort_values(["mode", "pnl_pct"], ascending=[True, False])
        .style.format({"initial_capital": "${:,.0f}", "total_value": "${:,.0f}", "pnl_pct": "{:+.2f}%"}),
    use_container_width=True,
    hide_index=True,
)

# ---- View 5: Pressure & flow per day ----
with st.expander("📊 Daily Flow & Pressure"):
    fig4 = go.Figure()
    for m in result.modes:
        sub = days_df[days_df["mode"] == m].sort_values("date")
        fig4.add_trace(go.Bar(
            x=sub["date"], y=sub["net_pressure"],
            name=f"{mode_names[m]} pressure",
            marker_color=mode_colors[m], opacity=0.6,
        ))
    fig4.update_layout(barmode="group", height=350, xaxis_title="Date", yaxis_title="Net pressure")
    st.plotly_chart(fig4, use_container_width=True)
