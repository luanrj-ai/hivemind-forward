"""Forward Predictor — dashboard for the leakage-free walk-forward experiment.

Reads forward_v01/results/ and shows: the running scorecard (direction hit-rate
with a Wilson CI + significance flag, CI calibration, MAE), the cumulative
hit-rate over time, the latest open predictions per ticker (consensus + 95% CI +
vote split + sample agent reasoning), and the population composition.

    streamlit run forward_v01/viz_app_forward.py --server.port 8503
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from forward_v01 import scoreboard as sb  # noqa: E402

RESULTS = HERE / "results"


def _read_jsonl(name: str) -> list[dict]:
    p = RESULTS / name
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


st.set_page_config(page_title="Forward Predictor", page_icon="📈", layout="wide")
st.title("📈 Forward Multi-Agent Predictor")
st.caption("~300 real LLM agents predict next-day moves each trading day; scored "
           "against LIVE data the next run. The target close doesn't exist at "
           "prediction time — so there's **no lookahead and no training-data leakage**.")

scored = _read_jsonl("scored.jsonl")
pending = _read_jsonl("pending.jsonl")
pop = json.load(open(RESULTS / "population.json")) if (RESULTS / "population.json").exists() else None

# ── Scorecard ─────────────────────────────────────────────────────────────────
st.header("Scorecard")
if not scored:
    st.info("No scored predictions yet. Run `python -m forward_v01.daily` across "
            "several trading days, then come back.")
else:
    board = sb.build(scored)
    o = board["overall"]
    c1, c2, c3, c4 = st.columns(4)
    hr = o["hit_rate"]
    c1.metric("Direction hit-rate",
              f"{hr*100:.1f}%" if hr is not None else "—",
              f"{o['n_directional']} directional calls")
    lo, hi = o["hit_ci95"]
    c2.metric("95% CI (Wilson)", f"[{lo*100:.0f}, {hi*100:.0f}]%",
              "excludes 50% ✓" if o["coinflip_excluded"] else "straddles 50%")
    cal = o["ci_calibration"]
    c3.metric("CI calibration", f"{cal*100:.0f}%" if cal is not None else "—",
              "target ≈ 95%")
    c4.metric("Mean abs error", f"{o['mae_pct']:.2f}%", f"n={o['n']}")

    if not o["coinflip_excluded"]:
        st.warning(f"With {o['n_directional']} directional observations, the hit-rate "
                   "is **not yet statistically distinguishable from a coin flip**. "
                   "Forward testing needs dozens of trading days to separate edge from noise.")

    st.subheader("Per ticker")
    rows = []
    for t, s in board["per_ticker"].items():
        rows.append({
            "ticker": t, "n": s["n"], "directional": s["n_directional"],
            "hit_rate %": round(s["hit_rate"] * 100, 1) if s["hit_rate"] is not None else None,
            "CI low %": round(s["hit_ci95"][0] * 100, 1),
            "CI high %": round(s["hit_ci95"][1] * 100, 1),
            "CI-calib %": round(s["ci_calibration"] * 100, 1) if s["ci_calibration"] is not None else None,
            "MAE %": s["mae_pct"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # cumulative hit-rate over time
    st.subheader("Cumulative direction hit-rate over time")
    dfh = pd.DataFrame([r for r in scored if r.get("pred_dir", 0) != 0])
    if not dfh.empty:
        dfh = dfh.sort_values("target_date")
        dfh["cum_hits"] = dfh["direction_correct"].cumsum()
        dfh["cum_n"] = range(1, len(dfh) + 1)
        dfh["cum_rate"] = dfh["cum_hits"] / dfh["cum_n"] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfh["target_date"], y=dfh["cum_rate"],
                                 mode="lines+markers", name="cum hit-rate"))
        fig.add_hline(y=50, line_dash="dash", line_color="gray",
                      annotation_text="coin flip (50%)")
        fig.update_layout(yaxis_title="hit-rate %", xaxis_title="target date",
                          height=340, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ── Latest open predictions ──────────────────────────────────────────────────
st.header("Open predictions (awaiting outcome)")
if not pending:
    st.info("No open predictions.")
else:
    for rec in sorted(pending, key=lambda r: (r["ticker"], r["as_of_date"])):
        fc = rec["forecast"]
        arrow = "🔺" if fc["consensus_lean"] == "long" else "🔻" if fc["consensus_lean"] == "short" else "➡️"
        with st.expander(
            f"{arrow} {rec['ticker']}  ·  as of {rec['as_of_date']}  ·  "
            f"T+{rec['horizon_days']} = {fc['expected_return_pct']:+.2f}% "
            f"[{fc['ci_low_pct']:+.2f}, {fc['ci_high_pct']:+.2f}]",
            expanded=False,
        ):
            a, b, c = st.columns(3)
            a.metric("Expected", f"{fc['expected_return_pct']:+.2f}%")
            b.metric("Votes L/S/N", f"{fc['n_long']}/{fc['n_short']}/{fc['n_neutral']}",
                     f"{fc['n_abstain']} abstain")
            c.metric("Consensus", fc["consensus_lean"],
                     f"strength {fc['consensus_strength']:.2f}")
            st.caption(f"close ${rec['t0_close']} · {rec.get('n_headlines', 0)} headlines · "
                       f"{rec['n_agents']} agents · model {rec['model']} · "
                       f"social rounds {fc.get('social_rounds', 0)}")
            for n in rec.get("sample_narratives", []):
                st.write(f"• {n}")

# ── Scored history ────────────────────────────────────────────────────────────
if scored:
    st.header("Scored history")
    hist = pd.DataFrame([{
        "as_of": r["as_of_date"], "target": r.get("target_date"), "ticker": r["ticker"],
        "pred %": r["forecast"]["expected_return_pct"], "actual %": r["actual_return_pct"],
        "dir": "✓" if r["direction_correct"] else "✗",
        "in CI": "✓" if r["in_ci"] else "✗", "abs err %": r["abs_error_pct"],
    } for r in scored])
    st.dataframe(hist.sort_values("target", ascending=False),
                 use_container_width=True, hide_index=True)

# ── Population ────────────────────────────────────────────────────────────────
if pop:
    with st.sidebar:
        st.header("Population")
        st.caption(f"{pop['n']} unique personas · seed {pop['seed']}")
        mix = pd.Series([p["archetype"] for p in pop["personas"]]).value_counts()
        st.bar_chart(mix)
        n_edges = sum(len(v) for v in pop["edges"].values())
        neg = sum(1 for v in pop["edges"].values() for _, w in v if w < 0)
        st.caption(f"Influence net: {n_edges} edges, {neg} contrarian (−), "
                   f"avg degree {n_edges/pop['n']:.1f}")
