"""Entry point: run toy v0.6 simulation, compute metrics, save & plot."""

import json
import sys
import time
from pathlib import Path

from agents import ALL_AGENTS
from metrics import compute_day_metrics
from simulation import run_simulation


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    start = time.time()

    def on_step(idx, total, msg):
        elapsed = time.time() - start
        eta = (elapsed / idx) * (total - idx) if idx > 0 else 0
        print(f"  [{idx:2d}/{total}] {msg} | elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print("=" * 70, flush=True)
    print("TOY v0.6 — 6 Agents, Influence Graph, Information Hierarchy, 5 days", flush=True)
    print("=" * 70, flush=True)
    print(f"Agents:", flush=True)
    for a in ALL_AGENTS:
        print(f"  {a.id:30s} tier={a.info_tier} out={a.influence_out:.2f} cap=${a.capital:,.0f}", flush=True)
    print("", flush=True)
    print("Running simulation...", flush=True)

    result = run_simulation(
        ticker="AAPL",
        start_date="2026-04-21",
        end_date="2026-04-25",
        use_llm_cache=True,
        on_step=on_step,
    )

    elapsed = time.time() - start
    print("", flush=True)
    print(f"Simulation done in {elapsed:.0f}s. Cost: ${result['total_cost_usd']:.3f}", flush=True)
    print("", flush=True)

    # Compute metrics per day
    metrics_series = []
    for day_record in result["days"]:
        m = compute_day_metrics(day_record)
        metrics_series.append(m)

    # Save full result + metrics
    result["metrics"] = metrics_series
    out_path = RESULTS_DIR / "toy_run_latest.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {out_path}", flush=True)
    print("", flush=True)

    # Print summary
    print("=" * 70, flush=True)
    print("DAILY METRICS", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Date':12} {'Nar.Conc':>9} {'B.Entrop':>9} {'Infl.Gini':>10} {'Fragil':>7} {'Pub-Priv':>9} {'Act-Pub':>8}", flush=True)
    for m in metrics_series:
        print(f"{m['date']:12} {m['narrative_concentration']:>9.3f} {m['belief_entropy']:>9.3f} {m['influence_centralization_gini']:>10.3f} {m['consensus_fragility']:>7.3f} {m['public_private_gap']:>9.3f} {m['action_public_gap']:>8.3f}", flush=True)
    print("", flush=True)

    print("=" * 70, flush=True)
    print("FINAL PORTFOLIO PnL", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Agent':35} {'Role':18} {'Initial':>15} {'Final':>15} {'PnL%':>8}", flush=True)
    for aid, p in result["final_portfolios"].items():
        print(f"{aid:35} {p['role']:18} ${p['initial_capital']:>14,.0f} ${p['final_total']:>14,.0f} {p['final_pnl_pct']:>+7.2f}%", flush=True)
    print("", flush=True)

    # Print sample of one day's outputs to inspect cascade quality
    print("=" * 70, flush=True)
    print(f"SAMPLE DAY: {result['days'][2]['date']} (middle day, all agents)", flush=True)
    print("=" * 70, flush=True)
    for aid, info in result["days"][2]["agent_outputs"].items():
        state = info["state"]
        print(f"\n[{info['name']}] ({info['role']})", flush=True)
        priv = state['private_belief']
        pub = state['public_statement']
        act = state['personal_action']
        print(f"  PRIVATE:  lean={priv.get('lean','?'):8s} conv={priv.get('conviction', 0):.2f}", flush=True)
        print(f"            thesis: \"{priv.get('actual_thesis', '')[:120]}\"", flush=True)
        print(f"  PUBLIC:   lean={pub.get('stated_lean','?'):8s} conv={pub.get('stated_conviction', 0):.2f}", flush=True)
        print(f"            says: \"{pub.get('narrative', '')[:120]}\"", flush=True)
        print(f"  DESIRED:  \"{state.get('desired_market_reaction','')[:120]}\"", flush=True)
        print(f"  ACTION:   {act.get('action_type','?')} size={act.get('size_pct', 0):.2f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
