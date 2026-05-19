# Stock Sim — You vs 11 AI Agents

A trading simulator where you go head-to-head against 11 LLM-driven agents (a hedge-fund PM, an activist short seller, a Cathie-Wood-style influencer, a CTA, retail FOMO, three economists, and more) trading AAPL over 32 days.

Two apps in this repo, both loading the same locked-in agent decisions from a prior LLM run:

| App | Purpose | File |
|---|---|---|
| **Stock Sim** (toC) | Play against the agents day-by-day, peek their private thoughts, project what-if scenarios | `toy_v06/viz_app_sim.py` |
| **Social Model** (research) | Inspect agent reputation, memory, deception, and the Ternus event case study | `toy_v06/viz_app_v07.py` |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Stock sim (port 8502)
streamlit run toy_v06/viz_app_sim.py --server.port 8502

# Social model viz (port 8501)
streamlit run toy_v06/viz_app_v07.py --server.port 8501
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already done if you're reading this).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **New app** → pick this repo, branch `main`, main file `toy_v06/viz_app_sim.py`.
4. Click **Deploy**. You'll get a `*.streamlit.app` URL in 2–3 minutes.

No environment variables required for the sim app — it loads everything from `toy_v06/results/v07_demo_latest.json` (the pre-computed agent decisions).

## What's inside

- `toy_v06/` — the sim engine, agents, price models, viz apps
- `toy_v06/results/v07_demo_latest.json` — locked-in agent run (32 days × 11 agents)
- `toy_v06/cache/news/` — pre-fetched news headlines used by the agents
- `mvp/data.py` — yfinance loader used by both apps

## Architecture

- 11 agents with hand-tuned personalities (8 structural parameters each + system prompts)
- 4-layer agent state: private belief / public statement / desired market reaction / personal action
- β-anchored price formation: virtual = prev × (1 + real_drift) × (1 + sensitivity × agent_pressure) × (1 - λ × deviation)
- Temporal-graph memory (Graphiti-style facts with validity windows)
- Dynamic reputation network (multiplier = 0.5 + rolling accuracy)
- Linear price-impact slippage (large orders pay)
- Per-role short-sale leverage (only activists, pods, CTAs, day-traders can short)

See `toy_v06/agents.py` for the full agent roster and `toy_v06/run_v07_demo.py` for the simulation loop.
