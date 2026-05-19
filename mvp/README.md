# MVP — Virtual Market Price Formation Test

Test which of 3 price-formation modes (α / β / γ) produces virtual prices that best track real market behavior, using 5 LLM personas trading AAPL over ~15 days of post-cutoff historical data.

## Modes

- **α (closed)**: virtual price = pure agent-driven (`prev_virtual × (1 + sensitivity × net_pressure)`)
- **β (anchored)**: virtual price = real-drift + agent pressure (`prev × (1 + real_drift) × (1 + sensitivity × net_pressure)`)
- **γ (dual)**: personas fill at real prices; virtual is auxiliary signal only

## Setup

```bash
cd ~/Desktop/stock-prediction/mvp
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY
streamlit run app.py
```

## What it does

- Fetches AAPL OHLCV from yfinance (cached locally)
- Replays 15 trading days (configurable, default 2026-04-14 → 2026-05-02)
- Each day, 5 personas read price history + own portfolio, output a vote + personal action via Haiku 4.5
- All 3 modes run in parallel — each has its own copy of persona portfolios
- Streamlit dashboard shows:
  1. Real price vs each mode's virtual price (time series)
  2. Persona PnL distribution per mode (bar chart)
  3. Predictive accuracy: today's virtual change vs tomorrow's real change (scatter)
  4. Per-persona portfolio detail table
- Sensitivity slider lets you re-run β with different agent influence weights

## Cost

~$0.50–$1.00 in Anthropic Haiku API spend per full run (with prompt caching).
LLM responses cached locally so UI reruns are free.

## Personas (MVP, hand-tuned subset of full v0.4 design)

| ID | Family | Archetype | Initial Capital |
|---|---|---|---|
| retail_fomo_bull_001 | Retail | WSB Ape style | $8,000 |
| retail_permabear_001 | Retail | Perma-Bear | $70,000 |
| inst_quant_pod_001 | Institutional | Citadel-style Pod | $300M |
| expert_buffett_001 | Expert | Quality + Moat | $1B |
| expert_burry_001 | Expert | Contrarian Short | $200M |

## Files

- `personas.py` — 5 persona definitions with system prompts
- `data.py` — yfinance loader with parquet cache
- `llm.py` — Anthropic SDK wrapper with prompt caching
- `price_engine.py` — 3 price formation modes
- `simulation.py` — Main orchestrator
- `app.py` — Streamlit dashboard

## Known limitations (deliberate)

- Single ticker (AAPL), single window
- No news input (personas see only price history)
- 5 personas only (real v0.4 has 90)
- No persona evolution (all 6 params frozen)
- No human-in-the-loop / Trader Agent / Group Leaders (MVP cuts straight to per-persona action)
