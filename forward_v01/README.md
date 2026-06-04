# forward_v01 — leakage-free, walk-forward multi-agent predictor

Every trading day, **~300 real LLM agents** each read today's real market context
(live prices + indicators + GDELT news) and reason **in language** to their own
directional view. Their views are coupled by a signed influence network (herding),
aggregated into a next-day forecast, and **scored against LIVE data the next run**.

The target close **does not exist** when the prediction is made, so this kills
both lookahead bias *and* LLM training-data contamination — the honest test of
whether the ensemble has any real edge.

## The one command

```bash
ANALYST_LLM=claude .venv/bin/python -m forward_v01.daily --agents 300
```

It does two things, in order:

1. **SCORE** — every pending prediction whose target trading day now has a live
   close is scored (direction / in-CI / abs-error) and moved to `scored.jsonl`.
   A prediction is *never* scored in the run that made it (its target bar doesn't
   exist yet) — that gap is the leakage guarantee.
2. **PREDICT** — for each of 5 tickers (AAPL MSFT NVDA GOOGL AMZN), gather today's
   live context, run the 300-agent population, aggregate, append to `pending.jsonl`.

Re-running the same day is cheap and safe: the LLM cache (`analyst_v01/cache`)
makes completed agents instant, so a re-run only refills agents that previously
failed/abstained. This is what lets a quota-paced run resume across windows.

## Scorecard & dashboard

```bash
.venv/bin/python -m forward_v01.scoreboard --json     # direction hit-rate + Wilson CI
.venv/bin/streamlit run forward_v01/viz_app_forward.py --server.port 8503
```

## Quota / cost (claude subscription)

300 agents × 5 tickers = **1500 LLM calls/day**, but a claude window is ~400
calls/~5h, so a full run paces across ~4–5 windows (most of a day) via low
concurrency + window-aware backoff. Subscription = $0 marginal, but it consumes
your personal claude headroom that day. Knobs (env): `FORWARD_MODEL` (default
`sonnet`; `haiku` is far lighter on quota), `FORWARD_CONCURRENCY`,
`FORWARD_QUOTA_RETRIES`, `FORWARD_SOCIAL_ROUNDS` (herding, default 1),
`LLM_PACE_SEC`.

## Files

| file | role |
|---|---|
| `population.py` | build 300 unique personas + signed influence net → `results/population.json` |
| `data_live.py` | force-live yfinance + GDELT (retry/fallback); as-of context & next-day actuals |
| `agent_think.py` | per-persona LLM reasoning; `think_population` (low concurrency + quota backoff) |
| `aggregate.py` | social propagation (herding) + capital/influence/horizon-weighted forecast |
| `daily.py` | the score-then-predict command |
| `scoreboard.py` | running hit-rate (Wilson CI), CI calibration, MAE |
| `viz_app_forward.py` | Streamlit dashboard |
| `results/*.jsonl` | `pending` (awaiting outcome) and `scored` (committed for the web view) |

## Scheduling (run daily, view on the web)

Preferred: a `/schedule` cloud routine running the command each trading day,
emitting a daily summary (read on claude.ai), then committing `results/` back so
the HF Space dashboard redeploys. **Caveat to verify at setup:** the routine must
(a) stay alive for the multi-window paced run and (b) have a usable `claude`
auth in the cloud env. If either fails, fall back to a local `launchd` job
(claude is already authenticated locally) that pushes results on completion.

## Honest expectation

Statistical significance needs **dozens of trading days** of accumulated, scored
predictions — the scoreboard explicitly flags when the hit-rate CI still straddles
50% (indistinguishable from a coin flip). That patience is the price of a clean,
leakage-free evaluation.
