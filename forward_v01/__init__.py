"""forward_v01 — walk-forward, leakage-free multi-agent stock predictor.

Every trading day: predict next-day moves for a small universe with ~300 real
LLM agents per stock, then score yesterday's predictions against LIVE data the
next run. Because the target close does not exist at prediction time, this kills
both lookahead bias and LLM training-data contamination.
"""
