"""Non-LLM time-series baseline — a quantitative next-day forecast.

Used two ways in forward_v01:
  1. an institutional-tier signal: shown ONLY to tier>=4 agents (info asymmetry),
  2. a scored benchmark: tracked alongside the LLM crowd so we can ask "does the
     750-agent crowd actually beat a plain time-series model on the SAME
     leakage-free forward test?"

By default a LIGHTWEIGHT statistical baseline runs (drift + vol quantiles,
zero deps, honest, labeled "naive-drift-vol"). Real TimesFM is opt-in via
FORWARD_TSFM=1 (heavy: torch + ~GB checkpoint download) — if it imports and
runs, it is used; otherwise we fall back to the statistical baseline. Either
way the architecture (institutional prior + scored benchmark) is identical.
"""
from __future__ import annotations

import math
import os

_NORM_80 = 1.2816  # z for the 10/90 percentiles (80% central interval)
_TSFM = None       # cached TimesFM model singleton


def _stat_forecast(closes: list[float], horizon: int) -> dict:
    """Random-walk-with-drift baseline. Mean = damped recent daily drift;
    80% interval from realized daily vol, scaled by sqrt(horizon)."""
    if len(closes) < 6:
        return {"mean_pct": 0.0, "q10_pct": -1.0, "q90_pct": 1.0, "model": "naive-drift-vol"}
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    recent = rets[-20:]
    mu = sum(recent) / len(recent)
    var = sum((r - mu) ** 2 for r in recent) / max(1, len(recent) - 1)
    sigma = math.sqrt(var)
    # damp the drift: daily drift is mostly noise — don't overstate direction
    mean = 0.5 * mu * horizon
    half = _NORM_80 * sigma * math.sqrt(horizon)
    return {
        "mean_pct": round(mean * 100, 3),
        "q10_pct": round((mean - half) * 100, 3),
        "q90_pct": round((mean + half) * 100, 3),
        "model": "naive-drift-vol",
    }


def _tsfm_forecast(closes: list[float], horizon: int) -> dict:
    """Best-effort TimesFM 2.5 adapter. Guarded — any import/API/runtime issue
    falls back to the statistical baseline. Verify the exact API when enabling."""
    global _TSFM
    import numpy as np
    import timesfm  # raises ImportError if not installed
    if _TSFM is None:
        # API differs across TimesFM versions; keep this isolated + guarded.
        _TSFM = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="cpu", horizon_len=max(8, horizon)),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.5-200m-pytorch"),
        )
    pt, qt = _TSFM.forecast([np.array(closes, dtype=float)], freq=[0])
    last = closes[-1]
    mean_px = float(pt[0][horizon - 1])
    # quantile tensor layout: [series][horizon][quantile]; q10≈idx1, q90≈idx9
    q = qt[0][horizon - 1]
    q10_px, q90_px = float(q[1]), float(q[9])
    return {
        "mean_pct": round((mean_px / last - 1) * 100, 3),
        "q10_pct": round((q10_px / last - 1) * 100, 3),
        "q90_pct": round((q90_px / last - 1) * 100, 3),
        "model": "timesfm-2.5-200m",
    }


def forecast(closes: list[float], horizon: int = 1) -> dict:
    """Next-h-day forecast as % return: {mean_pct, q10_pct, q90_pct, model}."""
    closes = [float(c) for c in closes if c is not None]
    if os.environ.get("FORWARD_TSFM") == "1":
        try:
            return _tsfm_forecast(closes, horizon)
        except Exception as e:  # not installed / API drift / OOM → degrade
            print(f"  ⚠ TimesFM unavailable ({str(e)[:60]}); using statistical baseline")
    return _stat_forecast(closes, horizon)


if __name__ == "__main__":
    demo = [100 + 5 * math.sin(i / 6) + i * 0.3 for i in range(80)]
    print(forecast(demo, 1))
