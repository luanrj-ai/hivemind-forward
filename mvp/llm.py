"""LLM wrapper with two backends: Anthropic SDK or claude CLI subprocess.

Both backends share the same on-disk response cache so re-runs are free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from anthropic import Anthropic, APIError
from dotenv import load_dotenv

from personas import PersonaDef


load_dotenv(Path(__file__).parent / ".env")

CACHE_DIR = Path(__file__).parent / "cache" / "llm_responses"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_HAIKU = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 512


@dataclass
class PersonaAction:
    vote: Literal["long", "flat", "short"]
    vote_conviction: float
    action_type: Literal["buy_strong", "buy_lite", "hold", "sell_lite", "sell_strong"]
    action_size_pct: float
    rationale: str
    raw_response: str = ""

    @classmethod
    def fallback_hold(cls, reason: str = "parse_error") -> "PersonaAction":
        return cls(
            vote="flat",
            vote_conviction=0.0,
            action_type="hold",
            action_size_pct=0.0,
            rationale=f"[fallback: {reason}]",
            raw_response="",
        )


def _client() -> Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to mvp/.env or export it before running."
        )
    return Anthropic(api_key=key)


def _build_user_message(
    ticker: str,
    price_history_csv: str,
    cash: float,
    shares: int,
    total_value: float,
    initial_capital: float,
    decision_date: str,
    mode: str,
) -> str:
    pnl_pct = (total_value - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0
    return f"""Decision date: {decision_date} (US market open)
Ticker: {ticker}
Simulation mode: {mode}

Last 30 trading days OHLCV:
{price_history_csv}

Your current portfolio:
- Cash: ${cash:,.2f}
- {ticker} shares: {shares}
- Total portfolio value: ${total_value:,.2f}
- Initial capital: ${initial_capital:,.2f}
- PnL: {pnl_pct:+.2f}%

Output your decision as JSON ONLY (no prose before or after):
{{"vote": ..., "vote_conviction": ..., "action_type": ..., "action_size_pct": ..., "rationale": "..."}}"""


def _cache_key(persona_id: str, ticker: str, decision_date: str, price_hash: str, portfolio_hash: str, mode: str, prompt_version: str = "v1") -> str:
    h = hashlib.sha256()
    h.update(f"{persona_id}|{ticker}|{decision_date}|{price_hash}|{portfolio_hash}|{mode}|{prompt_version}".encode())
    return h.hexdigest()[:16]


def _parse_json(raw: str) -> dict | None:
    """Try to extract JSON from response. Returns None on failure."""
    # find first { ... } block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # try fixing common issues
        cleaned = blob.replace("'", '"').replace("\n", " ")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


Backend = Literal["mock", "anthropic_sdk", "claude_cli"]


def _call_via_claude_cli(system_prompt: str, user_msg: str, max_budget_usd: float = 0.10) -> tuple[str, float]:
    """Subprocess the `claude` CLI. Returns (raw_text, cost_usd). Raises on failure."""
    cmd = [
        "claude", "-p",
        "--model", "haiku",
        "--no-session-persistence",
        "--output-format", "json",
        "--max-budget-usd", str(max_budget_usd),
        "--system-prompt", system_prompt,
        user_msg,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:300]}")
    data = json.loads(result.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude API error: {data.get('result', '')[:200]}")
    return data.get("result", ""), float(data.get("total_cost_usd", 0.0))


def persona_decide(
    persona: PersonaDef,
    ticker: str,
    price_history_csv: str,
    cash: float,
    shares: int,
    total_value: float,
    decision_date: str,
    mode: str,
    use_cache: bool = True,
    mock: bool = False,
    backend: Backend = "anthropic_sdk",
) -> PersonaAction:
    """Get one persona's decision via Haiku 4.5. Cached to disk by inputs."""
    # mock takes precedence for backward compat
    if mock:
        backend = "mock"

    price_hash = hashlib.sha256(price_history_csv.encode()).hexdigest()[:8]
    portfolio_hash = hashlib.sha256(f"{cash}|{shares}|{total_value}".encode()).hexdigest()[:8]
    key = _cache_key(persona.id, ticker, decision_date, price_hash, portfolio_hash, mode, persona.prompt_version)
    cache_file = CACHE_DIR / f"{key}.json"

    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        return PersonaAction(**data)

    if backend == "mock":
        # deterministic mock for smoke testing — derive from persona id
        pid = persona.id
        if "fomo_bull" in pid:
            action = PersonaAction(vote="long", vote_conviction=0.8, action_type="buy_lite", action_size_pct=0.5, rationale="[mock] FOMO momentum")
        elif "permabear" in pid:
            action = PersonaAction(vote="short", vote_conviction=0.7, action_type="sell_lite", action_size_pct=0.3, rationale="[mock] bearish trim")
        elif "quant" in pid:
            action = PersonaAction(vote="flat", vote_conviction=0.5, action_type="buy_lite", action_size_pct=0.3, rationale="[mock] modest signal long")
        elif "buffett" in pid:
            action = PersonaAction(vote="long", vote_conviction=0.6, action_type="buy_lite", action_size_pct=0.2, rationale="[mock] patient accumulation")
        else:  # burry
            action = PersonaAction(vote="short", vote_conviction=0.9, action_type="sell_lite", action_size_pct=0.5, rationale="[mock] bubble watch")
        with open(cache_file, "w") as f:
            json.dump(asdict(action), f)
        return action

    user_msg = _build_user_message(
        ticker=ticker,
        price_history_csv=price_history_csv,
        cash=cash,
        shares=shares,
        total_value=total_value,
        initial_capital=persona.initial_capital,
        decision_date=decision_date,
        mode=mode,
    )

    # claude CLI backend (uses your Claude Code subscription / auth, not API key)
    if backend == "claude_cli":
        for attempt in range(3):
            try:
                raw, cost = _call_via_claude_cli(persona.system_prompt, user_msg)
                parsed = _parse_json(raw)
                if parsed and all(k in parsed for k in ("vote", "action_type")):
                    action = PersonaAction(
                        vote=parsed.get("vote", "flat"),
                        vote_conviction=float(parsed.get("vote_conviction", 0.5)),
                        action_type=parsed.get("action_type", "hold"),
                        action_size_pct=float(parsed.get("action_size_pct", 0.0)),
                        rationale=str(parsed.get("rationale", ""))[:300],
                        raw_response=raw,
                    )
                    if action.vote not in ("long", "flat", "short"):
                        action.vote = "flat"
                    if action.action_type not in ACTION_TYPES:
                        action.action_type = "hold"
                    action.vote_conviction = max(0.0, min(1.0, action.vote_conviction))
                    action.action_size_pct = max(0.0, min(1.0, action.action_size_pct))
                    with open(cache_file, "w") as f:
                        json.dump(asdict(action), f)
                    return action
            except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return PersonaAction.fallback_hold(f"cli_error: {str(e)[:100]}")
            time.sleep(1)
        return PersonaAction.fallback_hold("cli_max_retries")

    # anthropic_sdk backend (default, needs ANTHROPIC_API_KEY)
    client = _client()
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=MODEL_HAIKU,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": persona.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text if resp.content else ""
            parsed = _parse_json(raw)
            if parsed and all(k in parsed for k in ("vote", "action_type")):
                action = PersonaAction(
                    vote=parsed.get("vote", "flat"),
                    vote_conviction=float(parsed.get("vote_conviction", 0.5)),
                    action_type=parsed.get("action_type", "hold"),
                    action_size_pct=float(parsed.get("action_size_pct", 0.0)),
                    rationale=str(parsed.get("rationale", ""))[:300],
                    raw_response=raw,
                )
                # normalize bad values
                if action.vote not in ("long", "flat", "short"):
                    action.vote = "flat"
                if action.action_type not in ACTION_TYPES:
                    action.action_type = "hold"
                action.vote_conviction = max(0.0, min(1.0, action.vote_conviction))
                action.action_size_pct = max(0.0, min(1.0, action.action_size_pct))

                with open(cache_file, "w") as f:
                    json.dump(asdict(action), f)
                return action
        except APIError as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return PersonaAction.fallback_hold(f"api_error: {e}")
        # parse failed, retry
        time.sleep(1)

    return PersonaAction.fallback_hold("max_retries_exhausted")


ACTION_TYPES = {"buy_strong", "buy_lite", "hold", "sell_lite", "sell_strong"}
