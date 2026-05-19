"""LLM wrapper for toy v0.6 — 4-layer output state.

Uses claude CLI subprocess. Disk-caches by input hash.
"""

import hashlib
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# v0.7.5 — Universal addendum appended to every agent's system prompt
SIZING_ADDENDUM = """

⚠️ V0.7.5 SIZING & RISK MANAGEMENT UPDATE:
- size_pct = LITERAL fraction (0.0–1.0). 0.5 = use 50% of cash. 1.0 = ALL IN. NEVER > 1.0.
- NO multipliers applied. What you say IS what gets deployed. You have FULL control.
- You decide your own sizing based on YOUR strategy and conviction. WSB can YOLO 0.95. Buffett can deploy 0.05.
- Optional stop-loss/take-profit fields in personal_action:
  - "stop_loss_pct": negative number e.g. -0.06 (force-close position if it drops 6% from purchase)
  - "take_profit_pct": positive e.g. +0.20 (force-close at +20% gain)
  - Omit (or null) if you don't want a stop/target (e.g. Buffett holds without stops)
- Example: {"action_type": "buy_strong", "size_pct": 0.65, "stop_loss_pct": -0.08, "take_profit_pct": 0.25, "rationale_internal": "..."}
"""


@dataclass
class FourLayerState:
    private_belief: dict       # {lean, conviction, actual_thesis}
    public_statement: dict     # {stated_lean, stated_conviction, narrative}
    desired_market_reaction: str
    personal_action: dict      # {action_type, size_pct, rationale_internal}
    raw_response: str = ""

    @classmethod
    def fallback(cls, reason: str = "parse_error") -> "FourLayerState":
        return cls(
            private_belief={"lean": "neutral", "conviction": 0.0, "actual_thesis": f"[fallback: {reason}]"},
            public_statement={"stated_lean": "neutral", "stated_conviction": 0.0, "narrative": "[silent]"},
            desired_market_reaction="[fallback]",
            personal_action={"action_type": "hold", "size_pct": 0.0, "rationale_internal": f"[fallback: {reason}]"},
            raw_response="",
        )

    def to_dict(self):
        return asdict(self)


def _cache_key(agent_id: str, day: str, info_hash: str, influence_hash: str, memory_hash: str = "") -> str:
    h = hashlib.sha256()
    # v0.7.5: bumped cache version (size_pct semantics changed + stop/take-profit added)
    h.update(f"v075|{agent_id}|{day}|{info_hash}|{influence_hash}|{memory_hash}".encode())
    return h.hexdigest()[:16]


def _parse_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        cleaned = m.group(0).replace("'", '"')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _call_claude_cli(system_prompt: str, user_msg: str, budget_usd: float = 0.08) -> tuple[str, float]:
    cmd = [
        "claude", "-p",
        "--model", "haiku",
        "--no-session-persistence",
        "--output-format", "json",
        "--max-budget-usd", str(budget_usd),
        "--system-prompt", system_prompt,
        user_msg,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:200]}")
    data = json.loads(result.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude API error: {data.get('result', '')[:200]}")
    return data.get("result", ""), float(data.get("total_cost_usd", 0.0))


def agent_decide(
    agent_id: str,
    system_prompt: str,
    info_view: str,             # tier-filtered market data
    influencer_posts: str,      # formatted influence content
    portfolio_state: str,       # current cash/shares
    day: str,
    use_cache: bool = True,
    memory_context: str = "",   # v0.7 NEW: formatted memory recall
) -> tuple[FourLayerState, float]:
    """Get one agent's 4-layer decision. Returns (state, cost_usd)."""
    info_hash = hashlib.sha256(info_view.encode()).hexdigest()[:8]
    inf_hash = hashlib.sha256(influencer_posts.encode()).hexdigest()[:8]
    mem_hash = hashlib.sha256(memory_context.encode()).hexdigest()[:8] if memory_context else ""
    key = _cache_key(agent_id, day, info_hash, inf_hash, mem_hash)
    cache_file = CACHE_DIR / f"{key}.json"

    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        return FourLayerState(**{k: v for k, v in data.items() if k != "cost"}), 0.0

    memory_block = f"\nYOUR MEMORY RECALL (recent relevant facts, ❌ = invalidated by market):\n{memory_context}\n" if memory_context else ""
    user_msg = f"""Day: {day}

YOUR PORTFOLIO:
{portfolio_state}

YOUR INFORMATION VIEW (filtered by your tier):
{info_view}

INFLUENCER POSTS YOU RECEIVED:
{influencer_posts}
{memory_block}
Now produce your 4-layer JSON decision per your system prompt schema."""

    # v0.7.5: append universal sizing addendum
    augmented_prompt = system_prompt + SIZING_ADDENDUM

    for attempt in range(3):
        try:
            raw, cost = _call_claude_cli(augmented_prompt, user_msg)
            parsed = _parse_json(raw)
            if parsed and "private_belief" in parsed and "public_statement" in parsed:
                state = FourLayerState(
                    private_belief=parsed.get("private_belief", {"lean": "neutral", "conviction": 0.5, "actual_thesis": ""}),
                    public_statement=parsed.get("public_statement", {"stated_lean": "neutral", "stated_conviction": 0.5, "narrative": ""}),
                    desired_market_reaction=parsed.get("desired_market_reaction", "")[:200],
                    personal_action=parsed.get("personal_action", {"action_type": "hold", "size_pct": 0.0, "rationale_internal": ""}),
                    raw_response=raw,
                )
                # Normalize leans
                for layer in ("private_belief", "public_statement"):
                    key_name = "lean" if layer == "private_belief" else "stated_lean"
                    val = getattr(state, layer).get(key_name, "neutral")
                    if val not in ("long", "neutral", "short", "flat"):
                        if layer == "private_belief":
                            state.private_belief[key_name] = "neutral"
                        else:
                            state.public_statement[key_name] = "neutral"
                if state.personal_action.get("action_type") not in {"buy_strong", "buy_lite", "hold", "sell_lite", "sell_strong"}:
                    state.personal_action["action_type"] = "hold"

                with open(cache_file, "w") as f:
                    out = state.to_dict()
                    out['cost'] = cost
                    json.dump(out, f)
                return state, cost
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return FourLayerState.fallback(f"{e}"), 0.0
        time.sleep(1)
    return FourLayerState.fallback("max_retries"), 0.0
