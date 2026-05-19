"""5 Market Ecology Metrics for toy v0.6."""

import math
from collections import Counter
from typing import Iterable


def shannon_entropy(values: Iterable[str]) -> float:
    """Shannon entropy of a categorical distribution (returns nats)."""
    vals = list(values)
    if not vals:
        return 0.0
    counts = Counter(vals)
    total = len(vals)
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log(p)
    return H


def herfindahl(values: Iterable[str]) -> float:
    """Herfindahl-Hirschman Index of categorical concentration (0 = uniform, 1 = single).
    Note: for >1 categories, max possible is 1 - 1/k → effective range adjusted."""
    vals = list(values)
    if not vals:
        return 0.0
    counts = Counter(vals)
    total = len(vals)
    return sum((c / total) ** 2 for c in counts.values())


def gini_coefficient(values: list[float]) -> float:
    """Gini coefficient of a non-negative list (0=equal, 1=max inequality)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    cumsum = 0
    for i, x in enumerate(sorted_v):
        cumsum += (i + 1) * x
    s = sum(sorted_v)
    if s == 0:
        return 0.0
    return (2 * cumsum) / (n * s) - (n + 1) / n


def compute_day_metrics(day_record: dict) -> dict:
    """Given one day's records (agent_outputs dict, trades), compute 5 ecology metrics."""
    outputs = day_record["agent_outputs"]

    private_leans = []
    public_leans = []
    public_actions = []
    narratives = []

    for agent_id, info in outputs.items():
        state = info["state"]
        if info["role"] == "cta_forced":
            continue  # skip deterministic agent for belief metrics
        priv = state["private_belief"].get("lean", "neutral")
        pub = state["public_statement"].get("stated_lean", "neutral")
        action = state["personal_action"].get("action_type", "hold")
        narr = state["public_statement"].get("narrative", "")[:50]  # short hash key

        private_leans.append(priv)
        public_leans.append(pub)
        public_actions.append(action)
        narratives.append(narr.lower())

    # 1. Narrative concentration (Herfindahl on top-3 keyword clusters from narratives)
    # Simple proxy: just dedup-bucket the narrative summary
    narrative_concentration = herfindahl(narratives) if narratives else 0

    # 2. Belief entropy (Shannon over private leans)
    belief_entropy = shannon_entropy(private_leans)

    # 3. Influence centralization (Gini weighted by total_value)
    portfolios = day_record["portfolios_close"]
    totals = [p["total"] for p in portfolios.values() if p["total"] > 0]
    influence_centralization = gini_coefficient(totals)

    # 4. Consensus fragility (proportion of dissenting voices)
    if private_leans:
        majority_lean = Counter(private_leans).most_common(1)[0][0]
        dissenting = sum(1 for l in private_leans if l != majority_lean)
        consensus_fragility = dissenting / len(private_leans)
    else:
        consensus_fragility = 0

    # 5. Public-vs-private gap (how often does each agent's vote ≠ private lean)
    mismatches = 0
    total = 0
    for agent_id, info in outputs.items():
        if info["role"] == "cta_forced":
            continue
        state = info["state"]
        priv = state["private_belief"].get("lean", "neutral")
        pub = state["public_statement"].get("stated_lean", "neutral")
        if priv != pub:
            mismatches += 1
        total += 1
    public_private_gap = mismatches / total if total > 0 else 0

    # 6. Bonus: action-vs-public gap (does public statement match action?)
    action_public_mismatches = 0
    action_total = 0
    for agent_id, info in outputs.items():
        if info["role"] == "cta_forced":
            continue
        state = info["state"]
        pub = state["public_statement"].get("stated_lean", "neutral")
        action = state["personal_action"].get("action_type", "hold")
        # Inferred lean from action
        action_lean = "neutral"
        if action.startswith("buy"):
            action_lean = "long"
        elif action.startswith("sell"):
            action_lean = "short"
        if pub != action_lean and pub != "neutral" and action_lean != "neutral":
            action_public_mismatches += 1
        action_total += 1
    action_public_gap = action_public_mismatches / action_total if action_total > 0 else 0

    return {
        "date": day_record["date"],
        "narrative_concentration": round(narrative_concentration, 4),
        "belief_entropy": round(belief_entropy, 4),
        "influence_centralization_gini": round(influence_centralization, 4),
        "consensus_fragility": round(consensus_fragility, 4),
        "public_private_gap": round(public_private_gap, 4),
        "action_public_gap": round(action_public_gap, 4),
        "private_leans": dict(Counter(private_leans)),
        "public_leans": dict(Counter(public_leans)),
        "action_types": dict(Counter(public_actions)),
    }
