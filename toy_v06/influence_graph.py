"""Influence Graph G(i,j): how much agent_i's public_statement influences agent_j.

Threshold-based: only edges with weight >= 0.4 result in agent_j seeing agent_i's post in their prompt.

Indexes match agents.ALL_AGENTS order:
  0=Super-Influencer Catherine, 1=Pod PM David, 2=Activist Short Sarah, 3=Sell-Side Michael,
  4=CTA, 5=Retail FOMO Alex, 6=Permabull Thomas, 7=Day Trader Devon,
  8=Economist Macro Bernanke, 9=Economist Political Krugman, 10=Economist Trader Druck
"""

INFLUENCE_GRAPH: dict[tuple[int, int], float] = {
    # ============================================================
    # ORIGINAL 6 (Catherine, Pod, Activist, Sell-Side, CTA, Retail)
    # ============================================================
    # Super-Influencer (0)
    (0, 1): 0.40, (0, 2): 0.10, (0, 3): 0.75, (0, 4): 0.00, (0, 5): 0.92,
    (0, 6): 0.45, (0, 7): 0.55, (0, 8): 0.10, (0, 9): 0.15, (0, 10): 0.20,
    # Pod PM (1)
    (1, 0): 0.05, (1, 2): 0.20, (1, 3): 0.35, (1, 4): 0.00, (1, 5): 0.08,
    (1, 6): 0.10, (1, 7): 0.05, (1, 8): 0.05, (1, 9): 0.05, (1, 10): 0.05,
    # Activist Short (2)
    (2, 0): 0.15, (2, 1): 0.45, (2, 3): 0.55, (2, 4): 0.00, (2, 5): 0.78,
    (2, 6): 0.30, (2, 7): 0.45, (2, 8): 0.10, (2, 9): 0.30, (2, 10): 0.15,
    # Sell-Side (3)
    (3, 0): 0.05, (3, 1): 0.40, (3, 2): 0.10, (3, 4): 0.00, (3, 5): 0.62,
    (3, 6): 0.35, (3, 7): 0.40, (3, 8): 0.05, (3, 9): 0.05, (3, 10): 0.10,
    # CTA (4) — silent
    (4, 0): 0.0, (4, 1): 0.0, (4, 2): 0.0, (4, 3): 0.0, (4, 5): 0.0,
    (4, 6): 0.0, (4, 7): 0.0, (4, 8): 0.0, (4, 9): 0.0, (4, 10): 0.0,
    # Retail (5) — noise
    (5, 0): 0.0, (5, 1): 0.0, (5, 2): 0.0, (5, 3): 0.05, (5, 4): 0.0,
    (5, 6): 0.0, (5, 7): 0.05, (5, 8): 0.0, (5, 9): 0.0, (5, 10): 0.0,

    # ============================================================
    # NEW 5 agents (6=Permabull, 7=Day Trader, 8/9/10 economists)
    # ============================================================
    # Permabull Thomas Lin (6) — moderate broadcast
    (6, 0): 0.10, (6, 1): 0.25, (6, 2): 0.10, (6, 3): 0.40, (6, 4): 0.00,
    (6, 5): 0.55, (6, 7): 0.30, (6, 8): 0.05, (6, 9): 0.05, (6, 10): 0.10,
    # Day Trader Devon (7) — low broadcast
    (7, 0): 0.02, (7, 1): 0.05, (7, 2): 0.02, (7, 3): 0.05, (7, 4): 0.00,
    (7, 5): 0.10, (7, 6): 0.05, (7, 8): 0.00, (7, 9): 0.00, (7, 10): 0.00,
    # Economist Macro Bernanke (8) — strong but quiet authority
    (8, 0): 0.30, (8, 1): 0.55, (8, 2): 0.20, (8, 3): 0.45, (8, 4): 0.00,
    (8, 5): 0.35, (8, 6): 0.40, (8, 7): 0.15, (8, 9): 0.25, (8, 10): 0.30,
    # Economist Political Krugman (9) — column reach
    (9, 0): 0.20, (9, 1): 0.30, (9, 2): 0.40, (9, 3): 0.35, (9, 4): 0.00,
    (9, 5): 0.45, (9, 6): 0.15, (9, 7): 0.15, (9, 8): 0.20, (9, 10): 0.10,
    # Economist Trader Druck (10) — legendary status, everyone watches
    (10, 0): 0.50, (10, 1): 0.75, (10, 2): 0.45, (10, 3): 0.60, (10, 4): 0.00,
    (10, 5): 0.55, (10, 6): 0.65, (10, 7): 0.45, (10, 8): 0.15, (10, 9): 0.10,
}

INFLUENCE_THRESHOLD = 0.4  # below this, j ignores i's posts


def get_inbound_influencers(receiver_idx: int) -> list[tuple[int, float]]:
    """Return list of (sender_idx, weight) whose influence on receiver exceeds threshold."""
    result = []
    for (i, j), w in INFLUENCE_GRAPH.items():
        if j == receiver_idx and w >= INFLUENCE_THRESHOLD:
            result.append((i, w))
    return sorted(result, key=lambda x: -x[1])


def get_out_degree(sender_idx: int) -> float:
    """Sum of outbound edges (proxy for influence_out)."""
    return sum(w for (i, j), w in INFLUENCE_GRAPH.items() if i == sender_idx)


def get_in_degree(receiver_idx: int) -> float:
    """Sum of inbound edges (proxy for influence_in)."""
    return sum(w for (i, j), w in INFLUENCE_GRAPH.items() if j == receiver_idx)
