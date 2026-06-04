"""Build a reproducible population of unique LLM-agent personas.

Each persona is one of the 11 `toy_v06` archetypes instantiated with a UNIQUE
sampled genome — same behavioral DNA, no two alike. They are coupled by a sparse
SIGNED influence network (positive = herd/follow, negative = fade/contrarian).

The population is built ONCE (fixed seed) and persisted to results/population.json,
so the same cohort persists and evolves day to day. Rebuild only if you change
N or the seed.

    python -m forward_v01.population --agents 300        # (re)build & save
    python -m forward_v01.population --show               # summarize saved pop
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "toy_v06"))
from agents import ALL_AGENTS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
POP_FILE = RESULTS / "population.json"

SEED = 20260602  # fixed → reproducible cohort

# Headcount mix across archetypes (retail dominates the long tail; hubs are rare).
# Keys are archetype roles; values are relative weights (normalized at build).
ARCHETYPE_MIX = {
    "retail_fomo": 0.30,
    "day_trader": 0.18,
    "pod_pm": 0.10,
    "sell_side": 0.08,
    "permabull": 0.07,
    "activist_short": 0.05,
    "super_influencer": 0.04,
    "cta_forced": 0.06,
    "economist_macro": 0.04,
    "economist_political": 0.02,
    "economist_trader": 0.06,
}

# Per-role contrarian prior (0 = pure herd, 1 = pure fade). Not on the archetype
# dataclass, so sampled from these role-based means.
CONTRARIAN_PRIOR = {
    "activist_short": 0.85, "economist_trader": 0.55, "pod_pm": 0.45,
    "economist_political": 0.50, "economist_macro": 0.45, "sell_side": 0.30,
    "super_influencer": 0.35, "cta_forced": 0.40, "day_trader": 0.30,
    "permabull": 0.10, "retail_fomo": 0.08,
}

AVG_DEGREE = 12  # how many sources each persona listens to


@dataclass
class Persona:
    pid: str
    archetype: str          # role of the parent archetype
    name: str
    # genome (sampled, unique)
    capital: float
    time_horizon_days: int
    career_risk: float
    info_tier: int
    influence_in: float     # susceptibility (how much peers move me)
    influence_out: float    # how much I move peers
    signaling_incentive: float
    reflexivity_awareness: float
    contrarian: float
    temperament: str        # one-line individualizing flavor


# ── samplers ────────────────────────────────────────────────────────────────
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _jitter01(rng: random.Random, mean: float, sd: float = 0.12) -> float:
    return round(_clip01(rng.gauss(mean, sd)), 3)


def _jitter_capital(rng: random.Random, base: float) -> float:
    if base <= 0:
        return 0.0
    # lognormal spread around the archetype's capital (±~1 order of magnitude)
    return round(base * (10 ** rng.gauss(0.0, 0.35)), 2)


def _jitter_horizon(rng: random.Random, base: int) -> int:
    return max(1, int(round(base * (10 ** rng.gauss(0.0, 0.18)))))


TEMPER_ADJ = [
    "cautious", "aggressive", "impatient", "methodical", "skeptical", "excitable",
    "stubborn", "nimble", "anxious", "cold-blooded", "narrative-driven", "data-obsessed",
    "overconfident", "burned-before", "thesis-locked", "tape-reading",
]


def _build(n: int) -> dict:
    rng = random.Random(SEED)
    by_role = {a.role: a for a in ALL_AGENTS}

    roles = list(ARCHETYPE_MIX)
    weights = [ARCHETYPE_MIX[r] for r in roles]
    role_seq = rng.choices(roles, weights=weights, k=n)

    personas: list[Persona] = []
    for i, role in enumerate(role_seq):
        a = by_role[role]
        personas.append(Persona(
            pid=f"{role}_{i:04d}",
            archetype=role,
            name=f"{a.name.split()[0]}-{i:04d}",
            capital=_jitter_capital(rng, a.capital),
            time_horizon_days=_jitter_horizon(rng, a.time_horizon_days),
            career_risk=_jitter01(rng, a.career_risk),
            info_tier=a.info_tier,
            influence_in=_jitter01(rng, a.influence_in),
            influence_out=_jitter01(rng, a.influence_out, sd=0.08),
            signaling_incentive=_jitter01(rng, a.signaling_incentive),
            reflexivity_awareness=_jitter01(rng, a.reflexivity_awareness),
            contrarian=_jitter01(rng, CONTRARIAN_PRIOR.get(role, 0.3)),
            temperament=rng.choice(TEMPER_ADJ),
        ))

    edges = _build_influence_network(rng, personas)
    return {
        "seed": SEED, "n": n, "avg_degree": AVG_DEGREE,
        "personas": [asdict(p) for p in personas],
        "edges": edges,  # {target_pid: [[source_idx, weight], ...]}
    }


def _build_influence_network(rng: random.Random, personas: list[Persona]) -> dict:
    """Sparse SIGNED graph. Each persona listens to ~AVG_DEGREE sources, chosen
    preferentially by the source's influence_out (hubs get many followers) and
    homophily (same archetype slightly preferred). Edge sign comes from the
    LISTENER's contrarian streak: contrarians fade their sources (negative)."""
    n = len(personas)
    out_pull = [0.05 + p.influence_out for p in personas]  # source attractiveness
    idx = list(range(n))
    edges: dict[str, list] = {}

    for j, p in enumerate(personas):
        k = max(1, int(round(rng.gauss(AVG_DEGREE, 3))))
        # weight candidate sources by influence_out × homophily, exclude self
        w = []
        for s in idx:
            if s == j:
                w.append(0.0)
                continue
            homophily = 1.6 if personas[s].archetype == p.archetype else 1.0
            w.append(out_pull[s] * homophily)
        sources = _weighted_sample_without_replacement(rng, idx, w, k)
        lst = []
        for s in sources:
            # base influence strength ∝ source out × listener susceptibility
            strength = round(out_pull[s] * (0.3 + 0.7 * p.influence_in), 3)
            sign = -1.0 if rng.random() < p.contrarian else 1.0
            lst.append([s, round(sign * strength, 3)])
        edges[p.pid] = lst
    return edges


def _weighted_sample_without_replacement(rng, items, weights, k):
    """Efraimidis-Spirakis A-Res: key = u**(1/w); take top-k."""
    keyed = []
    for it, wt in zip(items, weights):
        if wt <= 0:
            continue
        u = rng.random()
        keyed.append((u ** (1.0 / wt), it))
    keyed.sort(reverse=True)
    return [it for _, it in keyed[:k]]


# ── public API ────────────────────────────────────────────────────────────────
def load_population(n: int = 300, rebuild: bool = False) -> dict:
    """Load the saved population, building+saving it if missing or size mismatch."""
    if POP_FILE.exists() and not rebuild:
        pop = json.load(open(POP_FILE))
        if pop.get("n") == n:
            return pop
    pop = _build(n)
    with open(POP_FILE, "w") as f:
        json.dump(pop, f)
    return pop


def _summarize(pop: dict) -> None:
    from collections import Counter
    c = Counter(p["archetype"] for p in pop["personas"])
    n_edges = sum(len(v) for v in pop["edges"].values())
    neg = sum(1 for v in pop["edges"].values() for _, w in v if w < 0)
    print(f"Population: {pop['n']} personas, seed={pop['seed']}")
    print(f"Influence edges: {n_edges} ({neg} negative/contrarian, "
          f"{n_edges - neg} positive)  avg deg {n_edges / pop['n']:.1f}")
    print("Archetype mix:")
    for role, cnt in c.most_common():
        print(f"  {role:<22} {cnt:>4}  ({cnt / pop['n'] * 100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=300)
    ap.add_argument("--show", action="store_true", help="summarize saved population")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.show and POP_FILE.exists() and not args.rebuild:
        _summarize(json.load(open(POP_FILE)))
        return
    pop = load_population(args.agents, rebuild=True)
    print(f"✓ Saved {POP_FILE}")
    _summarize(pop)


if __name__ == "__main__":
    main()
