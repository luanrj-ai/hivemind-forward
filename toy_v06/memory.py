"""Simple in-memory fact store with temporal validity. NO Graphiti/Neo4j, dict-based for 1-hour demo."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Literal, Optional


@dataclass
class Fact:
    id: int
    content: str
    source: str                       # agent_id who said it, or "self_observation" / "market"
    ticker: Optional[str] = None
    valid_at: str = ""                # ISO date
    invalid_at: Optional[str] = None
    contradicted_by_fact_id: Optional[int] = None
    fact_type: Literal["price_event", "prediction", "belief", "social_observation"] = "belief"
    stated_lean: Optional[str] = None  # for prediction-type
    target_date: Optional[str] = None  # for prediction-type (when does the call mature)


class FactStore:
    """Per-agent fact store. List backed, simple temporal validity."""

    def __init__(self, owner_id: str):
        self.owner_id = owner_id
        self.facts: list[Fact] = []
        self._next_id = 0

    def add(
        self,
        content: str,
        source: str,
        valid_at: str,
        ticker: Optional[str] = None,
        fact_type: str = "belief",
        stated_lean: Optional[str] = None,
        target_date: Optional[str] = None,
    ) -> int:
        f = Fact(
            id=self._next_id,
            content=content,
            source=source,
            ticker=ticker,
            valid_at=valid_at,
            fact_type=fact_type,
            stated_lean=stated_lean,
            target_date=target_date,
        )
        self.facts.append(f)
        self._next_id += 1
        return f.id

    def invalidate(self, fact_id: int, when: str, contradicted_by: Optional[int] = None) -> None:
        for f in self.facts:
            if f.id == fact_id:
                f.invalid_at = when
                f.contradicted_by_fact_id = contradicted_by
                return

    def recall(
        self,
        ticker: Optional[str] = None,
        max_results: int = 10,
        as_of_date: Optional[str] = None,
        source_filter: Optional[list[str]] = None,
        only_active: bool = True,
    ) -> list[Fact]:
        """Simple recall: filter by ticker / validity / source, return most recent first."""
        results = []
        for f in self.facts:
            if ticker and f.ticker and f.ticker != ticker:
                continue
            if only_active and f.invalid_at:
                continue
            if as_of_date and f.valid_at > as_of_date:
                continue
            if source_filter and f.source not in source_filter:
                continue
            results.append(f)
        # most recent first
        results.sort(key=lambda x: x.valid_at, reverse=True)
        return results[:max_results]

    def format_for_prompt(self, facts: list[Fact]) -> str:
        if not facts:
            return "  (no relevant memory)"
        lines = []
        for f in facts:
            invalid_marker = " ❌INVALIDATED" if f.invalid_at else ""
            lines.append(
                f"  [{f.valid_at}] ({f.source}) {f.content[:120]}{invalid_marker}"
            )
        return "\n".join(lines)

    def count(self, only_active: bool = True) -> int:
        if only_active:
            return sum(1 for f in self.facts if not f.invalid_at)
        return len(self.facts)

    def to_dict(self) -> dict:
        return {
            "owner_id": self.owner_id,
            "fact_count": len(self.facts),
            "active_count": self.count(only_active=True),
            "facts": [asdict(f) for f in self.facts],
        }


def auto_invalidate_predictions(
    store: FactStore,
    today: str,
    price_change_per_ticker: dict[str, float],
) -> list[tuple[int, bool]]:
    """For predictions whose target_date has passed, mark invalid if wrong.

    Returns list of (fact_id, was_correct) tuples.
    """
    today_d = datetime.fromisoformat(today).date()
    results = []
    for f in list(store.facts):
        if f.fact_type != "prediction" or f.invalid_at:
            continue
        if not f.target_date or not f.ticker or not f.stated_lean:
            continue
        target_d = datetime.fromisoformat(f.target_date).date()
        if today_d < target_d:
            continue  # not yet
        # Evaluate
        change = price_change_per_ticker.get(f.ticker, 0.0)
        correct = False
        if f.stated_lean == "long" and change > 0.005:
            correct = True
        elif f.stated_lean == "short" and change < -0.005:
            correct = True
        elif f.stated_lean == "neutral" and abs(change) < 0.01:
            correct = True

        if not correct:
            store.invalidate(f.id, when=today)
        results.append((f.id, correct))
    return results
