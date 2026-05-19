"""Reputation tracker: rolling accuracy over past N days. Drives dynamic influence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PostRecord:
    date: str
    agent_id: str
    ticker: str
    stated_lean: str          # "long" | "short" | "neutral"
    private_lean: str
    action_type: str
    target_date: str          # date when prediction matures


@dataclass
class ReputationSnapshot:
    agent_id: str
    date: str
    n_posts_evaluated: int
    n_correct: int
    accuracy: float
    influence_multiplier: float   # 0.5 + accuracy (bound 0.05-1.5)


class ReputationTracker:
    """30-day rolling accuracy per agent."""

    def __init__(self, window_days: int = 30):
        self.window_days = window_days
        self.posts: list[PostRecord] = []
        # snapshot history: agent_id -> [ReputationSnapshot]
        self.history: dict[str, list[ReputationSnapshot]] = defaultdict(list)
        # current influence multiplier per agent
        self.current_multiplier: dict[str, float] = {}

    def record_post(self, post: PostRecord) -> None:
        self.posts.append(post)

    def update(
        self,
        today: str,
        price_data: dict[str, dict[str, float]],  # ticker -> date -> close
        agents: list,
    ) -> dict[str, ReputationSnapshot]:
        """Evaluate posts whose target_date <= today. Compute accuracy per agent."""
        today_d = datetime.fromisoformat(today).date()
        window_start = today_d.replace().fromordinal(today_d.toordinal() - self.window_days)

        # Aggregate by agent: how many predictions did each one make in window, how many correct?
        evaluated_by_agent: dict[str, list[bool]] = defaultdict(list)
        for post in self.posts:
            post_d = datetime.fromisoformat(post.date).date()
            target_d = datetime.fromisoformat(post.target_date).date()
            if target_d > today_d:
                continue  # not matured yet
            if post_d < window_start:
                continue  # outside window
            # Look up price change
            close_at_post = price_data.get(post.ticker, {}).get(post.date)
            close_at_target = None
            # find closest available close on or after target_date
            for d_key in sorted(price_data.get(post.ticker, {}).keys()):
                if d_key >= post.target_date:
                    close_at_target = price_data[post.ticker][d_key]
                    break
            if close_at_post is None or close_at_target is None:
                continue
            change = (close_at_target - close_at_post) / close_at_post
            correct = False
            if post.stated_lean == "long" and change > 0.005:
                correct = True
            elif post.stated_lean == "short" and change < -0.005:
                correct = True
            elif post.stated_lean == "neutral" and abs(change) < 0.01:
                correct = True
            evaluated_by_agent[post.agent_id].append(correct)

        snapshots = {}
        for agent in agents:
            aid = agent.id
            evals = evaluated_by_agent.get(aid, [])
            n = len(evals)
            n_correct = sum(evals)
            acc = n_correct / n if n > 0 else 0.5  # default neutral
            # influence multiplier: 0.5 + acc, bounded
            mult = max(0.05, min(1.5, 0.5 + acc))
            snap = ReputationSnapshot(
                agent_id=aid,
                date=today,
                n_posts_evaluated=n,
                n_correct=n_correct,
                accuracy=acc,
                influence_multiplier=mult,
            )
            self.history[aid].append(snap)
            self.current_multiplier[aid] = mult
            snapshots[aid] = snap

        return snapshots

    def get_multiplier(self, agent_id: str) -> float:
        return self.current_multiplier.get(agent_id, 1.0)

    def get_history(self, agent_id: str) -> list[ReputationSnapshot]:
        return self.history.get(agent_id, [])

    def to_dict(self) -> dict:
        return {
            "window_days": self.window_days,
            "current_multipliers": dict(self.current_multiplier),
            "history": {
                aid: [
                    {
                        "date": s.date,
                        "accuracy": s.accuracy,
                        "n_posts": s.n_posts_evaluated,
                        "influence_multiplier": s.influence_multiplier,
                    }
                    for s in snaps
                ]
                for aid, snaps in self.history.items()
            },
        }
