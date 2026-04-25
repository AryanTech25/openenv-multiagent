"""
MetaManagerAgent: a top-level scheduler that coordinates several sub-Manager
agents, each of which is in turn driving its own pool of worker LLMs.

Architecture
------------
                ┌────────────────────────┐
                │   MetaManagerAgent     │   (this file)
                └───────────┬────────────┘
            ┌───────┬───────┼───────┬───────┐
            ▼       ▼       ▼       ▼       ▼
        sub-mgr0 sub-mgr1 sub-mgr2 sub-mgr3 ...   (ParallelManagerAgent each)
            │       │       │       │
        env0/wp0 env1/wp1 env2/wp2 env3/wp3       (ManagerWorkerEnv each)
            │       │       │       │
        N workers N workers N workers N workers   (HFWorker / simulated)

The MetaManager doesn't move workers around between sub-envs. Each sub-env is
an independent task with its own budget. The MetaManager's only lever is
*time*: each step it picks ONE sub-env to advance. This keeps the design
simple while still giving the meta-agent meaningful policy choices:

* round_robin   — advance everyone equally (fair scheduler)
* priority      — advance the sub-env with the most pending work
* quality_aware — pause sub-envs that are burning budget on bad output

It also marks sub-envs as DONE / STUCK / OVER_BUDGET so the runner knows when
to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SubStatus(str, Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    STUCK = "STUCK"
    OVER_BUDGET = "OVER_BUDGET"


@dataclass
class SubEnvSnapshot:
    """A single sub-env's state as exposed to the MetaManager each tick."""

    sub_id: int
    status: SubStatus
    completion_pct: float        # subtasks complete / total
    budget_remaining: int
    budget_total: int
    last_reward: float
    cumulative_reward: float
    steps_taken: int
    avg_actual_quality: float    # mean over revealed worker outputs (proxy for "is this team good?")
    label: str = ""              # task type / human-friendly id


@dataclass
class MetaAction:
    """What the MetaManager decided this tick."""

    target_sub_id: int
    reason: str = ""


class MetaManagerAgent:
    """
    Round-robin scheduler with simple policy hooks.

    Strategies:
      - "round_robin": cycle through running sub-envs in order.
      - "priority":    pick the running sub-env with the lowest completion_pct
                       (catch up the laggard).
      - "quality":     pick the running sub-env with the HIGHEST
                       avg_actual_quality, i.e. invest in the strong horse.

    We always skip sub-envs whose status is not RUNNING.
    """

    SUPPORTED = {"round_robin", "priority", "quality"}

    def __init__(self, num_sub_managers: int, strategy: str = "round_robin") -> None:
        if strategy not in self.SUPPORTED:
            raise ValueError(f"Unknown meta strategy {strategy!r}; expected one of {self.SUPPORTED}")
        self.num_sub_managers = num_sub_managers
        self.strategy = strategy
        self._cursor = 0
        self.steps = 0
        self.history: List[MetaAction] = []

    def reset(self) -> None:
        self._cursor = 0
        self.steps = 0
        self.history = []

    def predict(self, snapshots: List[SubEnvSnapshot]) -> Optional[MetaAction]:
        """
        Choose which sub-env to advance this tick. Returns None when nothing
        is RUNNING (i.e. the whole hierarchy is done).
        """
        running = [s for s in snapshots if s.status == SubStatus.RUNNING]
        if not running:
            return None

        if self.strategy == "round_robin":
            # Walk forward from _cursor until we find a RUNNING sub-env.
            for offset in range(len(snapshots)):
                cand = (self._cursor + offset) % len(snapshots)
                if snapshots[cand].status == SubStatus.RUNNING:
                    self._cursor = (cand + 1) % len(snapshots)
                    action = MetaAction(target_sub_id=cand, reason="round_robin")
                    break
            else:  # pragma: no cover — guarded by `running` check above
                return None

        elif self.strategy == "priority":
            # Smallest completion_pct first → catch up the laggard.
            laggard = min(running, key=lambda s: (s.completion_pct, s.cumulative_reward))
            action = MetaAction(target_sub_id=laggard.sub_id, reason="catch_up_laggard")

        else:  # "quality"
            star = max(running, key=lambda s: (s.avg_actual_quality, s.completion_pct))
            action = MetaAction(target_sub_id=star.sub_id, reason="invest_in_best")

        self.steps += 1
        self.history.append(action)
        return action


__all__ = ["MetaManagerAgent", "MetaAction", "SubEnvSnapshot", "SubStatus"]
