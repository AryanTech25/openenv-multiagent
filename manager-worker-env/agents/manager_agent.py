"""
ManagerAgent: thin wrapper around a trained PPO policy that emits ManagerActions
for the ManagerWorkerEnv.

Two interchangeable backends:
- ``PPOManagerAgent``: loads a stable-baselines3 PPO checkpoint produced by
  ``training/train_manager.py``.
- ``HeuristicManagerAgent``: cheap rule-based fallback (assign → check →
  approve) useful for smoke-testing the wired-up environment without a trained
  model.

The agent operates on the dict observation produced by the gym wrapper used in
training (see ``training/train_manager.py``), but also accepts the raw
``ManagerWorkerObservation`` for convenience.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from env import ManagerAction, ManagerWorkerObservation


def _to_dict_obs(obs: Union[ManagerWorkerObservation, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Coerce either an observation object or a dict into the dict format SB3 expects."""
    if isinstance(obs, ManagerWorkerObservation):
        return {
            "task_embedding": np.array(obs.task_embedding, dtype=np.float32),
            "worker_states": np.array(obs.worker_states, dtype=np.float32),
            "subtask_status": np.array(obs.subtask_status, dtype=np.int8),
            "budget_remaining": np.array([obs.budget_remaining], dtype=np.float32),
            "steps_remaining": np.array([obs.steps_remaining], dtype=np.float32),
        }
    return obs


class HeuristicManagerAgent:
    """
    Stateless rule-of-thumb manager: cycle through assign → check → approve.

    Useful as a sanity baseline against the trained PPO policy and for
    end-to-end smoke tests when no checkpoint is available.
    """

    ACTION_CYCLE = [0, 1, 5]  # assign_subtask, check_worker_output, approve_output

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def predict(
        self,
        obs: Union[ManagerWorkerObservation, Dict[str, Any]],
        deterministic: bool = True,
    ) -> ManagerAction:
        action_id = self.ACTION_CYCLE[self._step % len(self.ACTION_CYCLE)]
        self._step += 1
        return ManagerAction(action_id=action_id)


class PPOManagerAgent:
    """
    Loads a PPO checkpoint produced by ``training/train_manager.py`` and emits
    ManagerActions for the env.
    """

    def __init__(self, model_path: str):
        # Defer the heavy import so the file is cheap to import for callers
        # that only want the heuristic baseline.
        from stable_baselines3 import PPO  # type: ignore

        self.model = PPO.load(model_path)
        self.model_path = model_path

    def reset(self) -> None:
        # PPO policies are stateless w.r.t. the env between episodes here; the
        # underlying RNG is handled by SB3.
        pass

    def predict(
        self,
        obs: Union[ManagerWorkerObservation, Dict[str, Any]],
        deterministic: bool = True,
    ) -> ManagerAction:
        action, _ = self.model.predict(_to_dict_obs(obs), deterministic=deterministic)
        return ManagerAction(action_id=int(action))


def load_manager_agent(model_path: Optional[str] = None) -> "HeuristicManagerAgent | PPOManagerAgent":
    """Convenience constructor — loads PPO if a path is given, else heuristic."""
    if model_path:
        return PPOManagerAgent(model_path)
    return HeuristicManagerAgent()


__all__ = [
    "HeuristicManagerAgent",
    "PPOManagerAgent",
    "load_manager_agent",
]
