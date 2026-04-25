#!/usr/bin/env python3
"""
End-to-end multi-agent demo: a Manager actually drives a pool of LLM Workers
through a complete task in ManagerWorkerEnv.

Modes (combine freely):
- ``--real``: spin up a real WorkerPool of Hugging Face workers and have them
  generate text. Requires ``HF_TOKEN`` for gated models.
- ``--parallel``: use the ParallelManagerAgent which fans subtasks out across
  ALL workers before checking — this is what makes multiple agents visibly
  active at the same time.
- ``--dashboard``: print a live multi-worker dashboard at every step instead
  of a one-line action log.
- ``--heterogeneous``: when combined with ``--real``, build the pool from a
  mix of HF models (one per worker) so you can compare different "brains".
- ``--ppo-checkpoint PATH``: replace the heuristic/parallel manager with a
  trained PPO policy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

# Make the package importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from agents import (
    HeuristicManagerAgent,
    ParallelManagerAgent,
    PPOManagerAgent,
    WorkerPool,
)
from agents.llm_worker import HFWorkerConfig
from env import ManagerWorkerEnv

load_dotenv()


DEFAULT_HETEROGENEOUS_MODELS: List[str] = [
    "HuggingFaceTB/SmolLM-135M",
    "HuggingFaceTB/SmolLM-360M",
    "microsoft/phi-2",
    "Qwen/Qwen2.5-0.5B",
]


def build_pool(model_id: str, mode: str, num_workers: int) -> WorkerPool:
    """Spin up a homogeneous WorkerPool of HF workers."""
    config = HFWorkerConfig(
        model_id=model_id,
        hf_token=os.getenv("HF_TOKEN"),
        worker_type=mode,
        max_tokens=128,
        temperature=0.7,
    )
    return WorkerPool(num_workers=num_workers, config=config)


def build_heterogeneous_pool(model_ids: List[str], mode: str) -> WorkerPool:
    """
    Build a pool where each worker uses a different HF model.

    We construct one worker at a time with its own HFWorkerConfig and stitch
    them into a WorkerPool-shaped object.
    """
    from agents.llm_worker import HFAPIWorker, HFWorker

    pool = WorkerPool.__new__(WorkerPool)  # bypass __init__'s homogeneous loader
    pool.num_workers = len(model_ids)
    pool.config = HFWorkerConfig(worker_type=mode, hf_token=os.getenv("HF_TOKEN"))
    pool.workers = []
    pool.worker_states = {}
    for wid, model_id in enumerate(model_ids):
        cfg = HFWorkerConfig(
            model_id=model_id,
            hf_token=os.getenv("HF_TOKEN"),
            worker_type=mode,
            max_tokens=128,
            temperature=0.7,
        )
        # Stagger skill so the dashboard shows visible variation.
        skill = 0.4 + 0.15 * wid
        worker_cls = HFAPIWorker if mode == "api" else HFWorker
        worker = worker_cls(worker_id=wid, skill_level=min(skill, 0.95), config=cfg)
        pool.workers.append(worker)
        pool.worker_states[wid] = {
            "skill_level": worker.skill_level,
            "is_active": False,
            "current_subtask": None,
            "output": None,
            "failure_mode": None,
        }
    return pool


def run_episode(env: ManagerWorkerEnv, manager, max_steps: int, dashboard: bool) -> None:
    obs = env.reset()
    if hasattr(manager, "reset"):
        manager.reset()

    print("=" * 96)
    print(f"Task: {env.state.task['task_type']} — {env.state.task['description']}")
    print(
        f"Subtasks: {env.state.task['num_subtasks']}  "
        f"Budget: {env.state.budget_remaining}  "
        f"Manager: {type(manager).__name__}"
    )
    print(
        "Workers: "
        + ", ".join(f"#{w.worker_id}(skill={w.skill_level:.2f})" for w in env.state.workers)
    )
    print("=" * 96)

    total_reward = 0.0
    for step in range(max_steps):
        action = manager.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if dashboard:
            print(f"\n>>> step {step + 1}  action={info.get('action_name', '?')}  reward={reward:+.2f}")
            print(env.render(mode="dashboard"))
            time.sleep(0.2)  # let humans actually read it
        else:
            n_active = sum(1 for w in env.state.workers if w.is_active)
            print(
                f"  step {step + 1:>2} "
                f"action={info.get('action_name', '?'):<22} "
                f"reward={reward:+.2f} "
                f"budget={info.get('budget_remaining', '?')} "
                f"active_workers={n_active} "
                f"subtasks={env.state.subtask_status}"
            )

        if done:
            break

    print("-" * 96)
    quality = env._compute_final_quality()
    print(
        f"Episode done. total_reward={total_reward:.2f}  "
        f"final_quality={quality:.2f}  "
        f"hallucinations_detected={env.state.hallucinations_detected}  "
        f"hallucinations_approved={env.state.hallucinations_approved}  "
        f"false_positives={env.state.false_positives}"
    )
    print("=" * 96)
    print()
    print("Final per-worker view:")
    for w in env.state.workers:
        buf = (w.output_buffer or "").replace("\n", " ")[:120]
        print(
            f"  #{w.worker_id} skill={w.skill_level:.2f} "
            f"failure_mode={w.failure_mode!s:<14} "
            f"actual_quality={w.actual_quality:.2f}  "
            f"buffer=\"{buf}\""
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end multi-agent demo")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use a real Hugging Face WorkerPool (otherwise simulator-only).",
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M",
        help="HF model id for the worker pool when --real is set (single-model pools).",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default="local",
        help="Worker backend when --real is set.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Pool size."
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        help="With --real, give each worker a different HF model from a built-in mix.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use ParallelManagerAgent (fans subtasks out to multiple workers).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Render a multi-worker dashboard at every step.",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        default=None,
        help="Optional path to a PPO checkpoint produced by train_manager.py.",
    )
    parser.add_argument("--task-difficulty", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=0.6,
        help="Failure injection rate passed to the env (0.0–1.0).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force HF/transformers into offline mode (uses cached weights, skips Hub network calls).",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    pool = None
    if args.real:
        if args.heterogeneous:
            models = DEFAULT_HETEROGENEOUS_MODELS[: args.num_workers]
            print(f"Spinning up heterogeneous WorkerPool: {models} ({args.mode})…")
            pool = build_heterogeneous_pool(models, args.mode)
        else:
            print(f"Spinning up real WorkerPool ({args.num_workers}× {args.model}, {args.mode})…")
            pool = build_pool(args.model, args.mode, args.num_workers)

    env = ManagerWorkerEnv(
        config={
            "max_workers": args.num_workers,
            "max_steps": args.max_steps,
            "token_budget": args.token_budget,
            "task_difficulty": args.task_difficulty,
            "failure_injection_rate": args.failure_rate,
        },
        worker_pool=pool,
    )

    if args.ppo_checkpoint:
        print(f"Loading PPO manager from {args.ppo_checkpoint}…")
        manager = PPOManagerAgent(args.ppo_checkpoint)
    elif args.parallel:
        manager = ParallelManagerAgent()
    else:
        manager = HeuristicManagerAgent()

    try:
        run_episode(env, manager, max_steps=args.max_steps, dashboard=args.dashboard)
    finally:
        if pool is not None:
            pool.cleanup()


if __name__ == "__main__":
    main()
