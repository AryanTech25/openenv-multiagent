#!/usr/bin/env python3
"""
End-to-end multi-agent demo: a Manager actually drives a pool of LLM Workers
through a complete task in ManagerWorkerEnv.

Two modes:
- ``--real``: spin up a real WorkerPool of Hugging Face workers and have them
  generate text. Requires ``HF_TOKEN`` for gated models.
- default: simulator-only mode (no LLM calls), useful for CI / fast smoke tests
  and to verify the wiring without network or model downloads.

Both modes go through the same ManagerWorkerEnv code paths; only the worker
backend differs.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the package importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from agents import HeuristicManagerAgent, PPOManagerAgent, WorkerPool
from agents.llm_worker import HFWorkerConfig
from env import ManagerWorkerEnv

load_dotenv()


def build_pool(model_id: str, mode: str, num_workers: int) -> WorkerPool:
    """Spin up a WorkerPool of HF workers (local or via the Inference API)."""
    config = HFWorkerConfig(
        model_id=model_id,
        hf_token=os.getenv("HF_TOKEN"),
        worker_type=mode,
        max_tokens=128,
        temperature=0.7,
    )
    return WorkerPool(num_workers=num_workers, config=config)


def run_episode(env: ManagerWorkerEnv, manager, max_steps: int = 50) -> None:
    obs = env.reset()
    if hasattr(manager, "reset"):
        manager.reset()

    print("=" * 80)
    print(f"Task: {env.state.task['task_type']} — {env.state.task['description']}")
    print(f"Subtasks: {env.state.task['num_subtasks']}  Budget: {env.state.budget_remaining}")
    print(
        f"Workers: "
        + ", ".join(f"#{w.worker_id}(skill={w.skill_level:.2f})" for w in env.state.workers)
    )
    print("=" * 80)

    total_reward = 0.0
    for step in range(max_steps):
        action = manager.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        active = next(
            (w for w in env.state.workers if w.is_active and w.has_output), None
        )
        snippet = ""
        if active is not None and active.output_buffer:
            snippet = active.output_buffer.replace("\n", " ")[:80]

        print(
            f"  step {step + 1:>2} "
            f"action={info.get('action_name', '?'):<22} "
            f"reward={reward:+.2f} "
            f"budget={info.get('budget_remaining', '?')} "
            f"subtasks={env.state.subtask_status} "
            + (f"output=\"{snippet}…\"" if snippet else "")
        )
        if done:
            break

    print("-" * 80)
    quality = env._compute_final_quality()
    print(
        f"Episode done. total_reward={total_reward:.2f}  "
        f"final_quality={quality:.2f}  "
        f"hallucinations_detected={env.state.hallucinations_detected}  "
        f"hallucinations_approved={env.state.hallucinations_approved}  "
        f"false_positives={env.state.false_positives}"
    )
    print("=" * 80)
    print()
    print("Worker outputs at end-of-episode:")
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
        help="HF model id for the worker pool when --real is set.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default="local",
        help="Worker backend when --real is set.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Pool size for --real mode."
    )
    parser.add_argument(
        "--ppo-checkpoint",
        default=None,
        help="Optional path to a PPO checkpoint produced by train_manager.py. "
        "Without it, a heuristic assign→check→approve manager is used.",
    )
    parser.add_argument("--task-difficulty", type=int, default=2)
    parser.add_argument("--token-budget", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    pool = None
    if args.real:
        print(f"Spinning up real WorkerPool ({args.num_workers}× {args.model}, {args.mode})…")
        pool = build_pool(args.model, args.mode, args.num_workers)

    env = ManagerWorkerEnv(
        config={
            "max_workers": args.num_workers if args.real else 4,
            "max_steps": args.max_steps,
            "token_budget": args.token_budget,
            "task_difficulty": args.task_difficulty,
            "failure_injection_rate": 0.6,
        },
        worker_pool=pool,
    )

    if args.ppo_checkpoint:
        print(f"Loading PPO manager from {args.ppo_checkpoint}…")
        manager = PPOManagerAgent(args.ppo_checkpoint)
    else:
        manager = HeuristicManagerAgent()

    try:
        run_episode(env, manager, max_steps=args.max_steps)
    finally:
        if pool is not None:
            pool.cleanup()


if __name__ == "__main__":
    main()
