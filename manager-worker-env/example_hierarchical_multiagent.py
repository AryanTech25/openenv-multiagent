#!/usr/bin/env python3
"""
Hierarchical multi-agent demo:

    1× MetaManagerAgent
        └── 4× ParallelManagerAgent (sub-managers)
                └── N× workers per sub-manager (sim or real HF LLMs)

The MetaManager picks one sub-manager to advance every meta-step. Each
sub-manager fans its task out across its own worker pool. A single terminal
dashboard renders all five agents (meta + 4 subs) on every meta-step, and a
final report summarises the entire hierarchy when the run ends.

Examples
--------
Sim only, watch live:
    ./.venv/bin/python3 example_hierarchical_multiagent.py --dashboard

Real heterogeneous LLMs (4 different HF models, one per sub-manager):
    ./.venv/bin/python3 example_hierarchical_multiagent.py \
        --real --heterogeneous --dashboard --workers-per-sub 1

Different meta strategy:
    ./.venv/bin/python3 example_hierarchical_multiagent.py \
        --dashboard --meta-strategy priority
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from agents import (
    MetaManagerAgent,
    ParallelManagerAgent,
    SubEnvSnapshot,
    SubStatus,
    WorkerPool,
)
from agents.llm_worker import HFAPIWorker, HFWorker, HFWorkerConfig
from env import ManagerWorkerEnv

load_dotenv()


# Default mix kept to *small* HF models so the demo stays runnable on a laptop.
# Override with --hetero-models if you want bigger/different LLMs.
DEFAULT_HETEROGENEOUS_MODELS: List[str] = [
    "HuggingFaceTB/SmolLM-135M",
    "HuggingFaceTB/SmolLM-360M",
    "Qwen/Qwen2-0.5B",
    "microsoft/phi-2",
]


# ───────────────────────────── sub-env construction ──────────────────────────


def build_homogeneous_pool(model_id: str, mode: str, num_workers: int) -> WorkerPool:
    cfg = HFWorkerConfig(
        model_id=model_id,
        hf_token=os.getenv("HF_TOKEN"),
        worker_type=mode,
        max_tokens=128,
        temperature=0.7,
    )
    return WorkerPool(num_workers=num_workers, config=cfg)


def build_single_model_pool(model_id: str, mode: str, num_workers: int, base_skill: float) -> WorkerPool:
    """Pool of N workers all running the same HF model, but with staggered skills."""
    pool = WorkerPool.__new__(WorkerPool)
    pool.num_workers = num_workers
    pool.config = HFWorkerConfig(worker_type=mode, hf_token=os.getenv("HF_TOKEN"))
    pool.workers = []
    pool.worker_states = {}
    worker_cls = HFAPIWorker if mode == "api" else HFWorker
    for wid in range(num_workers):
        cfg = HFWorkerConfig(
            model_id=model_id,
            hf_token=os.getenv("HF_TOKEN"),
            worker_type=mode,
            max_tokens=128,
            temperature=0.7,
        )
        skill = min(0.95, base_skill + 0.1 * wid)
        worker = worker_cls(worker_id=wid, skill_level=skill, config=cfg)
        pool.workers.append(worker)
        pool.worker_states[wid] = {
            "skill_level": worker.skill_level,
            "is_active": False,
            "current_subtask": None,
            "output": None,
            "failure_mode": None,
        }
    return pool


@dataclass
class SubUnit:
    """One sub-manager + its env + (optional) worker pool + bookkeeping."""

    sub_id: int
    label: str
    env: ManagerWorkerEnv
    manager: ParallelManagerAgent
    pool: Optional[WorkerPool]
    model_label: str
    obs: object
    cumulative_reward: float = 0.0
    last_reward: float = 0.0
    steps_taken: int = 0
    done: bool = False
    over_budget: bool = False
    stuck_strikes: int = 0  # how many recent steps produced negligible progress

    @property
    def status(self) -> SubStatus:
        if self.done:
            return SubStatus.DONE
        if self.over_budget:
            return SubStatus.OVER_BUDGET
        if self.stuck_strikes >= 8:
            return SubStatus.STUCK
        return SubStatus.RUNNING

    def snapshot(self) -> SubEnvSnapshot:
        st = self.env.state
        completed = sum(st.subtask_status)
        total = max(1, len(st.subtask_status))
        completion_pct = completed / total
        revealed = [w.actual_quality for w in st.workers if w.has_output]
        avg_q = sum(revealed) / len(revealed) if revealed else 0.0
        return SubEnvSnapshot(
            sub_id=self.sub_id,
            status=self.status,
            completion_pct=completion_pct,
            budget_remaining=st.budget_remaining,
            budget_total=self.env.token_budget,
            last_reward=self.last_reward,
            cumulative_reward=self.cumulative_reward,
            steps_taken=self.steps_taken,
            avg_actual_quality=avg_q,
            label=self.label,
        )


def build_sub_unit(
    sub_id: int,
    *,
    real: bool,
    mode: str,
    homogeneous_model: str,
    hetero_models: Optional[List[str]],
    workers_per_sub: int,
    task_difficulty: int,
    sub_token_budget: int,
    sub_max_steps: int,
    failure_rate: float,
) -> SubUnit:
    pool: Optional[WorkerPool] = None
    model_label = "simulator"

    if real:
        if hetero_models is not None:
            model_id = hetero_models[sub_id % len(hetero_models)]
        else:
            model_id = homogeneous_model
        # Stagger base skill across sub-envs so the dashboard shows variety.
        base_skill = 0.45 + 0.1 * (sub_id % 4)
        pool = build_single_model_pool(model_id, mode, workers_per_sub, base_skill)
        model_label = f"{model_id.split('/')[-1]} ({mode})"

    env = ManagerWorkerEnv(
        config={
            "max_workers": max(1, workers_per_sub),
            "max_steps": sub_max_steps,
            "token_budget": sub_token_budget,
            "task_difficulty": task_difficulty,
            "failure_injection_rate": failure_rate,
        },
        worker_pool=pool,
    )
    obs = env.reset()
    manager = ParallelManagerAgent()
    label = env.state.task["task_type"]
    return SubUnit(
        sub_id=sub_id,
        label=label,
        env=env,
        manager=manager,
        pool=pool,
        model_label=model_label,
        obs=obs,
    )


# ──────────────────────────────── dashboard ──────────────────────────────────

ANSI = {
    "reset": "\033[0m",
    "bold":  "\033[1m",
    "dim":   "\033[2m",
    "green": "\033[32m",
    "yellow":"\033[33m",
    "red":   "\033[31m",
    "cyan":  "\033[36m",
    "mag":   "\033[35m",
}


def color(text: str, name: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{ANSI.get(name, '')}{text}{ANSI['reset']}"


def status_chip(status: SubStatus, use_color: bool) -> str:
    mapping = {
        SubStatus.RUNNING: ("RUN ", "yellow"),
        SubStatus.DONE: ("DONE", "green"),
        SubStatus.STUCK: ("STK ", "red"),
        SubStatus.OVER_BUDGET: ("$$  ", "red"),
    }
    text, c = mapping[status]
    return color(text, c, use_color)


def progress_bar(pct: float, width: int = 12) -> str:
    fill = int(round(pct * width))
    return "[" + "█" * fill + "·" * (width - fill) + "]"


def render_meta_dashboard(
    meta: MetaManagerAgent,
    units: List[SubUnit],
    last_action_sub: Optional[int],
    use_color: bool = True,
) -> str:
    snaps = [u.snapshot() for u in units]
    total_completion = sum(s.completion_pct for s in snaps) / len(snaps)
    total_reward = sum(s.cumulative_reward for s in snaps)
    total_budget_remaining = sum(s.budget_remaining for s in snaps)
    total_budget = sum(s.budget_total for s in snaps)
    n_done = sum(1 for s in snaps if s.status == SubStatus.DONE)

    width = 128

    def row(text: str) -> str:
        # ljust on the *visible* width — color codes don't take screen columns.
        visible = _visible_len(text)
        pad = max(0, width - visible)
        return "║" + text + " " * pad + "║"

    lines: List[str] = []
    lines.append("╔" + "═" * width + "╗")
    title = " META-MANAGER DASHBOARD "
    pad = (width - len(title)) // 2
    lines.append("║" + " " * pad + color(title, "bold", use_color) + " " * (width - pad - len(title)) + "║")
    lines.append("╠" + "═" * width + "╣")
    line1 = (
        f" meta_step={meta.steps:>3}   strategy={meta.strategy:<12}   "
        f"hier_done={n_done}/{len(units)}   "
        f"avg_completion={progress_bar(total_completion)} {total_completion:>4.0%}"
    )
    line2 = (
        f" reward_total={total_reward:+.1f}   "
        f"budget={total_budget_remaining}/{total_budget}"
    )
    if last_action_sub is not None:
        line2 += f"   ↳ last: advance sub-env #{last_action_sub} ({units[last_action_sub].label})"
    lines.append(row(line1))
    lines.append(row(line2))
    lines.append("╠" + "═" * width + "╣")

    for unit, snap in zip(units, snaps):
        bar = progress_bar(snap.completion_pct, width=14)
        chip = status_chip(snap.status, use_color)
        head = (
            f" sub#{unit.sub_id} {chip}   task={snap.label:<22} "
            f"model={unit.model_label:<28} "
            f"step={snap.steps_taken:>3}  reward={snap.cumulative_reward:+7.1f}  "
            f"budget={snap.budget_remaining:>4}/{snap.budget_total}"
        )
        lines.append(row(head))
        sub_strip = []
        for i, done in enumerate(unit.env.state.subtask_status):
            owner = unit.env.state.subtask_assignments[i] if i < len(unit.env.state.subtask_assignments) else None
            if done:
                sub_strip.append(f"S{i}:✓")
            elif owner is not None:
                sub_strip.append(f"S{i}:w{owner}")
            else:
                sub_strip.append(f"S{i}:·")
        lines.append(row(
            f"   completion {bar} {snap.completion_pct:>4.0%}   "
            f"avg_q={snap.avg_actual_quality:.2f}   "
            f"subtasks: {' '.join(sub_strip)}"
        ))
        for w in unit.env.state.workers:
            state = "ACT" if w.is_active else "idl"
            sub = f"S{w.current_subtask_id}" if w.current_subtask_id is not None else " - "
            failure = (w.failure_mode or "—")[:10]
            actual = f"{w.actual_quality:.2f}" if w.has_output else "  - "
            preview = (w.output_buffer or "").replace("\n", " ")[:50]
            lines.append(row(
                f"     w#{w.worker_id} {state}  skill={w.skill_level:.2f}  "
                f"{sub:<3}  fail={failure:<10}  q={actual}  \"{preview}\""
            ))
        lines.append("╠" + "═" * width + "╣")
    lines[-1] = "╚" + "═" * width + "╝"
    return "\n".join(lines)


_ANSI_RE = None


def _visible_len(s: str) -> int:
    """Return the visible (printable) width of `s`, ignoring ANSI escapes."""
    global _ANSI_RE
    if _ANSI_RE is None:
        import re
        _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
    return len(_ANSI_RE.sub("", s))


def render_final_report(meta: MetaManagerAgent, units: List[SubUnit], use_color: bool = True) -> str:
    snaps = [u.snapshot() for u in units]
    total_completion = sum(s.completion_pct for s in snaps) / len(snaps)
    total_reward = sum(s.cumulative_reward for s in snaps)
    n_done = sum(1 for s in snaps if s.status == SubStatus.DONE)
    n_subtasks_total = sum(len(u.env.state.subtask_status) for u in units)
    n_subtasks_done = sum(sum(u.env.state.subtask_status) for u in units)
    halluc_detected = sum(u.env.state.hallucinations_detected for u in units)
    halluc_approved = sum(u.env.state.hallucinations_approved for u in units)
    false_pos = sum(u.env.state.false_positives for u in units)

    lines: List[str] = []
    lines.append("")
    lines.append(color("═" * 102, "cyan", use_color))
    lines.append(color(" FINAL HIERARCHICAL REPORT", "bold", use_color))
    lines.append(color("═" * 102, "cyan", use_color))
    lines.append(
        f"meta-strategy:        {meta.strategy}\n"
        f"meta-steps taken:     {meta.steps}\n"
        f"sub-envs done:        {n_done}/{len(units)}\n"
        f"subtasks complete:    {n_subtasks_done}/{n_subtasks_total}\n"
        f"avg completion:       {total_completion:.1%}\n"
        f"hierarchy reward:     {total_reward:+.2f}\n"
        f"hallucinations:       detected={halluc_detected}  approved={halluc_approved}  false_positives={false_pos}"
    )
    lines.append("")
    lines.append(color("Per sub-env breakdown", "bold", use_color))
    lines.append("─" * 102)
    lines.append(
        f"{'id':<4}{'task':<22}{'model':<32}{'status':<7}{'compl%':>7}{'avg_q':>8}{'reward':>10}{'steps':>7}"
    )
    lines.append("─" * 102)
    for unit, snap in zip(units, snaps):
        lines.append(
            f"{unit.sub_id:<4}{snap.label:<22}{unit.model_label:<32}"
            f"{snap.status.value:<7}{snap.completion_pct * 100:>6.0f}%{snap.avg_actual_quality:>8.2f}"
            f"{snap.cumulative_reward:>+10.1f}{snap.steps_taken:>7}"
        )
    lines.append("")
    lines.append(color("Worker outputs", "bold", use_color))
    lines.append("─" * 102)
    for unit in units:
        lines.append(f"sub#{unit.sub_id} ({unit.label}, {unit.model_label})")
        for w in unit.env.state.workers:
            buf = (w.output_buffer or "").replace("\n", " ")[:90]
            lines.append(
                f"  w#{w.worker_id} skill={w.skill_level:.2f} "
                f"actual_q={w.actual_quality:.2f} fail={w.failure_mode or '—':<14} "
                f"\"{buf}\""
            )
    lines.append(color("═" * 102, "cyan", use_color))
    return "\n".join(lines)


# ───────────────────────────── hierarchical loop ─────────────────────────────


def step_sub_unit(unit: SubUnit) -> None:
    """Advance one sub-manager by one env-step."""
    if unit.done or unit.over_budget:
        return
    action = unit.manager.predict(unit.obs)
    obs, reward, done, info = unit.env.step(action)
    unit.obs = obs
    unit.last_reward = reward
    unit.cumulative_reward += reward
    unit.steps_taken += 1
    if abs(reward) < 0.5 and not done:
        unit.stuck_strikes += 1
    else:
        unit.stuck_strikes = max(0, unit.stuck_strikes - 1)
    if done:
        unit.done = True
    elif unit.env.state.budget_remaining <= 0:
        unit.over_budget = True


def run_hierarchy(
    meta: MetaManagerAgent,
    units: List[SubUnit],
    *,
    max_meta_steps: int,
    dashboard: bool,
    sleep_secs: float,
    use_color: bool,
) -> None:
    print(color("Spinning up hierarchy:", "bold", use_color))
    print(f"  Meta-strategy: {meta.strategy}")
    for u in units:
        n_workers = len(u.env.state.workers)
        print(
            f"  sub#{u.sub_id}  task={u.label:<22} model={u.model_label:<32} "
            f"workers={n_workers}  budget={u.env.token_budget}"
        )
    print()

    last_action_sub: Optional[int] = None
    for meta_step in range(max_meta_steps):
        snaps = [u.snapshot() for u in units]
        action = meta.predict(snaps)
        if action is None:
            break
        step_sub_unit(units[action.target_sub_id])
        last_action_sub = action.target_sub_id

        if dashboard:
            os.system("clear" if os.name != "nt" else "cls")
            print(render_meta_dashboard(meta, units, last_action_sub, use_color=use_color))
            time.sleep(sleep_secs)
        else:
            u = units[action.target_sub_id]
            print(
                f"meta_step {meta.steps:>3}  → sub#{action.target_sub_id} "
                f"({u.label[:14]:<14}) {u.status.value:<5} "
                f"reward={u.last_reward:+5.1f}  cum={u.cumulative_reward:+7.1f}  "
                f"compl={sum(u.env.state.subtask_status)}/{len(u.env.state.subtask_status)}"
            )

    print(render_final_report(meta, units, use_color=use_color))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical multi-agent demo")
    parser.add_argument("--num-sub-managers", type=int, default=4)
    parser.add_argument(
        "--workers-per-sub",
        type=int,
        default=2,
        help="Workers per sub-manager. With --real keep this small (1-2) so HF loads stay light.",
    )
    parser.add_argument("--task-difficulty", type=int, default=3)
    parser.add_argument(
        "--sub-token-budget", type=int, default=2000, help="Per-sub-env token budget."
    )
    parser.add_argument("--sub-max-steps", type=int, default=60)
    parser.add_argument(
        "--max-meta-steps",
        type=int,
        default=240,
        help="Total scheduling ticks the MetaManager gets across all sub-envs.",
    )
    parser.add_argument("--failure-rate", type=float, default=0.6)
    parser.add_argument(
        "--meta-strategy",
        choices=["round_robin", "priority", "quality"],
        default="round_robin",
    )

    parser.add_argument("--real", action="store_true", help="Use real HF workers.")
    parser.add_argument(
        "--mode", choices=["local", "api"], default="local", help="HF backend when --real is set."
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M",
        help="Single HF model to use everywhere when --real (and not --heterogeneous).",
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        help="With --real, give each sub-manager a different HF model from a built-in list.",
    )
    parser.add_argument(
        "--hetero-models",
        nargs="+",
        default=None,
        help="Optional explicit list of HF model ids (overrides the built-in heterogeneous mix).",
    )

    parser.add_argument("--dashboard", action="store_true", help="Live multi-pane terminal dashboard.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument("--sleep", type=float, default=0.15, help="Seconds between dashboard frames.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force HF/transformers into offline mode (uses cached weights, skips Hub network calls).",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    use_color = not args.no_color and sys.stdout.isatty()

    hetero_models = None
    if args.real and (args.heterogeneous or args.hetero_models):
        source = args.hetero_models or DEFAULT_HETEROGENEOUS_MODELS
        hetero_models = source[: args.num_sub_managers]
        # If the user asked for more sub-managers than models, recycle the list.
        while len(hetero_models) < args.num_sub_managers:
            hetero_models.append(source[len(hetero_models) % len(source)])

    units: List[SubUnit] = []
    for sid in range(args.num_sub_managers):
        units.append(
            build_sub_unit(
                sid,
                real=args.real,
                mode=args.mode,
                homogeneous_model=args.model,
                hetero_models=hetero_models,
                workers_per_sub=args.workers_per_sub,
                task_difficulty=args.task_difficulty,
                sub_token_budget=args.sub_token_budget,
                sub_max_steps=args.sub_max_steps,
                failure_rate=args.failure_rate,
            )
        )

    meta = MetaManagerAgent(num_sub_managers=args.num_sub_managers, strategy=args.meta_strategy)

    try:
        run_hierarchy(
            meta,
            units,
            max_meta_steps=args.max_meta_steps,
            dashboard=args.dashboard,
            sleep_secs=args.sleep,
            use_color=use_color,
        )
    finally:
        for u in units:
            if u.pool is not None:
                u.pool.cleanup()


if __name__ == "__main__":
    main()
