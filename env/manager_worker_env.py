"""
ManagerWorkerEnv: Core OpenEnv environment for multi-agent coordination.

This environment simulates a workplace where one Manager Agent coordinates
multiple Worker Agents to complete complex multi-step tasks. Workers can fail
in realistic ways (hallucinations, off-task, incomplete, stuck), forcing the
Manager to learn detection and correction strategies under a limited token budget.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gym
from gym import spaces

from env.task_library import TaskLibrary, Task, Subtask
from env.hallucination_engine import HallucinationEngine
from env.reward_calculator import RewardCalculator


@dataclass
class WorkerState:
    """Represents the state of a single worker agent."""
    
    worker_id: int
    is_active: bool = False
    current_subtask_id: Optional[int] = None
    progress: float = 0.0  # 0.0 to 1.0
    hallucination_risk_score: float = 0.0  # 0.0 to 1.0
    output_quality_if_checked: float = 0.0  # 0.0 to 1.0
    tokens_consumed: int = 0
    failure_mode: Optional[str] = None  # 'hallucination', 'off_task', 'incomplete', 'stuck', None
    output_buffer: str = ""
    patience_counter: int = 0  # For detecting stuck loops
    skill_level: float = 0.5  # 0.3 to 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert worker state to observation array [is_active, progress, hallucination_risk, output_quality, tokens_consumed_ratio]."""
        return np.array([
            float(self.is_active),
            self.progress,
            self.hallucination_risk_score,
            self.output_quality_if_checked,
            # tokens_consumed_ratio will be set by environment
        ], dtype=np.float32)


@dataclass
class EpisodeLog:
    """Tracks events that occur during an episode."""
    
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_step(self, step_num: int, action: int, reward: float, info: Dict[str, Any]) -> None:
        """Log a step in the episode."""
        self.steps.append({
            'step': step_num,
            'action': action,
            'reward': reward,
            'info': info,
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode log to dictionary."""
        return {'steps': self.steps}


class ManagerWorkerEnv(gym.Env):
    """
    Multi-agent RL environment with Manager coordinating Workers.
    
    Inherits from gym.Env and implements standard gym interface.
    """
    
    # Action names for reference
    ACTION_NAMES = [
        'assign_subtask',      # 0: 10 tokens
        'check_worker_output', # 1: 50 tokens
        'correct_worker',      # 2: 30 tokens
        'reassign_task',       # 3: 40 tokens
        'fire_and_replace',    # 4: 100 tokens
        'approve_output',      # 5: 5 tokens
        'request_clarification', # 6: 20 tokens
    ]
    
    ACTION_COSTS = {
        0: 10,   # assign_subtask
        1: 50,   # check_worker_output
        2: 30,   # correct_worker
        3: 40,   # reassign_task
        4: 100,  # fire_and_replace
        5: 5,    # approve_output
        6: 20,   # request_clarification
    }
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize environment with configuration.
        
        Args:
            config: Dictionary with keys:
                - max_workers: int (default 4, range 1-4)
                - max_steps: int (default 50, range 10-100)
                - token_budget: int (default 1000, range 500-5000)
                - task_difficulty: int (default 3, range 1-5)
                - failure_injection_rate: float (default 0.6, range 0.0-1.0)
        """
        super().__init__()
        
        # Configuration
        self.max_workers = config.get('max_workers', 4)
        self.max_steps = config.get('max_steps', 50)
        self.token_budget = config.get('token_budget', 1000)
        self.task_difficulty = config.get('task_difficulty', 3)
        self.failure_injection_rate = config.get('failure_injection_rate', 0.6)
        
        # Validate configuration
        assert 1 <= self.max_workers <= 4, "max_workers must be in [1, 4]"
        assert 10 <= self.max_steps <= 100, "max_steps must be in [10, 100]"
        assert 500 <= self.token_budget <= 5000, "token_budget must be in [500, 5000]"
        assert 1 <= self.task_difficulty <= 5, "task_difficulty must be in [1, 5]"
        assert 0.0 <= self.failure_injection_rate <= 1.0, "failure_injection_rate must be in [0.0, 1.0]"
        
        # Initialize components
        self.task_library = TaskLibrary()
        self.hallucination_engine = HallucinationEngine()
        self.reward_calculator = RewardCalculator()
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'task_embedding': spaces.Box(
                low=-1.0, high=1.0,
                shape=(64,), dtype=np.float32
            ),
            'worker_states': spaces.Box(
                low=0.0, high=1.0,
                shape=(4, 5), dtype=np.float32
            ),
            'subtask_status': spaces.MultiBinary(4),
            'budget_remaining': spaces.Box(
                low=0.0, high=1.0,
                shape=(1,), dtype=np.float32
            ),
            'steps_remaining': spaces.Box(
                low=0.0, high=1.0,
                shape=(1,), dtype=np.float32
            ),
        })
        
        # Define action space
        self.action_space = spaces.Discrete(7)
        
        # Initialize internal state
        self.workers: List[WorkerState] = []
        self.task: Optional[Task] = None
        self.budget_remaining: int = self.token_budget
        self.step_counter: int = 0
        self.episode_log = EpisodeLog()
        
        # Episode tracking
        self.subtask_status: List[bool] = []  # True if subtask complete
        self.subtask_assignments: List[Optional[int]] = []  # worker_id assigned to each subtask
        self.hallucination_catch_rate: float = 0.0
        self.hallucinations_detected: int = 0
        self.hallucinations_approved: int = 0
        self.false_positives: int = 0
        
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Dictionary with keys:
                - task_embedding: np.ndarray shape (64,), dtype float32, range [-1, 1]
                - worker_states: np.ndarray shape (4, 5), dtype float32
                - subtask_status: np.ndarray shape (4,), dtype int32, values {0, 1}
                - budget_remaining: float, range [0, 1]
                - steps_remaining: float, range [0, 1]
        """
        # Select random task from task library
        self.task = self.task_library.sample_task(difficulty=self.task_difficulty)
        
        # Initialize workers with random skill levels
        self.workers = []
        for i in range(self.max_workers):
            skill_level = np.random.uniform(0.3, 1.0)
            worker = WorkerState(
                worker_id=i,
                is_active=False,
                skill_level=skill_level,
            )
            self.workers.append(worker)
        
        # Reset budget and counters
        self.budget_remaining = self.token_budget
        self.step_counter = 0
        self.episode_log = EpisodeLog()
        
        # Initialize subtask tracking
        num_subtasks = len(self.task.subtasks)
        self.subtask_status = [False] * num_subtasks
        self.subtask_assignments = [None] * num_subtasks
        
        # Reset hallucination tracking
        self.hallucination_catch_rate = 0.0
        self.hallucinations_detected = 0
        self.hallucinations_approved = 0
        self.false_positives = 0
        
        # Generate and return initial observation
        return self._generate_observation()
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute one step of environment.
        
        Args:
            action: int in range [0, 6], representing discrete action
        
        Returns:
            observation: Updated observation dictionary
            reward: float, cumulative reward for this step
            done: bool, True if episode terminates
            info: Dict with metadata (task_quality, tokens_used, etc.)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in range [0, 6]")
        
        # Check if action is affordable
        action_cost = self.ACTION_COSTS[action]
        if self.budget_remaining < action_cost:
            # Action rejected due to insufficient budget
            reward = -1.0
            done = True
            info = {
                'action_valid': False,
                'reason': 'insufficient_budget',
                'budget_remaining': self.budget_remaining,
                'action_cost': action_cost,
            }
            return self._generate_observation(), reward, done, info
        
        # Execute action
        self._execute_action(action)
        
        # Deduct tokens from budget
        self.budget_remaining -= action_cost
        
        # Increment step counter
        self.step_counter += 1
        
        # Check episode termination conditions
        done = self._check_termination()
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            final_quality=self._compute_final_quality() if done else 0.0,
            steps_used=self.step_counter,
            max_steps=self.max_steps,
            tokens_used=self.token_budget - self.budget_remaining,
            token_budget=self.token_budget,
            hallucination_interventions=self.hallucinations_detected,
            hallucination_approvals=self.hallucinations_approved,
            false_positives=self.false_positives,
        )
        
        # Generate observation
        observation = self._generate_observation()
        
        # Create info dictionary
        info = {
            'action_valid': True,
            'action': action,
            'action_name': self.ACTION_NAMES[action],
            'action_cost': action_cost,
            'budget_remaining': self.budget_remaining,
            'step': self.step_counter,
            'episode_log': self.episode_log.to_dict(),
            'hallucination_catch_rate': self.hallucination_catch_rate,
        }
        
        # Log step
        self.episode_log.add_step(self.step_counter, action, reward, info)
        
        return observation, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render environment state for visualization.
        
        Args:
            mode: 'human' for console, 'rgb_array' for image
        
        Returns:
            Rendered output or None
        """
        if mode == 'human':
            output = []
            output.append("=" * 80)
            output.append(f"Step {self.step_counter}/{self.max_steps}")
            output.append(f"Budget: {self.budget_remaining}/{self.token_budget} tokens")
            output.append(f"Task: {self.task.task_type if self.task else 'None'}")
            output.append(f"Subtasks: {sum(self.subtask_status)}/{len(self.subtask_status)} complete")
            output.append("")
            
            for worker in self.workers:
                status = "ACTIVE" if worker.is_active else "IDLE"
                output.append(
                    f"Worker {worker.worker_id}: {status} | "
                    f"Progress: {worker.progress:.2f} | "
                    f"Quality: {worker.output_quality_if_checked:.2f} | "
                    f"Skill: {worker.skill_level:.2f}"
                )
            
            output.append("=" * 80)
            return "\n".join(output)
        
        return None
    
    def _execute_action(self, action: int) -> None:
        """Execute manager action and update environment state."""
        if action == 0:  # assign_subtask
            self._action_assign_subtask()
        elif action == 1:  # check_worker_output
            self._action_check_worker_output()
        elif action == 2:  # correct_worker
            self._action_correct_worker()
        elif action == 3:  # reassign_task
            self._action_reassign_task()
        elif action == 4:  # fire_and_replace
            self._action_fire_and_replace()
        elif action == 5:  # approve_output
            self._action_approve_output()
        elif action == 6:  # request_clarification
            self._action_request_clarification()
    
    def _action_assign_subtask(self) -> None:
        """Assign next unassigned subtask to an idle worker."""
        # Find next unassigned subtask
        for i, assigned in enumerate(self.subtask_assignments):
            if assigned is None and not self.subtask_status[i]:
                # Find idle worker
                for worker in self.workers:
                    if not worker.is_active:
                        worker.is_active = True
                        worker.current_subtask_id = i
                        worker.progress = 0.0
                        worker.output_buffer = ""
                        worker.patience_counter = 0
                        self.subtask_assignments[i] = worker.worker_id
                        break
                break
    
    def _action_check_worker_output(self) -> None:
        """Check active worker's output quality."""
        # Find first active worker
        for worker in self.workers:
            if worker.is_active and worker.current_subtask_id is not None:
                # Simulate checking output
                worker.output_quality_if_checked = np.random.uniform(0.0, 1.0)
                break
    
    def _action_correct_worker(self) -> None:
        """Send corrective instructions to a worker."""
        # Find first active worker
        for worker in self.workers:
            if worker.is_active and worker.current_subtask_id is not None:
                # Improve output quality
                improvement = np.random.uniform(0.2, 0.5)
                worker.output_quality_if_checked = min(1.0, worker.output_quality_if_checked + improvement)
                break
    
    def _action_reassign_task(self) -> None:
        """Reassign current subtask to different worker."""
        # Find active worker
        for worker in self.workers:
            if worker.is_active and worker.current_subtask_id is not None:
                subtask_id = worker.current_subtask_id
                # Reset worker
                worker.is_active = False
                worker.current_subtask_id = None
                worker.progress = 0.0
                worker.output_buffer = ""
                # Find idle worker to reassign to
                for other_worker in self.workers:
                    if not other_worker.is_active:
                        other_worker.is_active = True
                        other_worker.current_subtask_id = subtask_id
                        other_worker.progress = 0.0
                        other_worker.output_buffer = ""
                        self.subtask_assignments[subtask_id] = other_worker.worker_id
                        break
                break
    
    def _action_fire_and_replace(self) -> None:
        """Replace a worker with a new one."""
        # Find first active worker
        for i, worker in enumerate(self.workers):
            if worker.is_active:
                subtask_id = worker.current_subtask_id
                # Replace worker
                new_skill = np.random.uniform(0.3, 1.0)
                self.workers[i] = WorkerState(
                    worker_id=i,
                    is_active=True if subtask_id is not None else False,
                    current_subtask_id=subtask_id,
                    skill_level=new_skill,
                )
                break
    
    def _action_approve_output(self) -> None:
        """Approve active worker's output and mark subtask complete."""
        # Find first active worker
        for worker in self.workers:
            if worker.is_active and worker.current_subtask_id is not None:
                subtask_id = worker.current_subtask_id
                self.subtask_status[subtask_id] = True
                worker.is_active = False
                worker.current_subtask_id = None
                break
    
    def _action_request_clarification(self) -> None:
        """Request clarification about task structure."""
        # This action increases task embedding specificity (no-op for now)
        pass
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if max steps reached
        if self.step_counter >= self.max_steps:
            return True
        
        # Terminate if budget exhausted
        if self.budget_remaining <= 0:
            return True
        
        # Terminate if all subtasks completed
        if all(self.subtask_status):
            return True
        
        # Terminate if worker stuck (patience counter too high)
        for worker in self.workers:
            if worker.patience_counter > 10:
                return True
        
        return False
    
    def _compute_final_quality(self) -> float:
        """Compute final quality score from completed subtasks."""
        if not self.task or not self.task.subtasks:
            return 0.0
        
        total_quality = 0.0
        completed_count = 0
        
        for i, subtask in enumerate(self.task.subtasks):
            if self.subtask_status[i]:
                # Get worker that completed this subtask
                worker_id = self.subtask_assignments[i]
                if worker_id is not None:
                    worker = self.workers[worker_id]
                    total_quality += worker.output_quality_if_checked
                    completed_count += 1
        
        if completed_count == 0:
            return 0.0
        
        return total_quality / completed_count
    
    def _generate_observation(self) -> Dict[str, np.ndarray]:
        """
        Generate observation dictionary from current environment state.
        
        Returns:
            Dict with 5 keys as specified in observation_space
        """
        # Generate task embedding (64-dim, normalized to [-1, 1])
        if self.task:
            task_embedding = self._get_task_embedding(self.task.task_type)
        else:
            task_embedding = np.zeros(64, dtype=np.float32)
        
        # Generate worker states (4x5 array)
        worker_states = np.zeros((4, 5), dtype=np.float32)
        for i, worker in enumerate(self.workers):
            tokens_consumed_ratio = (self.token_budget - self.budget_remaining) / self.token_budget if self.token_budget > 0 else 0.0
            worker_states[i] = np.array([
                float(worker.is_active),
                worker.progress,
                worker.hallucination_risk_score,
                worker.output_quality_if_checked,
                tokens_consumed_ratio,
            ], dtype=np.float32)
        
        # Generate subtask status (binary array)
        subtask_status = np.array(self.subtask_status[:4], dtype=np.int32)
        
        # Generate budget and steps remaining (normalized to [0, 1])
        budget_remaining = np.array([self.budget_remaining / self.token_budget], dtype=np.float32)
        steps_remaining = np.array([(self.max_steps - self.step_counter) / self.max_steps], dtype=np.float32)
        
        return {
            'task_embedding': task_embedding,
            'worker_states': worker_states,
            'subtask_status': subtask_status,
            'budget_remaining': budget_remaining,
            'steps_remaining': steps_remaining,
        }
    
    def _get_task_embedding(self, task_type: str) -> np.ndarray:
        """Get fixed embedding for task type."""
        # Simple lookup table for task embeddings
        embeddings = {
            'web_development': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
            'research': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
            'software_engineering': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
            'product_management': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
            'academic_writing': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
        }
        return embeddings.get(task_type, np.zeros(64, dtype=np.float32)).astype(np.float32)
