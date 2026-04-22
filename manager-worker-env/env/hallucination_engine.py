"""
Hallucination Engine: Injects realistic failure modes into worker outputs.

Implements 4 failure modes:
- Hallucination: High plausibility, low actual quality
- Off-task: Medium plausibility, near-zero quality
- Incomplete: Moderate plausibility, partial quality
- Stuck: Looping behavior, detected via patience counter
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class FailureMode:
    """Specification for a failure mode."""
    
    mode_type: str  # 'hallucination', 'off_task', 'incomplete', 'stuck'
    surface_plausibility: float  # How correct output appears
    actual_quality: float  # True quality of output
    detection_cost: int  # Tokens to detect via check_worker_output
    correction_cost: int  # Tokens to correct via correct_worker
    description: str


@dataclass
class WorkerOutput:
    """Output from a worker's execution."""
    
    content: str
    surface_plausibility: float  # What manager sees without checking
    actual_quality: float  # True quality (revealed by check_worker_output)
    failure_type: Optional[str]  # None, 'hallucination', 'off_task', 'incomplete', 'stuck'
    detectable: bool  # Whether failure is checkable
    detection_cost: int  # Tokens to detect


class HallucinationEngine:
    """Injects realistic failure modes into worker outputs."""
    
    # Failure mode specifications
    HALLUCINATION = FailureMode(
        mode_type='hallucination',
        surface_plausibility=0.85,  # Random in [0.8, 0.95]
        actual_quality=0.2,  # Random in [0.1, 0.3]
        detection_cost=50,
        correction_cost=30,
        description='Output looks correct but contains subtle errors'
    )
    
    OFF_TASK = FailureMode(
        mode_type='off_task',
        surface_plausibility=0.5,  # Random in [0.4, 0.6]
        actual_quality=0.05,  # Random in [0.0, 0.1]
        detection_cost=20,
        correction_cost=40,
        description='Worker produces output unrelated to subtask'
    )
    
    INCOMPLETE = FailureMode(
        mode_type='incomplete',
        surface_plausibility=0.55,  # Random in [0.5, 0.6]
        actual_quality=0.35,  # Random in [0.2, 0.5]
        detection_cost=50,
        correction_cost=30,
        description='Output is partial or missing key components'
    )
    
    STUCK = FailureMode(
        mode_type='stuck',
        surface_plausibility=0.0,
        actual_quality=0.0,
        detection_cost=20,
        correction_cost=100,
        description='Worker loops without progress'
    )
    
    # Failure mode distribution
    FAILURE_MODE_DISTRIBUTION = {
        'hallucination': 0.4,
        'off_task': 0.3,
        'incomplete': 0.2,
        'stuck': 0.1,
    }
    
    def __init__(self):
        """Initialize hallucination engine."""
        pass
    
    def generate_output(
        self,
        subtask_description: str,
        worker_skill: float,
        task_difficulty: int,
        failure_injection_rate: float,
    ) -> WorkerOutput:
        """
        Generate worker output with possible failure injection.
        
        Args:
            subtask_description: Description of the subtask
            worker_skill: Worker skill level (0.3 to 1.0)
            task_difficulty: Task difficulty (1 to 5)
            failure_injection_rate: Probability of injecting failure (0.0 to 1.0)
        
        Returns:
            WorkerOutput with content, quality, and failure info
        """
        # Calculate failure probability based on skill and difficulty
        failure_probability = self._calculate_failure_probability(worker_skill, task_difficulty)
        
        # Decide whether to inject failure
        should_inject_failure = np.random.random() < (failure_probability * failure_injection_rate)
        
        if should_inject_failure:
            # Select failure mode based on distribution
            failure_mode = self._select_failure_mode()
            return self._generate_failed_output(subtask_description, failure_mode)
        else:
            # Generate successful output
            return self._generate_successful_output(subtask_description, worker_skill)
    
    def _calculate_failure_probability(self, worker_skill: float, task_difficulty: int) -> float:
        """
        Calculate probability of failure based on skill and difficulty.
        
        Formula: failure_probability = (1 - worker_skill) * (task_difficulty / 5) * 0.9
        
        Examples:
        - Skill 1.0, Difficulty 1: (1 - 1.0) * (1 / 5) * 0.9 = 0.0 (5% success)
        - Skill 0.3, Difficulty 5: (1 - 0.3) * (5 / 5) * 0.9 = 0.63 (63% failure)
        """
        return (1.0 - worker_skill) * (task_difficulty / 5.0) * 0.9
    
    def _select_failure_mode(self) -> str:
        """Select a failure mode based on distribution."""
        modes = list(self.FAILURE_MODE_DISTRIBUTION.keys())
        probabilities = list(self.FAILURE_MODE_DISTRIBUTION.values())
        return np.random.choice(modes, p=probabilities)
    
    def _generate_successful_output(self, subtask_description: str, worker_skill: float) -> WorkerOutput:
        """Generate successful output."""
        # Quality increases with skill level
        quality = 0.6 + (worker_skill * 0.4)
        
        return WorkerOutput(
            content=f"Completed: {subtask_description}",
            surface_plausibility=quality,
            actual_quality=quality,
            failure_type=None,
            detectable=False,
            detection_cost=0,
        )
    
    def _generate_failed_output(self, subtask_description: str, failure_mode: str) -> WorkerOutput:
        """Generate failed output based on failure mode."""
        if failure_mode == 'hallucination':
            return self._generate_hallucination(subtask_description)
        elif failure_mode == 'off_task':
            return self._generate_off_task(subtask_description)
        elif failure_mode == 'incomplete':
            return self._generate_incomplete(subtask_description)
        elif failure_mode == 'stuck':
            return self._generate_stuck(subtask_description)
        else:
            raise ValueError(f"Unknown failure mode: {failure_mode}")
    
    def _generate_hallucination(self, subtask_description: str) -> WorkerOutput:
        """Generate hallucination failure."""
        surface_plausibility = np.random.uniform(0.8, 0.95)
        actual_quality = np.random.uniform(0.1, 0.3)
        
        return WorkerOutput(
            content=f"Completed: {subtask_description} (with subtle errors)",
            surface_plausibility=surface_plausibility,
            actual_quality=actual_quality,
            failure_type='hallucination',
            detectable=True,
            detection_cost=50,
        )
    
    def _generate_off_task(self, subtask_description: str) -> WorkerOutput:
        """Generate off-task failure."""
        surface_plausibility = np.random.uniform(0.4, 0.6)
        actual_quality = np.random.uniform(0.0, 0.1)
        
        return WorkerOutput(
            content=f"Completed: Different task (not {subtask_description})",
            surface_plausibility=surface_plausibility,
            actual_quality=actual_quality,
            failure_type='off_task',
            detectable=True,
            detection_cost=20,
        )
    
    def _generate_incomplete(self, subtask_description: str) -> WorkerOutput:
        """Generate incomplete failure."""
        surface_plausibility = np.random.uniform(0.5, 0.6)
        actual_quality = np.random.uniform(0.2, 0.5)
        
        return WorkerOutput(
            content=f"Partially completed: {subtask_description}",
            surface_plausibility=surface_plausibility,
            actual_quality=actual_quality,
            failure_type='incomplete',
            detectable=True,
            detection_cost=50,
        )
    
    def _generate_stuck(self, subtask_description: str) -> WorkerOutput:
        """Generate stuck failure."""
        return WorkerOutput(
            content=f"Processing: {subtask_description} (looping)",
            surface_plausibility=0.0,
            actual_quality=0.0,
            failure_type='stuck',
            detectable=True,
            detection_cost=20,
        )
