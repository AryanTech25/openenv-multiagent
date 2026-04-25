"""
Worker Pool Manager for managing multiple Hugging Face workers.
"""

from typing import List, Dict, Any, Optional
from agents.llm_worker import HFWorker, HFWorkerConfig
import logging
import random

logger = logging.getLogger(__name__)


class WorkerPool:
    """
    Manages a pool of Hugging Face worker agents.
    
    Handles:
    - Worker initialization
    - Task assignment
    - Output generation
    - Resource cleanup
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        config: Optional[HFWorkerConfig] = None,
    ):
        """
        Initialize worker pool.
        
        Args:
            num_workers: Number of workers in the pool
            config: HFWorkerConfig for all workers
        """
        self.num_workers = num_workers
        self.config = config or HFWorkerConfig()
        self.workers: List[HFWorker] = []
        self.worker_states: Dict[int, Dict[str, Any]] = {}
        
        self._initialize_workers()
    
    def _initialize_workers(self) -> None:
        """Initialize all workers with random skill levels."""
        logger.info(f"Initializing {self.num_workers} {self.config.model_id} workers ({self.config.worker_type} mode)...")
        
        for worker_id in range(self.num_workers):
            # Random skill level between 0.3 and 1.0
            skill_level = random.uniform(0.3, 1.0)
            
            if self.config.worker_type == "api":
                from agents.llm_worker import HFAPIWorker
                worker = HFAPIWorker(
                    worker_id=worker_id,
                    skill_level=skill_level,
                    config=self.config,
                )
            else:
                worker = HFWorker(
                    worker_id=worker_id,
                    skill_level=skill_level,
                    config=self.config,
                )
            
            self.workers.append(worker)
            self.worker_states[worker_id] = {
                'skill_level': skill_level,
                'is_active': False,
                'current_subtask': None,
                'output': None,
                'failure_mode': None,
            }
            
            logger.info(f"  Worker {worker_id}: skill={skill_level:.2f}")
    
    def assign_task(
        self,
        worker_id: int,
        subtask: Dict[str, Any],
    ) -> str:
        """
        Assign a subtask to a worker and get output.
        
        Args:
            worker_id: ID of worker to assign to
            subtask: Subtask dictionary
            
        Returns:
            Generated output string
        """
        if worker_id >= len(self.workers):
            raise ValueError(f"Invalid worker_id: {worker_id}")
        
        worker = self.workers[worker_id]
        
        # Mark worker as active
        self.worker_states[worker_id]['is_active'] = True
        self.worker_states[worker_id]['current_subtask'] = subtask
        
        # Generate output
        output = worker.work_on_task(subtask)
        
        # Store output
        self.worker_states[worker_id]['output'] = output
        
        return output
    
    def get_worker_state(self, worker_id: int) -> Dict[str, Any]:
        """Get current state of a worker."""
        return self.worker_states.get(worker_id, {})
    
    def reset_worker(self, worker_id: int) -> None:
        """Reset a worker's state."""
        self.worker_states[worker_id] = {
            'skill_level': self.workers[worker_id].skill_level,
            'is_active': False,
            'current_subtask': None,
            'output': None,
            'failure_mode': None,
        }
    
    def cleanup(self) -> None:
        """Clean up all workers."""
        logger.info("Cleaning up worker pool...")
        for worker in self.workers:
            worker.cleanup()
        logger.info("✓ Worker pool cleaned up")
