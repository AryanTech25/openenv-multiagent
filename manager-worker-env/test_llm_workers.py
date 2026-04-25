"""
Test script for Llama 2 worker agents.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.worker_pool import WorkerPool
from agents.llm_worker import LlamaWorkerConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_worker():
    """Test a single worker."""
    logger.info("=" * 60)
    logger.info("TEST 1: Single Worker Generation")
    logger.info("=" * 60)
    
    config = LlamaWorkerConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        max_tokens=256,
        temperature=0.7,
    )
    
    pool = WorkerPool(num_workers=1, config=config)
    
    subtask = {
        "description": "Write a Python function to calculate factorial",
        "context": "The function should handle edge cases",
    }
    
    logger.info(f"Assigning task to worker...")
    output = pool.assign_task(0, subtask)
    
    logger.info(f"\nGenerated Output:\n{output}\n")
    
    pool.cleanup()


def test_worker_pool():
    """Test worker pool with multiple workers."""
    logger.info("=" * 60)
    logger.info("TEST 2: Worker Pool (4 workers)")
    logger.info("=" * 60)
    
    config = LlamaWorkerConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        max_tokens=256,
        temperature=0.7,
    )
    
    pool = WorkerPool(num_workers=4, config=config)
    
    subtasks = [
        {
            "description": "Write HTML for a landing page header",
            "context": "Include navigation menu",
        },
        {
            "description": "Write CSS for styling the header",
            "context": "Make it responsive",
        },
        {
            "description": "Write JavaScript for menu interactions",
            "context": "Add smooth animations",
        },
        {
            "description": "Write a README for the project",
            "context": "Include setup instructions",
        },
    ]
    
    for worker_id, subtask in enumerate(subtasks):
        logger.info(f"\nWorker {worker_id} (skill={pool.workers[worker_id].skill_level:.2f}):")
        logger.info(f"Task: {subtask['description']}")
        
        output = pool.assign_task(worker_id, subtask)
        logger.info(f"Output: {output[:200]}...\n")
    
    pool.cleanup()


def test_failure_injection():
    """Test failure injection."""
    logger.info("=" * 60)
    logger.info("TEST 3: Failure Injection")
    logger.info("=" * 60)
    
    config = LlamaWorkerConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        max_tokens=256,
        temperature=0.7,
    )
    
    pool = WorkerPool(num_workers=1, config=config)
    
    # Low skill worker - more failures
    pool.workers[0].skill_level = 0.3
    
    subtask = {
        "description": "Implement a binary search algorithm",
        "context": "Should handle edge cases",
    }
    
    logger.info(f"Testing low-skill worker (skill=0.3)...")
    logger.info(f"Running 5 attempts to see failure injection:\n")
    
    for attempt in range(5):
        output = pool.assign_task(0, subtask)
        logger.info(f"Attempt {attempt + 1}: {output[:150]}...\n")
    
    pool.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Llama 2 workers")
    parser.add_argument(
        "--test",
        choices=["single", "pool", "failures", "all"],
        default="all",
        help="Which test to run",
    )
    
    args = parser.parse_args()
    
    try:
        if args.test in ["single", "all"]:
            test_single_worker()
        
        if args.test in ["pool", "all"]:
            test_worker_pool()
        
        if args.test in ["failures", "all"]:
            test_failure_injection()
        
        logger.info("\n✓ All tests completed successfully!")
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        sys.exit(1)
