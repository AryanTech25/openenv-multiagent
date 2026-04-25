"""
Test script for generic Hugging Face worker agents.
Example usage with different models.
"""

import sys
import os
from dotenv import load_dotenv

# Load .env file for HF_TOKEN if available
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.worker_pool import WorkerPool
from agents.llm_worker import HFWorkerConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hf_model(model_id: str, hf_token: str = None, task: str = None, context: str = None):
    """Test a specific Hugging Face model with a rich dashboard."""
    # Defaults
    test_task = task or "Explain the concept of multi-agent reinforcement learning in one sentence."
    test_context = context or "Keep it concise and technical."
    
    config = HFWorkerConfig(
        model_id=model_id,
        hf_token=hf_token,
        max_tokens=128,
        temperature=0.7,
        worker_type=args.api_mode,
    )
    
    try:
        pool = WorkerPool(num_workers=1, config=config)
        worker = pool.workers[0]
        
        # Calculate visual skill bar
        skill_pct = int(worker.skill_level * 100)
        bar_len = 20
        filled = int(bar_len * worker.skill_level)
        skill_bar = "█" * filled + "░" * (bar_len - filled)
        
        # Color based on skill
        color_code = "\033[92m" if skill_pct > 75 else ("\033[93m" if skill_pct > 40 else "\033[91m")
        reset = "\033[0m"

        print("\n" + "="*80)
        print(f"🤖  {color_code}ORCHESTRA-AI WORKER DASHBOARD{reset}")
        print("="*80)
        print(f"📄 MODEL:   {model_id}")
        print(f"🛠️  MODE:    {config.worker_type.upper()}")
        print(f"🎯 TASK:    {test_task}")
        print(f"🔍 CONTEXT: {test_context}")
        print("-" * 80)
        print(f"⭐ SKILL LEVEL: {color_code}{skill_pct}%{reset}  | {color_code}{skill_bar}{reset} |")
        print("-" * 80)
        
        subtask = {
            "description": test_task,
            "context": test_context,
        }
        
        import time
        start_time = time.time()
        output = pool.assign_task(0, subtask)
        elapsed = time.time() - start_time
        
        print(f"💬 GENERATED OUTPUT ({elapsed:.2f}s):")
        print(f"\n{output}\n")
        print("="*80 + "\n")
        
        pool.cleanup()
    except Exception as e:
        logger.error(f"Failed to test model {model_id}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hugging Face workers")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2", # Using phi-2 as a default lightweight model
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token (optional)",
    )
    parser.add_argument(
        "--api-mode",
        type=str,
        choices=["local", "api"],
        default="local",
        help="Whether to run model locally or via API",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Custom task description to test",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Custom context/instructions for the task",
    )
    
    args = parser.parse_args()
    
    # Example models you can try:
    # 1. microsoft/phi-2 (Lightweight, no token needed)
    # 2. mistralai/Mistral-7B-v0.1 (Needs token & gating approval)
    # 3. meta-llama/Llama-3-8B (Needs token & gating approval)
    
    test_hf_model(args.model, args.token, args.task, args.context)
