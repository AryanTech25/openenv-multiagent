#!/usr/bin/env python3
"""
Quick test script to verify the environment works correctly.
"""

import sys
sys.path.insert(0, '.')

from env import ManagerWorkerEnv
import numpy as np

def test_environment():
    """Test basic environment functionality."""
    print("=" * 80)
    print("Testing ManagerWorkerEnv")
    print("=" * 80)
    
    # Create environment
    config = {
        'max_workers': 4,
        'max_steps': 50,
        'token_budget': 1000,
        'task_difficulty': 3,
        'failure_injection_rate': 0.6,
    }
    
    env = ManagerWorkerEnv(config)
    print(f"✓ Environment created with config: {config}")
    
    # Test reset
    obs = env.reset()
    print(f"✓ Environment reset")
    print(f"  - Observation keys: {obs.keys()}")
    print(f"  - Task embedding shape: {obs['task_embedding'].shape}")
    print(f"  - Worker states shape: {obs['worker_states'].shape}")
    print(f"  - Subtask status shape: {obs['subtask_status'].shape}")
    print(f"  - Budget remaining: {obs['budget_remaining'][0]:.2f}")
    print(f"  - Steps remaining: {obs['steps_remaining'][0]:.2f}")
    
    # Test a few steps
    print(f"\n✓ Running 10 random steps...")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        action_name = env.ACTION_NAMES[action]
        print(f"  Step {step+1}: action={action_name}, reward={reward:.2f}, done={done}")
        
        if done:
            print(f"  Episode ended at step {step+1}")
            break
    
    print(f"\n✓ Total reward: {total_reward:.2f}")
    
    # Test render
    print(f"\n✓ Environment render:")
    print(env.render())
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)

if __name__ == '__main__':
    test_environment()
