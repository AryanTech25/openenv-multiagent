#!/usr/bin/env python3
"""
Example usage of ManagerWorkerEnv with stable-baselines3 PPO.

This demonstrates how to:
1. Create the environment
2. Wrap it for stable-baselines3
3. Train a PPO agent
4. Run inference
"""

import sys
sys.path.insert(0, '.')

from env import ManagerWorkerEnv, ManagerAction
import numpy as np


def example_basic_usage():
    """Basic usage example."""
    print("=" * 80)
    print("Example 1: Basic Environment Usage")
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
    print(f"✓ Environment created")
    
    # Reset and run episode
    obs = env.reset()
    print(f"✓ Episode started")
    print(f"  - Task: {env.state.task['task_type']}")
    print(f"  - Subtasks: {env.state.task['num_subtasks']}")
    print(f"  - Budget: {env.state.budget_remaining} tokens")
    
    # Run 20 steps with random actions
    total_reward = 0
    for step in range(20):
        action_id = np.random.randint(0, 7)
        action = ManagerAction(action_id=action_id)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: Reward={reward:.2f}, Budget={env.state.budget_remaining}, Done={done}")
        
        if done:
            print(f"  Episode ended at step {step+1}")
            break
    
    print(f"✓ Episode complete: Total Reward={total_reward:.2f}")
    print()


def example_custom_policy():
    """Example with a simple custom policy."""
    print("=" * 80)
    print("Example 2: Custom Policy (Greedy)")
    print("=" * 80)
    
    config = {
        'max_workers': 4,
        'max_steps': 50,
        'token_budget': 1000,
        'task_difficulty': 2,
        'failure_injection_rate': 0.5,
    }
    
    env = ManagerWorkerEnv(config)
    obs = env.reset()
    
    print(f"✓ Running greedy policy (assign → check → correct → approve)")
    
    action_sequence = [0, 1, 2, 5]  # assign, check, correct, approve
    action_idx = 0
    total_reward = 0
    
    for step in range(50):
        # Cycle through action sequence
        action_id = action_sequence[action_idx % len(action_sequence)]
        action = ManagerAction(action_id=action_id)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        action_idx += 1
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}: Action={env.ACTION_NAMES[action_id]}, Reward={reward:.2f}, Done={done}")
        
        if done:
            print(f"  Episode ended at step {step+1}")
            break
    
    print(f"✓ Episode complete: Total Reward={total_reward:.2f}")
    print()


def example_observation_structure():
    """Example showing observation structure."""
    print("=" * 80)
    print("Example 3: Observation Structure")
    print("=" * 80)
    
    config = {
        'max_workers': 4,
        'max_steps': 50,
        'token_budget': 1000,
        'task_difficulty': 3,
        'failure_injection_rate': 0.6,
    }
    
    env = ManagerWorkerEnv(config)
    obs = env.reset()
    
    print(f"✓ Observation structure:")
    print(f"  - task_embedding: {len(obs.task_embedding)}-dim vector")
    print(f"  - worker_states: {len(obs.worker_states)}x{len(obs.worker_states[0])} array")
    print(f"    - Columns: [is_active, progress, hallucination_risk, output_quality, tokens_consumed_ratio]")
    print(f"  - subtask_status: {obs.subtask_status}")
    print(f"  - budget_remaining: {obs.budget_remaining:.2f}")
    print(f"  - steps_remaining: {obs.steps_remaining:.2f}")
    print(f"  - hallucination_catch_rate: {obs.hallucination_catch_rate:.2f}")
    
    print(f"\n✓ Worker states detail:")
    for i, worker_state in enumerate(obs.worker_states):
        print(f"  Worker {i}: active={worker_state[0]}, progress={worker_state[1]:.2f}, "
              f"halluc_risk={worker_state[2]:.2f}, quality={worker_state[3]:.2f}, "
              f"tokens_ratio={worker_state[4]:.2f}")
    
    print()


def example_reward_breakdown():
    """Example showing reward breakdown."""
    print("=" * 80)
    print("Example 4: Reward Breakdown")
    print("=" * 80)
    
    config = {
        'max_workers': 4,
        'max_steps': 50,
        'token_budget': 1000,
        'task_difficulty': 2,
        'failure_injection_rate': 0.4,
    }
    
    env = ManagerWorkerEnv(config)
    obs = env.reset()
    
    print(f"✓ Running episode and tracking reward components...")
    
    for step in range(30):
        action_id = np.random.randint(0, 7)
        action = ManagerAction(action_id=action_id)
        obs, reward, done, info = env.step(action)
        
        if done:
            # Get reward breakdown
            breakdown = env.reward_calculator.get_reward_breakdown(
                final_quality=env._compute_final_quality(),
                steps_used=env.state.step_counter,
                max_steps=env.max_steps,
                tokens_used=env.token_budget - env.state.budget_remaining,
                token_budget=env.token_budget,
                hallucination_interventions=env.state.hallucinations_detected,
                hallucination_approvals=env.state.hallucinations_approved,
                false_positives=env.state.false_positives,
            )
            
            print(f"  Episode ended at step {step+1}")
            print(f"\n  Reward Breakdown:")
            print(f"    - Quality Reward: {breakdown['quality_reward']:.2f}")
            print(f"    - Time Reward: {breakdown['time_reward']:.2f}")
            print(f"    - Budget Reward: {breakdown['budget_reward']:.2f}")
            print(f"    - Hallucination Reward: {breakdown['hallucination_reward']:.2f}")
            print(f"    - Step Cost: {breakdown['step_cost']:.2f}")
            print(f"    - Total Reward: {breakdown['total_reward']:.2f}")
            print(f"\n  Metrics:")
            print(f"    - Budget Ratio: {breakdown['budget_ratio']:.2f}")
            print(f"    - Time Efficiency: {breakdown['time_efficiency']:.2f}")
            break
    
    print()


if __name__ == '__main__':
    example_basic_usage()
    example_custom_policy()
    example_observation_structure()
    example_reward_breakdown()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
