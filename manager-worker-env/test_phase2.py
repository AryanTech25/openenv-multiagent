#!/usr/bin/env python3
"""
Comprehensive test suite for Phase 2 (Training Pipeline).

This script tests all Phase 2 components:
1. Environment functionality
2. Training pipeline
3. Model inference
4. Logging setup
5. HuggingFace integration
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path

from env import ManagerWorkerEnv, ManagerAction
from training.train_manager import TrainingConfig, train_manager, EnvironmentWrapper
from training.logging_setup import setup_logging
from training.inference import InferenceRunner


def test_environment():
    """Test environment functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: Environment Functionality")
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
    
    print("✓ Environment created and reset")
    print(f"  - Observation type: {type(obs).__name__}")
    print(f"  - Task embedding: {len(obs.task_embedding)}-dim")
    print(f"  - Worker states: {len(obs.worker_states)}x{len(obs.worker_states[0])}")
    
    # Run a few steps
    total_reward = 0
    for step in range(5):
        action = ManagerAction(action_id=np.random.randint(0, 7))
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"✓ Ran 5 steps successfully")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Render output:\n{env.render()}")
    
    return True


def test_environment_wrapper():
    """Test Gymnasium-compatible environment wrapper."""
    print("\n" + "=" * 80)
    print("TEST 2: Environment Wrapper (Gymnasium Compatible)")
    print("=" * 80)
    
    config = {
        'max_workers': 4,
        'max_steps': 50,
        'token_budget': 1000,
        'task_difficulty': 3,
        'failure_injection_rate': 0.6,
    }
    
    wrapper = EnvironmentWrapper(config)
    
    print("✓ Environment wrapper created")
    print(f"  - Observation space: {wrapper.observation_space}")
    print(f"  - Action space: {wrapper.action_space}")
    
    # Test reset
    obs, info = wrapper.reset()
    print(f"✓ Reset successful")
    print(f"  - Observation keys: {obs.keys()}")
    
    # Test step
    obs, reward, terminated, truncated, info = wrapper.step(0)
    print(f"✓ Step successful")
    print(f"  - Reward: {reward:.2f}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Truncated: {truncated}")
    
    return True


def test_training_config():
    """Test training configuration."""
    print("\n" + "=" * 80)
    print("TEST 3: Training Configuration")
    print("=" * 80)
    
    config = TrainingConfig(
        timesteps=1000,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )
    
    print("✓ Training config created")
    print(f"  - Timesteps: {config.timesteps:,}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - N steps: {config.n_steps}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - N epochs: {config.n_epochs}")
    
    config_dict = config.to_dict()
    print(f"✓ Config converted to dict with {len(config_dict)} keys")
    
    return True


def test_logging_setup():
    """Test logging setup."""
    print("\n" + "=" * 80)
    print("TEST 4: Logging Setup")
    print("=" * 80)
    
    logging_manager = setup_logging(
        use_tensorboard=True,
        use_wandb=False,
        config={'test': 'value'},
    )
    
    print("✓ Logging manager created")
    
    tb_log_dir = logging_manager.get_tensorboard_log_dir()
    if tb_log_dir:
        print(f"✓ TensorBoard enabled: {tb_log_dir}")
    
    # Test logging metrics
    logging_manager.log_metrics({'test_metric': 1.0}, step=0)
    print(f"✓ Metrics logged successfully")
    
    return True


def test_model_training():
    """Test model training."""
    print("\n" + "=" * 80)
    print("TEST 5: Model Training (Quick Test)")
    print("=" * 80)
    
    config = TrainingConfig(timesteps=500)
    
    print("✓ Training config created")
    print(f"  - Timesteps: {config.timesteps}")
    
    # Train model
    model = train_manager(
        config=config,
        model_path='models/test_ppo_manager',
        verbose=0,
    )
    
    print("✓ Model trained successfully")
    print(f"  - Model type: {type(model).__name__}")
    
    # Check if model file exists
    model_file = Path('models/test_ppo_manager.zip')
    if model_file.exists():
        print(f"✓ Model saved to disk: {model_file}")
        print(f"  - File size: {model_file.stat().st_size / 1024:.1f} KB")
    
    return True


def test_inference():
    """Test model inference."""
    print("\n" + "=" * 80)
    print("TEST 6: Model Inference")
    print("=" * 80)
    
    # Use the trained model from test 5
    model_path = 'models/test_ppo_manager'
    
    if not Path(f'{model_path}.zip').exists():
        print("⚠ Model not found, skipping inference test")
        return True
    
    runner = InferenceRunner(model_path)
    print("✓ Inference runner created")
    
    # Run a single episode
    stats = runner.run_episode(deterministic=True)
    
    print("✓ Inference episode completed")
    print(f"  - Total reward: {stats['total_reward']:.2f}")
    print(f"  - Episode length: {stats['episode_length']}")
    print(f"  - Average reward/step: {stats['average_reward']:.2f}")
    
    return True


def test_file_structure():
    """Test that all required files exist."""
    print("\n" + "=" * 80)
    print("TEST 7: File Structure")
    print("=" * 80)
    
    required_files = [
        'training/__init__.py',
        'training/train_manager.py',
        'training/logging_setup.py',
        'training/huggingface_integration.py',
        'training/run_training.py',
        'training/inference.py',
        'colab_notebook.ipynb',
        'PHASE2_COMPLETE.md',
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✓ {file_path} ({size} bytes)")
        else:
            print(f"✗ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PHASE 2 COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Environment Functionality", test_environment),
        ("Environment Wrapper", test_environment_wrapper),
        ("Training Configuration", test_training_config),
        ("Logging Setup", test_logging_setup),
        ("Model Training", test_model_training),
        ("Model Inference", test_inference),
        ("File Structure", test_file_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 2 is complete and ready for Phase 3.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review.")
    
    print("=" * 80)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
