#!/usr/bin/env python3
"""
Complete training pipeline with logging and model saving.

This script runs the full training pipeline:
1. Create environment
2. Train PPO agent
3. Log metrics to TensorBoard and WandB
4. Save model to HuggingFace Hub

Usage:
    python training/run_training.py --timesteps 50000 --use-wandb
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Optional, Dict, Any

from training.train_manager import TrainingConfig, train_manager
from training.logging_setup import setup_logging
from training.huggingface_integration import HuggingFaceIntegration


def run_training_pipeline(
    timesteps: int = 50000,
    learning_rate: float = 3e-4,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: str = 'manager-worker-rl',
    wandb_entity: Optional[str] = None,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    model_path: str = 'models/ppo_manager',
    verbose: int = 1,
) -> None:
    """
    Run complete training pipeline.
    
    Args:
        timesteps: Number of training timesteps
        learning_rate: Learning rate for PPO
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable WandB logging
        wandb_project: WandB project name
        wandb_entity: WandB entity (username/team)
        push_to_hub: Push model to HuggingFace Hub
        hub_repo_id: HuggingFace Hub repository ID
        model_path: Path to save model
        verbose: Verbosity level
    """
    print("=" * 80)
    print("Manager-Worker RL Environment - Training Pipeline")
    print("=" * 80)
    
    # Setup logging
    print("\n[1/4] Setting up logging...")
    config = {
        'timesteps': timesteps,
        'learning_rate': learning_rate,
    }
    
    logging_manager = setup_logging(
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        config=config,
    )
    
    tensorboard_log_dir = logging_manager.get_tensorboard_log_dir()
    if tensorboard_log_dir:
        print(f"✓ TensorBoard logging enabled: {tensorboard_log_dir}")
    
    # Create training config
    print("\n[2/4] Creating training configuration...")
    training_config = TrainingConfig(
        timesteps=timesteps,
        learning_rate=learning_rate,
    )
    print(f"✓ Training config created")
    print(f"  - Timesteps: {timesteps:,}")
    print(f"  - Learning rate: {learning_rate}")
    
    # Train model
    print("\n[3/4] Training PPO agent...")
    print("-" * 80)
    
    model = train_manager(
        config=training_config,
        model_path=model_path,
        verbose=verbose,
    )
    
    print("-" * 80)
    print(f"✓ Training completed")
    
    # Push to HuggingFace Hub (optional)
    if push_to_hub and hub_repo_id:
        print("\n[4/4] Pushing model to HuggingFace Hub...")
        
        hf_integration = HuggingFaceIntegration(repo_id=hub_repo_id)
        
        hf_integration.push_model(
            model_path=model_path,
            training_timesteps=timesteps,
            hyperparameters=training_config.to_dict(),
        )
    else:
        print("\n[4/4] Skipping HuggingFace Hub push")
    
    # Finish logging
    logging_manager.finish()
    
    print("\n" + "=" * 80)
    print("Training Pipeline Complete!")
    print("=" * 80)
    print(f"\nModel saved to: {model_path}")
    if tensorboard_log_dir:
        print(f"TensorBoard logs: {tensorboard_log_dir}")
        print(f"View with: tensorboard --logdir {tensorboard_log_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run complete training pipeline for Manager-Worker RL Environment'
    )
    
    # Training arguments
    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Number of training timesteps (default: 50000)',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)',
    )
    
    # Logging arguments
    parser.add_argument(
        '--use-tensorboard',
        action='store_true',
        default=True,
        help='Enable TensorBoard logging (default: True)',
    )
    parser.add_argument(
        '--no-tensorboard',
        action='store_false',
        dest='use_tensorboard',
        help='Disable TensorBoard logging',
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable WandB logging',
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='manager-worker-rl',
        help='WandB project name (default: manager-worker-rl)',
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        help='WandB entity (username or team)',
    )
    
    # HuggingFace Hub arguments
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push model to HuggingFace Hub',
    )
    parser.add_argument(
        '--hub-repo-id',
        type=str,
        help='HuggingFace Hub repository ID (format: username/repo-name)',
    )
    
    # Model arguments
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/ppo_manager',
        help='Path to save model (default: models/ppo_manager)',
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (default: 1)',
    )
    
    args = parser.parse_args()
    
    # Run training pipeline
    run_training_pipeline(
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        model_path=args.model_path,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
