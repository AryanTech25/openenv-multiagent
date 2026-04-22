#!/usr/bin/env python3
"""
HuggingFace Hub integration for saving and loading trained models.

This module provides utilities for pushing trained models to HuggingFace Hub
with metadata, model cards, and training information.
"""

import os
import json
from typing import Optional, Dict, Any
from datetime import datetime


class HuggingFaceModelCard:
    """Generate model card for HuggingFace Hub."""
    
    def __init__(
        self,
        model_name: str,
        task: str = 'reinforcement-learning',
        library_name: str = 'stable-baselines3',
        tags: Optional[list] = None,
    ):
        """Initialize model card."""
        self.model_name = model_name
        self.task = task
        self.library_name = library_name
        self.tags = tags or []
    
    def generate(
        self,
        training_timesteps: int,
        mean_reward: float,
        mean_episode_length: float,
        hyperparameters: Dict[str, Any],
    ) -> str:
        """
        Generate model card markdown.
        
        Args:
            training_timesteps: Total timesteps trained
            mean_reward: Mean reward achieved
            mean_episode_length: Mean episode length
            hyperparameters: Training hyperparameters
        
        Returns:
            Model card markdown string
        """
        tags_str = ', '.join(self.tags) if self.tags else 'reinforcement-learning'
        
        hyperparams_str = '\n'.join(
            f'- {k}: {v}' for k, v in hyperparameters.items()
        )
        
        card = f"""---
library_name: {self.library_name}
tags:
- {tags_str}
model-index:
- name: {self.model_name}
  results:
  - task:
      name: Reinforcement Learning
      type: reinforcement-learning
    dataset:
      name: ManagerWorkerEnv
      type: ManagerWorkerEnv
    metrics:
    - type: mean_reward
      value: {mean_reward:.2f}
    - type: mean_episode_length
      value: {mean_episode_length:.2f}
---

# {self.model_name}

This is a PPO agent trained on the ManagerWorkerEnv environment.

## Model Details

- **Model Type**: PPO (Proximal Policy Optimization)
- **Framework**: Stable-Baselines3
- **Environment**: ManagerWorkerEnv (OpenEnv-based)
- **Training Timesteps**: {training_timesteps:,}
- **Mean Reward**: {mean_reward:.2f}
- **Mean Episode Length**: {mean_episode_length:.2f}

## Hyperparameters

{hyperparams_str}

## Usage

```python
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download

# Download model from HuggingFace Hub
model_path = hf_hub_download(
    repo_id="your-username/{self.model_name}",
    filename="model.zip"
)

# Load model
model = PPO.load(model_path)

# Use model for inference
obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
```

## Training Details

This model was trained using PPO on the ManagerWorkerEnv, a multi-agent
reinforcement learning environment where a Manager agent coordinates multiple
Worker agents to complete complex tasks under a limited token budget.

The environment includes realistic failure modes (hallucinations, off-task
behavior, incomplete work) that the Manager must learn to detect and correct.

## License

This model is released under the MIT License.
"""
        return card
    
    def save(self, path: str, content: str) -> None:
        """Save model card to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)


class HuggingFaceIntegration:
    """Integration with HuggingFace Hub."""
    
    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
    ):
        """
        Initialize HuggingFace integration.
        
        Args:
            repo_id: Repository ID (format: username/repo-name)
            token: HuggingFace API token
        """
        self.repo_id = repo_id
        self.token = token
    
    def push_model(
        self,
        model_path: str,
        model_name: str = 'ppo_manager',
        training_timesteps: int = 50000,
        mean_reward: float = 0.0,
        mean_episode_length: float = 0.0,
        hyperparameters: Optional[Dict[str, Any]] = None,
        commit_message: str = 'Add trained PPO model',
    ) -> str:
        """
        Push model to HuggingFace Hub.
        
        Args:
            model_path: Path to saved model
            model_name: Name for the model
            training_timesteps: Total timesteps trained
            mean_reward: Mean reward achieved
            mean_episode_length: Mean episode length
            hyperparameters: Training hyperparameters
            commit_message: Git commit message
        
        Returns:
            URL to the model on HuggingFace Hub
        """
        try:
            from huggingface_hub import HfApi, create_repo
            
            print(f"Pushing model to HuggingFace Hub: {self.repo_id}")
            
            # Create repo if it doesn't exist
            try:
                create_repo(
                    repo_id=self.repo_id,
                    repo_type='model',
                    exist_ok=True,
                    private=False,
                )
                print(f"✓ Repository created/verified: {self.repo_id}")
            except Exception as e:
                print(f"⚠ Repository creation warning: {e}")
            
            # Generate model card
            card_generator = HuggingFaceModelCard(
                model_name=model_name,
                tags=['reinforcement-learning', 'ppo', 'multi-agent'],
            )
            
            hyperparameters = hyperparameters or {}
            model_card = card_generator.generate(
                training_timesteps=training_timesteps,
                mean_reward=mean_reward,
                mean_episode_length=mean_episode_length,
                hyperparameters=hyperparameters,
            )
            
            # Save model card
            card_path = os.path.join(
                os.path.dirname(model_path),
                'README.md'
            )
            card_generator.save(card_path, model_card)
            print(f"✓ Model card generated: {card_path}")
            
            # Create metadata file
            metadata = {
                'model_name': model_name,
                'training_timesteps': training_timesteps,
                'mean_reward': mean_reward,
                'mean_episode_length': mean_episode_length,
                'hyperparameters': hyperparameters,
                'timestamp': datetime.now().isoformat(),
            }
            
            metadata_path = os.path.join(
                os.path.dirname(model_path),
                'metadata.json'
            )
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Metadata saved: {metadata_path}")
            
            # Upload to HuggingFace Hub
            api = HfApi()
            
            # Upload model file
            api.upload_file(
                path_or_fileobj=model_path + '.zip',
                path_in_repo='model.zip',
                repo_id=self.repo_id,
                repo_type='model',
                commit_message=commit_message,
                token=self.token,
            )
            print(f"✓ Model uploaded: {model_path}.zip")
            
            # Upload model card
            api.upload_file(
                path_or_fileobj=card_path,
                path_in_repo='README.md',
                repo_id=self.repo_id,
                repo_type='model',
                commit_message=commit_message,
                token=self.token,
            )
            print(f"✓ Model card uploaded")
            
            # Upload metadata
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo='metadata.json',
                repo_id=self.repo_id,
                repo_type='model',
                commit_message=commit_message,
                token=self.token,
            )
            print(f"✓ Metadata uploaded")
            
            url = f"https://huggingface.co/{self.repo_id}"
            print(f"\n✓ Model successfully pushed to: {url}")
            return url
        
        except ImportError:
            print("⚠ huggingface-hub not installed. Install with: pip install huggingface-hub")
            return ""
        except Exception as e:
            print(f"✗ Failed to push model to HuggingFace Hub: {e}")
            return ""
    
    @staticmethod
    def load_model(repo_id: str, model_name: str = 'model.zip'):
        """
        Load model from HuggingFace Hub.
        
        Args:
            repo_id: Repository ID
            model_name: Model filename
        
        Returns:
            Path to downloaded model
        """
        try:
            from huggingface_hub import hf_hub_download
            from stable_baselines3 import PPO
            
            print(f"Loading model from HuggingFace Hub: {repo_id}")
            
            # Download model
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_name,
            )
            print(f"✓ Model downloaded: {model_path}")
            
            # Load model
            model = PPO.load(model_path)
            print(f"✓ Model loaded successfully")
            
            return model
        
        except ImportError:
            print("⚠ huggingface-hub not installed. Install with: pip install huggingface-hub")
            return None
        except Exception as e:
            print(f"✗ Failed to load model from HuggingFace Hub: {e}")
            return None
