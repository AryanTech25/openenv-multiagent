# Project Setup Guide

## Environment Setup

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install openenv-core gymnasium stable-baselines3 torch numpy pandas matplotlib wandb tensorboard fastapi uvicorn websockets pydantic requests huggingface-hub python-dotenv
```

### 3. Verify Installation
```bash
python3 test_environment.py
```

## Project Structure

```
manager-worker-env/
├── env/
│   ├── __init__.py
│   ├── manager_worker_env.py      # Core environment class
│   ├── task_library.py             # 15+ task templates
│   ├── hallucination_engine.py     # Failure mode injection
│   └── reward_calculator.py        # 5-component reward
├── agents/                         # (To be implemented)
├── training/                       # (To be implemented)
├── backend/                        # (To be implemented)
├── frontend/                       # (To be implemented)
├── test_environment.py             # Quick test script
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
└── SETUP.md                        # This file
```

## Environment Features

### ManagerWorkerEnv
- **Inherits from**: `gym.Env` (compatible with stable-baselines3)
- **Observation Space**: Dict with 5 keys
  - `task_embedding`: 64-dim vector [-1, 1]
  - `worker_states`: 4x5 array (4 workers, 5 features each)
  - `subtask_status`: Binary array of length 4
  - `budget_remaining`: Float [0, 1]
  - `steps_remaining`: Float [0, 1]
- **Action Space**: Discrete(7)
  - 0: assign_subtask (10 tokens)
  - 1: check_worker_output (50 tokens)
  - 2: correct_worker (30 tokens)
  - 3: reassign_task (40 tokens)
  - 4: fire_and_replace (100 tokens)
  - 5: approve_output (5 tokens)
  - 6: request_clarification (20 tokens)

### Task Library
- 15 diverse task templates across 5 domains:
  - Web Development (3 tasks)
  - Research (4 tasks)
  - Software Engineering (4 tasks)
  - Product Management (2 tasks)
  - Academic Writing (2 tasks)
- Each task has 2-5 subtasks and difficulty 1-5

### Failure Modes
- **Hallucination**: High plausibility (0.8-0.95), low quality (0.1-0.3)
- **Off-task**: Medium plausibility (0.4-0.6), near-zero quality (0.0-0.1)
- **Incomplete**: Moderate plausibility (0.5-0.6), partial quality (0.2-0.5)
- **Stuck**: Looping behavior, detected via patience counter

### Reward Function (5 components)
1. **Quality Reward**: 50.0 × final_quality
2. **Time Efficiency**: 10.0 × (1 - steps_used/max_steps)
3. **Budget Efficiency**: -5.0 if tokens_used/budget > 80%
4. **Hallucination Detection**: +15.0 per intervention, -20.0 per approval, -3.0 per false positive
5. **Step Cost**: -0.5 per step

## Quick Start

### Test the Environment
```bash
source venv/bin/activate
python3 test_environment.py
```

### Use in Your Code
```python
from env import ManagerWorkerEnv

config = {
    'max_workers': 4,
    'max_steps': 50,
    'token_budget': 1000,
    'task_difficulty': 3,
    'failure_injection_rate': 0.6,
}

env = ManagerWorkerEnv(config)
obs = env.reset()

for step in range(50):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## Next Steps

1. **Part 1.3-1.7**: Implement remaining environment methods
2. **Part 2**: Create training pipeline with PPO
3. **Part 3**: Build FastAPI backend
4. **Part 4**: Create React dashboard
5. **Part 5**: Deploy to HuggingFace Spaces

## Notes

- Training will be on HuggingFace (Colab for development, HF Spaces for deployment)
- All code follows the spec documents in `.kiro/specs/manager-worker-rl-env/`
- Property-based testing with hypothesis for correctness validation
- Use `source venv/bin/activate` to activate the virtual environment before running any commands
