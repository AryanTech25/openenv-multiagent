# OpenEnv Implementation Guide

## Overview

The ManagerWorkerEnv is fully implemented using the **OpenEnv framework**, which provides a standardized interface for building RL environments that work seamlessly with LLM agents and modern training pipelines.

## Architecture

### Core Classes

#### 1. **ManagerWorkerEnv** (inherits from `openenv.core.Environment`)
The main environment class that orchestrates the multi-agent coordination simulation.

```python
from env import ManagerWorkerEnv, ManagerAction

# Create environment
env = ManagerWorkerEnv(config={
    'max_workers': 4,
    'max_steps': 50,
    'token_budget': 1000,
    'task_difficulty': 3,
    'failure_injection_rate': 0.6,
})

# Reset and get initial observation
obs = env.reset()

# Take a step
action = ManagerAction(action_id=0)  # assign_subtask
obs, reward, done, info = env.step(action)
```

#### 2. **ManagerWorkerObservation** (inherits from `openenv.core.Observation`)
Pydantic-based observation model with automatic validation and serialization.

```python
class ManagerWorkerObservation(Observation):
    task_embedding: List[float]           # 64-dim task vector
    worker_states: List[List[float]]      # 4x5 worker state matrix
    subtask_status: List[int]             # Binary completion status
    budget_remaining: float               # Normalized budget [0, 1]
    steps_remaining: float                # Normalized steps [0, 1]
    episode_log: List[Dict]               # Episode history
    hallucination_catch_rate: float       # Detection rate metric
```

#### 3. **ManagerAction** (inherits from `openenv.core.Action`)
Pydantic-based action model with validation.

```python
class ManagerAction(Action):
    action_id: int                        # 0-6 discrete action
    target_worker_id: Optional[int]       # Optional target worker
```

#### 4. **ManagerWorkerState** (inherits from `openenv.core.State`)
Internal environment state representation.

```python
class ManagerWorkerState(State):
    workers: List[WorkerStateModel]       # Worker states
    task: Optional[Dict]                  # Current task
    budget_remaining: int                 # Token budget
    step_counter: int                     # Episode step count
    # ... other state fields
```

## Key Features

### 1. **Pydantic Integration**
All observations and actions use Pydantic models for:
- Automatic validation
- Type checking
- JSON serialization/deserialization
- API documentation

### 2. **OpenEnv Compliance**
- Inherits from `openenv.core.Environment`
- Implements required abstract methods
- Compatible with OpenEnv's training infrastructure
- Supports MCP (Model Context Protocol) integration

### 3. **Observation Space**
```
task_embedding (64-dim)
├─ Encodes task type and complexity
└─ Normalized to [-1, 1]

worker_states (4x5 matrix)
├─ is_active: Worker active status
├─ progress: Task progress [0, 1]
├─ hallucination_risk: Hidden risk score
├─ output_quality: Revealed quality if checked
└─ tokens_consumed_ratio: Budget usage

subtask_status (4-element binary)
├─ 1 if subtask complete
└─ 0 if in progress

budget_remaining [0, 1]
└─ Normalized token budget

steps_remaining [0, 1]
└─ Normalized step count
```

### 4. **Action Space**
```
0: assign_subtask (10 tokens)
   └─ Assign next unassigned subtask to idle worker

1: check_worker_output (50 tokens)
   └─ Inspect worker output quality

2: correct_worker (30 tokens)
   └─ Send corrective instructions

3: reassign_task (40 tokens)
   └─ Reassign subtask to different worker

4: fire_and_replace (100 tokens)
   └─ Replace worker with new one

5: approve_output (5 tokens)
   └─ Accept worker output and mark complete

6: request_clarification (20 tokens)
   └─ Request more task information
```

### 5. **Reward Function** (5 components)
```
Total Reward = Quality + Time + Budget + Hallucination + StepCost

Quality (0-50):
  50.0 × final_quality

Time Efficiency (0-10):
  10.0 × (1 - steps_used/max_steps)

Budget Efficiency (penalty):
  -5.0 if tokens_used/budget > 80%

Hallucination Detection:
  +15.0 per correct intervention
  -20.0 per hallucination approval
  -3.0 per false positive

Step Cost:
  -0.5 per step
```

## Usage Examples

### Basic Usage
```python
from env import ManagerWorkerEnv, ManagerAction
import numpy as np

# Create environment
env = ManagerWorkerEnv()

# Reset
obs = env.reset()

# Run episode
for step in range(50):
    action = ManagerAction(action_id=np.random.randint(0, 7))
    obs, reward, done, info = env.step(action)
    
    if done:
        break
```

### With Custom Policy
```python
# Define action sequence
actions = [0, 1, 2, 5]  # assign, check, correct, approve

for step in range(50):
    action_id = actions[step % len(actions)]
    action = ManagerAction(action_id=action_id)
    obs, reward, done, info = env.step(action)
```

### Accessing State
```python
# Current state
state = env.state

# Worker information
for worker in state.workers:
    print(f"Worker {worker.worker_id}: skill={worker.skill_level:.2f}")

# Task information
print(f"Task: {state.task['task_type']}")
print(f"Subtasks: {state.task['num_subtasks']}")

# Budget tracking
print(f"Budget remaining: {state.budget_remaining}")
print(f"Tokens used: {env.token_budget - state.budget_remaining}")
```

### Reward Breakdown
```python
# Get detailed reward components
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

print(f"Quality Reward: {breakdown['quality_reward']:.2f}")
print(f"Time Reward: {breakdown['time_reward']:.2f}")
print(f"Total Reward: {breakdown['total_reward']:.2f}")
```

## Integration with Training

### With Stable-Baselines3
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = ManagerWorkerEnv()

# Wrap for stable-baselines3
vec_env = DummyVecEnv([lambda: env])

# Create PPO agent
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train
model.learn(total_timesteps=100000)
```

### With OpenEnv Server
```python
from openenv.core import HTTPEnvServer

# Create environment
env = ManagerWorkerEnv()

# Start HTTP server
server = HTTPEnvServer(env, host="0.0.0.0", port=8000)
server.run()
```

## Testing

### Run Tests
```bash
# Basic environment test
python3 test_environment.py

# Example usage
python3 example_usage.py
```

### Test Coverage
- ✅ Environment initialization
- ✅ Reset functionality
- ✅ Step execution
- ✅ Observation generation
- ✅ Reward calculation
- ✅ Episode termination
- ✅ Rendering

## File Structure

```
env/
├── __init__.py                    # Exports all classes
├── manager_worker_env.py          # Main environment (OpenEnv-based)
├── task_library.py                # 15+ task templates
├── hallucination_engine.py        # Failure mode injection
└── reward_calculator.py           # 5-component reward

test_environment.py                # Basic tests
example_usage.py                   # Usage examples
OPENENV_IMPLEMENTATION.md          # This file
```

## Next Steps

1. **Training Pipeline** (Part 2)
   - Create `training/train_manager.py` with PPO
   - Set up TensorBoard and WandB logging
   - Implement model checkpointing

2. **Backend API** (Part 3)
   - Create FastAPI server
   - Implement REST endpoints
   - Add WebSocket support

3. **React Dashboard** (Part 4)
   - Create visualization components
   - Connect to backend via WebSocket
   - Real-time monitoring

4. **Deployment** (Part 5)
   - Deploy to HuggingFace Spaces
   - Create Colab notebook
   - Write documentation

## References

- [OpenEnv Documentation](https://github.com/openenv-ai/openenv)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
