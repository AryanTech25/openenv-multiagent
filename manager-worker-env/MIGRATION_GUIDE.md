# OpenEnv Structure Migration Guide

## Overview

This document outlines the migration from the current codebase structure to OpenEnv-compliant structure.

## Current Structure
```
manager-worker-env/
├── env/
│   ├── __init__.py
│   ├── manager_worker_env.py
│   ├── task_library.py
│   ├── hallucination_engine.py
│   └── reward_calculator.py
├── training/
├── BE/
├── FE/
└── [other files]
```

## Target OpenEnv Structure
```
manager-worker-env/
├── src/
│   └── manager_worker_env/
│       ├── __init__.py
│       ├── environment.py (renamed from manager_worker_env.py)
│       ├── models.py (extracted models)
│       └── utils/
│           ├── __init__.py
│           ├── task_library.py
│           ├── hallucination_engine.py
│           └── reward_calculator.py
├── tests/
│   ├── __init__.py
│   ├── test_environment.py
│   └── test_integration.py
├── training/
│   ├── train_manager.py
│   ├── logging_setup.py
│   ├── huggingface_integration.py
│   ├── run_training.py
│   ├── inference.py
│   └── __init__.py
├── BE/
│   ├── config.py
│   ├── database.py
│   ├── episode_manager.py
│   ├── websocket_manager.py
│   ├── metrics_tracker.py
│   ├── models.py
│   ├── errors.py
│   ├── main.py
│   ├── __init__.py
│   ├── .env
│   └── README.md
├── pyproject.toml (NEW)
├── setup.py (NEW)
├── README.md
├── requirements.txt
└── Dockerfile (NEW)
```

## Migration Steps

### Step 1: Create src/ directory structure
```bash
mkdir -p src/manager_worker_env/utils
mkdir -p tests
```

### Step 2: Move and rename files
- `env/manager_worker_env.py` → `src/manager_worker_env/environment.py`
- `env/task_library.py` → `src/manager_worker_env/utils/task_library.py`
- `env/hallucination_engine.py` → `src/manager_worker_env/utils/hallucination_engine.py`
- `env/reward_calculator.py` → `src/manager_worker_env/utils/reward_calculator.py`

### Step 3: Extract models
- Create `src/manager_worker_env/models.py`
- Move Pydantic models from `environment.py` to `models.py`
- Update imports in `environment.py`

### Step 4: Update imports
All imports need to be updated from:
```python
from env.task_library import TaskLibrary
```

To:
```python
from manager_worker_env.utils.task_library import TaskLibrary
```

### Step 5: Create pyproject.toml
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "manager-worker-env"
version = "1.0.0"
description = "OpenEnv-compatible multi-agent RL environment"
requires-python = ">=3.10"
dependencies = [
    "openenv-core>=0.2.3",
    "gymnasium>=0.29.1",
    "pydantic>=2.0.0",
    "numpy>=1.18.0",
]

[project.optional-dependencies]
training = [
    "stable-baselines3>=2.0.0",
    "torch>=2.3.0",
    "wandb>=0.12.0",
    "tensorboard>=2.0.0",
]
backend = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "motor>=3.0.0",
    "pydantic-settings>=2.0.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
```

### Step 6: Create setup.py
```python
from setuptools import setup, find_packages

setup(
    name="manager-worker-env",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
```

### Step 7: Update all imports across codebase

**In training/train_manager.py:**
```python
# OLD
from env import ManagerWorkerEnv, ManagerAction

# NEW
from manager_worker_env import ManagerWorkerEnv, ManagerAction
```

**In training/inference.py:**
```python
# OLD
from env import ManagerWorkerEnv, ManagerAction

# NEW
from manager_worker_env import ManagerWorkerEnv, ManagerAction
```

**In BE/episode_manager.py:**
```python
# OLD
from env import ManagerWorkerEnv, ManagerAction

# NEW
from manager_worker_env import ManagerWorkerEnv, ManagerAction
```

### Step 8: Create __init__.py files

**src/manager_worker_env/__init__.py:**
```python
from .environment import ManagerWorkerEnv
from .models import (
    ManagerWorkerObservation,
    ManagerAction,
    ManagerWorkerState,
    WorkerStateModel,
)

__version__ = "1.0.0"
__all__ = [
    "ManagerWorkerEnv",
    "ManagerWorkerObservation",
    "ManagerAction",
    "ManagerWorkerState",
    "WorkerStateModel",
]
```

**src/manager_worker_env/utils/__init__.py:**
```python
from .task_library import TaskLibrary, Task, Subtask
from .hallucination_engine import HallucinationEngine, FailureMode
from .reward_calculator import RewardCalculator

__all__ = [
    "TaskLibrary",
    "Task",
    "Subtask",
    "HallucinationEngine",
    "FailureMode",
    "RewardCalculator",
]
```

### Step 9: Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "BE.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 10: Update requirements.txt
```
openenv-core>=0.2.3
gymnasium>=0.29.1
stable-baselines3>=2.0.0
torch>=2.3.0
numpy>=1.18.0
pandas>=1.0.0
matplotlib>=3.0.0
wandb>=0.12.0
tensorboard>=2.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
websockets>=15.0.1
pydantic>=2.0.0
pydantic-settings>=2.0.0
requests>=2.25.0
huggingface-hub>=0.20.0
motor>=3.0.0
python-dotenv>=1.1.0
```

## Installation

After migration:

```bash
# Install in development mode
pip install -e .

# Install with training dependencies
pip install -e ".[training]"

# Install with backend dependencies
pip install -e ".[backend]"

# Install all dependencies
pip install -e ".[training,backend,dev]"
```

## Usage

After migration, usage remains the same:

```python
from manager_worker_env import ManagerWorkerEnv, ManagerAction

env = ManagerWorkerEnv()
obs = env.reset()

for step in range(50):
    action = ManagerAction(action_id=0)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_environment.py

# Run with coverage
pytest --cov=src/manager_worker_env
```

## Key Changes

1. **Package Structure**: Now follows OpenEnv conventions with `src/` layout
2. **Imports**: All imports use `manager_worker_env` package name
3. **Installation**: Can be installed as a package with `pip install -e .`
4. **Distribution**: Can be published to PyPI
5. **Docker**: Can be containerized for deployment
6. **Testing**: Organized test structure

## Backward Compatibility

All logic remains unchanged:
- Environment behavior identical
- API unchanged
- Models unchanged
- Training pipeline unchanged
- Backend API unchanged

Only the file organization and import paths have changed.

## Next Steps

1. Run migration steps 1-10
2. Update all imports across codebase
3. Test with: `python -c "from manager_worker_env import ManagerWorkerEnv; print('Success!')"`
4. Run test suite: `pytest`
5. Install package: `pip install -e .`
6. Run training: `python training/run_training.py`
7. Run backend: `python -m uvicorn BE.main:app`

## Troubleshooting

**Import Error: "No module named 'manager_worker_env'"**
- Ensure you've run `pip install -e .` from the project root
- Check that `src/manager_worker_env/__init__.py` exists

**Import Error: "No module named 'env'"**
- Update imports to use `manager_worker_env` instead of `env`
- Check all files in `training/`, `BE/`, and `tests/`

**Module not found in tests**
- Ensure `tests/__init__.py` exists
- Run tests from project root: `pytest`

