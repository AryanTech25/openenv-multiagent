# 🎻 OrchestraAI: Multi-Agent RL Training Environment

**OrchestraAI** is a state-of-the-art Reinforcement Learning (RL) environment designed to train a **Manager Agent** to coordinate and oversee a pool of **Worker Agents** (powered by Hugging Face LLMs). 

The core challenge is realism: Worker agents have varying skill levels and randomly exhibit realistic failure modes—hallucinations, off-task behavior, and incomplete work. The Manager must learn to spend a limited **Token Budget** to check, correct, or replace workers to complete complex tasks successfully.

---

## 🌟 Key Features

- **Brain-Swappable Workers**: Support for ANY Hugging Face model (Llama 3, Phi-2, SmolLM, etc.).
- **Dual-Inference Modes**:
  - **Local Mode**: Runs smaller models (like SmolLM) on your local hardware (optimized for Mac MPS).
  - **API Mode**: Leverages the Hugging Face Inference API for massive models (Llama 3 70B) with zero local memory footprint.
- **Realistic Failure Injection**: An internal "Hallucination Engine" that simulates agent errors based on skill level.
- **Multi-Task SFT Pipeline**: A dedicated supervised fine-tuning pipeline to teach small LLMs how to perform specifically within the OrchestraAI ecosystem.
- **Rich Visualization**: A real-time dashboard showing Manager actions, Worker state alerts, and budget consumption.

---

## 🏗️ Project Structure

```text
manager-worker-env/
├── agents/
│   ├── manager_agent.py    # The RL-based orchestrator (Manager)
│   ├── llm_worker.py       # Generic Hugging Face worker implementation
│   └── worker_pool.py      # Management logic for multiple workers
├── env/
│   ├── manager_worker_env.py # The RL Environment (Gymnasium)
│   ├── task_library.py       # 400+ complex multi-step tasks across 5 domains
│   ├── hallucination_engine.py # The failure injection logic
│   └── reward_calculator.py  # 5-component reward system
├── training/
│   ├── train_manager.py    # PPO training for the Manager Agent
│   ├── train_worker_sft.py # Fine-tuning script for Worker Agents (LoRA/SFT)
│   ├── generate_multitask_data.py # Dataset generator for worker SFT
│   └── worker_finetune_colab.ipynb # One-click Colab training notebook
├── WORKER_GUIDE.md         # Detailed guide for model swapping & training
└── test_hf_workers.py      # Dashboard for testing individual worker performance
```

---

## 🚀 Getting Started

### 1. Installation
Ensure you have the virtual environment active:
```bash
source manager-worker-env/.venv/bin/activate
pip install -e ./manager-worker-env
```

### 2. Configure Hugging Face
Create a `.env` file in the root directory:
```text
HF_TOKEN=your_token_here
```

### 3. Testing a Worker
Run the interactive dashboard to test a specific model (e.g., Llama 3 via API):
```bash
python3 manager-worker-env/test_hf_workers.py --model "meta-llama/Meta-Llama-3-8B" --api-mode api
```

---

## 🧠 The Agent Architecture

### The Manager (RL Agent)
The Manager acts as the "Conductor." It observes the task progress and worker states but cannot see the "ground truth" of a worker's output quality without spending the budget to **Check** it. It must learn the optimal balance between trusting workers and spending resources on quality control.

### The Workers (LLM Agents)
Workers are the "Musicians." They receive subtasks and generate outputs.
- **High Skill (0.8-1.0)**: Produces accurate work with minimal errors.
- **Low Skill (0.3-0.5)**: Prone to high-plausibility hallucinations (looks good, but it's wrong).

Workers can be fine-tuned using the `training/train_worker_sft.py` script to behave more like the "Expert" models in the environment.

---

## 📊 Training Pipeline

### Manager RL (PPO)
Uses Stable Baselines3 to train the Manager in the `ManagerWorkerEnv`.
```bash
python3 manager-worker-env/training/train_manager.py
```

### Worker SFT (LoRA)
Uses Hugging Face `trl` and `peft` to fine-tune worker agents for domain-specific mastery.
```bash
python3 manager-worker-env/training/train_worker_sft.py --model "HuggingFaceTB/SmolLM-135M" --steps 500
```

---

## 🛡️ License
MIT License. Created by Tirth and the OpenEnv team.
