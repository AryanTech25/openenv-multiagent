# 🧠 OrchestraAI: Hugging Face Worker Guide

Welcome to your upgraded Worker Agent system. You can now use any model from Hugging Face to power your multi-agent environment.

## 🚀 Quick Start: Running Workers

### Option A: API Mode (No Memory/Zero Download)
Use this for huge models like **Llama 3** or **Qwen-72B**.
```bash
./manager-worker-env/.venv/bin/python3 manager-worker-env/test_hf_workers.py \
    --model "meta-llama/Meta-Llama-3-8B" \
    --api-mode api
```

### Option B: Local Mode (Fast Inference)
Use this for tiny models like **SmolLM** that run directly on your Mac.
```bash
./manager-worker-env/.venv/bin/python3 manager-worker-env/test_hf_workers.py \
    --model "HuggingFaceTB/SmolLM-135M"
```

---

## 🛠️ Fine-Tuning Your Workers

### 1. Generate Data
Created a multi-task dataset from your library:
```bash
./manager-worker-env/.venv/bin/python3 manager-worker-env/training/generate_multitask_data.py
```

### 2. Run Training (Mac optimized)
Train a custom adapter for your environment:
```bash
./manager-worker-env/.venv/bin/python3 manager-worker-env/training/train_worker_sft.py \
    --model "HuggingFaceTB/SmolLM-135M" \
    --dataset "manager-worker-env/training/multitask_worker_data.jsonl" \
    --steps 200 \
    --output "models/my_custom_worker"
```

---

## 📁 Key Files Created
- `manager-worker-env/agents/llm_worker.py`: The heart of the HF integration.
- `manager-worker-env/training/train_worker_sft.py`: The fine-tuning engine.
- `manager-worker-env/test_hf_workers.py`: The testing dashboard.
- `manager-worker-env/training/worker_finetune_colab.ipynb`: One-click Colab trainer.

## 🔐 Configuration
Your Hugging Face token is stored in `.env`. Never commit this file to GitHub!
