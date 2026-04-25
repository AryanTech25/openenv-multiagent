# Llama 2 Worker Integration - Quick Start

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd manager-worker-env

# Option A: Using uv (recommended)
uv pip install -e ".[llm]"

# Option B: Using pip
pip install transformers>=4.36.0 accelerate>=0.24.0 bitsandbytes>=0.41.0 peft>=0.7.0 datasets>=2.14.0 trl>=0.7.0
```

### Step 2: Authenticate with HuggingFace

```bash
# Get token from https://huggingface.co/settings/tokens
huggingface-cli login

# Or set environment variable
export HF_TOKEN="your_token_here"
```

### Step 3: Run Test

```bash
# Test single worker
python test_llm_workers.py --test single

# Test worker pool
python test_llm_workers.py --test pool

# Test failure injection
python test_llm_workers.py --test failures

# Run all tests
python test_llm_workers.py --test all
```

---

## 📁 What Was Created

```
manager-worker-env/
├── agents/
│   ├── __init__.py              # Package init
│   ├── llm_worker.py            # Llama 2 worker implementation
│   └── worker_pool.py           # Worker pool manager
├── test_llm_workers.py          # Test suite
└── pyproject.toml               # Updated with LLM dependencies
```

---

## 🔧 Key Files Explained

### `agents/llm_worker.py`
- **LlamaWorkerConfig**: Configuration class for model settings
- **LlamaWorker**: Main worker class that:
  - Loads Llama 2 7B from HuggingFace
  - Generates task outputs
  - Injects realistic failures (hallucination, off-task, incomplete)

### `agents/worker_pool.py`
- **WorkerPool**: Manages multiple workers
  - Initializes workers with random skill levels (0.3-1.0)
  - Assigns tasks to workers
  - Tracks worker states

### `test_llm_workers.py`
- Test suite with 3 test cases:
  1. Single worker generation
  2. Worker pool with 4 workers
  3. Failure injection demonstration

---

## 💡 Usage Example

```python
from agents.worker_pool import WorkerPool
from agents.llm_worker import LlamaWorkerConfig

# Create worker pool
config = LlamaWorkerConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    max_tokens=512,
    temperature=0.7,
)

pool = WorkerPool(num_workers=4, config=config)

# Assign task to worker
subtask = {
    "description": "Write a Python function to calculate factorial",
    "context": "Handle edge cases",
}

output = pool.assign_task(worker_id=0, subtask=subtask)
print(output)

# Cleanup
pool.cleanup()
```

---

## ⚙️ Configuration Options

### Model Settings
```python
config = LlamaWorkerConfig(
    model_name="meta-llama/Llama-2-7b-hf",  # Model to use
    device="cuda",                           # "cuda" or "cpu"
    use_8bit=True,                          # 8-bit quantization (saves memory)
    use_flash_attention=True,               # Flash attention (faster)
    max_tokens=512,                         # Max output length
    temperature=0.7,                        # Randomness (0.0-1.0)
    top_p=0.9,                              # Nucleus sampling
)
```

### Worker Pool Settings
```python
pool = WorkerPool(
    num_workers=4,                          # Number of workers
    config=config,                          # LlamaWorkerConfig
)
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
```python
# Solution: Enable 8-bit quantization
config = LlamaWorkerConfig(use_8bit=True)
```

### Issue: Model Download Fails
```bash
# Solution: Set cache directory
export HF_HOME=/path/to/cache

# Or download manually
huggingface-cli download meta-llama/Llama-2-7b-hf
```

### Issue: Slow Generation
```python
# Solution: Reduce max tokens and temperature
config = LlamaWorkerConfig(
    max_tokens=256,
    temperature=0.5,
)
```

### Issue: "No module named 'transformers'"
```bash
# Solution: Install transformers
pip install transformers>=4.36.0
```

---

## 📊 Performance Tips

### For GPU (CUDA)
- ✅ Use 8-bit quantization (saves 75% memory)
- ✅ Use Flash Attention (2-3x faster)
- ✅ Reduce max_tokens if OOM

### For CPU
- ✅ Use smaller model (Llama 2 7B is minimum)
- ✅ Reduce batch size
- ✅ Increase temperature for faster sampling

### For Production
- ✅ Use quantization
- ✅ Cache model after first load
- ✅ Use batch generation for multiple tasks
- ✅ Add timeout handling

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Authenticate with HuggingFace
3. ✅ Run tests
4. ⏭️ Integrate with environment (`env/manager_worker_env.py`)
5. ⏭️ Add to training pipeline
6. ⏭️ Update FastAPI backend
7. ⏭️ Deploy to HuggingFace Spaces

---

## 📚 Resources

- [Llama 2 Model Card](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

---

## ✅ Checklist

- [ ] Dependencies installed
- [ ] HuggingFace authenticated
- [ ] Single worker test passes
- [ ] Worker pool test passes
- [ ] Failure injection test passes
- [ ] Ready to integrate with environment

---

**Status**: 🟢 Ready to use!

Run `python test_llm_workers.py --test all` to verify everything works.
