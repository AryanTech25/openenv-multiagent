# Llama 2 Worker Setup Checklist

## 📋 Pre-Setup Requirements

- [ ] Python 3.10+ installed
- [ ] CUDA 11.8+ installed (for GPU, optional)
- [ ] HuggingFace account created
- [ ] HuggingFace token generated
- [ ] 8GB+ GPU VRAM or 28GB+ RAM available
- [ ] Internet connection for model download

---

## 🔧 Installation Phase

### Step 1: Install Dependencies

- [ ] Navigate to `manager-worker-env` directory
- [ ] Run: `uv pip install -e ".[llm]"` (or `pip install ...`)
- [ ] Verify installation:
  ```bash
  python -c "import torch; print(torch.__version__)"
  python -c "import transformers; print(transformers.__version__)"
  ```

### Step 2: Authenticate with HuggingFace

- [ ] Get token from https://huggingface.co/settings/tokens
- [ ] Run: `huggingface-cli login`
- [ ] Paste token when prompted
- [ ] Verify authentication:
  ```bash
  huggingface-cli whoami
  ```

### Step 3: Verify GPU (Optional)

- [ ] Check CUDA availability:
  ```bash
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
  ```
- [ ] Check GPU memory:
  ```bash
  python -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
  ```

---

## 📁 File Structure Verification

### Core Implementation Files

- [ ] `manager-worker-env/agents/__init__.py` exists
- [ ] `manager-worker-env/agents/llm_worker.py` exists (220+ lines)
- [ ] `manager-worker-env/agents/worker_pool.py` exists (110+ lines)
- [ ] `manager-worker-env/test_llm_workers.py` exists (150+ lines)
- [ ] `manager-worker-env/config/llm_config.yaml` exists

### Configuration Files

- [ ] `manager-worker-env/pyproject.toml` updated with LLM dependencies
- [ ] `manager-worker-env/pyproject.toml` has `[llm]` optional dependency group

### Documentation Files

- [ ] `LLAMA2_WORKER_INTEGRATION_GUIDE.md` exists
- [ ] `LLAMA2_QUICKSTART.md` exists
- [ ] `LLAMA2_IMPLEMENTATION_SUMMARY.md` exists
- [ ] `LLAMA2_ARCHITECTURE.md` exists
- [ ] `LLAMA2_COMPLETE_GUIDE.md` exists
- [ ] `LLAMA2_SETUP_CHECKLIST.md` exists (this file)

---

## 🧪 Testing Phase

### Test 1: Single Worker

```bash
cd manager-worker-env
python test_llm_workers.py --test single
```

- [ ] Command runs without errors
- [ ] Model loads successfully
- [ ] Output is generated
- [ ] No CUDA out of memory errors
- [ ] Execution completes in <5 minutes

**Expected Output:**
```
Loading meta-llama/Llama-2-7b-hf on cuda...
✓ Model loaded successfully on cuda
Assigning task to worker...

Generated Output:
[... generated text ...]
```

### Test 2: Worker Pool

```bash
python test_llm_workers.py --test pool
```

- [ ] Command runs without errors
- [ ] 4 workers initialize successfully
- [ ] Each worker has different skill level
- [ ] All workers generate outputs
- [ ] No memory leaks
- [ ] Execution completes in <10 minutes

**Expected Output:**
```
Initializing 4 Llama workers...
  Worker 0: skill=0.45
  Worker 1: skill=0.78
  Worker 2: skill=0.62
  Worker 3: skill=0.91

Worker 0 (skill=0.45):
Task: Write HTML for a landing page header
Output: [... generated text ...]
```

### Test 3: Failure Injection

```bash
python test_llm_workers.py --test failures
```

- [ ] Command runs without errors
- [ ] Low-skill worker (0.3) generates outputs
- [ ] Failures are injected in some outputs
- [ ] Different failure modes appear
- [ ] Execution completes in <5 minutes

**Expected Output:**
```
Testing low-skill worker (skill=0.3)...
Running 5 attempts to see failure injection:

Attempt 1: [... normal output ...]
Attempt 2: I'm not sure how to approach this task...
Attempt 3: [... truncated output ...]
```

### Test 4: All Tests

```bash
python test_llm_workers.py --test all
```

- [ ] All three tests pass
- [ ] No errors or warnings
- [ ] Total execution time < 20 minutes
- [ ] Final message: "✓ All tests completed successfully!"

---

## 🔍 Verification Checks

### Code Quality

- [ ] No syntax errors in `agents/llm_worker.py`
- [ ] No syntax errors in `agents/worker_pool.py`
- [ ] No syntax errors in `test_llm_workers.py`
- [ ] All imports resolve correctly
- [ ] Type hints are present

### Functionality

- [ ] LlamaWorkerConfig initializes correctly
- [ ] LlamaWorker loads model successfully
- [ ] WorkerPool creates multiple workers
- [ ] Tasks can be assigned to workers
- [ ] Outputs are generated
- [ ] Failures are injected
- [ ] Resources are cleaned up

### Performance

- [ ] Single task generation: <200ms (GPU) or <5s (CPU)
- [ ] Worker pool initialization: <5 seconds
- [ ] Memory usage is reasonable
- [ ] No memory leaks after cleanup
- [ ] GPU memory is freed after cleanup

### Documentation

- [ ] All documentation files are readable
- [ ] Code examples are correct
- [ ] Configuration options are documented
- [ ] Troubleshooting section is helpful
- [ ] Next steps are clear

---

## 🚀 Integration Readiness

### Before Integration

- [ ] All tests pass
- [ ] No errors in logs
- [ ] Memory usage is acceptable
- [ ] Generation speed is acceptable
- [ ] Failure injection works correctly

### Integration Preparation

- [ ] Review `env/manager_worker_env.py` structure
- [ ] Identify integration points
- [ ] Plan environment modifications
- [ ] Prepare integration test cases
- [ ] Document integration changes

### Integration Steps

- [ ] Add import: `from agents.worker_pool import WorkerPool`
- [ ] Add worker pool initialization in `__init__`
- [ ] Add `_get_worker_output()` method
- [ ] Add cleanup in `__del__`
- [ ] Test environment with LLM workers
- [ ] Verify all environment tests pass

---

## 📊 Performance Baseline

### Record Your Baseline

**Hardware:**
- [ ] GPU Model: _______________
- [ ] GPU Memory: _______________
- [ ] CPU: _______________
- [ ] RAM: _______________

**Configuration:**
- [ ] Model: meta-llama/Llama-2-7b-hf
- [ ] Device: cuda / cpu
- [ ] 8-bit quantization: yes / no
- [ ] Flash Attention: yes / no
- [ ] Max tokens: _______________

**Performance Metrics:**
- [ ] Model load time: _____ seconds
- [ ] Single task generation: _____ ms
- [ ] Worker pool initialization: _____ seconds
- [ ] GPU memory used: _____ GB
- [ ] CPU memory used: _____ GB

---

## 🐛 Troubleshooting Checklist

### If Tests Fail

- [ ] Check Python version: `python --version`
- [ ] Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Check HuggingFace authentication: `huggingface-cli whoami`
- [ ] Check disk space: `df -h`
- [ ] Check internet connection: `ping huggingface.co`

### If Memory Issues

- [ ] Enable 8-bit quantization
- [ ] Reduce max_tokens
- [ ] Use CPU instead of GPU
- [ ] Close other applications
- [ ] Increase swap space

### If Generation is Slow

- [ ] Enable Flash Attention
- [ ] Reduce max_tokens
- [ ] Reduce temperature
- [ ] Use GPU instead of CPU
- [ ] Check GPU utilization: `nvidia-smi`

### If Authentication Fails

- [ ] Verify token is valid
- [ ] Check token permissions
- [ ] Re-authenticate: `huggingface-cli login`
- [ ] Set environment variable: `export HF_TOKEN="..."`

---

## ✅ Final Verification

### Before Declaring Success

- [ ] All tests pass ✓
- [ ] No errors in logs ✓
- [ ] Performance is acceptable ✓
- [ ] Memory usage is reasonable ✓
- [ ] Documentation is complete ✓
- [ ] Code is clean and well-commented ✓
- [ ] Ready for integration ✓

### Sign-Off

- [ ] I have completed all checklist items
- [ ] All tests pass successfully
- [ ] System is ready for integration
- [ ] Documentation is complete
- [ ] Performance is acceptable

**Date Completed:** _______________  
**Completed By:** _______________  
**Notes:** _______________

---

## 📞 Support Resources

### Documentation
- LLAMA2_WORKER_INTEGRATION_GUIDE.md - Complete guide
- LLAMA2_QUICKSTART.md - Quick reference
- LLAMA2_ARCHITECTURE.md - Architecture details
- LLAMA2_COMPLETE_GUIDE.md - Full guide

### External Resources
- [Llama 2 Model Card](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Transformers Docs](https://huggingface.co/docs/transformers/)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

### Common Issues
- See "Troubleshooting" section in LLAMA2_COMPLETE_GUIDE.md
- See "Common Issues & Solutions" in LLAMA2_IMPLEMENTATION_SUMMARY.md

---

## 🎯 Next Steps After Setup

1. ✅ Complete this checklist
2. ⏭️ Integrate with environment (`env/manager_worker_env.py`)
3. ⏭️ Update training pipeline (`training/train_manager.py`)
4. ⏭️ Add to FastAPI backend (`server/app.py`)
5. ⏭️ Update React dashboard
6. ⏭️ Deploy to HuggingFace Spaces

---

## 📝 Notes

Use this space to record any issues, solutions, or observations:

```
Issue 1: _______________________________________________
Solution: _______________________________________________

Issue 2: _______________________________________________
Solution: _______________________________________________

Observation 1: _______________________________________________

Observation 2: _______________________________________________
```

---

**Status**: 🟢 Ready to Begin Setup!

**First Command to Run:**
```bash
cd manager-worker-env
uv pip install -e ".[llm]"
```

Good luck! 🚀
