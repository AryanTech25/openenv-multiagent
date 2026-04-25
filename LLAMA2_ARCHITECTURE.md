# Llama 2 Worker Architecture & Design

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    OrchestraAI Environment                       │
│                  (manager_worker_env.py)                         │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                      WorkerPool Manager                          │
│                   (worker_pool.py)                               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  • Initialize 4 workers with random skill levels          │  │
│  │  • Assign tasks to workers                                │  │
│  │  • Track worker states                                    │  │
│  │  • Manage resource cleanup                                │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                    Worker Instances (4x)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │   Worker 0   │  │   Worker 1   │  │   Worker 2   │  ...       │
│  │  skill=0.45  │  │  skill=0.78  │  │  skill=0.62  │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                    LlamaWorker Class                             │
│                   (llm_worker.py)                                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  1. Load Model                                             │  │
│  │     • Download from HuggingFace                            │  │
│  │     • Apply 8-bit quantization (optional)                  │  │
│  │     • Enable Flash Attention (optional)                    │  │
│  │                                                            │  │
│  │  2. Build Prompt                                           │  │
│  │     • Task description                                     │  │
│  │     • Context information                                  │  │
│  │     • Skill level hint                                     │  │
│  │                                                            │  │
│  │  3. Generate Output                                        │  │
│  │     • Forward pass through model                           │  │
│  │     • Sampling with temperature & top_p                   │  │
│  │     • Decode tokens to text                                │  │
│  │                                                            │  │
│  │  4. Inject Failures                                        │  │
│  │     • Hallucination (subtle errors)                        │  │
│  │     • Off-task (unrelated output)                          │  │
│  │     • Incomplete (partial output)                          │  │
│  │     • Probability based on skill level                     │  │
│  │                                                            │  │
│  │  5. Cleanup                                                │  │
│  │     • Delete model from memory                             │  │
│  │     • Clear GPU cache                                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│              Llama 2 7B Model (HuggingFace)                      │
│  • 7 billion parameters                                          │
│  • Trained on 2 trillion tokens                                  │
│  • Supports 8-bit quantization                                   │
│  • Compatible with Flash Attention                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Task Assignment Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Environment creates subtask                                  │
│    {                                                             │
│      "description": "Write HTML for landing page",              │
│      "context": "Include navigation menu"                       │
│    }                                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. WorkerPool.assign_task(worker_id=0, subtask)                │
│    • Mark worker as active                                      │
│    • Store current subtask                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. LlamaWorker.work_on_task(subtask)                            │
│    • Build prompt with task description                         │
│    • Add skill level hint                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. LlamaWorker._generate_output(prompt)                         │
│    • Tokenize prompt                                            │
│    • Forward pass through model                                 │
│    • Sample tokens with temperature                             │
│    • Decode to text                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. LlamaWorker._inject_failures(output)                         │
│    • Calculate failure probability                              │
│    • Choose failure mode (if applicable)                        │
│    • Corrupt output                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Return output to environment                                 │
│    "<!DOCTYPE html>                                             │
│     <html>                                                       │
│       <head>                                                     │
│         <title>Landing Page</title>                             │
│       </head>                                                    │
│       <body>                                                     │
│         <!-- Navigation menu here -->                           │
│       </body>                                                    │
│     </html>"                                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Failure Injection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Generated Output (from Llama 2)                                 │
│ "def factorial(n):                                              │
│      if n <= 1:                                                 │
│          return 1                                               │
│      return n * factorial(n-1)"                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Calculate Failure Probability                                   │
│ failure_prob = (1 - skill_level) * 0.7                         │
│                                                                 │
│ Examples:                                                       │
│ • skill=0.3 → failure_prob = 0.49 (49% chance)                │
│ • skill=0.5 → failure_prob = 0.35 (35% chance)                │
│ • skill=0.8 → failure_prob = 0.14 (14% chance)                │
│ • skill=1.0 → failure_prob = 0.00 (0% chance)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │ Random Check    │
                    └─────────────────┘
                    /                 \
                   /                   \
            No Failure              Failure
            (Clean Output)          (Choose Mode)
                   |                    |
                   |                    ├─→ Hallucination (40%)
                   |                    ├─→ Off-task (30%)
                   |                    └─→ Incomplete (30%)
                   |                    |
                   └────────┬───────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Final Output                                                    │
│                                                                 │
│ HALLUCINATION:                                                  │
│ "def factorial(n):                                              │
│      if n <= 1:                                                 │
│          return 1                                               │
│      return n * factorial(n-1)                                  │
│  <!-- TODO: Fix styling -->"                                   │
│                                                                 │
│ OFF-TASK:                                                       │
│ "I'm not sure how to approach this task.                        │
│  Let me think about something else."                            │
│                                                                 │
│ INCOMPLETE:                                                     │
│ "def factorial(n):                                              │
│      if n <= 1:                                                 │
│          return 1                                               │
│  [Output incomplete - worker ran out of time]"                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    LlamaWorkerConfig                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Attributes:                                               │  │
│  │ • model_name: str                                         │  │
│  │ • device: str ("cuda" or "cpu")                           │  │
│  │ • use_8bit: bool                                          │  │
│  │ • use_flash_attention: bool                               │  │
│  │ • max_tokens: int                                         │  │
│  │ • temperature: float                                      │  │
│  │ • top_p: float                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                              │ uses
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      LlamaWorker                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Attributes:                                               │  │
│  │ • worker_id: int                                          │  │
│  │ • skill_level: float (0.0-1.0)                            │  │
│  │ • config: LlamaWorkerConfig                               │  │
│  │ • model: AutoModelForCausalLM                             │  │
│  │ • tokenizer: AutoTokenizer                                │  │
│  │ • device: str                                             │  │
│  │                                                           │  │
│  │ Methods:                                                  │  │
│  │ • work_on_task(subtask) → str                             │  │
│  │ • _load_model() → None                                    │  │
│  │ • _build_prompt(subtask) → str                            │  │
│  │ • _generate_output(prompt) → str                          │  │
│  │ • _inject_failures(output, subtask) → str                 │  │
│  │ • _inject_hallucination(output) → str                     │  │
│  │ • _inject_off_task(output) → str                          │  │
│  │ • _inject_incomplete(output) → str                        │  │
│  │ • cleanup() → None                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                              │ contains (4x)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      WorkerPool                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Attributes:                                               │  │
│  │ • num_workers: int                                        │  │
│  │ • config: LlamaWorkerConfig                               │  │
│  │ • workers: List[LlamaWorker]                              │  │
│  │ • worker_states: Dict[int, Dict[str, Any]]                │  │
│  │                                                           │  │
│  │ Methods:                                                  │  │
│  │ • _initialize_workers() → None                            │  │
│  │ • assign_task(worker_id, subtask) → str                   │  │
│  │ • get_worker_state(worker_id) → Dict                      │  │
│  │ • reset_worker(worker_id) → None                          │  │
│  │ • cleanup() → None                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## State Transitions

### Worker State Machine

```
┌──────────────┐
│   INACTIVE   │  (Initial state)
└──────────────┘
       ↓
       │ assign_task()
       ↓
┌──────────────┐
│   ACTIVE     │  (Working on task)
└──────────────┘
       ↓
       │ work_on_task() completes
       ↓
┌──────────────┐
│   COMPLETED  │  (Output generated)
└──────────────┘
       ↓
       │ reset_worker()
       ↓
┌──────────────┐
│   INACTIVE   │  (Ready for next task)
└──────────────┘
```

### Worker State Dictionary

```python
worker_state = {
    'skill_level': 0.45,              # 0.0-1.0
    'is_active': True,                # Currently working
    'current_subtask': {              # Current task
        'description': '...',
        'context': '...'
    },
    'output': '...',                  # Generated output
    'failure_mode': 'hallucination',  # None or failure type
}
```

---

## Memory Management

### Model Loading

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Download Model (first time only)                            │
│    • ~13 GB from HuggingFace Hub                                │
│    • Cached in ~/.cache/huggingface/                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Load Model into Memory                                       │
│    • Without quantization: ~14 GB (GPU)                         │
│    • With 8-bit quantization: ~3.5 GB (GPU)                     │
│    • CPU: ~28 GB (not recommended)                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Generate Outputs                                             │
│    • Inference memory: ~1-2 GB per batch                        │
│    • Temporary allocations during generation                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Cleanup                                                      │
│    • Delete model from memory                                   │
│    • Clear GPU cache                                            │
│    • Free all temporary allocations                             │
└─────────────────────────────────────────────────────────────────┘
```

### Optimization Strategies

```
Strategy 1: 8-bit Quantization
┌─────────────────────────────────────────────────────────────────┐
│ • Reduces model size by 75%                                     │
│ • Minimal quality loss                                          │
│ • Requires bitsandbytes library                                 │
│ • Recommended for GPU with <8GB VRAM                            │
└─────────────────────────────────────────────────────────────────┘

Strategy 2: Flash Attention
┌─────────────────────────────────────────────────────────────────┐
│ • 2-3x faster inference                                         │
│ • Reduces memory usage during generation                        │
│ • Requires flash-attn library                                   │
│ • Recommended for all GPU setups                                │
└─────────────────────────────────────────────────────────────────┘

Strategy 3: Batch Generation
┌─────────────────────────────────────────────────────────────────┐
│ • Process multiple tasks in parallel                            │
│ • Better GPU utilization                                        │
│ • Requires padding and masking                                  │
│ • Recommended for production                                    │
└─────────────────────────────────────────────────────────────────┘

Strategy 4: Model Caching
┌─────────────────────────────────────────────────────────────────┐
│ • Keep model in memory between tasks                            │
│ • Avoid reload overhead                                         │
│ • Requires careful resource management                          │
│ • Recommended for continuous operation                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Latency

```
Task: Generate 256 tokens

GPU (CUDA) without optimization:
┌─────────────────────────────────────────────────────────────────┐
│ Model Load: 2-3 seconds (first time)                            │
│ Tokenization: 1-2 ms                                            │
│ Generation: 100-200 ms                                          │
│ Decoding: 1-2 ms                                                │
│ Total: ~100-200 ms                                              │
└─────────────────────────────────────────────────────────────────┘

GPU (CUDA) with 8-bit + Flash Attention:
┌─────────────────────────────────────────────────────────────────┐
│ Model Load: 1-2 seconds (first time)                            │
│ Tokenization: 1-2 ms                                            │
│ Generation: 50-100 ms (2-3x faster)                             │
│ Decoding: 1-2 ms                                                │
│ Total: ~50-100 ms                                               │
└─────────────────────────────────────────────────────────────────┘

CPU:
┌─────────────────────────────────────────────────────────────────┐
│ Model Load: 5-10 seconds (first time)                           │
│ Tokenization: 5-10 ms                                           │
│ Generation: 2-5 seconds                                         │
│ Decoding: 5-10 ms                                               │
│ Total: ~2-5 seconds                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Throughput

```
Scenario: 4 workers, 50 tasks per episode

Sequential (one worker at a time):
┌─────────────────────────────────────────────────────────────────┐
│ Time per task: 100 ms                                           │
│ Total time: 50 tasks × 100 ms = 5 seconds                       │
│ Throughput: 10 tasks/second                                     │
└─────────────────────────────────────────────────────────────────┘

Parallel (4 workers simultaneously):
┌─────────────────────────────────────────────────────────────────┐
│ Time per task: 100 ms (same)                                    │
│ Total time: 50 tasks ÷ 4 workers × 100 ms = 1.25 seconds       │
│ Throughput: 40 tasks/second                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### With Environment

```python
# In manager_worker_env.py
from agents.worker_pool import WorkerPool

class ManagerWorkerEnv:
    def __init__(self, ...):
        self.worker_pool = WorkerPool(num_workers=4)
    
    def _get_worker_output(self, worker_id, subtask):
        return self.worker_pool.assign_task(worker_id, subtask)
    
    def __del__(self):
        if self.worker_pool:
            self.worker_pool.cleanup()
```

### With Training Pipeline

```python
# In training/train_manager.py
from env.manager_worker_env import ManagerWorkerEnv

env = ManagerWorkerEnv(use_llm_workers=True)
model = PPO(MultiInputActorCriticPolicy, env, ...)
model.learn(total_timesteps=500000)
```

### With FastAPI Backend

```python
# In server/app.py
from agents.worker_pool import WorkerPool

worker_pool = WorkerPool(num_workers=4)

@app.post("/episode/step")
async def step_episode(episode_id: str, action: int):
    # Get worker output
    output = worker_pool.assign_task(worker_id, subtask)
    # ... rest of logic
```

---

## Deployment Considerations

### Development
- Single GPU (8GB+)
- 8-bit quantization enabled
- Flash Attention enabled
- Max tokens: 512

### Production
- Multiple GPUs (distributed)
- 8-bit quantization enabled
- Flash Attention enabled
- Batch generation
- Model caching
- Max tokens: 256

### Edge/Mobile
- CPU only
- 4-bit quantization
- Smaller model (3B)
- Max tokens: 128

---

## Summary

The Llama 2 worker architecture provides:

✅ **Scalability**: Multiple workers with independent skill levels  
✅ **Realism**: LLM-based outputs with realistic failure modes  
✅ **Performance**: GPU acceleration with quantization and Flash Attention  
✅ **Flexibility**: Configurable for different hardware constraints  
✅ **Integration**: Clean API for environment and training pipeline  
✅ **Production-Ready**: Error handling, logging, resource cleanup  

Ready for integration with the training pipeline!
