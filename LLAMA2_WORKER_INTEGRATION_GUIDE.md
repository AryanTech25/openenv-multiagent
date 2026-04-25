# Llama 2 7B Worker Agent Integration Guide

## Complete Step-by-Step Implementation

This guide walks you through integrating Llama 2 7B from HuggingFace as worker agents in your OrchestraAI environment.

---

## STEP 1: Update Dependencies

### 1.1 Add LLM Dependencies to pyproject.toml

Add these packages to the `dependencies` section:

```toml
dependencies = [
    # ... existing dependencies ...
    "transformers>=4.36.0",           # HuggingFace transformers
    "accelerate>=0.24.0",             # GPU acceleration
    "bitsandbytes>=0.41.0",           # 8-bit quantization (optional but recommended)
    "peft>=0.7.0",                    # Parameter-Efficient Fine-Tuning
    "datasets>=2.14.0",               # For fine-tuning data
    "trl>=0.7.0",                     # Transformer Reinforcement Learning
]
```

Also add a new optional dependency group:

```toml
[project.optional-dependencies]
llm = [
    "transformers>=4.36.0",
    "accelerate>=0.24.0",
    "bitsandbytes>=0.41.0",
    "peft>=0.7.0",
    "datasets>=2.14.0",
    "trl>=0.7.0",
]
```

### 1.2 Install Dependencies

```bash
cd manager-worker-env
uv pip install -e ".[llm]"
```

Or with pip:
```bash
pip install transformers>=4.36.0 accelerate>=0.24.0 bitsandbytes>=0.41.0 peft>=0.7.0 datasets>=2.14.0 trl>=0.7.0
```

---

## STEP 2: Create LLM Worker Base Class

### 2.1 Create `agents/llm_worker.py`

```python
"""
LLM-based Worker Agent using Llama 2 7B from HuggingFace.
"""

import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)


class LlamaWorkerConfig:
    """Configuration for Llama 2 Worker."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_8bit: bool = True,
        use_flash_attention: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.device = device
        self.use_8bit = use_8bit
        self.use_flash_attention = use_flash_attention
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p


class LlamaWorker:
    """
    Worker Agent powered by Llama 2 7B.
    
    This worker generates task outputs using the Llama 2 model.
    Failure modes are injected post-generation based on skill level.
    """
    
    def __init__(
        self,
        worker_id: int,
        skill_level: float,
        config: Optional[LlamaWorkerConfig] = None,
    ):
        """
        Initialize Llama Worker.
        
        Args:
            worker_id: Unique worker identifier
            skill_level: Worker competence (0.0-1.0)
            config: LlamaWorkerConfig instance
        """
        self.worker_id = worker_id
        self.skill_level = skill_level
        self.config = config or LlamaWorkerConfig()
        
        self.model = None
        self.tokenizer = None
        self.device = self.config.device
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Llama 2 model and tokenizer from HuggingFace."""
        logger.info(f"Loading {self.config.model_name} on {self.device}...")
        
        # Configure 8-bit quantization if enabled
        quantization_config = None
        if self.config.use_8bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else "eager",
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"✓ Model loaded successfully on {self.device}")
    
    def work_on_task(self, subtask: Dict[str, Any]) -> str:
        """
        Generate output for a subtask.
        
        Args:
            subtask: Dictionary with 'description' and 'context' keys
            
        Returns:
            Generated output string
        """
        # Build prompt
        prompt = self._build_prompt(subtask)
        
        # Generate output
        output = self._generate_output(prompt)
        
        # Inject failures based on skill level
        output = self._inject_failures(output, subtask)
        
        return output
    
    def _build_prompt(self, subtask: Dict[str, Any]) -> str:
        """Build prompt for the subtask."""
        description = subtask.get("description", "")
        context = subtask.get("context", "")
        
        prompt = f"""You are a worker with skill level {self.skill_level:.1f}/1.0.

Task: {description}

Context: {context}

Generate a complete and accurate output for this task. Your skill level affects quality:
- 0.9-1.0: Excellent, complete, well-structured work
- 0.6-0.8: Good work with minor issues
- 0.3-0.5: Partial work with some errors
- 0.0-0.3: Poor work with many errors

Output:"""
        
        return prompt
    
    def _generate_output(self, prompt: str) -> str:
        """Generate output using the model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract only the generated part
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_output[len(prompt):].strip()
        
        return generated_text
    
    def _inject_failures(self, output: str, subtask: Dict[str, Any]) -> str:
        """
        Inject realistic failures based on skill level.
        
        Failure modes:
        - Hallucination: Output looks good but has subtle errors
        - Off-task: Output is unrelated to the task
        - Incomplete: Output is partial
        - Stuck: Worker loops (handled at environment level)
        """
        import random
        
        # Probability of failure increases as skill decreases
        failure_probability = (1 - self.skill_level) * 0.7
        
        if random.random() > failure_probability:
            # No failure - return clean output
            return output
        
        # Choose failure mode
        failure_mode = random.choice(['hallucination', 'off_task', 'incomplete'])
        
        if failure_mode == 'hallucination':
            return self._inject_hallucination(output)
        elif failure_mode == 'off_task':
            return self._inject_off_task(output)
        else:  # incomplete
            return self._inject_incomplete(output)
    
    def _inject_hallucination(self, output: str) -> str:
        """
        Inject hallucination: output looks good but has subtle errors.
        
        Examples:
        - Missing closing tags in HTML
        - Incorrect variable names in code
        - Inconsistent formatting
        """
        import random
        
        hallucinations = [
            lambda x: x + "\n<!-- TODO: Fix styling -->",  # Incomplete comment
            lambda x: x.replace("function", "func"),  # Wrong syntax
            lambda x: x[:len(x)//2] + "\n[... rest of output ...]",  # Truncated
            lambda x: x + "\n\nNote: This might need review",  # Uncertain
        ]
        
        corrupted = random.choice(hallucinations)(output)
        return corrupted
    
    def _inject_off_task(self, output: str) -> str:
        """
        Inject off-task: output is unrelated to the task.
        """
        off_task_outputs = [
            "I'm not sure how to approach this task. Let me think about something else.",
            "This reminds me of a different problem I solved before...",
            "I'll just generate some random content instead.",
            "Here's what I think about this topic in general...",
        ]
        
        import random
        return random.choice(off_task_outputs)
    
    def _inject_incomplete(self, output: str) -> str:
        """
        Inject incomplete: output is partial.
        """
        # Truncate output to 30-70% of original length
        import random
        truncation_ratio = random.uniform(0.3, 0.7)
        truncated_length = int(len(output) * truncation_ratio)
        return output[:truncated_length] + "\n[Output incomplete - worker ran out of time]"
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache()
        logger.info(f"✓ Worker {self.worker_id} cleaned up")
```

---

## STEP 3: Create Worker Pool Manager

### 3.1 Create `agents/worker_pool.py`

```python
"""
Worker Pool Manager for managing multiple Llama workers.
"""

from typing import List, Dict, Any, Optional
from agents.llm_worker import LlamaWorker, LlamaWorkerConfig
import logging

logger = logging.getLogger(__name__)


class WorkerPool:
    """
    Manages a pool of Llama worker agents.
    
    Handles:
    - Worker initialization
    - Task assignment
    - Output generation
    - Resource cleanup
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        config: Optional[LlamaWorkerConfig] = None,
    ):
        """
        Initialize worker pool.
        
        Args:
            num_workers: Number of workers in the pool
            config: LlamaWorkerConfig for all workers
        """
        self.num_workers = num_workers
        self.config = config or LlamaWorkerConfig()
        self.workers: List[LlamaWorker] = []
        self.worker_states: Dict[int, Dict[str, Any]] = {}
        
        self._initialize_workers()
    
    def _initialize_workers(self) -> None:
        """Initialize all workers with random skill levels."""
        logger.info(f"Initializing {self.num_workers} Llama workers...")
        
        import random
        
        for worker_id in range(self.num_workers):
            # Random skill level between 0.3 and 1.0
            skill_level = random.uniform(0.3, 1.0)
            
            worker = LlamaWorker(
                worker_id=worker_id,
                skill_level=skill_level,
                config=self.config,
            )
            
            self.workers.append(worker)
            self.worker_states[worker_id] = {
                'skill_level': skill_level,
                'is_active': False,
                'current_subtask': None,
                'output': None,
                'failure_mode': None,
            }
            
            logger.info(f"  Worker {worker_id}: skill={skill_level:.2f}")
    
    def assign_task(
        self,
        worker_id: int,
        subtask: Dict[str, Any],
    ) -> str:
        """
        Assign a subtask to a worker and get output.
        
        Args:
            worker_id: ID of worker to assign to
            subtask: Subtask dictionary
            
        Returns:
            Generated output string
        """
        if worker_id >= len(self.workers):
            raise ValueError(f"Invalid worker_id: {worker_id}")
        
        worker = self.workers[worker_id]
        
        # Mark worker as active
        self.worker_states[worker_id]['is_active'] = True
        self.worker_states[worker_id]['current_subtask'] = subtask
        
        # Generate output
        output = worker.work_on_task(subtask)
        
        # Store output
        self.worker_states[worker_id]['output'] = output
        
        return output
    
    def get_worker_state(self, worker_id: int) -> Dict[str, Any]:
        """Get current state of a worker."""
        return self.worker_states.get(worker_id, {})
    
    def reset_worker(self, worker_id: int) -> None:
        """Reset a worker's state."""
        self.worker_states[worker_id] = {
            'skill_level': self.workers[worker_id].skill_level,
            'is_active': False,
            'current_subtask': None,
            'output': None,
            'failure_mode': None,
        }
    
    def cleanup(self) -> None:
        """Clean up all workers."""
        logger.info("Cleaning up worker pool...")
        for worker in self.workers:
            worker.cleanup()
        logger.info("✓ Worker pool cleaned up")
```

---

## STEP 4: Integrate with Environment

### 4.1 Update `env/manager_worker_env.py`

Add this import at the top:

```python
from agents.worker_pool import WorkerPool
```

Update the `__init__` method to use the worker pool:

```python
def __init__(
    self,
    max_workers: int = 4,
    max_steps: int = 50,
    token_budget: int = 1000,
    task_difficulty: int = 3,
    failure_injection_rate: float = 0.6,
    use_llm_workers: bool = True,  # NEW
):
    """Initialize environment."""
    # ... existing code ...
    
    # Initialize worker pool
    if use_llm_workers:
        logger.info("Initializing Llama 2 worker pool...")
        self.worker_pool = WorkerPool(num_workers=max_workers)
    else:
        self.worker_pool = None
```

Add method to get worker output:

```python
def _get_worker_output(self, worker_id: int, subtask: Dict[str, Any]) -> str:
    """Get output from worker (LLM or rule-based)."""
    if self.worker_pool is not None:
        return self.worker_pool.assign_task(worker_id, subtask)
    else:
        # Fallback to rule-based worker
        return self._generate_rule_based_output(worker_id, subtask)
```

Add cleanup in `__del__`:

```python
def __del__(self):
    """Cleanup resources."""
    if hasattr(self, 'worker_pool') and self.worker_pool is not None:
        self.worker_pool.cleanup()
```

---

## STEP 5: Create Configuration File

### 5.1 Create `config/llm_config.yaml`

```yaml
# Llama 2 Worker Configuration

llm:
  model_name: "meta-llama/Llama-2-7b-hf"
  device: "cuda"  # or "cpu"
  use_8bit: true
  use_flash_attention: true
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9

worker_pool:
  num_workers: 4
  skill_level_range: [0.3, 1.0]

generation:
  timeout_seconds: 30
  retry_attempts: 3
  cache_outputs: true

failure_injection:
  hallucination_probability: 0.4
  off_task_probability: 0.3
  incomplete_probability: 0.3
```

### 5.2 Create `config/llm_loader.py`

```python
"""Load LLM configuration from YAML."""

import yaml
from typing import Dict, Any
import os


def load_llm_config(config_path: str = "config/llm_config.yaml") -> Dict[str, Any]:
    """Load LLM configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_llm_config_from_env() -> Dict[str, Any]:
    """Get LLM config from environment variables."""
    return {
        'model_name': os.getenv('LLM_MODEL_NAME', 'meta-llama/Llama-2-7b-hf'),
        'device': os.getenv('LLM_DEVICE', 'cuda'),
        'use_8bit': os.getenv('LLM_USE_8BIT', 'true').lower() == 'true',
        'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '512')),
        'temperature': float(os.getenv('LLM_TEMPERATURE', '0.7')),
    }
```

---

## STEP 6: Create Test Script

### 6.1 Create `test_llm_workers.py`

```python
"""
Test script for Llama 2 worker agents.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.worker_pool import WorkerPool
from agents.llm_worker import LlamaWorkerConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_worker():
    """Test a single worker."""
    logger.info("=" * 60)
    logger.info("TEST 1: Single Worker Generation")
    logger.info("=" * 60)
    
    config = LlamaWorkerConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        max_tokens=256,
        temperature=0.7,
    )
    
    pool = WorkerPool(num_workers=1, config=config)
    
    subtask = {
        "description": "Write a Python function to calculate factorial",
        "context": "The function should handle edge cases",
    }
    
    logger.info(f"Assigning task to worker...")
    output = pool.assign_task(0, subtask)
    
    logger.info(f"\nGenerated Output:\n{output}\n")
    
    pool.cleanup()


def test_worker_pool():
    """Test worker pool with multiple workers."""
    logger.info("=" * 60)
    logger.info("TEST 2: Worker Pool (4 workers)")
    logger.info("=" * 60)
    
    config = LlamaWorkerConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        max_tokens=256,
        temperature=0.7,
    )
    
    pool = WorkerPool(num_workers=4, config=config)
    
    subtasks = [
        {
            "description": "Write HTML for a landing page header",
            "context": "Include navigation menu",
        },
        {
            "description": "Write CSS for styling the header",
            "context": "Make it responsive",
        },
        {
            "description": "Write JavaScript for menu interactions",
            "context": "Add smooth animations",
        },
        {
            "description": "Write a README for the project",
            "context": "Include setup instructions",
        },
    ]
    
    for worker_id, subtask in enumerate(subtasks):
        logger.info(f"\nWorker {worker_id} (skill={pool.workers[worker_id].skill_level:.2f}):")
        logger.info(f"Task: {subtask['description']}")
        
        output = pool.assign_task(worker_id, subtask)
        logger.info(f"Output: {output[:200]}...\n")
    
    pool.cleanup()


def test_failure_injection():
    """Test failure injection."""
    logger.info("=" * 60)
    logger.info("TEST 3: Failure Injection")
    logger.info("=" * 60)
    
    config = LlamaWorkerConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        max_tokens=256,
        temperature=0.7,
    )
    
    pool = WorkerPool(num_workers=1, config=config)
    
    # Low skill worker - more failures
    pool.workers[0].skill_level = 0.3
    
    subtask = {
        "description": "Implement a binary search algorithm",
        "context": "Should handle edge cases",
    }
    
    logger.info(f"Testing low-skill worker (skill=0.3)...")
    logger.info(f"Running 5 attempts to see failure injection:\n")
    
    for attempt in range(5):
        output = pool.assign_task(0, subtask)
        logger.info(f"Attempt {attempt + 1}: {output[:150]}...\n")
    
    pool.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Llama 2 workers")
    parser.add_argument(
        "--test",
        choices=["single", "pool", "failures", "all"],
        default="all",
        help="Which test to run",
    )
    
    args = parser.parse_args()
    
    try:
        if args.test in ["single", "all"]:
            test_single_worker()
        
        if args.test in ["pool", "all"]:
            test_worker_pool()
        
        if args.test in ["failures", "all"]:
            test_failure_injection()
        
        logger.info("\n✓ All tests completed successfully!")
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        sys.exit(1)
```

---

## STEP 7: HuggingFace Authentication

### 7.1 Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "read" access
3. Copy the token

### 7.2 Authenticate Locally

```bash
huggingface-cli login
# Paste your token when prompted
```

Or set environment variable:

```bash
export HF_TOKEN="your_token_here"
```

---

## STEP 8: Run Tests

### 8.1 Test Single Worker

```bash
cd manager-worker-env
python test_llm_workers.py --test single
```

### 8.2 Test Worker Pool

```bash
python test_llm_workers.py --test pool
```

### 8.3 Test Failure Injection

```bash
python test_llm_workers.py --test failures
```

### 8.4 Run All Tests

```bash
python test_llm_workers.py --test all
```

---

## STEP 9: Optimize for Performance

### 9.1 Use Quantization (Recommended)

The code already includes 8-bit quantization. This reduces memory usage by 75%.

### 9.2 Use Flash Attention (Optional)

Requires `flash-attn` package:

```bash
pip install flash-attn
```

### 9.3 Batch Processing (Advanced)

For multiple workers, use batch generation:

```python
def batch_generate(self, prompts: List[str]) -> List[str]:
    """Generate outputs for multiple prompts in parallel."""
    inputs = self.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(self.device)
    
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
    
    return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
```

---

## STEP 10: Fine-Tuning (Optional - Post-Hackathon)

### 10.1 Prepare Training Data

```python
from datasets import Dataset

training_data = [
    {
        "instruction": "Write HTML for a landing page",
        "output": "<html>...</html>"
    },
    # ... more examples
]

dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in training_data],
    "output": [d["output"] for d in training_data],
})
```

### 10.2 Fine-Tune with LoRA

```python
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./llama-worker-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```python
config = LlamaWorkerConfig(
    use_8bit=True,  # Enable 8-bit quantization
    max_tokens=256,  # Reduce max tokens
)
```

### Issue: Model Download Fails

**Solution:**
```bash
# Set cache directory
export HF_HOME=/path/to/cache

# Or download manually
huggingface-cli download meta-llama/Llama-2-7b-hf
```

### Issue: Slow Generation

**Solution:**
```python
config = LlamaWorkerConfig(
    use_flash_attention=True,  # Enable flash attention
    temperature=0.5,  # Lower temperature = faster
    top_p=0.9,  # Reduce diversity
)
```

---

## Next Steps

1. ✅ Update `pyproject.toml` with LLM dependencies
2. ✅ Create `agents/llm_worker.py`
3. ✅ Create `agents/worker_pool.py`
4. ✅ Integrate with environment
5. ✅ Create configuration files
6. ✅ Run tests
7. ⏭️ Integrate with training pipeline
8. ⏭️ Add to FastAPI backend
9. ⏭️ Update React dashboard
10. ⏭️ Deploy to HuggingFace Spaces

---

## Summary

You now have:
- ✅ Llama 2 7B worker agents
- ✅ Worker pool management
- ✅ Failure injection system
- ✅ Configuration management
- ✅ Test suite
- ✅ Performance optimizations

Ready to start implementing!
