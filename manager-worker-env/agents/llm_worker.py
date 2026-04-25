"""
LLM-based Worker Agent using Llama 2 7B from HuggingFace.
"""

import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import random

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
        
        # Load model (try flash attention on CUDA; fall back to "eager" if flash-attn is missing)
        attn = "eager"
        if self.config.use_flash_attention and self.device == "cuda":
            attn = "flash_attention_2"
        load_kw = dict(
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation=attn,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **load_kw
            )
        except Exception as e:  # noqa: BLE001 — broad so missing flash deps still work
            if attn == "flash_attention_2":
                logger.warning(
                    "Flash attention 2 unavailable (%s); loading with eager attention.",
                    e,
                )
                load_kw["attn_implementation"] = "eager"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name, **load_kw
                )
            else:
                raise
        
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
        
        return random.choice(off_task_outputs)
    
    def _inject_incomplete(self, output: str) -> str:
        """
        Inject incomplete: output is partial.
        """
        # Truncate output to 30-70% of original length
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
