"""
Generic Hugging Face Worker Agent for multi-agent coordination.
"""

import os
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import InferenceClient
import logging
import random

logger = logging.getLogger(__name__)


def _is_offline() -> bool:
    """True if the user / env wants us to use local cache only."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        v = os.environ.get(var, "")
        if v and v not in ("0", "false", "False"):
            return True
    return False


class HFWorkerConfig:
    """Configuration for Hugging Face Worker."""
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_8bit: bool = True,
        use_flash_attention: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        hf_token: Optional[str] = None,
        worker_type: str = "local", # "local" or "api"
    ):
        self.model_id = model_id
        self.device = device
        self.use_8bit = use_8bit
        self.use_flash_attention = use_flash_attention
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.hf_token = hf_token
        self.worker_type = worker_type


# For backward compatibility
LlamaWorkerConfig = HFWorkerConfig


class HFWorker:
    """
    Worker Agent powered by any Hugging Face Causal LM.
    
    This worker generates task outputs using a model from the Hugging Face Hub.
    Failure modes are injected post-generation based on skill level.
    """
    
    def __init__(
        self,
        worker_id: int,
        skill_level: float,
        config: Optional[HFWorkerConfig] = None,
    ):
        """
        Initialize HF Worker.
        
        Args:
            worker_id: Unique worker identifier
            skill_level: Worker competence (0.0-1.0)
            config: HFWorkerConfig instance
        """
        self.worker_id = worker_id
        self.skill_level = skill_level
        self.config = config or HFWorkerConfig()
        
        self.model = None
        self.tokenizer = None
        self.device = self.config.device
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model and tokenizer from Hugging Face.

        We auto-fall-back to ``local_files_only=True`` whenever the network
        round-trip fails (e.g. the HF Hub returns 401 when looking up
        ``additional_chat_templates`` for an unauthenticated user). This keeps
        the demo working with cached weights even without an HF token.
        """
        logger.info(f"Loading {self.config.model_id} on {self.device}...")

        offline_pref = _is_offline() or not self.config.hf_token

        # Configure 8-bit quantization if enabled
        quantization_config = None
        if self.config.use_8bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

        self.tokenizer = self._load_with_offline_fallback(
            AutoTokenizer.from_pretrained,
            self.config.model_id,
            kwargs=dict(token=self.config.hf_token, trust_remote_code=True),
            prefer_offline=offline_pref,
            label="tokenizer",
        )
        if self.tokenizer.pad_token is None:
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
            token=self.config.hf_token,
        )

        try:
            self.model = self._load_with_offline_fallback(
                AutoModelForCausalLM.from_pretrained,
                self.config.model_id,
                kwargs=load_kw,
                prefer_offline=offline_pref,
                label="model",
            )
        except Exception as e:
            if attn == "flash_attention_2":
                logger.warning(
                    "Flash attention 2 unavailable (%s); loading with eager attention.",
                    e,
                )
                load_kw["attn_implementation"] = "eager"
                self.model = self._load_with_offline_fallback(
                    AutoModelForCausalLM.from_pretrained,
                    self.config.model_id,
                    kwargs=load_kw,
                    prefer_offline=offline_pref,
                    label="model",
                )
            else:
                raise

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info(f"✓ Model {self.config.model_id} loaded successfully on {self.device}")

    @staticmethod
    def _load_with_offline_fallback(loader, model_id: str, *, kwargs: Dict[str, Any], prefer_offline: bool, label: str):
        """
        Call a transformers/HF loader, retrying with ``local_files_only=True``
        if the online attempt fails on a network/auth error. If we already
        prefer offline (e.g. HF_HUB_OFFLINE set, or no token available) we go
        straight to the local-cache attempt first.
        """
        attempts = []
        if prefer_offline:
            attempts.append({**kwargs, "local_files_only": True})
            attempts.append(kwargs)
        else:
            attempts.append(kwargs)
            attempts.append({**kwargs, "local_files_only": True})

        last_exc: Optional[Exception] = None
        for i, kw in enumerate(attempts):
            try:
                return loader(model_id, **kw)
            except Exception as exc:  # noqa: BLE001 — broad on purpose
                last_exc = exc
                if i + 1 < len(attempts):
                    logger.warning(
                        "Loading %s for %s failed (%s); retrying with local_files_only=%s",
                        label,
                        model_id,
                        exc.__class__.__name__,
                        attempts[i + 1].get("local_files_only", False),
                    )
        # All attempts failed.
        raise last_exc  # type: ignore[misc]
    
    def work_on_task(self, subtask: Dict[str, Any]) -> str:
        """
        Generate output for a subtask.
        
        Args:
            subtask: Dictionary with 'description' and 'context' keys
            
        Returns:
            Generated output string
        """
        if isinstance(subtask, dict):
            description = subtask.get("description") or subtask.get("task") or str(subtask)
            context = subtask.get("context")
        else:
            description = str(subtask)
            context = None

        prompt = self._build_prompt(description, context)
        output = self._generate_output(prompt)
        output = self._inject_failures(output, subtask)
        return output
    
    def _build_prompt(self, task_description: str, context: Optional[str] = None) -> str:
        """
        Build a high-quality instruction prompt.
        """
        skill_description = (
            "expert-level, professional, and accurate" if self.skill_level > 0.8 else
            "reliable and competent" if self.skill_level > 0.5 else
            "basic and preliminary"
        )
        
        prompt = f"### System: You are an AI worker agent with {skill_description} proficiency.\n"
        prompt += f"### Instruction: Please complete the following task to the best of your ability.\n\n"
        prompt += f"TASK:\n{task_description}\n\n"
        
        if context:
            prompt += f"CONTEXT & CONSTRAINTS:\n{context}\n\n"
            
        prompt += "### Output:\n"
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
        """Inject hallucination: output looks good but has subtle errors."""
        hallucinations = [
            lambda x: x + "\n<!-- TODO: Fix styling -->",
            lambda x: x.replace("function", "func"),
            lambda x: x[:len(x)//2] + "\n[... rest of output ...]",
            lambda x: x + "\n\nNote: This might need review",
        ]
        
        corrupted = random.choice(hallucinations)(output)
        return corrupted
    
    def _inject_off_task(self, output: str) -> str:
        """Inject off-task: output is unrelated to the task."""
        off_task_outputs = [
            "I'm not sure how to approach this task. Let me think about something else.",
            "This reminds me of a different problem I solved before...",
            "I'll just generate some random content instead.",
            "Here's what I think about this topic in general...",
        ]
        
        return random.choice(off_task_outputs)
    
    def _inject_incomplete(self, output: str) -> str:
        """Inject incomplete: output is partial."""
        truncation_ratio = random.uniform(0.3, 0.7)
        truncated_length = int(len(output) * truncation_ratio)
        return output[:truncated_length] + "\n[Output incomplete - worker ran out of time]"
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"✓ Worker {self.worker_id} cleaned up")


class HFAPIWorker(HFWorker):
    """
    Worker Agent that uses the Hugging Face Inference API.
    Zero local memory footprint.
    """
    
    def __init__(
        self,
        worker_id: int,
        skill_level: float,
        config: Optional[HFWorkerConfig] = None,
    ):
        self.worker_id = worker_id
        self.skill_level = skill_level
        self.config = config or HFWorkerConfig()
        self.client = InferenceClient(token=self.config.hf_token)
        logger.info(f"✓ API Worker {self.worker_id} initialized (Model: {self.config.model_id})")

    def _load_model(self) -> None:
        """No local loading needed for API worker."""
        pass

    def _generate_output(self, prompt: str) -> str:
        """Generate output via HF Inference API with Chat fallback."""
        try:
            # Try standard text generation first
            response = self.client.text_generation(
                prompt,
                model=self.config.model_id,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )
            return response.strip()
        except Exception as e:
            # If text-generation isn't supported, try conversational (Chat) API
            if "conversational" in str(e).lower() or "chat" in str(e).lower():
                try:
                    chat_response = self.client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.config.model_id,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                    return chat_response.choices[0].message.content.strip()
                except Exception as chat_e:
                    logger.error(f"Chat API Error for worker {self.worker_id}: {chat_e}")
                    return f"[Error: Chat API failed - {chat_e}]"
            
            logger.error(f"API Error for worker {self.worker_id}: {e}")
            return f"[Error: API failed - {e}]"

    def cleanup(self) -> None:
        """No local resources to clean up."""
        logger.info(f"✓ API Worker {self.worker_id} cleaned up")


# For backward compatibility
LlamaWorker = HFWorker
