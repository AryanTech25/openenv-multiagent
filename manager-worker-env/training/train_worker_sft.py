"""
Fine-tuning script for Worker Agents.
Optimized for Apple Silicon (M1/M2/M3) and CUDA.
"""

import os
import torch
from typing import Optional
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_worker(
    model_id: str,
    dataset_name: str,
    output_dir: str = "models/worker_finetuned",
    max_steps: int = 100,
    batch_size: int = 1, # Minimal batch size for 8GB RAM
    learning_rate: float = 2e-4,
    push_to_hub: bool = False,
    hf_token: Optional[str] = None,
):
    """
    Fine-tune a worker model. Multi-device compatible.
    """
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        use_4bit = True
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16 # MPS prefers float16 or float32
        use_4bit = False # bitsandbytes doesn't support MPS yet
    else:
        device = "cpu"
        dtype = torch.float32
        use_4bit = False

    logger.info(f"Starting training on {device}...")

    # Load Dataset
    if dataset_name.endswith(('.jsonl', '.json')):
        dataset = load_dataset('json', data_files=dataset_name, split='train')
    else:
        dataset = load_dataset(dataset_name, split='train')

    # Model Loading
    load_kwargs = {
        "token": hf_token,
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,
    }

    # Only use 4-bit if on CUDA
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    
    if device == "mps":
        model = model.to("mps")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA Configuration
    peft_config = LoraConfig(
        r=8, # Lower rank for 8GB RAM
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Training / SFT Configuration (Optimized for trl 1.2.0+)
    from trl import SFTConfig
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8 if device != "cuda" else 4,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="no",
        fp16=(device == "cuda"),
        push_to_hub=push_to_hub,
        hub_token=hf_token,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=256, # In newer trl versions, this is max_length
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    logger.info(f"✓ Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Worker Agents")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/worker_finetuned")
    parser.add_argument("--steps", type=int, default=50)
    
    args = parser.parse_args()
    train_worker(
        model_id=args.model,
        dataset_name=args.dataset,
        output_dir=args.output,
        max_steps=args.steps,
        hf_token=os.getenv("HF_TOKEN")
    )
