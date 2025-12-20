#!/usr/bin/env python3
"""
Phase 4 Enhanced: Korean Medical Instruction Tuning with More Data

Train on enhanced Korean medical instruction dataset (15K+ examples)
to improve KorMedMCQA accuracy toward 90%.
"""

import sys
import os
sys.path.insert(0, '.')

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
import json


class MetricsCallback(TrainerCallback):
    """Callback to print training metrics."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            metrics = []
            if "loss" in logs:
                metrics.append(f"loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                metrics.append(f"eval_loss={logs['eval_loss']:.4f}")
            if "learning_rate" in logs:
                metrics.append(f"lr={logs['learning_rate']:.2e}")
            if metrics:
                print(f"[Step {step}] {', '.join(metrics)}")


# GPU setup
from config.gpu_utils import setup_gpu, print_memory_usage, clear_memory
device = setup_gpu("config/gpu_config.json")

print_memory_usage()

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL_DIR = "models/final/korean_medgemma_expanded"
DATA_DIR = "data/processed/korean_medical_instruction_enhanced"
OUTPUT_DIR = "models/instruction_tuned_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Base model: {BASE_MODEL_DIR}")
print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# Enhanced configuration for better accuracy
CONFIG = {
    "learning_rate": 1e-5,  # Lower LR for more stable training
    "num_epochs": 5,  # More epochs
    "batch_size": 1,
    "gradient_accumulation_steps": 16,  # Larger effective batch
    "max_seq_length": 2048,
    "warmup_ratio": 0.05,
    # LoRA config - higher rank for more capacity
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
}

print("\n" + "=" * 60)
print("Enhanced Instruction Tuning Configuration")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# =============================================================================
# Load Model
# =============================================================================
print("\n" + "=" * 60)
print("Loading Model")
print("=" * 60)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Korean MedGemma model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded!")
print(f"Vocab size: {len(tokenizer)}")
print_memory_usage()

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)
print("Model prepared for k-bit training")

# Apply LoRA with higher rank
lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

print("\nLoRA applied (higher rank for better accuracy)")
model.print_trainable_parameters()

# =============================================================================
# Load Enhanced Instruction Data
# =============================================================================
print("\n" + "=" * 60)
print("Loading Enhanced Instruction Data")
print("=" * 60)

dataset = load_from_disk(DATA_DIR)
print(f"Loaded dataset: {dataset}")
print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")

# Preview
print("\nSample instruction (first 500 chars):")
print(dataset["train"][0]["text"][:500])

# =============================================================================
# Training
# =============================================================================
print("\n" + "=" * 60)
print("Setting up Training")
print("=" * 60)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
    gradient_checkpointing=False,
    max_length=CONFIG["max_seq_length"],
    dataset_text_field="text",
)

print("SFT Config configured")

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    callbacks=[MetricsCallback()],
)

print("SFT Trainer created")

# Train!
print("\n" + "=" * 60)
print("Starting Enhanced Instruction Tuning")
print("=" * 60)
print_memory_usage()

trainer.train()

print("\nTraining complete!")
print_memory_usage()

# =============================================================================
# Save Model
# =============================================================================
print("\n" + "=" * 60)
print("Saving Model")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")

# Save training info
training_info = {
    "phase": "instruction_tuning_v2",
    "base_model": BASE_MODEL_DIR,
    "config": CONFIG,
    "train_samples": len(dataset["train"]),
    "eval_samples": len(dataset["validation"]),
    "data_improvement": "6.8x more data than v1",
}

with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print("Training info saved")

print("\n" + "=" * 60)
print("Phase 4 Enhanced Complete!")
print("=" * 60)
print(f"\nModel saved to: {OUTPUT_DIR}")
print("\nNext: Run evaluation to check accuracy improvement")
