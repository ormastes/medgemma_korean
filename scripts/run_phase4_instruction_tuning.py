#!/usr/bin/env python3
"""
Phase 4: Korean Medical Instruction Tuning

Fine-tune the Korean-adapted model on medical instruction data.

Purpose:
- Train on Korean medical QA format
- Enable instruction-following capabilities
- Use KorMedMCQA and other instruction data
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
    """Callback to print training metrics at each logging step."""

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
# Use expanded model from Phase 3 Stage 7 (hybrid: identity layers + QLoRA)
BASE_MODEL_DIR = "models/final/korean_medgemma_expanded"
DATA_DIR = "data/processed"
OUTPUT_DIR = "models/instruction_tuned"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Base model: {BASE_MODEL_DIR}")
print(f"Output dir: {OUTPUT_DIR}")
print("\nNote: Using hybrid expanded model with +2 identity layers")

# Instruction tuning configuration
CONFIG = {
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 2048,
    "warmup_ratio": 0.03,
    # LoRA config for instruction tuning
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
}

print("\n" + "=" * 60)
print("Instruction Tuning Configuration")
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

# Ensure padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded!")
print(f"Vocab size: {len(tokenizer)}")
print_memory_usage()

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)
print("Model prepared for k-bit training")

# Apply LoRA for instruction tuning
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

print("\nLoRA applied for instruction tuning")
model.print_trainable_parameters()

# =============================================================================
# Load Instruction Data
# =============================================================================
print("\n" + "=" * 60)
print("Loading Instruction Data")
print("=" * 60)

instruction_data_path = f"{DATA_DIR}/korean_medical_instruction"

if os.path.exists(instruction_data_path):
    dataset = load_from_disk(instruction_data_path)
    print(f"Loaded instruction dataset: {dataset}")
else:
    print(f"Dataset not found at {instruction_data_path}")
    print("Run Phase 0 notebooks to prepare data.")
    sys.exit(1)

# Preview instruction data
print("\nSample instruction (first 500 chars):")
print(dataset["train"][0]["text"][:500])

# =============================================================================
# Training
# =============================================================================
print("\n" + "=" * 60)
print("Setting up Training")
print("=" * 60)

# SFT Config (combines TrainingArguments + SFT-specific settings)
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
    eval_strategy="epoch" if "validation" in dataset else "no",
    save_total_limit=2,
    load_best_model_at_end=True if "validation" in dataset else False,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
    gradient_checkpointing=False,  # Disabled for PEFT compatibility
    # SFT-specific settings
    max_length=CONFIG["max_seq_length"],
    dataset_text_field="text",
)

print("SFT Config configured")

# Create SFT Trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"] if "validation" in dataset else None,
    processing_class=tokenizer,
    callbacks=[MetricsCallback()],
)

print("SFT Trainer created")

# Train!
print("\n" + "=" * 60)
print("Starting Instruction Tuning")
print("=" * 60)
print_memory_usage()

trainer.train()

print("\nTraining complete!")
print_memory_usage()

# =============================================================================
# Test Generation
# =============================================================================
print("\n" + "=" * 60)
print("Testing Instruction-Tuned Model")
print("=" * 60)

test_prompts = [
    """<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
고혈압의 주요 증상과 치료법에 대해 설명해주세요.
<|im_end|>
<|im_start|>assistant
""",
    """<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
당뇨병 환자가 주의해야 할 식이요법은 무엇인가요?
<|im_end|>
<|im_start|>assistant
""",
]

for i, prompt in enumerate(test_prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print(f"\n--- Test {i+1} ---")
    print(response[:1000])  # Limit output length
    print()

# =============================================================================
# Save Model
# =============================================================================
print("\n" + "=" * 60)
print("Saving Model")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Instruction-tuned model saved to {OUTPUT_DIR}")

# Save training info
training_info = {
    "phase": "instruction_tuning",
    "base_model": BASE_MODEL_DIR,
    "config": CONFIG,
    "train_samples": len(dataset["train"]),
    "eval_samples": len(dataset["validation"]) if "validation" in dataset else 0,
}

with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print("Training info saved")

print("\n" + "=" * 60)
print("Phase 4 Complete: Instruction Tuning Done!")
print("=" * 60)
print(f"\nModel saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Run phase5_evaluation/01_evaluate_korean.ipynb")
print("  2. Run phase5_evaluation/02_evaluate_english.ipynb")
