#!/usr/bin/env python3
"""
Stage 3: Train BOTH new input and output embeddings
This script trains both new input and output embeddings jointly while keeping
all other parameters and original embeddings frozen.
"""

import sys
import os
sys.path.insert(0, '.')

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)


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


from datasets import load_from_disk
import json
import shutil

# GPU setup
from config.gpu_utils import setup_gpu, print_memory_usage, clear_memory
device = setup_gpu("config/gpu_config.json")

print_memory_usage()

# Directories
STAGE2_MODEL_DIR = "models/staged_training/stage2_new_output_embeds"
DATA_DIR = "data/processed"
OUTPUT_DIR = "models/staged_training/stage3_both_new_embeds"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Input model: {STAGE2_MODEL_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# Load token mapping
mapping_path = f"{STAGE2_MODEL_DIR}/token_mapping.json"
with open(mapping_path, "r", encoding="utf-8") as f:
    token_mapping = json.load(f)

original_vocab_size = token_mapping["original_vocab_size"]
new_vocab_size = token_mapping["new_vocab_size"]

print(f"\nOriginal vocab: {original_vocab_size}")
print(f"New vocab: {new_vocab_size}")
print(f"New tokens: {new_vocab_size - original_vocab_size}")

# Load model from Stage 2
print("\nLoading model from Stage 2...")
model = AutoModelForCausalLM.from_pretrained(
    STAGE2_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(STAGE2_MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded!")
print_memory_usage()

# Get actual vocab size from model
if hasattr(model.config, 'text_config'):
    actual_new_vocab = model.config.text_config.vocab_size
    actual_original_vocab = actual_new_vocab - (new_vocab_size - original_vocab_size)
else:
    actual_original_vocab = original_vocab_size
    actual_new_vocab = new_vocab_size

print(f"Actual original vocab: {actual_original_vocab}")
print(f"Actual new vocab: {actual_new_vocab}")

# Freeze ALL parameters first
for param in model.parameters():
    param.requires_grad = False
print("Froze all parameters")

# Enable BOTH input and output embeddings
input_embeddings = model.get_input_embeddings()
output_embeddings = model.get_output_embeddings()

input_embeddings.weight.requires_grad = True
output_embeddings.weight.requires_grad = True

print(f"Enabled input embeddings training: {input_embeddings.weight.shape}")
print(f"Enabled output embeddings training: {output_embeddings.weight.shape}")

# Create hooks to freeze OLD embeddings during training
class FreezeOldEmbeddingsHook:
    """Hook to zero out gradients for original token embeddings"""

    def __init__(self, original_vocab_size):
        self.original_vocab_size = original_vocab_size

    def __call__(self, grad):
        grad[:self.original_vocab_size] = 0
        return grad

# Register hooks for both embeddings
freeze_hook_input = FreezeOldEmbeddingsHook(actual_original_vocab)
freeze_hook_output = FreezeOldEmbeddingsHook(actual_original_vocab)

hook_handle_input = input_embeddings.weight.register_hook(freeze_hook_input)
hook_handle_output = output_embeddings.weight.register_hook(freeze_hook_output)
print(f"Registered gradient hooks to freeze original {actual_original_vocab} embeddings")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
print(f"Percentage: {100 * trainable_params / total_params:.4f}%")

# Load dataset
lm_data_path = f"{DATA_DIR}/korean_medical_lm"
dataset = load_from_disk(lm_data_path)
print(f"\nDataset: {dataset}")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4,
)
print(f"Tokenized dataset: {tokenized_dataset}")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments - lower LR for joint training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,  # Lower LR for joint training
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    optim="adamw_torch",
    max_grad_norm=1.0,
    report_to="none",
    dataloader_num_workers=2,
    eval_strategy="steps",
    eval_steps=500,
)

# Create trainer with eval dataset and metrics callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    callbacks=[MetricsCallback()],
)

print("\n" + "=" * 60)
print("Starting Stage 3 Training: Both New Embeddings")
print("=" * 60)
print_memory_usage()

trainer.train()

print("\nTraining complete!")
print_memory_usage()

# Remove hooks before saving
hook_handle_input.remove()
hook_handle_output.remove()
print("Removed gradient hooks")

# Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")

# Save stage info
stage_info = {
    "stage": 3,
    "name": "stage3_both_new_embeds",
    "description": "Train both NEW input and output embeddings",
    "trainable_params": trainable_params,
    "total_params": total_params,
    "original_vocab_size": actual_original_vocab,
    "new_vocab_size": actual_new_vocab,
    "previous_stage": STAGE2_MODEL_DIR,
}

with open(f"{OUTPUT_DIR}/stage_info.json", "w") as f:
    json.dump(stage_info, f, indent=2)

# Copy token mapping
shutil.copy(f"{STAGE2_MODEL_DIR}/token_mapping.json", f"{OUTPUT_DIR}/token_mapping.json")

print("\n" + "=" * 60)
print("Stage 3 Complete!")
print("=" * 60)
print(f"\nCheckpoint saved to: {OUTPUT_DIR}")
print("Next: Run Stage 4 (train all output embeddings)")
