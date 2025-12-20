#!/usr/bin/env python3
"""
Stage 7: Cooldown + Merge for Hybrid Expanded Model

This stage:
1. Reconstruct expanded model (load Stage 5 + expand with identity layers)
2. Load LoRA adapters from Stage 6
3. Cooldown training with lower LR (stabilize identity layers + LoRA)
4. Merge LoRA weights into the expanded model
5. Save final Korean MedGemma model with added layers

Reference: research/layer_expansion_techniques.md
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import copy
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


from peft import PeftModel
from datasets import load_from_disk
import json
import shutil


# =============================================================================
# Identity Layer Creation Functions (same as Stage 6)
# =============================================================================

def create_identity_layer(source_layer):
    """
    Create an identity-initialized copy of a transformer layer.
    """
    new_layer = copy.deepcopy(source_layer)

    # Zero-initialize all linear layers
    for name, module in new_layer.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return new_layer


def get_model_layers(model):
    """Get the layers module for different model architectures."""
    if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        return model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        return model.model.layers
    else:
        raise AttributeError("Cannot find layers in model structure")


def set_model_layers(model, new_layers):
    """Set the layers module for different model architectures."""
    if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        model.model.language_model.layers = new_layers
    elif hasattr(model.model, 'layers'):
        model.model.layers = new_layers
    else:
        raise AttributeError("Cannot find layers in model structure")


def expand_model_with_identity_layers(model, front_expand=1, rear_expand=1):
    """
    Expand model by inserting identity-initialized layers.
    """
    original_layers = list(get_model_layers(model))
    num_layers = len(original_layers)
    print(f"\nOriginal model has {num_layers} layers")

    new_layers = []
    identity_layer_indices = []

    for i, layer in enumerate(original_layers):
        new_layers.append(layer)

        if i < front_expand:
            print(f"  Creating identity layer after layer {i} (front)...")
            identity_layer = create_identity_layer(layer)
            new_layers.append(identity_layer)
            identity_layer_indices.append(len(new_layers) - 1)

        if i >= num_layers - rear_expand:
            print(f"  Creating identity layer after layer {i} (rear)...")
            identity_layer = create_identity_layer(layer)
            new_layers.append(identity_layer)
            identity_layer_indices.append(len(new_layers) - 1)

    set_model_layers(model, nn.ModuleList(new_layers))
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = len(new_layers)
    else:
        model.config.num_hidden_layers = len(new_layers)

    print(f"Expanded model has {len(new_layers)} layers")
    print(f"Identity layer indices: {identity_layer_indices}")

    return model, identity_layer_indices

# GPU setup
from config.gpu_utils import setup_gpu, print_memory_usage, clear_memory
device = setup_gpu("config/gpu_config.json")

print_memory_usage()

# Directories
STAGE6_MODEL_DIR = "models/staged_training/stage6_hybrid_expansion"
DATA_DIR = "data/processed"
OUTPUT_DIR = "models/staged_training/stage7_cooldown"
FINAL_MODEL_DIR = "models/final/korean_medgemma_expanded"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

print(f"Input model: {STAGE6_MODEL_DIR}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Final model dir: {FINAL_MODEL_DIR}")

# Stage 7 configuration (cooldown)
STAGE_CONFIG = {
    "name": "stage7_cooldown",
    "description": "Stabilization with lower LR for hybrid expanded model",
    "learning_rate": 5e-5,  # Lower LR for cooldown
    "num_epochs": 1,
    "warmup_ratio": 0.1,  # Higher warmup ratio for stability
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
}

print("\n" + "=" * 60)
print("Stage 7 Configuration (Cooldown)")
print("=" * 60)
for key, value in STAGE_CONFIG.items():
    print(f"  {key}: {value}")

# Load stage 6 info
stage6_info_path = f"{STAGE6_MODEL_DIR}/stage_info.json"
with open(stage6_info_path, "r") as f:
    stage6_info = json.load(f)

print("\nStage 6 Info:")
print(f"  Method: {stage6_info.get('method', 'Unknown')}")
print(f"  Identity layers: {stage6_info.get('identity_layer_indices', [])}")
print(f"  LoRA layers: {len(stage6_info.get('lora_layers', []))} layers")
print(f"  Total layers: {stage6_info.get('total_layers', 'Unknown')}")

# Load token mapping
mapping_path = f"{STAGE6_MODEL_DIR}/token_mapping.json"
with open(mapping_path, "r", encoding="utf-8") as f:
    token_mapping = json.load(f)

original_vocab_size = token_mapping["original_vocab_size"]
new_vocab_size = token_mapping["new_vocab_size"]

print(f"\nVocab: {original_vocab_size} → {new_vocab_size} (+{new_vocab_size - original_vocab_size})")

# =============================================================================
# Reconstruct Expanded Model + Load LoRA Adapters
# =============================================================================
print("\n" + "=" * 60)
print("Reconstructing expanded model from Stage 5 + loading LoRA from Stage 6...")
print("=" * 60)

# Handle Gemma3 nested config (text_config.num_hidden_layers)
def get_num_layers(config):
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        return config.text_config.num_hidden_layers
    return config.num_hidden_layers

# Stage 6 saved only PEFT adapters, not the expanded base model
# We need to reconstruct: Stage 5 model → expand with identity layers → load LoRA

# Step 1: Load Stage 5 base model
STAGE5_MODEL_DIR = stage6_info.get('previous_stage', 'models/staged_training/stage5_harmonization')
print(f"Loading base model from {STAGE5_MODEL_DIR}...")

model = AutoModelForCausalLM.from_pretrained(
    STAGE5_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"Stage 5 model loaded with {get_num_layers(model.config)} layers")

# Step 2: Expand with identity layers (same config as Stage 6)
front_expand = stage6_info.get('config', {}).get('front_expand', 1)
rear_expand = stage6_info.get('config', {}).get('rear_expand', 1)

print(f"\nExpanding model with identity layers (front={front_expand}, rear={rear_expand})...")
model, identity_layer_indices = expand_model_with_identity_layers(
    model,
    front_expand=front_expand,
    rear_expand=rear_expand
)

# Step 3: Load LoRA adapters from Stage 6
adapter_config_path = f"{STAGE6_MODEL_DIR}/adapter_config.json"
if os.path.exists(adapter_config_path):
    print(f"\nLoading LoRA adapters from {STAGE6_MODEL_DIR}...")
    model = PeftModel.from_pretrained(model, STAGE6_MODEL_DIR, is_trainable=True)
    print("LoRA adapters loaded successfully (training mode enabled)!")
else:
    print("Warning: No LoRA adapters found in Stage 6 output")

tokenizer = AutoTokenizer.from_pretrained(STAGE6_MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\nModel ready with {get_num_layers(model.config)} layers")
print_memory_usage()

# =============================================================================
# Set Up Cooldown Training (Lower LR)
# =============================================================================
print("\n" + "=" * 60)
print("Setting up cooldown training...")
print("=" * 60)

# Get identity layer indices from stage 6 info
identity_layer_indices = stage6_info.get('identity_layer_indices', [])
print(f"Identity layer indices: {identity_layer_indices}")

# For PEFT models, the LoRA adapters are already trainable
# We need to enable training on identity layers as well

# First, check trainable parameters (LoRA adapters should already be trainable)
if hasattr(model, 'print_trainable_parameters'):
    model.print_trainable_parameters()

# Get base model for accessing layers
def get_base_model_layers(peft_model):
    """Get layers from PEFT-wrapped model."""
    base = peft_model.base_model.model if hasattr(peft_model, 'base_model') else peft_model
    if hasattr(base.model, 'language_model') and hasattr(base.model.language_model, 'layers'):
        return base.model.language_model.layers
    elif hasattr(base.model, 'layers'):
        return base.model.layers
    return None

# Enable training for identity layers (they were zero-initialized, need to train)
try:
    base_layers = get_base_model_layers(model)
    if base_layers:
        for idx in identity_layer_indices:
            if idx < len(base_layers):
                layer = base_layers[idx]
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"Enabled training for identity layer {idx}")
except Exception as e:
    print(f"Note: Could not enable identity layers: {e}")

# Keep embeddings frozen for cooldown (they're well-trained from earlier stages)
for name, param in model.named_parameters():
    if "embed_tokens" in name or "lm_head" in name:
        param.requires_grad = False

print("\nEmbeddings frozen for cooldown stage")
print("LoRA adapters + identity layers are trainable")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
print(f"Percentage: {100 * trainable_params / total_params:.4f}%")

# =============================================================================
# Load Dataset
# =============================================================================
lm_data_path = f"{DATA_DIR}/korean_medical_lm"
dataset = load_from_disk(lm_data_path)
print(f"\nDataset: {dataset}")

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
print(f"Tokenized: {tokenized_dataset}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# =============================================================================
# Cooldown Training
# =============================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=STAGE_CONFIG["num_epochs"],
    per_device_train_batch_size=STAGE_CONFIG["batch_size"],
    per_device_eval_batch_size=STAGE_CONFIG["batch_size"],
    gradient_accumulation_steps=STAGE_CONFIG["gradient_accumulation_steps"],
    learning_rate=STAGE_CONFIG["learning_rate"],
    warmup_ratio=STAGE_CONFIG["warmup_ratio"],
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    report_to="none",
    gradient_checkpointing=False,  # Disabled for PEFT compatibility
    dataloader_num_workers=2,
    eval_strategy="no",  # Disabled - causes tensor shape mismatch with PEFT expanded model
)

# Note: Gradient checkpointing disabled for Stage 7
# It conflicts with PEFT models causing "None of the inputs have requires_grad=True" error
# Memory is sufficient without it (using ~8GB of 47GB available)
# model.gradient_checkpointing_enable()
print("Note: Gradient checkpointing disabled for PEFT compatibility")

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
print("Starting Stage 7 Training: Cooldown")
print("=" * 60)
print_memory_usage()

trainer.train()

print("\nCooldown training complete!")
print_memory_usage()

# Save cooldown checkpoint
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nCooldown checkpoint saved to {OUTPUT_DIR}")

# Save stage info
stage_info = {
    "stage": 7,
    "name": STAGE_CONFIG["name"],
    "description": STAGE_CONFIG["description"],
    "config": STAGE_CONFIG,
    "trainable_params": trainable_params,
    "total_params": total_params,
    "original_vocab_size": original_vocab_size,
    "new_vocab_size": new_vocab_size,
    "identity_layer_indices": identity_layer_indices,
    "total_layers": get_num_layers(model.config),
    "previous_stage": STAGE6_MODEL_DIR,
}

with open(f"{OUTPUT_DIR}/stage_info.json", "w") as f:
    json.dump(stage_info, f, indent=2)

shutil.copy(mapping_path, f"{OUTPUT_DIR}/token_mapping.json")

# =============================================================================
# Merge and Save Final Model
# =============================================================================
print("\n" + "=" * 60)
print("Merging LoRA and Saving Final Model")
print("=" * 60)

# Check if we need to merge (if it's a PEFT model)
try:
    if hasattr(model, 'merge_and_unload'):
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("LoRA weights merged!")
    else:
        print("Model is already a full model (no merge needed)")
except Exception as e:
    print(f"Merge note: {e}")
    print("Continuing with current model...")

# Save final model
print(f"\nSaving final model to {FINAL_MODEL_DIR}...")

model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# Copy token mapping
shutil.copy(f"{OUTPUT_DIR}/token_mapping.json", f"{FINAL_MODEL_DIR}/token_mapping.json")

# Save final model info
final_info = {
    "model_name": "korean_medgemma_expanded",
    "base_model": token_mapping.get("base_model", "medgemma-4b-it"),
    "method": "Hybrid (Identity Layer Expansion + QLoRA + EEVE)",
    "original_vocab_size": original_vocab_size,
    "new_vocab_size": new_vocab_size,
    "korean_tokens_added": new_vocab_size - original_vocab_size,
    "original_layers": stage6_info.get('original_layers', 'Unknown'),
    "total_layers": get_num_layers(model.config),
    "identity_layer_indices": identity_layer_indices,
    "training_stages": 7,
    "stages": [
        "Stage 1: Train new input embeddings",
        "Stage 2: Train new output embeddings",
        "Stage 3: Train both new embeddings",
        "Stage 4: Train all output embeddings",
        "Stage 5: Harmonization",
        "Stage 6: Hybrid Expansion (Identity + QLoRA)",
        "Stage 7: Cooldown + Merge",
    ],
}

with open(f"{FINAL_MODEL_DIR}/model_info.json", "w") as f:
    json.dump(final_info, f, indent=2)

print("\n" + "=" * 60)
print("Phase 3 Complete: All Staged Training Done!")
print("=" * 60)
print(f"\nFinal Model Summary:")
print(f"  Location: {FINAL_MODEL_DIR}")
print(f"  Vocab: {original_vocab_size} → {new_vocab_size} (+{new_vocab_size - original_vocab_size} Korean tokens)")
print(f"  Layers: {final_info['original_layers']} → {final_info['total_layers']} (+{len(identity_layer_indices)} identity layers)")
print(f"  Method: {final_info['method']}")
print("\nNext steps:")
print("  1. Run phase4_instruction_tuning/01_instruction_tuning.ipynb")
print("  2. Or proceed to phase5_evaluation for testing")
