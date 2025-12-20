#!/usr/bin/env python3
"""
Stage 6: Hybrid Expansion (Identity Layers + QLoRA)

This stage implements the recommended hybrid approach:
1. Identity Layer Expansion: Add 2 new layers (front + rear) with zero-init
2. QLoRA on Middle Layers: Light adaptation with r=32
3. Full Embeddings Training: For Korean vocabulary

Reference: research/layer_expansion_techniques.md
- LLaMA Pro: Block Expansion with Identity Initialization
- SOLAR: Depth Up-Scaling concepts
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
    BitsAndBytesConfig,
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


from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_from_disk
import json
import shutil

# GPU setup
from config.gpu_utils import setup_gpu, print_memory_usage, clear_memory
device = setup_gpu("config/gpu_config.json")

print_memory_usage()

# Directories
STAGE5_MODEL_DIR = "models/staged_training/stage5_harmonization"
DATA_DIR = "data/processed"
OUTPUT_DIR = "models/staged_training/stage6_hybrid_expansion"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Input model: {STAGE5_MODEL_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# =============================================================================
# Stage 6 Hybrid Configuration
# =============================================================================
STAGE_CONFIG = {
    "name": "stage6_hybrid_expansion",
    "description": "Identity Layer Expansion + QLoRA + Full Embeddings",

    # Identity Layer Expansion
    "front_expand": 1,  # Add 1 identity layer after L1
    "rear_expand": 1,   # Add 1 identity layer after last layer

    # QLoRA configuration (lighter than before)
    "lora_r": 32,       # Reduced from 64 (identity layers add capacity)
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_layer_range": (4, 28),  # Apply LoRA only to middle layers

    # Training
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "warmup_ratio": 0.03,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
}

print("\n" + "=" * 60)
print("Stage 6 Hybrid Configuration")
print("=" * 60)
for key, value in STAGE_CONFIG.items():
    print(f"  {key}: {value}")

# =============================================================================
# Identity Layer Creation Functions
# =============================================================================

def create_identity_layer(source_layer):
    """
    Create an identity-initialized copy of a transformer layer.

    Identity initialization:
    - Zero-initialize all linear projections
    - Due to residual connection: output = input + layer(input)
    - If layer(input) = 0, then output = input (identity mapping)

    This allows inserting new layers without disrupting model behavior.
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
    # Gemma3 multimodal: model.model.language_model.layers
    if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        return model.model.language_model.layers
    # Standard: model.model.layers
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

    Based on LLaMA Pro block expansion approach.

    Args:
        model: Base model to expand
        front_expand: Number of identity layers to add after front layers
        rear_expand: Number of identity layers to add after rear layers

    Returns:
        Expanded model and list of identity layer indices
    """
    original_layers = list(get_model_layers(model))
    num_layers = len(original_layers)
    print(f"\nOriginal model has {num_layers} layers")

    new_layers = []
    identity_layer_indices = []

    for i, layer in enumerate(original_layers):
        new_layers.append(layer)

        # Add identity layer after first `front_expand` layers
        if i < front_expand:
            print(f"  Creating identity layer after layer {i} (front)...")
            identity_layer = create_identity_layer(layer)
            new_layers.append(identity_layer)
            identity_layer_indices.append(len(new_layers) - 1)

        # Add identity layer after last `rear_expand` layers
        if i >= num_layers - rear_expand:
            print(f"  Creating identity layer after layer {i} (rear)...")
            identity_layer = create_identity_layer(layer)
            new_layers.append(identity_layer)
            identity_layer_indices.append(len(new_layers) - 1)

    set_model_layers(model, nn.ModuleList(new_layers))
    # Update config - handle Gemma3 nested config
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = len(new_layers)
    else:
        model.config.num_hidden_layers = len(new_layers)

    print(f"Expanded model has {len(new_layers)} layers")
    print(f"Identity layer indices: {identity_layer_indices}")

    return model, identity_layer_indices

# =============================================================================
# Load Token Mapping
# =============================================================================
mapping_path = f"{STAGE5_MODEL_DIR}/token_mapping.json"
with open(mapping_path, "r", encoding="utf-8") as f:
    token_mapping = json.load(f)

original_vocab_size = token_mapping["original_vocab_size"]
new_vocab_size = token_mapping["new_vocab_size"]

print(f"\nOriginal vocab: {original_vocab_size}")
print(f"New vocab: {new_vocab_size}")
print(f"New tokens: {new_vocab_size - original_vocab_size}")

# =============================================================================
# Load Model (Full Precision for Expansion)
# =============================================================================
print("\n" + "=" * 60)
print("Loading model from Stage 5...")
print("=" * 60)

# Load in full precision first for layer expansion
model = AutoModelForCausalLM.from_pretrained(
    STAGE5_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(STAGE5_MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Handle Gemma3 nested config (text_config.num_hidden_layers)
def get_num_layers(config):
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        return config.text_config.num_hidden_layers
    return config.num_hidden_layers

def set_num_layers(config, num_layers):
    if hasattr(config, 'text_config'):
        config.text_config.num_hidden_layers = num_layers
    else:
        config.num_hidden_layers = num_layers

print(f"Model loaded with {get_num_layers(model.config)} layers")
print_memory_usage()

# =============================================================================
# Expand Model with Identity Layers
# =============================================================================
print("\n" + "=" * 60)
print("Expanding model with identity layers...")
print("=" * 60)

model, identity_layer_indices = expand_model_with_identity_layers(
    model,
    front_expand=STAGE_CONFIG["front_expand"],
    rear_expand=STAGE_CONFIG["rear_expand"]
)

print_memory_usage()

# =============================================================================
# Freeze All, Then Selectively Unfreeze
# =============================================================================
print("\n" + "=" * 60)
print("Setting up trainable parameters...")
print("=" * 60)

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False
print("Froze all parameters")

# Unfreeze identity layers (fully trainable)
layers = get_model_layers(model)
for idx in identity_layer_indices:
    layer = layers[idx]
    for param in layer.parameters():
        param.requires_grad = True
    print(f"Unfroze layer {idx} (identity layer - full training)")

# Unfreeze embeddings (for Korean tokens) - handle Gemma3 structure
if hasattr(model.model, 'language_model'):
    # Gemma3 multimodal
    model.model.language_model.embed_tokens.weight.requires_grad = True
else:
    model.model.embed_tokens.weight.requires_grad = True
model.lm_head.weight.requires_grad = True
print("Unfroze embeddings and lm_head")

# =============================================================================
# Apply QLoRA to Middle Layers
# =============================================================================
print("\n" + "=" * 60)
print("Applying QLoRA to middle layers...")
print("=" * 60)

# Determine which layers to apply LoRA (skip identity layers)
total_layers = len(get_model_layers(model))
lora_start, lora_end = STAGE_CONFIG["lora_layer_range"]

layers_to_transform = []
for i in range(total_layers):
    # Skip identity layers
    if i in identity_layer_indices:
        continue
    # Only include middle layers
    if lora_start <= i <= lora_end:
        layers_to_transform.append(i)

print(f"Applying LoRA to layers: {layers_to_transform}")

# Prepare for k-bit training (enables gradient computation on quantized layers)
model = prepare_model_for_kbit_training(model)

# LoRA configuration with layer targeting
lora_config = LoraConfig(
    r=STAGE_CONFIG["lora_r"],
    lora_alpha=STAGE_CONFIG["lora_alpha"],
    target_modules=STAGE_CONFIG["lora_target_modules"],
    lora_dropout=STAGE_CONFIG["lora_dropout"],
    layers_to_transform=layers_to_transform,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

print(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
print(f"Target modules: {lora_config.target_modules}")

# Apply LoRA
model = get_peft_model(model, lora_config)
print("\nPEFT model created")
model.print_trainable_parameters()

# Re-enable training for identity layers and embeddings after PEFT wrapping
# (PEFT might have changed the model structure)
def get_peft_model_layers(peft_model):
    """Get layers from PEFT-wrapped model."""
    base = peft_model.base_model.model
    # Gemma3 multimodal
    if hasattr(base.model, 'language_model') and hasattr(base.model.language_model, 'layers'):
        return base.model.language_model.layers
    # Standard
    elif hasattr(base.model, 'layers'):
        return base.model.layers
    return None

try:
    peft_layers = get_peft_model_layers(model)
    if peft_layers:
        for idx in identity_layer_indices:
            layer = peft_layers[idx]
            for param in layer.parameters():
                param.requires_grad = True
except Exception as e:
    print(f"Note: Could not re-enable identity layers: {e}")

try:
    base = model.base_model.model
    if hasattr(base.model, 'language_model'):
        base.model.language_model.embed_tokens.weight.requires_grad = True
    else:
        base.model.embed_tokens.weight.requires_grad = True
    base.lm_head.weight.requires_grad = True
except Exception as e:
    print(f"Note: Could not re-enable embeddings: {e}")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"\nTotal trainable parameters: {trainable_params:,} / {total_params:,}")
print(f"Percentage: {100 * trainable_params / total_params:.4f}%")
print_memory_usage()

# =============================================================================
# Load Dataset
# =============================================================================
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

# =============================================================================
# Training
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
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    eval_strategy="steps",
    eval_steps=500,
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
print("Gradient checkpointing enabled")

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
print("Starting Stage 6 Training: Hybrid Expansion")
print("  - Identity layers: Full training")
print("  - Middle layers: QLoRA adaptation")
print("  - Embeddings: Full training")
print("=" * 60)
print_memory_usage()

trainer.train()

print("\nTraining complete!")
print_memory_usage()

# =============================================================================
# Save Model
# =============================================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to {OUTPUT_DIR}")

# Save stage info
stage_info = {
    "stage": 6,
    "name": STAGE_CONFIG["name"],
    "description": STAGE_CONFIG["description"],
    "method": "Hybrid (Identity Expansion + QLoRA)",
    "config": STAGE_CONFIG,
    "identity_layer_indices": identity_layer_indices,
    "lora_layers": layers_to_transform,
    "original_layers": get_num_layers(model.config) - STAGE_CONFIG["front_expand"] - STAGE_CONFIG["rear_expand"],
    "total_layers": get_num_layers(model.config),
    "trainable_params": trainable_params,
    "total_params": total_params,
    "original_vocab_size": original_vocab_size,
    "new_vocab_size": new_vocab_size,
    "previous_stage": STAGE5_MODEL_DIR,
}

with open(f"{OUTPUT_DIR}/stage_info.json", "w") as f:
    json.dump(stage_info, f, indent=2)

# Copy token mapping
shutil.copy(f"{STAGE5_MODEL_DIR}/token_mapping.json", f"{OUTPUT_DIR}/token_mapping.json")

print("\n" + "=" * 60)
print("Stage 6 Complete!")
print("=" * 60)
print(f"\nExpanded layers: {stage_info['original_layers']} → {stage_info['total_layers']}")
print(f"Identity layers at: {identity_layer_indices}")
print(f"LoRA applied to: {len(layers_to_transform)} middle layers")
print(f"\nCheckpoint saved to: {OUTPUT_DIR}")
print("Next: Run Stage 7 (cooldown and merge)")
