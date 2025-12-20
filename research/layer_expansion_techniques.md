# Layer Expansion Techniques for Language Adaptation

**Research Date:** 2025-12-07

## Overview

When adapting LLMs to new languages, there are techniques that **actually add layers** to increase model capacity, rather than just fine-tuning existing parameters. This document covers two main approaches.

---

## Two Main Approaches: Layer Duplication

### 1. SOLAR - Depth Up-Scaling (DUS)

One prominent approach is the Depth Up-Scaling (DUS) method introduced by Kim et al. in the SOLAR 10.7B model, which focuses on increasing the depth of transformers by duplicating top and bottom layers, effectively expanding the model's depth while maintaining simplicity.

```
┌─────────────────────────────────────────────────────────────┐
│              SOLAR Depth Up-Scaling (DUS)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Mistral 7B (32 layers)                                     │
│  ┌────────────────────────────────────┐                     │
│  │ Layer 0-7   (front 8)              │ ← Copy these        │
│  │ Layer 8-23  (middle 16)            │                     │
│  │ Layer 24-31 (rear 8)               │ ← Copy these        │
│  └────────────────────────────────────┘                     │
│                     ↓                                       │
│  Step 1: Duplicate entire model                             │
│  Step 2: Remove middle layers to create "seam"              │
│                     ↓                                       │
│  SOLAR 10.7B (48 layers)                                    │
│  ┌────────────────────────────────────┐                     │
│  │ Layer 0-23  (from original)        │                     │
│  │ --- seam ---                       │                     │
│  │ Layer 24-47 (from duplicate)       │                     │
│  └────────────────────────────────────┘                     │
│                                                             │
│  7B → 10.7B parameters (+53%)                               │
│  32 → 48 layers (+50%)                                      │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. LLaMA Pro - Block Expansion (Interleaved)

We initialize our base model with LLaMA2-7B and expand the number of blocks from 32 to 40 using an interleaved approach. In the block expansion process, we configure the parameters as N = 8, resulting in 8 groups where each group expands from 4 blocks to 5 blocks.

```
┌─────────────────────────────────────────────────────────────┐
│              LLaMA Pro Block Expansion                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLaMA2-7B (32 layers) → LLaMA Pro (40 layers)              │
│                                                             │
│  Original: [0,1,2,3] [4,5,6,7] ... [28,29,30,31]            │
│               ↓         ↓              ↓                    │
│  Expanded: [0,1,2,3,3'] [4,5,6,7,7'] ... [28,29,30,31,31']  │
│                    ↑            ↑                  ↑        │
│               New layer    New layer          New layer     │
│            (identity init) (identity init)  (identity init) │
│                                                             │
│  Key: New layers initialized as IDENTITY (zero linear)      │
│       → Output unchanged initially                          │
│       → Train only new layers, freeze original              │
│                                                             │
│  7B → 8.3B parameters (+18%)                                │
│  32 → 40 layers (+25%)                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparison: SOLAR vs LLaMA Pro vs LoRA

| Aspect | SOLAR (DUS) | LLaMA Pro | LoRA/QLoRA |
|--------|-------------|-----------|------------|
| **What changes** | Adds layers (front+rear duplication) | Adds layers (interleaved) | Adds adapters (parallel) |
| **Layer count** | 32 → 48 | 32 → 40 | 32 → 32 (unchanged) |
| **Parameters** | +53% | +18% | +1-3% |
| **Initialization** | Copy existing weights | Identity (zeros) | Random/zeros |
| **Training** | All layers (continued pretraining) | Only new layers | Only adapters |
| **Inference cost** | Higher (more layers) | Higher (more layers) | Same (can merge) |
| **Best for** | General capability boost | Domain adaptation | Fine-tuning |

---

## Implementation: SOLAR-Style Layer Duplication

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

def solar_depth_upscale(model, duplicate_front=2, duplicate_rear=2):
    """
    SOLAR-style: Duplicate front and rear layers

    Example: 32 layers with front=2, rear=2
    Original: [0,1,2,...,29,30,31]
    Result:   [0,1,0',1',2,...,29,30,31,30',31']  (36 layers)
    """

    config = model.config
    original_layers = list(model.model.layers)
    num_layers = len(original_layers)

    # Deep copy front layers
    front_copies = []
    for i in range(duplicate_front):
        layer_copy = copy.deepcopy(original_layers[i])
        front_copies.append(layer_copy)

    # Deep copy rear layers
    rear_copies = []
    for i in range(num_layers - duplicate_rear, num_layers):
        layer_copy = copy.deepcopy(original_layers[i])
        rear_copies.append(layer_copy)

    # Assemble new layer list
    # [original_front, copied_front, middle, original_rear, copied_rear]
    new_layers = nn.ModuleList([
        *original_layers[:duplicate_front],      # original front
        *front_copies,                            # duplicated front
        *original_layers[duplicate_front:-duplicate_rear],  # middle
        *original_layers[-duplicate_rear:],      # original rear
        *rear_copies                              # duplicated rear
    ])

    model.model.layers = new_layers

    # Update config
    model.config.num_hidden_layers = len(new_layers)

    return model

# Usage
model = AutoModelForCausalLM.from_pretrained("medgemma-27b")
model = solar_depth_upscale(model, duplicate_front=2, duplicate_rear=2)
# 32 layers → 36 layers
# 27B → ~30.4B parameters
```

---

## Implementation: LLaMA Pro-Style (Identity Initialization)

```python
import torch
import torch.nn as nn
import copy

def llama_pro_expand(model, expansion_positions=[7, 15, 23, 31]):
    """
    LLaMA Pro-style: Insert identity-initialized layers at positions

    expansion_positions: After which layers to insert new layer
    """

    original_layers = list(model.model.layers)
    new_layers = []

    for i, layer in enumerate(original_layers):
        new_layers.append(layer)

        if i in expansion_positions:
            # Create identity-initialized copy
            new_layer = copy.deepcopy(layer)

            # Zero-initialize linear layers for identity mapping
            # Due to residual: output = input + layer(input)
            # If layer(input) = 0, then output = input (identity)
            for name, param in new_layer.named_parameters():
                if 'weight' in name and 'norm' not in name:
                    # Zero out projection weights
                    nn.init.zeros_(param)

            new_layers.append(new_layer)
            print(f"Inserted identity layer after layer {i}")

    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)

    return model

# Usage for 32-layer model
model = AutoModelForCausalLM.from_pretrained("medgemma-27b")
model = llama_pro_expand(model, expansion_positions=[1, 30])  # After front and before rear
# 32 layers → 34 layers
```

---

## Application to Korean MedGemma

### Option A: Front/Rear Duplication (SOLAR-style)

```
┌─────────────────────────────────────────────────────────────┐
│         MedGemma 27B + Layer Duplication                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Original (32 layers, 27B params):                          │
│  [L0, L1, L2, ..., L29, L30, L31]                           │
│                                                             │
│  After duplication (36 layers, ~30.4B params):              │
│  [L0, L1, L0', L1', L2...L29, L30, L31, L30', L31']         │
│   ^^^^  ^^^^^^^^              ^^^^^^^^^  ^^^^^^^^^          │
│   orig  new (for Korean)      orig       new (for Korean)   │
│                                                             │
│  Training strategy:                                         │
│  - Freeze: L0, L1, L2...L29, L30, L31 (original)            │
│  - Train: L0', L1', L30', L31' (new Korean layers)          │
│  - Train: Embeddings, LM Head (for Korean tokens)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Memory Calculation (36 layers, training 4 new)

| Component | VRAM |
|-----------|------|
| Original 32 layers (4-bit frozen) | 27B × 0.5 = 13.5 GB |
| New 4 layers (FP16, trainable) | 3.4B × 2 = 6.8 GB |
| New layer gradients | 3.4B × 2 = 6.8 GB |
| New layer optimizer (8-bit) | 3.4B × 1 = 3.4 GB |
| Embeddings (FP16, trainable) | 3B × 2 = 6 GB |
| Embedding gradients | 3B × 2 = 6 GB |
| Embedding optimizer | 3B × 1 = 3 GB |
| Activations (grad checkpoint) | ~4-6 GB |
| CUDA overhead | ~2 GB |
| **TOTAL** | **~52-56 GB** |

**RTX A5000 (48GB):** Too large - needs optimization or multi-GPU

---

### Option B: Identity Expansion (LLaMA Pro-style) - More Efficient

```
┌─────────────────────────────────────────────────────────────┐
│         MedGemma 27B + Identity Layer Expansion             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Original (32 layers):                                      │
│  [L0, L1, L2, ..., L29, L30, L31]                           │
│                                                             │
│  After expansion (34 layers):                               │
│  [L0, L1, L1'(id), L2, ..., L30, L31, L31'(id)]             │
│            ^^^^^^                      ^^^^^^^              │
│           identity                    identity              │
│           (zeros)                     (zeros)               │
│                                                             │
│  Key advantage:                                             │
│  - Identity init → model output unchanged at start          │
│  - Only train NEW layers (2 layers = ~1.7B params)          │
│  - Much less memory than full layer duplication             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Memory Calculation (Identity Expansion, 2 new layers)

| Component | VRAM |
|-----------|------|
| Original 32 layers (4-bit frozen) | 27B × 0.5 = 13.5 GB |
| New 2 layers (FP16, trainable) | 1.7B × 2 = 3.4 GB |
| New layer gradients | 1.7B × 2 = 3.4 GB |
| New layer optimizer (8-bit) | 1.7B × 1 = 1.7 GB |
| Embeddings (FP16, trainable) | 3B × 2 = 6 GB |
| Embedding gradients | 3B × 2 = 6 GB |
| Embedding optimizer | 3B × 1 = 3 GB |
| Activations (grad checkpoint) | ~4-5 GB |
| CUDA overhead | ~2 GB |
| **TOTAL** | **~43-45 GB** |

**RTX A5000 (48GB):** Fits!

---

## Complete Implementation: Korean MedGemma with Layer Expansion

```python
import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def create_identity_layer(source_layer):
    """Create identity-initialized copy of a transformer layer"""
    new_layer = copy.deepcopy(source_layer)

    # Zero-initialize all linear projections
    # This makes layer output = 0, so residual = input (identity)
    for name, module in new_layer.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return new_layer

def expand_model_for_korean(model_name, front_expand=1, rear_expand=1):
    """
    Expand MedGemma with identity layers for Korean adaptation

    Args:
        model_name: Base model path
        front_expand: Number of identity layers after front
        rear_expand: Number of identity layers before end
    """

    # Load base model in 4-bit for frozen layers
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    original_layers = list(model.model.layers)
    num_layers = len(original_layers)
    print(f"Original layers: {num_layers}")

    new_layers = []

    for i, layer in enumerate(original_layers):
        new_layers.append(layer)

        # Add identity layer after first `front_expand` layers
        if i < front_expand:
            identity_layer = create_identity_layer(layer)
            # Convert to FP16 for training (not quantized)
            identity_layer = identity_layer.to(torch.float16)
            new_layers.append(identity_layer)
            print(f"Added identity layer after layer {i} (front)")

        # Add identity layer after last `rear_expand` layers
        if i >= num_layers - rear_expand:
            identity_layer = create_identity_layer(layer)
            identity_layer = identity_layer.to(torch.float16)
            new_layers.append(identity_layer)
            print(f"Added identity layer after layer {i} (rear)")

    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)

    print(f"New total layers: {len(new_layers)}")

    # Set trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False  # Freeze all first

    # Unfreeze new identity layers (they're in FP16)
    for i, layer in enumerate(model.model.layers):
        # Identity layers are in FP16, original are quantized
        if layer.self_attn.q_proj.weight.dtype == torch.float16:
            for param in layer.parameters():
                param.requires_grad = True
            print(f"Layer {i} set to trainable (identity layer)")

    # Unfreeze embeddings for Korean tokens
    model.model.embed_tokens.weight.requires_grad = True
    model.lm_head.weight.requires_grad = True

    return model

# Usage
model = expand_model_for_korean(
    "medgemma-27b",
    front_expand=1,  # 1 identity layer after L0
    rear_expand=1    # 1 identity layer after L31
)

# Training config
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./korean-medgemma-expanded",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-4,
    num_train_epochs=3,
    optim="paged_adamw_8bit",
)
```

---

## Summary Comparison

| Method | New Params | Layers | Training VRAM | A5000 (48GB) |
|--------|------------|--------|---------------|--------------|
| **QLoRA (r=64)** | 0.3B (1%) | 32 | ~30 GB | Easy |
| **QLoRA variable rank** | 0.5B (2%) | 32 | ~35 GB | Good |
| **Identity expansion (2 layers)** | 1.7B (6%) | 34 | ~43 GB | Fits |
| **Identity expansion (4 layers)** | 3.4B (12%) | 36 | ~50 GB | Tight |
| **Full duplication (4 layers)** | 6.4B (23%) | 36 | ~55 GB | No |

---

## Recommendation for Korean MedGemma

### Best Approach: Hybrid

1. **Identity Layer Expansion (2 new layers)**
   - 1 after L1 (front, language input)
   - 1 after L31 (rear, language output)
   - Train these fully (FP16)

2. **QLoRA on Middle Layers (r=32)**
   - Light adaptation of reasoning layers
   - Small memory footprint

3. **Full Training on Embeddings + LM Head**
   - Essential for Korean vocabulary

**Total VRAM: ~40-45 GB** ← Fits A5000!

This gives you **actual new capacity** (identity layers) for Korean language processing while keeping memory manageable!

---

## References

- SOLAR 10.7B: Depth Up-Scaling (Kim et al., 2023)
- LLaMA Pro: Progressive LLaMA with Block Expansion (Wu et al., 2024)
- EEVE: Efficient and Effective Vocabulary Expansion (Kim et al., 2024)
