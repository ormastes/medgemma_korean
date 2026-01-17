#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Training with Dual-LoRA Strategy

LoRA_0: Frozen (Korean language from Phase 0)
LoRA_1: Trainable (Medical dictionary)
Embeddings: Trainable (continue learning Korean medical vocab)

Usage:
    python script/train/train_01_dual_lora.py
    python script/train/train_01_dual_lora.py --epochs 5
"""

import torch
from pathlib import Path
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import json
import argparse

from training_config import MODEL_CONFIGS, LORA_TARGET_MODULES, TRAINING_DEFAULTS


def load_phase0_model_with_frozen_adapter(phase0_path, device="cuda:0"):
    """
    Load Phase 0 model with frozen LoRA adapter

    Returns:
        model with adapter_0 (frozen)
    """
    print("=" * 70)
    print("Loading Phase 0 Model with Frozen Adapter")
    print("=" * 70)

    # Load base model (8-bit)
    print("Loading base model in 8-bit...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-4b-it",
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    # Load Phase 0 adapter (includes LoRA + embeddings)
    print(f"Loading Phase 0 adapter from: {phase0_path}")
    model = PeftModel.from_pretrained(
        base_model,
        phase0_path,
        adapter_name="lora_0",  # Name this adapter
        is_trainable=False  # CRITICAL: Freeze this adapter
    )

    print("✓ Phase 0 adapter loaded as 'lora_0' (FROZEN)")
    print(f"Active adapters: {model.active_adapters}")

    return model


def add_trainable_lora_adapter(model, adapter_name="lora_1", lora_r=64, lora_alpha=128):
    """
    Add a new trainable LoRA adapter to the model

    Args:
        model: Model with existing frozen adapter
        adapter_name: Name for new adapter
    """
    print("=" * 70)
    print(f"Adding Trainable Adapter: {adapter_name}")
    print("=" * 70)

    # Create LoRA config for new adapter
    # Note: Don't include modules_to_save here since embeddings are already
    # loaded from lora_0. We only add LoRA adapters for the transformer layers.
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=None,  # Don't modify embeddings - use lora_0's embeddings
        lora_dropout=TRAINING_DEFAULTS['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Add adapter to model
    model.add_adapter(adapter_name, lora_config)

    # Set this adapter as active (trainable)
    model.set_adapter(adapter_name)  # Single string, not list

    print(f"✓ Adapter '{adapter_name}' added (TRAINABLE)")
    print(f"Active adapters: {model.active_adapters}")

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def setup_dual_lora_model(phase0_path, device="cuda:0"):
    """
    Complete setup: Load Phase 0 (frozen) + Add Phase 1 (trainable)
    """
    # Step 1: Load Phase 0 with frozen adapter
    model = load_phase0_model_with_frozen_adapter(phase0_path, device)

    # Step 2: Add new trainable adapter for Phase 1
    cfg = MODEL_CONFIGS["medgemma-4b"]
    model = add_trainable_lora_adapter(
        model,
        adapter_name="lora_1",
        lora_r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha']
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(phase0_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_medical_dict_data(max_samples=None):
    """Load medical dictionary data"""
    data = []

    # Load Korean medical dictionary
    dict_file = Path("data/02_refined/01_medical_dict.json")
    with open(dict_file, 'r', encoding='utf-8') as f:
        dict_data = json.load(f)

    # Load character dictionary
    char_file = Path("data/02_refined/02_char_dict.json")
    with open(char_file, 'r', encoding='utf-8') as f:
        char_data = json.load(f)

    all_entries = dict_data + char_data
    if max_samples:
        all_entries = all_entries[:max_samples]

    # Format as training examples
    for entry in all_entries:
        term = entry['term']
        definition = entry['definition']

        text = (
            f"<start_of_turn>user\n"
            f"Meaning of word {term}:<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"{definition}<end_of_turn>"
        )
        data.append({"text": text})

    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Training: Dual-LoRA Strategy")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training samples")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1 Training: Dual-LoRA Strategy")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")

    # Configuration
    phase0_path = "model/00_trained/medgemma-4b"
    output_dir = Path("model/01_dual_lora/medgemma-4b")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify Phase 0 exists
    if not Path(phase0_path).exists():
        print(f"❌ Phase 0 model not found: {phase0_path}")
        print("Please run train_00_plain_text.py first!")
        return 1

    # Setup model with dual LoRA
    model, tokenizer = setup_dual_lora_model(phase0_path, args.device)

    # Load data
    print("\nLoading medical dictionary data...")
    train_data = load_medical_dict_data(max_samples=args.max_samples)
    print(f"Training samples: {len(train_data)}")

    # Training config
    cfg = MODEL_CONFIGS["medgemma-4b"]
    training_args = SFTConfig(
        output_dir=str(output_dir / "training"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=cfg['batch'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        weight_decay=TRAINING_DEFAULTS['weight_decay'],
        warmup_ratio=TRAINING_DEFAULTS['warmup_ratio'],
        lr_scheduler_type=TRAINING_DEFAULTS['lr_scheduler_type'],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=False,
        max_length=256,  # Short for dictionary entries
        gradient_checkpointing=False,
        optim=TRAINING_DEFAULTS['optim'],
        max_grad_norm=TRAINING_DEFAULTS['max_grad_norm'],
        report_to="none",
        dataloader_num_workers=0,
    )

    # Train
    print("\nStarting training...")
    print("=" * 70)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save model
    final_dir = output_dir / "final"
    print(f"\nSaving model to: {final_dir}")

    # Save both adapters
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training info
    info = {
        "script": "train_01_dual_lora",
        "model": "medgemma-4b",
        "phase0_model": phase0_path,
        "adapters": {
            "lora_0": "frozen (Korean language from Phase 0)",
            "lora_1": "trained (Medical dictionary)"
        },
        "epochs": args.epochs,
        "train_samples": len(train_data),
        "strategy": "dual_lora_frozen_phase0"
    }

    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    # Create README
    readme = f"""# Phase 1: Dual-LoRA Training

## Training Strategy

**Dual-LoRA Approach:**
- LoRA_0: FROZEN (Korean language from Phase 0)
- LoRA_1: TRAINED (Medical dictionary)
- Embeddings: TRAINED (Korean medical vocabulary)

## Training Info

- Script: train_01_dual_lora.py
- Epochs: {args.epochs}
- Training Samples: {len(train_data)}
- Base Model: {phase0_path}

## Adapters

### LoRA_0 (Frozen)
- Source: Phase 0 (train_00_plain_text.py)
- Purpose: Preserve Korean language knowledge
- Status: FROZEN (not updated during training)

### LoRA_1 (Trained)
- Purpose: Learn Korean-English medical terminology
- Status: TRAINED (updated during this phase)
- Data: Medical dictionary (4,138 terms) + Special symbols (89 chars)

## Benefits

1. ✅ Preserves Korean language fluency from Phase 0
2. ✅ Adds medical vocabulary without catastrophic forgetting
3. ✅ Can be used with both adapters or just lora_0 for general Korean

## Usage

### Load Model with Both Adapters
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    device_map="cuda:0",
    load_in_8bit=True
)

model = PeftModel.from_pretrained(
    model,
    "model/01_dual_lora/medgemma-4b/final"
)

# Both adapters are active
print(model.active_adapters)
```

### Use Only Korean Adapter
```python
model.set_adapter(["lora_0"])  # General Korean only
```

### Use Both Adapters
```python
model.set_adapter(["lora_0", "lora_1"])  # Korean + Medical
```

## Next Steps

Continue to Phase 2 with triple-LoRA:
```bash
python script/train/train_02_triple_lora.py
```
"""

    with open(final_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme)

    print("\n✓ Training complete!")
    print("=" * 70)
    print(f"Output: {final_dir}")
    print(f"\nAdapters saved:")
    print(f"  - lora_0: Frozen (Korean language preservation)")
    print(f"  - lora_1: Trained (Medical dictionary)")
    print(f"\nNext step:")
    print(f"  python script/train/train_02_triple_lora.py")

    return 0


if __name__ == "__main__":
    exit(main())
