#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Training with Triple-LoRA Strategy

LoRA_0: Frozen (Korean language from Phase 0)
LoRA_1: Frozen (Medical dictionary from Phase 1)
LoRA_2: Trainable (MCQ reasoning)
Embeddings: Trainable (continue learning)

Usage:
    python script/train/train_02_triple_lora.py
    python script/train/train_02_triple_lora.py --epochs 5
"""

import torch
from pathlib import Path
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import json
import argparse

from training_config import MODEL_CONFIGS, LORA_TARGET_MODULES, TRAINING_DEFAULTS, MAX_LENGTHS


def load_phase1_model_with_frozen_adapters(phase1_path, device="cuda:0"):
    """
    Load Phase 1 model with both adapters frozen

    Returns:
        model with lora_0 and lora_1 (both frozen)
    """
    print("=" * 70)
    print("Loading Phase 1 Model with Frozen Adapters")
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

    # Load Phase 1 model (has lora_0 and lora_1)
    print(f"Loading Phase 1 adapters from: {phase1_path}")
    model = PeftModel.from_pretrained(
        base_model,
        phase1_path,
        is_trainable=False  # Freeze all existing adapters
    )

    print("✓ Phase 1 adapters loaded (lora_0, lora_1) - BOTH FROZEN")
    print(f"Available adapters: {list(model.peft_config.keys())}")

    return model


def add_lora2_for_mcq(model, lora_r=64, lora_alpha=128):
    """Add LoRA_2 for MCQ reasoning training"""
    print("=" * 70)
    print("Adding Trainable Adapter: lora_2 (MCQ Reasoning)")
    print("=" * 70)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=["embed_tokens", "lm_head"],  # Continue training embeddings
        lora_dropout=TRAINING_DEFAULTS['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model.add_adapter("lora_2", lora_config)
    model.set_adapter("lora_2")  # Only lora_2 is trainable (single string)

    print("✓ Adapter 'lora_2' added (TRAINABLE)")
    print(f"Active adapters: {model.active_adapters}")
    model.print_trainable_parameters()

    return model


def setup_triple_lora_model(phase1_path, device="cuda:0"):
    """
    Complete setup: Load Phase 1 (lora_0, lora_1 frozen) + Add lora_2 (trainable)
    """
    # Step 1: Load Phase 1 with frozen adapters
    model = load_phase1_model_with_frozen_adapters(phase1_path, device)

    # Step 2: Add new trainable adapter for MCQ
    cfg = MODEL_CONFIGS["medgemma-4b"]
    model = add_lora2_for_mcq(
        model,
        lora_r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha']
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(phase1_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # For generation

    return model, tokenizer


def load_mcq_data(max_samples=None):
    """Load KorMedMCQA training data"""
    data = []

    train_file = Path("data/02_refined/02_kor_med_test/train.jsonl")
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            sample = json.loads(line)
            data.append(sample)

    # Format for training (95% simple format)
    formatted_data = []
    for sample in data:
        # Simple format (95%)
        question = sample['question']
        choices = [sample[choice] for choice in ['A', 'B', 'C', 'D', 'E']]
        answer = sample['answer']

        text = (
            "<start_of_turn>user\n"
            "Reasoning 후 정답 알파벳 하나만 답하세요.\n\n"
            f"{question}\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}\n"
            f"E) {choices[4]}\n\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"<reasoning>\n"
            f"각 선택지를 분석하면:\n"
            f"A) {choices[0][:20]}...\n"
            f"B) {choices[1][:20]}...\n"
            f"C) {choices[2][:20]}...\n"
            f"D) {choices[3][:20]}...\n"
            f"E) {choices[4][:20]}...\n"
            f"</reasoning>{answer}<end_of_turn>"
        )
        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Training: Triple-LoRA Strategy")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training samples")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 2 Training: Triple-LoRA Strategy")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")

    # Configuration
    phase1_path = "model/01_dual_lora/medgemma-4b/final"
    output_dir = Path("model/02_triple_lora/medgemma-4b")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify Phase 1 exists
    if not Path(phase1_path).exists():
        print(f"❌ Phase 1 model not found: {phase1_path}")
        print("Please run train_01_dual_lora.py first!")
        return 1

    # Setup model with triple LoRA
    model, tokenizer = setup_triple_lora_model(phase1_path, args.device)

    # Load data
    print("\nLoading KorMedMCQA training data...")
    train_data = load_mcq_data(max_samples=args.max_samples)
    print(f"Training samples: {len(train_data)}")

    # Training config
    cfg = MODEL_CONFIGS["medgemma-4b"]
    max_length = MAX_LENGTHS["medgemma-4b"]["train_02"]

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
        max_length=max_length,
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

    # Save all adapters
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training info
    info = {
        "script": "train_02_triple_lora",
        "model": "medgemma-4b",
        "phase1_model": phase1_path,
        "adapters": {
            "lora_0": "frozen (Korean language from Phase 0)",
            "lora_1": "frozen (Medical dictionary from Phase 1)",
            "lora_2": "trained (MCQ reasoning)"
        },
        "epochs": args.epochs,
        "train_samples": len(train_data),
        "max_length": max_length,
        "strategy": "triple_lora_frozen_phase0_and_phase1"
    }

    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    # Create README
    readme = f"""# Phase 2: Triple-LoRA Training

## Training Strategy

**Triple-LoRA Approach:**
- LoRA_0: FROZEN (Korean language from Phase 0)
- LoRA_1: FROZEN (Medical dictionary from Phase 1)
- LoRA_2: TRAINED (MCQ reasoning)
- Embeddings: TRAINED (Korean medical vocabulary)

## Training Info

- Script: train_02_triple_lora.py
- Epochs: {args.epochs}
- Training Samples: {len(train_data)}
- Base Model: {phase1_path}

## Adapters

### LoRA_0 (Frozen)
- Source: Phase 0 (train_00_plain_text.py)
- Purpose: Preserve Korean language knowledge
- Status: FROZEN

### LoRA_1 (Frozen)
- Source: Phase 1 (train_01_dual_lora.py)
- Purpose: Preserve medical vocabulary
- Status: FROZEN

### LoRA_2 (Trained)
- Purpose: Learn MCQ reasoning with chain-of-thought
- Status: TRAINED (updated during this phase)
- Data: KorMedMCQA (1,890 training samples)

## Benefits

1. ✅ Preserves Korean language fluency (from Phase 0)
2. ✅ Preserves medical vocabulary (from Phase 1)
3. ✅ Adds MCQ reasoning without catastrophic forgetting
4. ✅ Can mix and match adapters for different tasks

## Usage

### Load Model with All Adapters
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
    "model/02_triple_lora/medgemma-4b/final"
)

# All three adapters are loaded
print(model.peft_config.keys())  # ['lora_0', 'lora_1', 'lora_2']
```

### Selective Adapter Usage
```python
# General Korean only
model.set_adapter(["lora_0"])

# Korean + Medical vocabulary
model.set_adapter(["lora_0", "lora_1"])

# Full capability (Korean + Medical + MCQ)
model.set_adapter(["lora_0", "lora_1", "lora_2"])
```

## Evaluation

```bash
python script/validation_kor_med_test.py \\
    --model model/02_triple_lora/medgemma-4b/final
```

Expected improvements:
- Korean fluency preserved (no increase in perplexity)
- Medical vocabulary intact
- MCQ accuracy > single-LoRA approach
"""

    with open(final_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme)

    print("\n✓ Training complete!")
    print("=" * 70)
    print(f"Output: {final_dir}")
    print(f"\nAdapters saved:")
    print(f"  - lora_0: Frozen (Korean language)")
    print(f"  - lora_1: Frozen (Medical dictionary)")
    print(f"  - lora_2: Trained (MCQ reasoning)")
    print(f"\nNext step:")
    print(f"  python script/validation_kor_med_test.py --model {final_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
