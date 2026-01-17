#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Training with Mixed Data Strategy (Alternative to Dual-LoRA)

Instead of freezing adapters (which has implementation issues),
mix Korean plain text with medical dictionary to prevent catastrophic forgetting.

Strategy:
- 70% Medical dictionary (main focus)
- 30% Korean plain text (prevent forgetting)

Usage:
    python script/train/train_01_mixed_data.py --epochs 3
"""

import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer
import json
import argparse
import random
from datasets import Dataset

from training_utils import (
    load_model_8bit,
    create_training_args,
    run_training,
    save_training_info
)
from training_config import MODEL_CONFIGS


def load_medical_dict_data():
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
        data.append({"text": text, "source": "medical"})

    return data


def load_korean_plain_text(max_samples=2000):
    """Load Korean plain text from Phase 0 data"""
    data = []

    plain_text_file = Path("data/02_refined/00_plain_text/train.jsonl")
    if not plain_text_file.exists():
        print("Warning: Korean plain text not found, skipping Korean mixing")
        return data

    with open(plain_text_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            sample = json.loads(line)
            data.append({"text": sample['text'], "source": "korean"})

    return data


def create_mixed_dataset(medical_ratio=0.7):
    """
    Create mixed dataset: medical + Korean

    Args:
        medical_ratio: Ratio of medical samples (default: 0.7)
    """
    print("\nLoading datasets...")

    # Load medical dictionary
    medical_data = load_medical_dict_data()
    print(f"Medical dictionary: {len(medical_data)} samples")

    # Load Korean plain text (sample from Phase 0 data)
    korean_samples = int(len(medical_data) * (1 - medical_ratio) / medical_ratio)
    korean_data = load_korean_plain_text(max_samples=korean_samples)
    print(f"Korean plain text: {len(korean_data)} samples")

    # Mix datasets
    all_data = medical_data + korean_data
    random.shuffle(all_data)

    print(f"\nMixed dataset:")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Medical: {len(medical_data)} ({100*medical_ratio:.1f}%)")
    print(f"  Korean: {len(korean_data)} ({100*(1-medical_ratio):.1f}%)")

    return Dataset.from_list(all_data)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Training: Mixed Data Strategy")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    parser.add_argument("--medical-ratio", type=float, default=0.7,
                       help="Ratio of medical samples (default: 0.7)")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1 Training: Mixed Data Strategy")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Medical ratio: {args.medical_ratio:.1%}")
    print(f"Korean ratio: {1-args.medical_ratio:.1%}")

    # Configuration
    phase0_path = "model/00_trained/medgemma-4b"
    output_dir = Path("model/01_mixed/medgemma-4b")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify Phase 0 exists
    if not Path(phase0_path).exists():
        print(f"❌ Phase 0 model not found: {phase0_path}")
        print("Please run train_00_plain_text.py first!")
        return 1

    # Load Phase 0 model (continue training)
    print("\nLoading Phase 0 model with extended vocabulary...")

    # Load tokenizer first (has extended vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(phase0_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load base model
    model = load_model_8bit("google/medgemma-4b-it", device=args.device, model_name="medgemma-4b")

    # Resize embeddings to match tokenizer
    print(f"Resizing model embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # Load Phase 0 adapter
    model = PeftModel.from_pretrained(model, phase0_path, is_trainable=True)

    model.print_trainable_parameters()

    # Create mixed dataset
    train_data = create_mixed_dataset(medical_ratio=args.medical_ratio)

    # Training config
    cfg = MODEL_CONFIGS["medgemma-4b"]
    training_args = create_training_args(
        output_dir=str(output_dir / "training"),
        num_epochs=args.epochs,
        batch_size=cfg['batch'],
        grad_accum=cfg['grad_accum'],
        learning_rate=cfg['lr'] * 0.5,  # Lower LR to prevent forgetting
        max_length=256
    )

    # Train
    print("\nStarting training...")
    print("=" * 70)

    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save model
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training info
    info = {
        "script": "train_01_mixed_data",
        "model": "medgemma-4b",
        "phase0_model": phase0_path,
        "strategy": "mixed_data_training",
        "medical_ratio": args.medical_ratio,
        "korean_ratio": 1 - args.medical_ratio,
        "epochs": args.epochs,
        "train_samples": len(train_data),
        "learning_rate": cfg['lr'] * 0.5
    }

    save_training_info(output_dir, info)

    print("\n✓ Training complete!")
    print("=" * 70)
    print(f"Output: {final_dir}")
    print(f"\nStrategy: Mixed data (Medical {args.medical_ratio:.0%} + Korean {1-args.medical_ratio:.0%})")
    print(f"This approach prevents catastrophic forgetting by continuously")
    print(f"exposing the model to both medical and Korean data.")

    return 0


if __name__ == "__main__":
    exit(main())
