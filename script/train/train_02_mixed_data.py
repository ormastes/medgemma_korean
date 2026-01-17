#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Training with Mixed Data Strategy

Mix MCQ data with medical dictionary to prevent catastrophic forgetting

Strategy:
- 70% MCQ (KorMedMCQA) - main focus
- 30% Medical dictionary - prevent forgetting

Usage:
    python script/train/train_02_mixed_data.py --epochs 5
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
    save_training_info
)
from training_config import MODEL_CONFIGS, MAX_LENGTHS


def load_mcq_data():
    """Load KorMedMCQA training data"""
    data = []

    train_file = Path("data/02_refined/02_kor_med_test/train.jsonl")
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)

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
                f"A) {choices[0][:30]}...\n"
                f"B) {choices[1][:30]}...\n"
                f"C) {choices[2][:30]}...\n"
                f"D) {choices[3][:30]}...\n"
                f"E) {choices[4][:30]}...\n"
                f"</reasoning>{answer}<end_of_turn>"
            )
            data.append({"text": text, "source": "mcq"})

    return data


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


def create_mixed_dataset(mcq_ratio=0.7):
    """
    Create mixed dataset: MCQ + Medical dictionary

    Args:
        mcq_ratio: Ratio of MCQ samples (default: 0.7)
    """
    print("\nLoading datasets...")

    # Load MCQ data
    mcq_data = load_mcq_data()
    print(f"MCQ data: {len(mcq_data)} samples")

    # Load medical dictionary (sample to match ratio)
    medical_samples = int(len(mcq_data) * (1 - mcq_ratio) / mcq_ratio)
    all_medical = load_medical_dict_data()
    medical_data = random.sample(all_medical, min(medical_samples, len(all_medical)))
    print(f"Medical dictionary: {len(medical_data)} samples")

    # Mix datasets
    all_data = mcq_data + medical_data
    random.shuffle(all_data)

    print(f"\nMixed dataset:")
    print(f"  Total samples: {len(all_data)}")
    print(f"  MCQ: {len(mcq_data)} ({100*len(mcq_data)/len(all_data):.1f}%)")
    print(f"  Medical: {len(medical_data)} ({100*len(medical_data)/len(all_data):.1f}%)")

    return Dataset.from_list(all_data)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Training: Mixed Data Strategy")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    parser.add_argument("--mcq-ratio", type=float, default=0.7,
                       help="Ratio of MCQ samples (default: 0.7)")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 2 Training: Mixed Data Strategy")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"MCQ ratio: {args.mcq_ratio:.1%}")
    print(f"Medical dict ratio: {1-args.mcq_ratio:.1%}")

    # Configuration
    phase1_path = "model/01_mixed/medgemma-4b/final"
    output_dir = Path("model/02_mixed/medgemma-4b")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify Phase 1 exists
    if not Path(phase1_path).exists():
        print(f"❌ Phase 1 model not found: {phase1_path}")
        print("Please run train_01_mixed_data.py first!")
        return 1

    # Load Phase 1 model (continue training)
    print("\nLoading Phase 1 model...")

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(phase1_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # For generation

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load base model
    model = load_model_8bit("google/medgemma-4b-it", device=args.device, model_name="medgemma-4b")

    # Resize embeddings
    print(f"Resizing model embeddings to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # Load Phase 1 adapter
    model = PeftModel.from_pretrained(model, phase1_path, is_trainable=True)

    model.print_trainable_parameters()

    # Create mixed dataset
    train_data = create_mixed_dataset(mcq_ratio=args.mcq_ratio)

    # Training config
    cfg = MODEL_CONFIGS["medgemma-4b"]
    max_length = MAX_LENGTHS["medgemma-4b"]["train_02"]

    training_args = create_training_args(
        output_dir=str(output_dir / "training"),
        num_epochs=args.epochs,
        batch_size=cfg['batch'],
        grad_accum=cfg['grad_accum'],
        learning_rate=cfg['lr'] * 0.3,  # Even lower LR for Phase 2
        max_length=max_length
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
        "script": "train_02_mixed_data",
        "model": "medgemma-4b",
        "phase1_model": phase1_path,
        "strategy": "mixed_data_training",
        "mcq_ratio": args.mcq_ratio,
        "medical_ratio": 1 - args.mcq_ratio,
        "epochs": args.epochs,
        "train_samples": len(train_data),
        "max_length": max_length,
        "learning_rate": cfg['lr'] * 0.3
    }

    save_training_info(output_dir, info)

    print("\n✓ Training complete!")
    print("=" * 70)
    print(f"Output: {final_dir}")
    print(f"\nStrategy: Mixed data (MCQ {args.mcq_ratio:.0%} + Medical {1-args.mcq_ratio:.0%})")
    print(f"\nNext step: Validate on KorMedMCQA test set")
    print(f"  python script/validation_kor_med_test.py --model {final_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
