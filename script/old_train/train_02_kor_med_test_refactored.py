#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 02: Korean Medical Test (MCQ)
Refactored version using shared utilities
"""

import argparse
import sys
from pathlib import Path
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS
from training_utils import (
    load_jsonl_data,
    setup_model_with_lora,
    create_training_args,
    save_training_info
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"
OUTPUT_DIR = BASE_DIR / "models" / "train_02_kor_med_test"


def main():
    parser = argparse.ArgumentParser(description="Train 02: Korean Medical Test")
    parser.add_argument("--model", default="medgemma-4b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--base-model", default=None, help="Base model path (optional)")
    parser.add_argument("--output", default=None, help="Output directory (optional)")
    args = parser.parse_args()
    
    cfg = MODEL_CONFIGS[args.model]
    output_dir = Path(args.output) if args.output else OUTPUT_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Train 02: Korean Medical Test (MCQ)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    
    # Load data
    train_data, val_data = load_jsonl_data(DATA_DIR, max_samples=args.max_samples)
    if len(train_data) == 0:
        print("❌ No training data found!")
        return
    
    # Setup model
    model_path = args.base_model if args.base_model else cfg['path']
    model, tokenizer = setup_model_with_lora(
        model_path,
        lora_r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        include_embeddings=False,
        device="cuda:0"
    )
    
    # Training arguments
    training_args = create_training_args(
        output_dir=str(output_dir),
        num_epochs=args.epochs,
        batch_size=cfg['batch'],
        grad_accum=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        max_length=cfg['max_length']
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    # Save training info
    save_training_info(output_dir, {
        "script": "train_02_kor_med_test",
        "model": args.model,
        "base_model": model_path,
        "epochs": args.epochs,
        "train_samples": len(train_data),
        "val_samples": len(val_data)
    })
    
    print(f"\n✓ Training complete! Model saved to: {final_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
