#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical Dictionary Training
- Korean term -> English definition (translation)
- Single word/phrase answer
- Target: Learn medical terminology mapping
"""

import argparse
import json
import random
import torch
import sys
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS, LORA_TARGET_MODULES, TRAINING_DEFAULTS

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "02_refined" / "01_medical_dict.json"
OUTPUT_DIR = BASE_DIR / "models" / "medical_dict"

# System prompts for different training modes
SYSTEM_PROMPTS = {
    "ko_to_en": "당신은 의료 용어 번역 전문가입니다. 한국어 의료 용어를 영어로 번역하세요.",
    "en_to_ko": "You are a medical terminology translator. Translate the English medical term to Korean.",
}


def load_medical_dict(filepath: Path) -> list[dict]:
    """Load medical dictionary from JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both list format [{"term": ..., "definition": ...}] and dict format {"term": "definition"}
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [{"term": k, "definition": v} for k, v in data.items()]
    else:
        raise ValueError(f"Unknown data format: {type(data)}")


def create_training_samples(entries: list[dict], mode: str = "ko_to_en") -> list[dict]:
    """
    Create training samples from dictionary entries

    Modes:
    - ko_to_en: Korean term -> English definition
    - en_to_ko: English definition -> Korean term
    - both: Both directions
    """
    samples = []

    for entry in entries:
        term = entry["term"]
        definition = entry["definition"]

        if mode in ["ko_to_en", "both"]:
            # Korean -> English
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPTS['ko_to_en']}<|im_end|>\n<|im_start|>user\n{term}<|im_end|>\n<|im_start|>assistant\n"
            completion = f"{definition}<|im_end|>"
            samples.append({
                "prompt": prompt,
                "completion": completion,
                "text": prompt + completion
            })

        if mode in ["en_to_ko", "both"]:
            # English -> Korean
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPTS['en_to_ko']}<|im_end|>\n<|im_start|>user\n{definition}<|im_end|>\n<|im_start|>assistant\n"
            completion = f"{term}<|im_end|>"
            samples.append({
                "prompt": prompt,
                "completion": completion,
                "text": prompt + completion
            })

    return samples


def split_data(samples: list[dict], val_ratio: float = 0.1) -> tuple:
    """Split into train and validation sets"""
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    return samples[:split_idx], samples[split_idx:]


def evaluate_accuracy(model, tokenizer, eval_data, device, max_samples=200):
    """Evaluate translation accuracy"""
    model.eval()
    correct = 0
    total = 0

    eval_subset = eval_data.select(range(min(max_samples, len(eval_data))))

    with torch.no_grad():
        for example in eval_subset:
            prompt = example['prompt']
            expected = example['completion'].replace('<|im_end|>', '').strip()

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract answer after the prompt
            if len(response) > len(prompt):
                answer = response[len(prompt):].strip()
                # Normalize for comparison
                answer_norm = answer.lower().strip()
                expected_norm = expected.lower().strip()

                if answer_norm == expected_norm or expected_norm in answer_norm:
                    correct += 1

            total += 1

    model.train()
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma-4b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mode", default="both", choices=["ko_to_en", "en_to_ko", "both"],
                        help="Training mode: ko_to_en, en_to_ko, or both directions")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--eval-samples", type=int, default=200, help="Samples for evaluation")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    output_dir = OUTPUT_DIR / args.model / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Medical Dictionary Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")

    # Load dictionary
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        print("Run refine_medical_dict.py first!")
        return

    entries = load_medical_dict(DATA_FILE)
    print(f"Loaded {len(entries)} dictionary entries")

    # Create training samples
    samples = create_training_samples(entries, mode=args.mode)
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Created {len(samples)} training samples")

    # Split data
    train_samples, val_samples = split_data(samples, args.val_ratio)
    train_data = Dataset.from_list(train_samples)
    eval_data = Dataset.from_list(val_samples)
    print(f"Train: {len(train_data)}, Validation: {len(eval_data)}")

    # Show sample
    print("\nSample training data:")
    sample = train_samples[0]
    print(f"  Prompt: {sample['prompt'][:100]}...")
    print(f"  Completion: {sample['completion']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 8-bit quantization
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg['path'],
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=TRAINING_DEFAULTS['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=cfg['batch'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        weight_decay=TRAINING_DEFAULTS['weight_decay'],
        warmup_ratio=TRAINING_DEFAULTS['warmup_ratio'],
        lr_scheduler_type=TRAINING_DEFAULTS['lr_scheduler_type'],
        logging_steps=TRAINING_DEFAULTS['logging_steps'],
        save_strategy=TRAINING_DEFAULTS['save_strategy'],
        save_total_limit=TRAINING_DEFAULTS['save_total_limit'],
        fp16=False,
        bf16=False,
        max_length=256,  # Shorter for dictionary entries
        gradient_checkpointing=False,
        optim=TRAINING_DEFAULTS['optim'],
        max_grad_norm=TRAINING_DEFAULTS['max_grad_norm'],
        report_to="none",
        dataloader_num_workers=0,
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

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nModel saved to: {final_dir}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    device = torch.device("cuda:0")
    accuracy, correct, total = evaluate_accuracy(
        model, tokenizer, eval_data, device, max_samples=args.eval_samples
    )

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

    # Save results
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "epochs": args.epochs,
        "model": args.model,
        "mode": args.mode,
        "train_samples": len(train_data),
        "val_samples": len(eval_data),
        "dictionary_entries": len(entries)
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nTraining complete! Results saved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
