#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type 3: Multiple Choice Question (MCQ) Training
- Single letter answer (A, B, C, D, E)
- NO reasoning tokens
- Target: >=90% accuracy
"""

import argparse
import json
import torch
import sys
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS, LORA_TARGET_MODULES, TRAINING_DEFAULTS

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "refined" / "type3_word"
OUTPUT_DIR = BASE_DIR / "models" / "type3_mcq"


def load_data(data_dir: Path, max_samples: int = None):
    """Load training and validation data"""
    train, val = [], []
    
    for split, samples in [("train", train), ("validation", val)]:
        file_path = data_dir / split / "data.jsonl"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    samples.append(json.loads(line))
    
    print(f"Loaded {len(train)} train, {len(val)} validation samples")
    return Dataset.from_list(train), Dataset.from_list(val)


def evaluate_accuracy(model, tokenizer, eval_data, device, max_samples=500):
    """Simple evaluation without callbacks"""
    model.eval()
    correct = 0
    total = 0
    
    eval_subset = eval_data.select(range(min(max_samples, len(eval_data))))
    
    with torch.no_grad():
        for example in eval_subset:
            prompt = example['prompt']
            expected = example['completion'].strip()
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract answer after the prompt
            if len(response) > len(prompt):
                answer = response[len(prompt):].strip()
                # Take first character/word
                answer = answer.split()[0] if answer else ""
                
                if answer == expected:
                    correct += 1
            
            total += 1
    
    model.train()
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma-4b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--eval-samples", type=int, default=500, help="Samples to use for evaluation")
    args = parser.parse_args()
    
    cfg = MODEL_CONFIGS[args.model]
    output_dir = OUTPUT_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Type 3: MCQ Training (No Reasoning)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    
    # Load data
    train_data, eval_data = load_data(DATA_DIR, max_samples=args.max_samples)
    if len(train_data) == 0:
        print("No training data found! Run data preparation first.")
        return
    
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
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=False,
        max_length=cfg['max_length'],
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        dataloader_num_workers=0,  # Avoid multiprocessing issues
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
        "model": args.model
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete! Results saved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
