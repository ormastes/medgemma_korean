#!/usr/bin/env python3
"""
Training script with division-aware evaluation
Tracks performance per medical division
Optimized for A6000 GPU (40GB usage target)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import argparse
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Optimized model configurations for A6000
MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 128, "lora_alpha": 256,
        "lr": 1e-4, "batch": 8, "grad_accum": 4,
        "max_length": 1024, "grad_ckpt": False
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 384, "lora_alpha": 768,  # Optimized for 40GB A6000
        "lr": 5e-5, "batch": 4, "grad_accum": 8,
        "max_length": 1024, "grad_ckpt": True
    },
    "gemma-2-2b": {
        "path": "google/gemma-2-2b-it",
        "lora_r": 64, "lora_alpha": 128,
        "lr": 2e-4, "batch": 16, "grad_accum": 2,
        "max_length": 1024, "grad_ckpt": False
    }
}


class DivisionTracker:
    """Track performance metrics per division"""
    
    def __init__(self):
        self.division_metrics = defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'losses': []
        })
    
    def update(self, division: str, correct: bool, loss: float):
        """Update metrics for a division"""
        self.division_metrics[division]['total'] += 1
        if correct:
            self.division_metrics[division]['correct'] += 1
        self.division_metrics[division]['losses'].append(loss)
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics per division"""
        stats = {}
        for div, metrics in self.division_metrics.items():
            total = metrics['total']
            if total > 0:
                stats[div] = {
                    'accuracy': metrics['correct'] / total,
                    'count': total,
                    'avg_loss': np.mean(metrics['losses']) if metrics['losses'] else 0.0
                }
        return stats
    
    def print_report(self, title: str = "Division Performance"):
        """Print performance report"""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        # Sort by count descending
        sorted_divs = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
        
        print(f"{'Division':<20} {'Count':>8} {'Accuracy':>10} {'Avg Loss':>10}")
        print(f"{'-'*60}")
        
        for div, metrics in sorted_divs:
            print(f"{div:<20} {metrics['count']:>8} {metrics['accuracy']:>9.2%} {metrics['avg_loss']:>10.4f}")
        
        # Overall stats
        total = sum(m['count'] for m in stats.values())
        total_correct = sum(m['accuracy'] * m['count'] for m in stats.values())
        overall_acc = total_correct / total if total > 0 else 0
        
        print(f"{'-'*60}")
        print(f"{'OVERALL':<20} {total:>8} {overall_acc:>9.2%}")
        print(f"{'='*60}\n")


class DivisionAwareTrainer(Trainer):
    """Custom trainer that tracks division performance"""
    
    def __init__(self, *args, division_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.division_tracker = division_tracker or DivisionTracker()
    
    def evaluation_loop(self, *args, **kwargs):
        """Override evaluation to track per-division metrics"""
        output = super().evaluation_loop(*args, **kwargs)
        
        # Print division report after evaluation
        if self.division_tracker:
            self.division_tracker.print_report("Evaluation Results")
        
        return output


def load_division_data(data_file: str) -> Dataset:
    """Load data with division annotations"""
    
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                data.append({
                    'text': sample.get('text', ''),
                    'prompt': sample.get('prompt', ''),
                    'completion': sample.get('completion', ''),
                    'divisions': sample.get('divisions', ['UNKNOWN']),
                    'primary_division': sample.get('primary_division', 'UNKNOWN')
                })
    
    return Dataset.from_list(data)


def evaluate_with_divisions(
    model,
    tokenizer,
    eval_dataset: Dataset,
    division_tracker: DivisionTracker,
    device: str = "cuda"
):
    """Evaluate model and track per-division performance"""
    
    model.eval()
    division_tracker = DivisionTracker()
    
    for sample in tqdm(eval_dataset, desc="Evaluating"):
        # Get primary division
        primary_div = sample['primary_division']
        
        # Prepare input
        prompt = sample['prompt']
        expected = sample['completion']
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Simple accuracy check (can be improved based on task type)
        correct = expected.strip().lower() in generated.strip().lower()
        
        # Calculate loss
        target_ids = tokenizer(expected, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            loss_outputs = model(input_ids=target_ids, labels=target_ids)
            loss = loss_outputs.loss.item()
        
        # Update tracker
        division_tracker.update(primary_div, correct, loss)
    
    return division_tracker


def main():
    parser = argparse.ArgumentParser(description="Train with division tracking")
    parser.add_argument('--train-data', type=str, required=True, help='Training data')
    parser.add_argument('--val-data', type=str, required=True, help='Validation data')
    parser.add_argument('--model', type=str, default='medgemma-27b',
                       choices=['medgemma-4b', 'medgemma-27b', 'gemma-2-2b'],
                       help='Model configuration')
    parser.add_argument('--output-dir', type=str, default='phase5_subject_training/models',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (use cuda:0 for A6000)')
    parser.add_argument('--eval-steps', type=int, default=100, help='Evaluation frequency')
    
    args = parser.parse_args()
    
    # Get model config
    cfg = MODEL_CONFIGS.get(args.model)
    if not cfg:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"{'='*60}")
    print(f"Division-Aware Training - {args.model}")
    print(f"{'='*60}")
    print(f"LoRA rank: {cfg['lora_r']}, Batch: {cfg['batch']}, Max length: {cfg['max_length']}")
    
    # Load data
    print("Loading datasets...")
    train_dataset = load_division_data(args.train_data)
    val_dataset = load_division_data(args.val_data)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Load tokenizer
    print(f"Loading model: {cfg['path']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg['path'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=cfg['batch'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_steps=args.eval_steps,
        save_steps=500,
        save_strategy="steps",
        evaluation_strategy="steps",
        bf16=True,
        max_length=cfg['max_length'],
        gradient_checkpointing=cfg['grad_ckpt'],
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    # Initialize division tracker
    division_tracker = DivisionTracker()
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Final evaluation with division tracking
    print("\nFinal evaluation with division tracking...")
    final_tracker = evaluate_with_divisions(
        model, tokenizer, val_dataset, division_tracker, device=args.device
    )
    final_tracker.print_report("Final Validation Results")
    
    # Save division report
    stats = final_tracker.get_stats()
    report_file = os.path.join(args.output_dir, 'division_report.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nDivision report saved to: {report_file}")
    
    # Save model
    final_dir = os.path.join(args.output_dir, 'final')
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print("Training completed!")
    print(f"Model saved to: {final_dir}")


if __name__ == "__main__":
    main()
