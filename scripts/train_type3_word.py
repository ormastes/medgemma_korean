#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Type 3: WORD
- Single word/letter answer, NO reasoning tokens
- Evaluation: Exact match accuracy
- Target: >=90% accuracy
"""

import argparse
import json
import re
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "refined" / "type3_word"
OUTPUT_DIR = BASE_DIR / "models" / "type3_word"

# Reasoning tokens (skip if present - model may have learned from Type 2/4)
R_START = "<R>"
R_END = "<R/>"


def skip_reasoning_block(text: str) -> str:
    """Skip reasoning block if present, return text after <R/>"""
    if R_END in text:
        # Extract everything after the reasoning block
        return text.split(R_END)[-1].strip()
    elif R_START in text:
        # Has start but no end - take text before start
        return text.split(R_START)[0].strip()
    return text.strip()


MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 64, "lora_alpha": 128,
        "lr": 1e-4, "batch": 4, "grad_accum": 8,
        "max_length": 512, "grad_ckpt": False
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 64, "lora_alpha": 128,
        "lr": 1e-4, "batch": 2, "grad_accum": 16,  # Increased batch, reduced grad_accum
        "max_length": 512, "grad_ckpt": False
    }
}


class WordAccuracyCallback(TrainerCallback):
    """Evaluate word/letter exact match accuracy"""

    def __init__(self, eval_dataset, tokenizer, model, eval_steps=500,
                 target_accuracy=90.0, output_dir=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.eval_steps = eval_steps
        self.target = target_accuracy
        self.output_dir = output_dir
        self.best_acc = 0.0
        self.current_acc = 0.0  # Current accuracy (0.0 until first eval)
        self.best_step = 0
        self.train_correct = 0
        self.train_total = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log current accuracy with training loss"""
        if logs and state.global_step > 0:
            logs['current_accuracy'] = self.current_acc
            logs['best_accuracy'] = self.best_acc
            
            print(f"Step {state.global_step} | Loss: {logs.get('loss', 0):.4f} | "
                  f"Val Acc: {self.current_acc:.1f}% | Best: {self.best_acc:.1f}%")

    def on_step_end(self, args, state, control, **kwargs):
        # Full evaluation every eval_steps
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            acc, no_reason_pct = self._evaluate(state.global_step)
            self.current_acc = acc  # Update current accuracy

            if acc > self.best_acc:
                self.best_acc = acc
                self.best_step = state.global_step
                self._save_checkpoint(state.global_step)

            if acc >= self.target:
                print(f"\nTarget accuracy {self.target}% reached!")
                control.should_training_stop = True

    def _evaluate(self, step: int):
        print(f"\n{'='*50}")
        print(f"Type 3 (WORD) Evaluation - Step {step}")
        print(f"{'='*50}")

        self.model.eval()
        correct = 0
        total = 0
        had_reasoning = 0  # Track how many had reasoning (will be skipped)

        samples = list(self.eval_dataset)[:200]

        with torch.no_grad():
            for sample in samples:
                try:
                    prompt = sample['prompt']
                    expected = sample['answer'].strip().upper()

                    inputs = self.tokenizer(prompt, return_tensors="pt",
                                           truncation=True, max_length=400).to(self.model.device)

                    # Generate - may have reasoning, so allow more tokens
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,  # Allow space for reasoning if model produces it
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=False
                    ).strip()

                    # Check if had reasoning tokens (for stats)
                    if R_START in response or R_END in response:
                        had_reasoning += 1

                    # Skip reasoning block if present, extract answer after <R/>
                    clean_response = skip_reasoning_block(response)

                    # Extract answer (first word/letter from clean response)
                    predicted = clean_response.split()[0] if clean_response.split() else ""
                    predicted = re.sub(r'[^A-Za-z가-힣0-9]', '', predicted).upper()

                    # For MCQ, just compare first letter
                    if len(expected) == 1 and expected in "ABCDE":
                        if predicted and predicted[0] == expected:
                            correct += 1
                    else:
                        # Word comparison
                        expected_clean = re.sub(r'[^가-힣A-Za-z0-9]', '', expected)
                        if predicted == expected_clean:
                            correct += 1

                    total += 1

                except Exception as e:
                    continue

        acc = 100 * correct / total if total > 0 else 0
        reasoning_pct = 100 * had_reasoning / total if total > 0 else 0

        print(f"  Samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {acc:.1f}%")
        print(f"  Had reasoning (skipped): {had_reasoning} ({reasoning_pct:.1f}%)")
        print(f"  Best: {self.best_acc:.1f}% (step {self.best_step})")
        print(f"  Target: {self.target}%")

        self.model.train()
        return acc, reasoning_pct

    def _save_checkpoint(self, step: int):
        if self.output_dir:
            ckpt = Path(self.output_dir) / "best_checkpoint"
            ckpt.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(ckpt)
            self.tokenizer.save_pretrained(ckpt)
            with open(ckpt / "info.json", 'w') as f:
                json.dump({"step": step, "accuracy": self.best_acc}, f)


def load_data(data_dir: Path, max_samples: int = None):
    train, val = [], []
    for split, samples in [("train", train), ("validation", val)]:
        f = data_dir / split / "data.jsonl"
        if f.exists():
            with open(f, 'r', encoding='utf-8') as file:
                for i, l in enumerate(file):
                    if max_samples and i >= max_samples:
                        break
                    samples.append(json.loads(l))
    print(f"Loaded {len(train)} train, {len(val)} val")
    return Dataset.from_list(train), Dataset.from_list(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma-4b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--target-accuracy", type=float, default=90.0)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training samples for testing")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    output_dir = str(OUTPUT_DIR / args.model)

    print("=" * 60)
    print("Type 3: WORD Training (No Reasoning)")
    print("=" * 60)
    print("Model should output ONLY the answer word/letter")
    print(f"Model should NOT use {R_START}...{R_END} tokens")

    train_data, eval_data = load_data(DATA_DIR, max_samples=args.max_samples)
    if len(train_data) == 0:
        print("No data! Run refine_4types.py first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(cfg['path'], quantization_config=bnb,
                                                  device_map="cuda:0", trust_remote_code=True,
                                                  attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=cfg['batch'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'], weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type="cosine", logging_steps=10,
        save_strategy="steps", save_steps=1000,
        fp16=False, bf16=False, max_length=cfg['max_length'],
        gradient_checkpointing=cfg['grad_ckpt'],
        optim="paged_adamw_8bit", max_grad_norm=0.3, report_to="none"
    )

    callback = WordAccuracyCallback(eval_data, tokenizer, model,
                                    args.eval_steps, args.target_accuracy, output_dir)

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_data,
                         processing_class=tokenizer, callbacks=[callback])
    trainer.train()

    trainer.save_model(output_dir + "/final")
    tokenizer.save_pretrained(output_dir + "/final")
    print(f"\nDone! Best accuracy: {callback.best_acc:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
