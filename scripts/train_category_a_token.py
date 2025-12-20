#!/usr/bin/env python3
"""
Category A Training: Token/Letter Answer

- Input: MCQ question with options
- Output: Single letter (A, B, C, D, E)
- Method: Direct token prediction
- Evaluation: Exact match accuracy
- Target: 90%+ accuracy
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "refined" / "category_a_token"

MODELS = {
    "medgemma-4b": "google/medgemma-4b-it",
    "medgemma-27b": "google/medgemma-27b-text-it",
}


class TokenAccuracyCallback(TrainerCallback):
    """Evaluate accuracy for token (letter) prediction."""

    def __init__(self, model, tokenizer, eval_dataset, target_acc, eval_steps, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.target_acc = target_acc
        self.eval_steps = eval_steps
        self.output_dir = Path(output_dir)
        self.best_acc = 0.0
        self.history = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            acc = self._evaluate()
            self.history.append({
                "step": state.global_step,
                "accuracy": acc,
                "time": str(datetime.now())
            })
            print(f"\n[Step {state.global_step}] Token Accuracy: {acc:.2f}%")

            if acc > self.best_acc:
                self.best_acc = acc
                self._save_best()

            self._save_history()

            if acc >= self.target_acc:
                print(f"\n🎉 Target {self.target_acc}% reached!")
                control.should_training_stop = True

        return control

    def _evaluate(self, max_samples=300):
        """Evaluate token prediction accuracy."""
        self.model.eval()
        correct = 0
        total = 0

        indices = list(range(min(len(self.eval_dataset), max_samples)))

        for idx in tqdm(indices, desc="Eval", leave=False):
            try:
                item = self.eval_dataset[idx]
                prompt = item.get("prompt", "")
                expected = item.get("answer", "").upper().strip()

                if not expected or expected not in "ABCDE":
                    continue

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=3,  # Only need 1 letter
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().upper()

                # Get first letter
                predicted = None
                for char in response[:3]:
                    if char in "ABCDE":
                        predicted = char
                        break

                if predicted == expected:
                    correct += 1
                total += 1

            except Exception:
                continue

            if total % 50 == 0:
                torch.cuda.empty_cache()

        self.model.train()
        return (correct / total * 100) if total > 0 else 0

    def _save_best(self):
        best_path = self.output_dir / "best_checkpoint"
        best_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(best_path))
        self.tokenizer.save_pretrained(str(best_path))

    def _save_history(self):
        with open(self.output_dir / "accuracy_history.json", "w") as f:
            json.dump({
                "category": "A",
                "type": "token_prediction",
                "target": self.target_acc,
                "best": self.best_acc,
                "history": self.history
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Category A: Token Answer Training")
    parser.add_argument("--model", default="medgemma-4b", choices=list(MODELS.keys()))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--target-accuracy", type=float, default=90.0)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    # Config based on model size
    is_27b = "27b" in args.model
    config = {
        "batch_size": 1 if is_27b else 4,
        "grad_accum": 32 if is_27b else 8,
        "lr": 5e-5 if is_27b else 1e-4,
        "lora_r": 64,
        "lora_alpha": 128,
        "max_length": 512,  # Short for MCQ
    }

    output_dir = args.output_dir or str(
        BASE_DIR / "models" / f"category_a_{args.model.replace('-', '_')}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Category A: Token Answer Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Target: {args.target_accuracy}% accuracy")
    print(f"Output: Letter (A-E)")
    print("=" * 60)

    # Load data
    print("\nLoading Category A data...")
    train_ds = load_from_disk(str(DATA_DIR / "train"))
    eval_ds = load_from_disk(str(DATA_DIR / "validation"))
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Load model
    print(f"\nLoading {args.model}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    sft_config = SFTConfig(
        output_dir=str(Path(output_dir) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=is_27b,
        optim="paged_adamw_8bit",
        max_grad_norm=0.5,
        report_to="none",
        max_length=config["max_length"],
        dataset_text_field="text",
    )

    # Callback
    callback = TokenAccuracyCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        target_acc=args.target_accuracy,
        eval_steps=args.eval_steps,
        output_dir=output_dir,
    )

    # Train
    print("\nStarting Token Answer Training...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    trainer.train()

    # Save final
    print("\nSaving final model...")
    model.save_pretrained(str(Path(output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(output_dir) / "final"))

    print(f"\nComplete! Best accuracy: {callback.best_acc:.2f}%")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
