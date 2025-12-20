#!/usr/bin/env python3
"""
Category B Training: Text Generation

- Input: Question/Instruction
- Output: Full text response
- Method: Sequence-to-sequence generation
- Evaluation: Perplexity
- Target: Perplexity < 3.0
"""

import os
import sys
import json
import torch
import argparse
import math
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
DATA_DIR = BASE_DIR / "data" / "refined" / "category_b_generation"

MODELS = {
    "medgemma-4b": "google/medgemma-4b-it",
    "medgemma-27b": "google/medgemma-27b-text-it",
}


class PerplexityCallback(TrainerCallback):
    """Evaluate perplexity for generation training."""

    def __init__(self, model, tokenizer, eval_dataset, target_ppl, eval_steps, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.target_ppl = target_ppl
        self.eval_steps = eval_steps
        self.output_dir = Path(output_dir)
        self.best_ppl = float('inf')
        self.history = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            ppl = self._evaluate()
            self.history.append({
                "step": state.global_step,
                "perplexity": ppl,
                "time": str(datetime.now())
            })
            print(f"\n[Step {state.global_step}] Perplexity: {ppl:.3f}")

            if ppl < self.best_ppl:
                self.best_ppl = ppl
                self._save_best()

            self._save_history()

            if ppl <= self.target_ppl:
                print(f"\n🎉 Target perplexity {self.target_ppl} reached!")
                control.should_training_stop = True

        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        ppl = self._evaluate()
        self.history.append({
            "epoch": state.epoch,
            "step": state.global_step,
            "perplexity": ppl,
            "time": str(datetime.now())
        })
        print(f"\n[Epoch {state.epoch}] Perplexity: {ppl:.3f}")

        if ppl < self.best_ppl:
            self.best_ppl = ppl
            self._save_best()

        self._save_history()
        return control

    def _evaluate(self, max_samples=200):
        """Evaluate perplexity on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        indices = list(range(min(len(self.eval_dataset), max_samples)))

        for idx in tqdm(indices, desc="Eval PPL", leave=False):
            try:
                item = self.eval_dataset[idx]
                text = item.get("text", "")

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss

                if not torch.isnan(loss) and not torch.isinf(loss):
                    num_tokens = inputs["input_ids"].numel()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

            except Exception:
                continue

            if idx % 50 == 0:
                torch.cuda.empty_cache()

        self.model.train()

        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            return min(perplexity, 1000.0)  # Cap at 1000
        return float('inf')

    def _save_best(self):
        best_path = self.output_dir / "best_checkpoint"
        best_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(best_path))
        self.tokenizer.save_pretrained(str(best_path))

    def _save_history(self):
        with open(self.output_dir / "perplexity_history.json", "w") as f:
            json.dump({
                "category": "B",
                "type": "generation",
                "target_perplexity": self.target_ppl,
                "best_perplexity": self.best_ppl,
                "history": self.history
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Category B: Generation Training")
    parser.add_argument("--model", default="medgemma-4b", choices=list(MODELS.keys()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--target-perplexity", type=float, default=3.0)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    # Config based on model size
    is_27b = "27b" in args.model
    config = {
        "batch_size": 1 if is_27b else 2,
        "grad_accum": 32 if is_27b else 16,
        "lr": 2e-5,  # Lower LR for generation
        "lora_r": 64,
        "lora_alpha": 128,
        "max_length": 1024,  # Longer for generation
    }

    output_dir = args.output_dir or str(
        BASE_DIR / "models" / f"category_b_{args.model.replace('-', '_')}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Category B: Generation Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Target: Perplexity < {args.target_perplexity}")
    print(f"Output: Full text response")
    print("=" * 60)

    # Load data
    print("\nLoading Category B data...")
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
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=is_27b,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        max_length=config["max_length"],
        dataset_text_field="text",
    )

    # Callback
    callback = PerplexityCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        target_ppl=args.target_perplexity,
        eval_steps=args.eval_steps,
        output_dir=output_dir,
    )

    # Train
    print("\nStarting Generation Training...")
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

    print(f"\nComplete! Best perplexity: {callback.best_ppl:.3f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
