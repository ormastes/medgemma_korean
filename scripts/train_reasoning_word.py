#!/usr/bin/env python3
"""
Train reasoning with word answer (Category C)

Format:
    System: 단계별로 추론하고 최종 진단명을 한 단어로 답하세요.
    User: [question]
    Assistant: <reasoning_start>[reasoning]<reasoning_end>[answer_word]

Evaluation:
    - Base score: Exact match on answer word (1.0 or 0.0)
    - Bonus +0.1: Reasoning has >= 10 tokens
    - Bonus +0.05 per key term match (max 0.3)
    - Total max score: 1.4

Training uses custom loss that:
    1. Emphasizes answer word prediction
    2. Gives small weight to reasoning tokens
"""

import argparse
import json
import os
import re
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "refined" / "category_c_reasoning_word"
OUTPUT_DIR = BASE_DIR / "models" / "reasoning_word"

# Special tokens
REASONING_START = "<reasoning_start>"
REASONING_END = "<reasoning_end>"

# Model configurations
MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "grad_accum": 16,
        "max_length": 1024,
        "gradient_checkpointing": False
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 128,
        "lora_alpha": 256,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "grad_accum": 32,
        "max_length": 512,
        "gradient_checkpointing": True
    },
    "stage5": {
        "path": str(BASE_DIR / "models" / "staged_training" / "stage5_harmonization"),
        "lora_r": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "grad_accum": 16,
        "max_length": 1024,
        "gradient_checkpointing": False
    }
}


class ReasoningWordCallback(TrainerCallback):
    """
    Callback for evaluating reasoning with word answer.

    Scoring:
        - Base: Exact match on answer (1.0 or 0.0)
        - Bonus: +0.1 if reasoning >= 10 tokens
        - Bonus: +0.05 per key term match (max 0.3)
    """

    def __init__(self, eval_dataset, tokenizer, model, eval_steps=500,
                 target_score=1.2, output_dir=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.eval_steps = eval_steps
        self.target_score = target_score
        self.output_dir = output_dir
        self.best_score = 0.0
        self.best_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            score = self._evaluate(state.global_step)

            if score > self.best_score:
                self.best_score = score
                self.best_step = state.global_step
                self._save_best_checkpoint(state.global_step)
                print(f"  New best score: {score:.3f} at step {state.global_step}")

            if score >= self.target_score:
                print(f"\nTarget score {self.target_score} reached! Stopping training.")
                control.should_training_stop = True

    def _evaluate(self, step: int) -> float:
        """Evaluate on validation set with scoring"""
        print(f"\n{'='*60}")
        print(f"Evaluating at step {step}...")
        print(f"{'='*60}")

        self.model.eval()
        total_score = 0.0
        num_samples = 0
        exact_matches = 0

        # Sample subset for evaluation
        eval_samples = list(self.eval_dataset)[:100]

        with torch.no_grad():
            for sample in eval_samples:
                try:
                    prompt = sample['prompt']
                    expected_answer = sample['answer']
                    expected_reasoning = sample.get('reasoning', '')
                    key_terms = set(sample.get('key_terms', []))

                    # Generate response
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(self.model.device)

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=False
                    )

                    # Parse response
                    score = self._score_response(
                        response, expected_answer, key_terms
                    )
                    total_score += score

                    if score >= 1.0:
                        exact_matches += 1

                    num_samples += 1

                except Exception as e:
                    continue

        if num_samples == 0:
            return 0.0

        avg_score = total_score / num_samples
        exact_match_rate = exact_matches / num_samples

        print(f"  Samples evaluated: {num_samples}")
        print(f"  Exact matches: {exact_matches} ({100*exact_match_rate:.1f}%)")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  Best score so far: {self.best_score:.3f} (step {self.best_step})")

        self.model.train()
        return avg_score

    def _score_response(self, response: str, expected_answer: str,
                        key_terms: set) -> float:
        """Score a single response"""
        score = 0.0

        # Extract answer after <reasoning_end>
        if REASONING_END in response:
            parts = response.split(REASONING_END)
            if len(parts) >= 2:
                predicted_answer = parts[1].strip()
                reasoning = parts[0].replace(REASONING_START, "").strip()
            else:
                predicted_answer = ""
                reasoning = ""
        else:
            # No reasoning tokens, take first word
            predicted_answer = response.split()[0] if response.split() else ""
            reasoning = ""

        # Clean predicted answer (remove spaces, special chars)
        predicted_answer = re.sub(r'[^가-힣A-Za-z0-9]', '', predicted_answer)
        expected_clean = re.sub(r'[^가-힣A-Za-z0-9]', '', expected_answer)

        # Base score: exact match
        if predicted_answer == expected_clean:
            score = 1.0
        elif expected_clean in predicted_answer or predicted_answer in expected_clean:
            score = 0.5  # Partial match

        # Bonus: reasoning length
        reasoning_tokens = len(reasoning.split())
        if reasoning_tokens >= 10:
            score += 0.1

        # Bonus: key term matches
        if key_terms and reasoning:
            matches = sum(1 for term in key_terms if term in reasoning)
            score += min(0.3, matches * 0.05)

        return score

    def _save_best_checkpoint(self, step: int):
        """Save best checkpoint"""
        if self.output_dir:
            checkpoint_dir = Path(self.output_dir) / "best_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

            with open(checkpoint_dir / "best_info.json", 'w') as f:
                json.dump({
                    "step": step,
                    "score": self.best_score,
                    "target": self.target_score
                }, f, indent=2)


def load_data(data_dir: Path):
    """Load refined reasoning word data"""
    train_file = data_dir / "train" / "data.jsonl"
    val_file = data_dir / "validation" / "data.jsonl"

    train_samples = []
    val_samples = []

    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            train_samples = [json.loads(line) for line in f]

    if val_file.exists():
        with open(val_file, 'r', encoding='utf-8') as f:
            val_samples = [json.loads(line) for line in f]

    print(f"Loaded {len(train_samples)} train, {len(val_samples)} validation samples")

    return Dataset.from_list(train_samples), Dataset.from_list(val_samples)


def add_special_tokens(tokenizer):
    """Add reasoning special tokens to tokenizer"""
    special_tokens = {
        "additional_special_tokens": [REASONING_START, REASONING_END]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens: {REASONING_START}, {REASONING_END}")
    return num_added


def main():
    parser = argparse.ArgumentParser(description="Train reasoning with word answer")
    parser.add_argument("--model", type=str, default="medgemma-4b",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--target-score", type=float, default=1.2,
                        help="Target average score (max 1.4)")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    output_dir = args.output_dir or str(OUTPUT_DIR / args.model)

    print("=" * 60)
    print(f"Reasoning Word Training - {args.model}")
    print("=" * 60)
    print(f"Model: {config['path']}")
    print(f"Target score: {args.target_score}")
    print(f"Output: {output_dir}")

    # Load data
    print("\nLoading data...")
    train_dataset, eval_dataset = load_data(DATA_DIR)

    if len(train_dataset) == 0:
        print("No training data found! Run refine_reasoning_split.py first.")
        return

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['path'],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    num_new_tokens = add_special_tokens(tokenizer)

    # Load model with quantization
    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        config['path'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # Resize embeddings for new tokens
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['grad_accum'],
        learning_rate=config['learning_rate'],
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        bf16=True,
        max_length=config['max_length'],
        gradient_checkpointing=config['gradient_checkpointing'],
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none"
    )

    # Callback
    callback = ReasoningWordCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model=model,
        eval_steps=args.eval_steps,
        target_score=args.target_score,
        output_dir=output_dir
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[callback]
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final
    print("\nSaving final model...")
    trainer.save_model(output_dir + "/final")
    tokenizer.save_pretrained(output_dir + "/final")

    print(f"\nTraining complete!")
    print(f"Best score: {callback.best_score:.3f} at step {callback.best_step}")


if __name__ == "__main__":
    main()
