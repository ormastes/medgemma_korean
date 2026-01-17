#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 02 Long: Extended MCQ training with periodic full evaluation

Trains on KorMedMCQA with full 604-sample evaluation every N epochs.
Uses NORMAL mode (simple prompt) for efficient training.

Usage:
    python train_02_long.py --epochs 100 --eval-every 10
    python train_02_long.py --epochs 50 --eval-every 5 --target-accuracy 90
"""

import sys
import json
import torch
import gc
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS, MEMORY_CONFIGS
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, TrainerCallback
)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from trl import SFTTrainer

BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

# Model paths
INPUT_DIR = BASE_DIR / "model" / "01_trained"  # From train_01
OUTPUT_DIR = BASE_DIR / "model" / "02_long_trained"

# Log file
LOG_DIR = BASE_DIR / "log"
LOG_FILE = LOG_DIR / "train_02_long.log"


def log(msg: str, level: str = "INFO"):
    """Write log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    print(log_msg)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    except:
        pass


# Simple prompt template (NORMAL mode)
PROMPT_TEMPLATE = """<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
E) {E}
<end_of_turn>
<start_of_turn>model
"""

# Response template with reasoning
RESPONSE_TEMPLATE = """<reasoning>
{reasoning}
</reasoning>{answer}<end_of_turn>"""


def load_jsonl(filepath: Path) -> list:
    """Load JSONL file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def generate_reasoning(sample: dict) -> str:
    """Generate simple reasoning for each choice."""
    answer = sample['answer']
    lines = []
    for choice in ['A', 'B', 'C', 'D', 'E']:
        if choice == answer:
            lines.append(f"{choice}) {sample[choice]} - 정답")
        else:
            lines.append(f"{choice}) {sample[choice]} - 오답")
    return "\n".join(lines)


def format_sample(sample: dict) -> dict:
    """Format a single sample for training."""
    prompt = PROMPT_TEMPLATE.format(
        question=sample['question'],
        A=sample['A'], B=sample['B'], C=sample['C'],
        D=sample['D'], E=sample['E']
    )
    reasoning = generate_reasoning(sample)
    response = RESPONSE_TEMPLATE.format(reasoning=reasoning, answer=sample['answer'])

    return {"text": prompt + response}


def extract_answer(response: str) -> str:
    """Extract answer letter from model response."""
    response = response.strip().upper()
    for char in response:
        if char in 'ABCDE':
            return char
    return response[:1] if response else ""


def evaluate_full(model, tokenizer, test_data: list, device: str) -> dict:
    """Run full evaluation on all test samples."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in tqdm(test_data, desc="Full Evaluation", leave=False):
            prompt = f"""<start_of_turn>user
{sample['question']}

A) {sample['A']}
B) {sample['B']}
C) {sample['C']}
D) {sample['D']}
E) {sample['E']}

정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).<end_of_turn>
<start_of_turn>model
"""
            expected = sample['answer']

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted = extract_answer(response)

            if predicted == expected:
                correct += 1
            total += 1

    model.train()
    accuracy = correct / total * 100 if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


class FullEvalCallback(TrainerCallback):
    """Callback for full evaluation at epoch boundaries."""

    def __init__(self, model, tokenizer, test_data, device, eval_every, target_accuracy, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = test_data
        self.device = device
        self.eval_every = eval_every
        self.target_accuracy = target_accuracy
        self.output_dir = output_dir
        self.history = []
        self.best_accuracy = 0
        self.target_reached = False

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)

        if epoch % self.eval_every == 0:
            log(f"\n{'='*60}")
            log(f"Full Evaluation at Epoch {epoch}")
            log(f"{'='*60}")

            result = evaluate_full(self.model, self.tokenizer, self.test_data, self.device)

            log(f"Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")

            self.history.append({
                "epoch": epoch,
                "accuracy": result['accuracy'],
                "correct": result['correct'],
                "total": result['total']
            })

            # Save best model
            if result['accuracy'] > self.best_accuracy:
                self.best_accuracy = result['accuracy']
                best_dir = self.output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(best_dir)
                self.tokenizer.save_pretrained(best_dir)
                log(f"New best model saved: {result['accuracy']:.2f}%")

            # Check target
            if result['accuracy'] >= self.target_accuracy and not self.target_reached:
                self.target_reached = True
                log(f"TARGET REACHED: {result['accuracy']:.2f}% >= {self.target_accuracy}%")
                control.should_training_stop = True

            # Save history
            history_file = self.output_dir / "eval_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.history, f, indent=2)

            log(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train 02 Long with periodic full evaluation")
    parser.add_argument("--model", default="medgemma-4b", choices=["medgemma-4b", "medgemma-27b"])
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every N epochs")
    parser.add_argument("--target-accuracy", type=float, default=90.0, help="Target accuracy to stop")
    parser.add_argument("--base-model", type=str, default=None, help="Base model path")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length")
    args = parser.parse_args()

    log("="*70)
    log("TRAIN 02 LONG - Extended MCQ Training")
    log("="*70)
    log(f"Model: {args.model}")
    log(f"Total epochs: {args.epochs}")
    log(f"Eval every: {args.eval_every} epochs")
    log(f"Target accuracy: {args.target_accuracy}%")

    # Get config
    cfg = MODEL_CONFIGS[args.model].copy()
    mem_cfg = MEMORY_CONFIGS.get(args.model, {})

    # Load data
    log("\nLoading data...")
    train_data = load_jsonl(TRAIN_FILE)
    test_data = load_jsonl(TEST_FILE)
    log(f"Train samples: {len(train_data)}")
    log(f"Test samples: {len(test_data)}")

    # Format training data
    formatted = [format_sample(s) for s in train_data]
    train_dataset = Dataset.from_list(formatted)

    # Load model
    log("\nLoading model...")
    base_model_path = args.base_model or str(INPUT_DIR / args.model)

    # Check if base_model_path is a LoRA adapter or full model
    adapter_config_path = Path(base_model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        # Load as LoRA adapter
        log(f"Loading LoRA adapter from: {base_model_path}")
        peft_config = PeftConfig.from_pretrained(base_model_path)

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map=args.device,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # Load tokenizer from adapter (has extended vocab)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Resize embeddings
        if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
            log(f"Resizing embeddings: {base_model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, base_model_path, is_trainable=True)

        if mem_cfg.get('use_gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
            log("Gradient checkpointing: ENABLED")
    else:
        # Load as full model
        log(f"Loading full model from: {base_model_path}")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map=args.device,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Output directory
    output_dir = OUTPUT_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=cfg['batch'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        bf16=True,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    # Evaluation callback
    eval_callback = FullEvalCallback(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        device=args.device,
        eval_every=args.eval_every,
        target_accuracy=args.target_accuracy,
        output_dir=output_dir,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        max_seq_length=args.max_length,
        callbacks=[eval_callback],
    )

    # Initial evaluation
    log("\nInitial Evaluation (before training)...")
    initial_result = evaluate_full(model, tokenizer, test_data, args.device)
    log(f"Initial accuracy: {initial_result['accuracy']:.2f}%")
    eval_callback.history.append({
        "epoch": 0,
        "accuracy": initial_result['accuracy'],
        "correct": initial_result['correct'],
        "total": initial_result['total']
    })

    # Train
    log(f"\nStarting training for {args.epochs} epochs...")
    log(f"Will evaluate every {args.eval_every} epochs on {len(test_data)} test samples")

    trainer.train()

    # Final save
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Final evaluation
    log("\nFinal Evaluation...")
    final_result = evaluate_full(model, tokenizer, test_data, args.device)
    log(f"Final accuracy: {final_result['accuracy']:.2f}%")

    # Save training info
    info = {
        "model": args.model,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "target_accuracy": args.target_accuracy,
        "initial_accuracy": initial_result['accuracy'],
        "final_accuracy": final_result['accuracy'],
        "best_accuracy": eval_callback.best_accuracy,
        "target_reached": eval_callback.target_reached,
        "history": eval_callback.history,
    }

    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    log("\n" + "="*70)
    log("TRAINING COMPLETE")
    log("="*70)
    log(f"Initial accuracy: {initial_result['accuracy']:.2f}%")
    log(f"Final accuracy: {final_result['accuracy']:.2f}%")
    log(f"Best accuracy: {eval_callback.best_accuracy:.2f}%")
    log(f"Target reached: {eval_callback.target_reached}")
    log(f"Output: {output_dir}")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        log(f"FATAL ERROR: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        exit(1)
