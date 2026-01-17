#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 02 Answer Only: MCQ training with REASONING support

KEY FIXES:
1. Uses "Reasoning 후" prompt to match train_01 validation
2. Trains model to output reasoning block then answer
3. Evaluation skips reasoning block to extract final answer

Uses completion_only_loss=True with prompt-completion format.
"""

import sys
import json
import torch
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from datasets import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS, MEMORY_CONFIGS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

INPUT_DIR = BASE_DIR / "model" / "01_trained"
OUTPUT_DIR = BASE_DIR / "model" / "03_answer_only"
LOG_DIR = BASE_DIR / "log"
LOG_FILE = LOG_DIR / "train_02_answer_only.log"


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    print(log_msg, flush=True)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    except:
        pass


# Prompt-completion format for loss masking
# With completion_only_loss=True, loss is only computed on completion (answer)
# IMPORTANT: Use "Reasoning 후" to match train_01 validation prompt
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

# Completion includes simple reasoning then answer
# Format: <reasoning>brief analysis</reasoning>ANSWER
COMPLETION_TEMPLATE = "<reasoning>정답은 {answer}입니다.</reasoning>{answer}<end_of_turn>"


def load_jsonl(filepath):
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def format_sample(sample):
    """Format sample as prompt-completion pair for loss masking."""
    prompt = PROMPT_TEMPLATE.format(
        question=sample['question'],
        A=sample['A'], B=sample['B'], C=sample['C'],
        D=sample['D'], E=sample['E'],
    )
    completion = COMPLETION_TEMPLATE.format(answer=sample['answer'])
    return {"prompt": prompt, "completion": completion}


def extract_answer(response):
    """Extract answer from response, skipping reasoning block.

    Handles formats like:
    - "<reasoning>...</reasoning>B"
    - "B"
    - "The answer is B"
    """
    response = response.strip()

    # Skip reasoning block if present
    if '</reasoning>' in response:
        # Get content after </reasoning>
        response = response.split('</reasoning>')[-1]
    elif '<reasoning>' in response:
        # Incomplete reasoning block - try to get answer anyway
        pass

    response = response.strip().upper()
    for char in response:
        if char in 'ABCDE':
            return char
    return ""


def evaluate(model, tokenizer, test_data, device, max_samples=None):
    """Evaluate accuracy with reasoning-aware extraction."""
    model.eval()
    correct = 0
    total = 0

    samples = test_data[:max_samples] if max_samples else test_data

    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating", leave=False):
            # Use same prompt as training (Reasoning 후)
            prompt = """<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
E) {E}

<end_of_turn>
<start_of_turn>model
""".format(
                question=sample['question'],
                A=sample['A'], B=sample['B'], C=sample['C'],
                D=sample['D'], E=sample['E']
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate more tokens to allow for reasoning block
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,  # Increased to allow reasoning
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted = extract_answer(response)

            if predicted == sample['answer']:
                correct += 1
            total += 1

    model.train()
    accuracy = correct / total * 100 if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def load_model(model_path, device, model_cfg, mem_cfg):
    """Load model from path."""
    adapter_config_path = Path(model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        log(f"Loading LoRA adapter from: {model_path}")
        peft_config = PeftConfig.from_pretrained(model_path)

        log("Loading model in bfloat16")
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
            log(f"Resizing embeddings: {base_model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)

        if mem_cfg.get('use_gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
            log("Gradient checkpointing: ENABLED")
    else:
        raise ValueError(f"Expected LoRA adapter at {model_path}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma-4b", choices=["medgemma-4b", "medgemma-27b"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-model", type=str, help="Override base model path")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--eval-samples", type=int, default=604, help="Evaluation samples")
    args = parser.parse_args()

    log("="*70)
    log("TRAIN 02 ANSWER ONLY - WITH LOSS MASKING")
    log("="*70)
    log(f"Model: {args.model}")
    log(f"Epochs: {args.epochs}")
    log(f"Eval every: {args.eval_every} epochs")

    model_cfg = MODEL_CONFIGS[args.model].copy()
    mem_cfg = MEMORY_CONFIGS.get(args.model, {})

    # Load data
    log("\nLoading data...")
    train_data = load_jsonl(TRAIN_FILE)
    test_data = load_jsonl(TEST_FILE)
    log(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Format training data
    formatted = [format_sample(s) for s in train_data]
    train_dataset = Dataset.from_list(formatted)

    # Determine input model path
    base_model_path = args.base_model or str(INPUT_DIR / args.model)
    output_dir = OUTPUT_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    log(f"\nLoading model from: {base_model_path}")
    model, tokenizer = load_model(base_model_path, args.device, model_cfg, mem_cfg)

    log("\nUsing completion_only_loss=True (loss only on reasoning+answer)")
    log("Prompt format: 'Reasoning 후 정답 알파벳 하나만 답하세요.'")
    log("Completion format: '<reasoning>정답은 X입니다.</reasoning>X'")

    # Track accuracy history
    history = []

    # Initial evaluation
    log("\n" + "="*60)
    log(f"INITIAL EVALUATION ({args.eval_samples} samples)")
    log("="*60)
    result = evaluate(model, tokenizer, test_data, args.device, args.eval_samples)
    log(f"Initial Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
    history.append({
        "epoch": 0,
        "accuracy": result['accuracy'],
        "correct": result['correct'],
        "total": result['total']
    })

    best_accuracy = result['accuracy']

    # Training loop - epoch by epoch with evaluation
    for epoch in range(1, args.epochs + 1):
        log(f"\n{'='*60}")
        log(f"EPOCH {epoch}/{args.epochs}")
        log(f"{'='*60}")

        training_args = SFTConfig(
            output_dir=str(output_dir / f"epoch_{epoch}"),
            num_train_epochs=1,
            per_device_train_batch_size=model_cfg['batch'],
            gradient_accumulation_steps=model_cfg['grad_accum'],
            learning_rate=model_cfg['lr'],
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            bf16=True,
            dataloader_pin_memory=False,
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            remove_unused_columns=False,
            # KEY: Only compute loss on completion (answer), not prompt
            completion_only_loss=True,
            max_length=512,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate after each epoch
        if epoch % args.eval_every == 0:
            log(f"\nEvaluating after epoch {epoch}...")
            result = evaluate(model, tokenizer, test_data, args.device, args.eval_samples)
            log(f"Epoch {epoch} Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
            history.append({
                "epoch": epoch,
                "accuracy": result['accuracy'],
                "correct": result['correct'],
                "total": result['total']
            })

            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                # Save best model
                best_dir = output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                log(f"New best model saved: {best_accuracy:.2f}%")

            # Save history
            with open(output_dir / "accuracy_history.json", 'w') as f:
                json.dump(history, f, indent=2)

    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Print summary
    log("\n" + "="*70)
    log("TRAINING COMPLETE - ACCURACY HISTORY")
    log("="*70)
    log(f"{'Epoch':<10}{'Accuracy':<15}{'Change':<10}")
    log("-"*35)

    prev_acc = None
    for h in history:
        change = ""
        if prev_acc is not None:
            diff = h['accuracy'] - prev_acc
            change = f"{'+' if diff >= 0 else ''}{diff:.2f}%"
        log(f"{h['epoch']:<10}{h['accuracy']:.2f}%{'':<7}{change}")
        prev_acc = h['accuracy']

    log("-"*35)
    log(f"Best accuracy: {best_accuracy:.2f}%")
    log(f"Output: {output_dir}")

    # Save training info
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump({
            "model": args.model,
            "epochs": args.epochs,
            "best_accuracy": best_accuracy,
            "history": history,
            "loss_masking": "completion_only_loss=True",
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        log(f"FATAL ERROR: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        exit(1)
