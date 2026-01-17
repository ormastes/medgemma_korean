#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 02 Alternating: FULL mode → NORMAL mode with validation

Features:
- Configurable loop counts via config struct
- Saves model after each mode (can resume)
- --check mode: all counts=1 to verify pipeline works
- --real mode: actual training with full counts

Usage:
    python train_02_alternating.py --check    # Quick test (all counts=1)
    python train_02_alternating.py --real     # Full training
    python train_02_alternating.py --cycles 3 --epochs-per-mode 5  # Custom
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
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
)
from peft import PeftModel, PeftConfig
from trl import SFTTrainer

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

INPUT_DIR = BASE_DIR / "model" / "02_trained"
OUTPUT_DIR = BASE_DIR / "model" / "02_alternating"
LOG_DIR = BASE_DIR / "log"
LOG_FILE = LOG_DIR / "train_02_alternating.log"


@dataclass
class TrainConfig:
    """Training configuration with all loop counts."""
    # Model
    model_name: str = "medgemma-4b"
    device: str = "cuda:0"  # A6000

    # Loop counts
    cycles: int = 5                    # Number of FULL+NORMAL cycles
    epochs_full: int = 10              # Epochs per FULL mode
    epochs_normal: int = 10            # Epochs per NORMAL mode
    max_steps: int = -1                # -1 = use epochs, >0 = limit steps

    # Evaluation
    eval_samples: int = 604            # All test samples (604)
    target_accuracy: float = 90.0      # Stop if reached

    # Paths
    base_model_path: str = None        # Override input model
    output_dir: str = None             # Override output dir

    @classmethod
    def check_config(cls):
        """Quick test config - 1 step only"""
        return cls(
            cycles=1,
            epochs_full=1,
            epochs_normal=1,
            max_steps=1,          # Only 1 training step!
            eval_samples=5,       # Only 5 samples for eval
        )

    @classmethod
    def real_config(cls):
        """Real training config"""
        return cls(
            cycles=5,
            epochs_full=10,
            epochs_normal=10,
            max_steps=-1,         # Use epochs
            eval_samples=604,
        )


def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    print(log_msg)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    except:
        pass


# ============================================================
# FULL MODE: With example (answer only, no reasoning)
# ============================================================
FULL_PROMPT = """<start_of_turn>user
다음 의료 문제를 읽고 정답 알파벳 하나만 답하세요.

예시:
문제: 폐렴의 가장 흔한 원인균은?
A) 대장균
B) 폐렴구균
C) 황색포도상구균
D) 녹농균
E) 인플루엔자균
정답: B

문제: {question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}
정답:<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>"""

# ============================================================
# NORMAL MODE: Simple prompt (answer only)
# ============================================================
NORMAL_PROMPT = """<start_of_turn>user
정답 알파벳 하나만 답하세요.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
E) {E}
<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>"""

# No reasoning - just answer letter
RESPONSE_TEMPLATE = None  # Not used anymore


def load_jsonl(filepath: Path) -> list:
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def format_full(sample: dict) -> dict:
    """Format for FULL mode - answer only, no reasoning."""
    text = FULL_PROMPT.format(
        question=sample['question'],
        A=sample['A'], B=sample['B'], C=sample['C'],
        D=sample['D'], E=sample['E'],
        answer=sample['answer']
    )
    return {"text": text}


def format_normal(sample: dict) -> dict:
    """Format for NORMAL mode - answer only, no reasoning."""
    text = NORMAL_PROMPT.format(
        question=sample['question'],
        A=sample['A'], B=sample['B'], C=sample['C'],
        D=sample['D'], E=sample['E'],
        answer=sample['answer']
    )
    return {"text": text}


def extract_answer(response: str) -> str:
    """Extract first A/B/C/D/E from response."""
    response = response.strip().upper()
    for char in response:
        if char in 'ABCDE':
            return char
    return ""


def evaluate(model, tokenizer, test_data: list, device: str, max_samples: int = None) -> dict:
    """Evaluate on test samples - expects just answer letter output."""
    model.eval()
    correct = 0
    total = 0

    samples = test_data[:max_samples] if max_samples else test_data

    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating", leave=False):
            # Simple prompt - just ask for answer
            prompt = f"""<start_of_turn>user
정답 알파벳 하나만 답하세요.

{sample['question']}

A) {sample['A']}
B) {sample['B']}
C) {sample['C']}
D) {sample['D']}
E) {sample['E']}
<end_of_turn>
<start_of_turn>model
"""
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Only need a few tokens for just the answer letter
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Extract first A/B/C/D/E
            predicted = extract_answer(response)

            if predicted == sample['answer']:
                correct += 1
            total += 1

    model.train()
    accuracy = correct / total * 100 if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def save_checkpoint(model, tokenizer, output_dir: Path, name: str):
    """Save model checkpoint."""
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    log(f"Saved checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def train_mode(model, tokenizer, train_data, mode, epochs, cfg, output_dir, device, max_steps=-1):
    """Train for specified mode and epochs (or max_steps if >0)."""
    log(f"\n{'='*60}")
    if max_steps > 0:
        log(f"Training {mode.upper()} mode for {max_steps} steps")
    else:
        log(f"Training {mode.upper()} mode for {epochs} epochs")
    log(f"{'='*60}")

    # Format data based on mode
    if mode == "full":
        formatted = [format_full(s) for s in train_data]
    else:
        formatted = [format_normal(s) for s in train_data]

    dataset = Dataset.from_list(formatted)
    log(f"Samples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=str(output_dir / f"{mode}_checkpoints"),
        num_train_epochs=epochs if max_steps <= 0 else 1,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=cfg['batch'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        logging_steps=1 if max_steps > 0 else 20,
        save_strategy="no",
        report_to="none",
        bf16=True,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_path: str, device: str, model_cfg: dict, mem_cfg: dict):
    """Load model from path (supports LoRA adapters)."""
    adapter_config_path = Path(model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        log(f"Loading LoRA adapter from: {model_path}")
        peft_config = PeftConfig.from_pretrained(model_path)

        # 32-bit (bfloat16) training - no quantization
        log("Loading model in bfloat16 (32-bit training, no quantization)")
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
    parser.add_argument("--check", action="store_true", help="Quick test mode (all counts=1)")
    parser.add_argument("--real", action="store_true", help="Real training mode")
    parser.add_argument("--model", default="medgemma-4b", choices=["medgemma-4b", "medgemma-27b"])
    parser.add_argument("--cycles", type=int, help="Number of FULL+NORMAL cycles")
    parser.add_argument("--epochs-full", type=int, help="Epochs per FULL mode")
    parser.add_argument("--epochs-normal", type=int, help="Epochs per NORMAL mode")
    parser.add_argument("--epochs-per-mode", type=int, help="Epochs for both modes (shorthand)")
    parser.add_argument("--eval-samples", type=int, help="Evaluation samples")
    parser.add_argument("--base-model", type=str, help="Override base model path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-accuracy", type=float, help="Target accuracy to stop")
    args = parser.parse_args()

    # Build config
    if args.check:
        config = TrainConfig.check_config()
        log("Using CHECK config (quick test)")
    elif args.real:
        config = TrainConfig.real_config()
        log("Using REAL config (full training)")
    else:
        config = TrainConfig()

    # Override with CLI args
    config.model_name = args.model
    config.device = args.device
    if args.cycles:
        config.cycles = args.cycles
    if args.epochs_full:
        config.epochs_full = args.epochs_full
    if args.epochs_normal:
        config.epochs_normal = args.epochs_normal
    if args.epochs_per_mode:
        config.epochs_full = args.epochs_per_mode
        config.epochs_normal = args.epochs_per_mode
    if args.eval_samples:
        config.eval_samples = args.eval_samples
    if args.base_model:
        config.base_model_path = args.base_model
    if args.target_accuracy:
        config.target_accuracy = args.target_accuracy

    log("="*70)
    log("TRAIN 02 ALTERNATING: FULL → NORMAL → FULL → NORMAL ...")
    log("="*70)
    log(f"Config: {asdict(config)}")

    model_cfg = MODEL_CONFIGS[config.model_name].copy()
    mem_cfg = MEMORY_CONFIGS.get(config.model_name, {})

    # Load data
    log("\nLoading data...")
    train_data = load_jsonl(TRAIN_FILE)
    test_data = load_jsonl(TEST_FILE)
    log(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Determine input model path
    base_model_path = config.base_model_path or str(INPUT_DIR / config.model_name)

    # Output directory
    output_dir = Path(config.output_dir) if config.output_dir else OUTPUT_DIR / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to resume
    last_checkpoint = None
    checkpoint_info_file = output_dir / "checkpoint_info.json"
    if checkpoint_info_file.exists():
        with open(checkpoint_info_file, 'r') as f:
            checkpoint_info = json.load(f)
        last_checkpoint = checkpoint_info.get('last_checkpoint')
        if last_checkpoint and Path(last_checkpoint).exists():
            log(f"Found checkpoint to resume: {last_checkpoint}")
            base_model_path = last_checkpoint

    # Load model
    log(f"\nLoading model from: {base_model_path}")
    model, tokenizer = load_model(base_model_path, config.device, model_cfg, mem_cfg)

    # Track accuracy history
    history = []

    # Initial evaluation
    log("\n" + "="*60)
    log(f"INITIAL EVALUATION ({config.eval_samples} samples)")
    log("="*60)
    result = evaluate(model, tokenizer, test_data, config.device, config.eval_samples)
    log(f"Initial Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
    history.append({
        "cycle": 0,
        "mode": "initial",
        "epochs": 0,
        "accuracy": result['accuracy'],
        "correct": result['correct'],
        "total": result['total']
    })

    best_accuracy = result['accuracy']
    target_reached = False

    # Training cycles
    for cycle in range(1, config.cycles + 1):
        log(f"\n{'#'*70}")
        log(f"CYCLE {cycle}/{config.cycles}")
        log(f"{'#'*70}")

        # FULL mode
        train_mode(model, tokenizer, train_data, "full", config.epochs_full,
                   model_cfg, output_dir, config.device, config.max_steps)

        # Save checkpoint after FULL (overwrite 'latest' to save disk space)
        checkpoint_path = save_checkpoint(model, tokenizer, output_dir, "latest")
        with open(checkpoint_info_file, 'w') as f:
            json.dump({"last_checkpoint": str(checkpoint_path), "cycle": cycle, "mode": "full"}, f)

        log(f"\nEvaluating after FULL mode ({config.eval_samples} samples)...")
        result = evaluate(model, tokenizer, test_data, config.device, config.eval_samples)
        log(f"Accuracy after FULL: {result['accuracy']:.2f}%")
        history.append({
            "cycle": cycle,
            "mode": "full",
            "epochs": config.epochs_full,
            "accuracy": result['accuracy'],
            "correct": result['correct'],
            "total": result['total']
        })

        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            save_checkpoint(model, tokenizer, output_dir, "best")
            log(f"New best model saved: {best_accuracy:.2f}%")

        if result['accuracy'] >= config.target_accuracy:
            log(f"TARGET REACHED: {result['accuracy']:.2f}% >= {config.target_accuracy}%")
            target_reached = True
            break

        # NORMAL mode
        train_mode(model, tokenizer, train_data, "normal", config.epochs_normal,
                   model_cfg, output_dir, config.device, config.max_steps)

        # Save checkpoint after NORMAL (overwrite 'latest' to save disk space)
        checkpoint_path = save_checkpoint(model, tokenizer, output_dir, "latest")
        with open(checkpoint_info_file, 'w') as f:
            json.dump({"last_checkpoint": str(checkpoint_path), "cycle": cycle, "mode": "normal"}, f)

        log(f"\nEvaluating after NORMAL mode ({config.eval_samples} samples)...")
        result = evaluate(model, tokenizer, test_data, config.device, config.eval_samples)
        log(f"Accuracy after NORMAL: {result['accuracy']:.2f}%")
        history.append({
            "cycle": cycle,
            "mode": "normal",
            "epochs": config.epochs_normal,
            "accuracy": result['accuracy'],
            "correct": result['correct'],
            "total": result['total']
        })

        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            save_checkpoint(model, tokenizer, output_dir, "best")
            log(f"New best model saved: {best_accuracy:.2f}%")

        if result['accuracy'] >= config.target_accuracy:
            log(f"TARGET REACHED: {result['accuracy']:.2f}% >= {config.target_accuracy}%")
            target_reached = True
            break

        # Save history after each cycle
        with open(output_dir / "accuracy_history.json", 'w') as f:
            json.dump(history, f, indent=2)

    # Final save
    save_checkpoint(model, tokenizer, output_dir, "final")

    # Print summary
    log("\n" + "="*70)
    log("TRAINING COMPLETE - ACCURACY HISTORY")
    log("="*70)
    log(f"{'Cycle':<8}{'Mode':<10}{'Epochs':<10}{'Accuracy':<12}{'Change':<10}")
    log("-"*50)

    prev_acc = None
    for h in history:
        change = ""
        if prev_acc is not None:
            diff = h['accuracy'] - prev_acc
            change = f"{'+' if diff >= 0 else ''}{diff:.2f}%"
        log(f"{h['cycle']:<8}{h['mode']:<10}{h['epochs']:<10}{h['accuracy']:.2f}%{'':<4}{change}")
        prev_acc = h['accuracy']

    log("-"*50)
    log(f"Best accuracy: {best_accuracy:.2f}%")
    log(f"Target reached: {target_reached}")
    log(f"Output: {output_dir}")

    # Save final info
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump({
            "config": asdict(config),
            "best_accuracy": best_accuracy,
            "target_reached": target_reached,
            "history": history,
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
