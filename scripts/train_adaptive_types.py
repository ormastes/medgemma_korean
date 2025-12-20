#!/usr/bin/env python3
"""
Adaptive Multi-Type Training for KorMedMCQA

Strategy:
1. Load all 4 training types
2. Evaluate on KorMedMCQA after each training round
3. Train a few steps on each type, measure KorMedMCQA improvement
4. Focus on the type with best improvement
5. When improvement < 1%, move to next best type
6. Stop when all types show < 1% improvement

This helps identify which data type most benefits KorMedMCQA performance.
"""

import argparse
import json
import re
import time
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "models" / "adaptive_training"
LOG_DIR = BASE_DIR / "logs"

# Training types
TYPES = ["type1_text", "type2_text_reasoning", "type3_word", "type4_word_reasoning"]
TYPE_NAMES = {
    "type1_text": "TEXT (Full text, NO reasoning)",
    "type2_text_reasoning": "TEXT_REASONING (Full text, WITH reasoning)",
    "type3_word": "WORD (MCQ answer, NO reasoning)",
    "type4_word_reasoning": "WORD_REASONING (Word answer, WITH reasoning)"
}

MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 64, "lora_alpha": 128,
        "lr": 1e-4, "batch": 4, "grad_accum": 4,
        "max_length": 512
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 256, "lora_alpha": 512,
        "lr": 5e-5, "batch": 2, "grad_accum": 8,
        "max_length": 768
    }
}


@dataclass
class TypeStats:
    """Track statistics for each training type"""
    name: str
    total_steps: int = 0
    total_samples: int = 0
    best_accuracy: float = 0.0
    last_accuracy: float = 0.0
    accuracy_history: List[Tuple[int, float]] = field(default_factory=list)
    improvement_history: List[float] = field(default_factory=list)
    avg_improvement: float = 0.0
    is_exhausted: bool = False  # No more improvement
    current_position: int = 0  # Position in dataset


@dataclass
class TrainingState:
    """Global training state"""
    global_step: int = 0
    best_accuracy: float = 0.0
    best_step: int = 0
    best_type: str = ""
    current_type: str = ""
    type_stats: Dict[str, TypeStats] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)


def load_kormedmcqa_eval(data_dir: Path) -> Dataset:
    """Load KorMedMCQA validation set for evaluation"""
    samples = []
    val_file = data_dir / "type3_word" / "validation" / "data.jsonl"

    if val_file.exists():
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get('source', '').lower() == 'kormedmcqa':
                    samples.append(data)

    print(f"Loaded {len(samples)} KorMedMCQA evaluation samples")
    return Dataset.from_list(samples)


def load_type_data(data_dir: Path, type_name: str) -> Dataset:
    """Load training data for a specific type"""
    samples = []
    train_file = data_dir / type_name / "train" / "data.jsonl"

    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))

    return Dataset.from_list(samples)


def evaluate_kormedmcqa(
    model,
    tokenizer,
    eval_dataset: Dataset,
    max_samples: int = 200,
    show_examples: int = 3
) -> Tuple[float, int, int]:
    """
    Evaluate model on KorMedMCQA
    Returns: (accuracy, correct, total)
    """
    model.eval()
    correct = 0
    total = 0

    samples = list(eval_dataset)[:max_samples]

    print(f"\n{'='*60}")
    print("KORMEDMCQA EVALUATION")
    print(f"{'='*60}")

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
            try:
                prompt = sample['prompt']
                expected = sample.get('answer', sample.get('completion', '')).strip().upper()
                expected = re.sub(r'[^A-E]', '', expected)[:1]

                if not expected:
                    continue

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=800
                ).to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                # Extract answer letter
                predicted = response.split()[0] if response.split() else ""
                predicted = re.sub(r'[^A-E]', '', predicted.upper())
                predicted = predicted[0] if predicted else ""

                is_correct = (predicted == expected)
                if is_correct:
                    correct += 1
                total += 1

                # Show examples
                if idx < show_examples:
                    status = "CORRECT" if is_correct else "WRONG"
                    print(f"\n  Example {idx+1}: Expected={expected}, Got={predicted} [{status}]")
                    print(f"    Response: {response[:60]}...")

            except Exception as e:
                continue

    accuracy = 100 * correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{total} = {accuracy:.2f}%")
    print(f"{'='*60}\n")

    model.train()
    return accuracy, correct, total


def train_steps_on_type(
    model,
    tokenizer,
    type_dataset: Dataset,
    type_stats: TypeStats,
    num_steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_length: int,
    device: str
) -> float:
    """
    Train for num_steps on a specific type
    Returns average loss
    """
    model.train()

    # Create optimizer for this round
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    total_loss = 0
    steps_done = 0
    accum_loss = 0

    # Get data starting from current position
    start_pos = type_stats.current_position
    data_list = list(type_dataset)

    print(f"\n  Training {num_steps} steps on {type_stats.name}...")
    print(f"  Starting from position {start_pos}/{len(data_list)}")

    pbar = tqdm(total=num_steps, desc=f"  {type_stats.name[:10]}")

    batch_texts = []

    for i in range(num_steps * batch_size * grad_accum + batch_size):
        # Get sample (loop around if needed)
        idx = (start_pos + i) % len(data_list)
        sample = data_list[idx]

        text = sample.get('text', '')
        if not text:
            prompt = sample.get('prompt', '')
            completion = sample.get('completion', sample.get('answer', ''))
            text = f"{prompt}{completion}"

        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            # Forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['input_ids']
            )

            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            # Gradient accumulation step
            if (i // batch_size + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

                total_loss += accum_loss
                accum_loss = 0
                steps_done += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{total_loss/steps_done:.4f}"})

                if steps_done >= num_steps:
                    break

            batch_texts = []

    pbar.close()

    # Update position
    type_stats.current_position = (start_pos + steps_done * batch_size * grad_accum) % len(data_list)
    type_stats.total_steps += steps_done
    type_stats.total_samples += steps_done * batch_size * grad_accum

    avg_loss = total_loss / steps_done if steps_done > 0 else 0
    return avg_loss


def select_best_type(state: TrainingState, min_improvement: float = 1.0) -> Optional[str]:
    """
    Select the type with best average improvement
    Returns None if all types are exhausted
    """
    candidates = []

    for type_name, stats in state.type_stats.items():
        if stats.is_exhausted:
            continue

        # Calculate recent improvement
        if len(stats.improvement_history) > 0:
            recent_improvements = stats.improvement_history[-3:]  # Last 3
            avg_recent = sum(recent_improvements) / len(recent_improvements)
        else:
            avg_recent = float('inf')  # Unknown, try it

        candidates.append((type_name, avg_recent, stats.total_steps))

    if not candidates:
        return None

    # Sort by improvement (higher is better), then by total_steps (prefer less trained)
    candidates.sort(key=lambda x: (-x[1], x[2]))

    best_type = candidates[0][0]
    best_improvement = candidates[0][1]

    # If best improvement is below threshold, mark as exhausted
    if best_improvement < min_improvement and len(state.type_stats[best_type].improvement_history) >= 2:
        state.type_stats[best_type].is_exhausted = True
        print(f"\n  Type {best_type} exhausted (avg improvement {best_improvement:.2f}% < {min_improvement}%)")
        return select_best_type(state, min_improvement)  # Try next

    return best_type


def save_checkpoint(
    model,
    tokenizer,
    state: TrainingState,
    output_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    if is_best:
        ckpt_dir = output_dir / "best_checkpoint"
    else:
        ckpt_dir = output_dir / f"checkpoint_step{state.global_step}"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # Save state
    state_dict = {
        "global_step": state.global_step,
        "best_accuracy": state.best_accuracy,
        "best_step": state.best_step,
        "best_type": state.best_type,
        "type_stats": {
            k: {
                "name": v.name,
                "total_steps": v.total_steps,
                "total_samples": v.total_samples,
                "best_accuracy": v.best_accuracy,
                "last_accuracy": v.last_accuracy,
                "accuracy_history": v.accuracy_history,
                "improvement_history": v.improvement_history,
                "is_exhausted": v.is_exhausted
            }
            for k, v in state.type_stats.items()
        },
        "history": state.history
    }

    with open(ckpt_dir / "training_state.json", 'w') as f:
        json.dump(state_dict, f, indent=2)

    print(f"  Saved checkpoint to {ckpt_dir}")


def print_type_stats(state: TrainingState):
    """Print statistics for all types"""
    print(f"\n{'='*80}")
    print("TYPE PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Type':<25} {'Steps':>8} {'Samples':>10} {'Best Acc':>10} {'Last Acc':>10} {'Avg Impr':>10} {'Status':<10}")
    print(f"{'-'*80}")

    for type_name in TYPES:
        stats = state.type_stats.get(type_name)
        if stats:
            status = "EXHAUSTED" if stats.is_exhausted else "ACTIVE"
            avg_impr = f"{stats.avg_improvement:.2f}%" if stats.improvement_history else "N/A"
            print(f"{type_name:<25} {stats.total_steps:>8} {stats.total_samples:>10} {stats.best_accuracy:>9.2f}% {stats.last_accuracy:>9.2f}% {avg_impr:>10} {status:<10}")

    print(f"{'-'*80}")
    print(f"Global Best: {state.best_accuracy:.2f}% at step {state.best_step} (type: {state.best_type})")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Multi-Type Training")
    parser.add_argument("--model", default="medgemma-27b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--base-model", type=str, default=None, help="Path to base model")
    parser.add_argument("--steps-per-round", type=int, default=50, help="Training steps per evaluation round")
    parser.add_argument("--min-improvement", type=float, default=1.0, help="Min improvement % to continue type")
    parser.add_argument("--target-accuracy", type=float, default=90.0, help="Target KorMedMCQA accuracy")
    parser.add_argument("--max-rounds", type=int, default=100, help="Maximum training rounds")
    parser.add_argument("--eval-samples", type=int, default=200, help="KorMedMCQA samples for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"adaptive_training_{timestamp}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = MODEL_CONFIGS[args.model]

    # Determine model path
    if args.base_model:
        model_path = args.base_model
    else:
        stage6 = BASE_DIR / "models" / "staged_training" / "stage6"
        model_path = str(stage6) if stage6.exists() else cfg["path"]

    print(f"\n{'#'*80}")
    print("ADAPTIVE MULTI-TYPE TRAINING")
    print(f"{'#'*80}")
    print(f"Model: {model_path}")
    print(f"Steps per round: {args.steps_per_round}")
    print(f"Min improvement: {args.min_improvement}%")
    print(f"Target accuracy: {args.target_accuracy}%")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Log file: {log_file}")
    print(f"{'#'*80}\n")

    # Load model
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": args.device},
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    trainable = model.num_parameters(only_trainable=True)
    total = model.num_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load evaluation data
    data_dir = BASE_DIR / "data" / "reviewed"
    eval_dataset = load_kormedmcqa_eval(data_dir)

    # Load all type datasets
    print("\nLoading training data for all types...")
    type_datasets = {}
    for type_name in TYPES:
        type_datasets[type_name] = load_type_data(data_dir, type_name)
        print(f"  {type_name}: {len(type_datasets[type_name])} samples")

    # Initialize state
    state = TrainingState()
    for type_name in TYPES:
        state.type_stats[type_name] = TypeStats(name=type_name)

    # Initial evaluation
    print("\n" + "="*80)
    print("INITIAL EVALUATION (before training)")
    print("="*80)
    initial_acc, _, _ = evaluate_kormedmcqa(
        model, tokenizer, eval_dataset, args.eval_samples
    )
    state.best_accuracy = initial_acc

    for type_name in TYPES:
        state.type_stats[type_name].last_accuracy = initial_acc

    # Log initial state
    with open(log_file, 'w') as f:
        f.write(f"Adaptive Training Log - {timestamp}\n")
        f.write(f"Initial accuracy: {initial_acc:.2f}%\n\n")

    # Main training loop
    print("\n" + "#"*80)
    print("STARTING ADAPTIVE TRAINING")
    print("#"*80)

    round_num = 0
    while round_num < args.max_rounds:
        round_num += 1

        print(f"\n{'='*80}")
        print(f"ROUND {round_num}")
        print(f"{'='*80}")

        # Select best type to train
        best_type = select_best_type(state, args.min_improvement)

        if best_type is None:
            print("\nAll types exhausted! Stopping training.")
            break

        state.current_type = best_type
        stats = state.type_stats[best_type]

        print(f"\nSelected type: {best_type}")
        print(f"  {TYPE_NAMES[best_type]}")
        print(f"  Previous accuracy: {stats.last_accuracy:.2f}%")

        # Train on selected type
        prev_accuracy = stats.last_accuracy

        avg_loss = train_steps_on_type(
            model=model,
            tokenizer=tokenizer,
            type_dataset=type_datasets[best_type],
            type_stats=stats,
            num_steps=args.steps_per_round,
            batch_size=cfg["batch"],
            grad_accum=cfg["grad_accum"],
            lr=cfg["lr"],
            max_length=cfg["max_length"],
            device=args.device
        )

        state.global_step += args.steps_per_round

        # Evaluate on KorMedMCQA
        print(f"\n  Evaluating on KorMedMCQA...")
        new_accuracy, correct, total = evaluate_kormedmcqa(
            model, tokenizer, eval_dataset, args.eval_samples
        )

        # Calculate improvement
        improvement = new_accuracy - prev_accuracy
        stats.last_accuracy = new_accuracy
        stats.accuracy_history.append((state.global_step, new_accuracy))
        stats.improvement_history.append(improvement)

        # Update average improvement
        if len(stats.improvement_history) > 0:
            stats.avg_improvement = sum(stats.improvement_history) / len(stats.improvement_history)

        # Update best
        if new_accuracy > stats.best_accuracy:
            stats.best_accuracy = new_accuracy

        if new_accuracy > state.best_accuracy:
            state.best_accuracy = new_accuracy
            state.best_step = state.global_step
            state.best_type = best_type
            save_checkpoint(model, tokenizer, state, OUTPUT_DIR, is_best=True)
            print(f"\n  NEW BEST! {new_accuracy:.2f}%")

        # Log round result
        round_result = {
            "round": round_num,
            "type": best_type,
            "steps": state.global_step,
            "accuracy": new_accuracy,
            "improvement": improvement,
            "loss": avg_loss
        }
        state.history.append(round_result)

        with open(log_file, 'a') as f:
            f.write(f"Round {round_num}: {best_type} | Acc: {new_accuracy:.2f}% | Impr: {improvement:+.2f}% | Loss: {avg_loss:.4f}\n")

        # Print summary
        print(f"\n  Round {round_num} Summary:")
        print(f"    Type: {best_type}")
        print(f"    Accuracy: {prev_accuracy:.2f}% -> {new_accuracy:.2f}% ({improvement:+.2f}%)")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    Best: {state.best_accuracy:.2f}%")

        # Check if improvement is below threshold
        if improvement < args.min_improvement:
            consecutive_low = sum(1 for x in stats.improvement_history[-3:] if x < args.min_improvement)
            if consecutive_low >= 2:
                stats.is_exhausted = True
                print(f"\n  Type {best_type} marked as exhausted (low improvement)")

        # Print all type stats
        print_type_stats(state)

        # Check target
        if new_accuracy >= args.target_accuracy:
            print(f"\n{'#'*80}")
            print(f"TARGET ACCURACY {args.target_accuracy}% REACHED!")
            print(f"Final: {new_accuracy:.2f}%")
            print(f"{'#'*80}")
            break

    # Final save
    save_checkpoint(model, tokenizer, state, OUTPUT_DIR, is_best=False)

    # Print final summary
    print(f"\n{'#'*80}")
    print("TRAINING COMPLETE")
    print(f"{'#'*80}")
    print(f"Total rounds: {round_num}")
    print(f"Total steps: {state.global_step}")
    print(f"Best accuracy: {state.best_accuracy:.2f}% (step {state.best_step}, type: {state.best_type})")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Log file: {log_file}")

    # Save final report
    report = {
        "final_accuracy": state.best_accuracy,
        "best_step": state.best_step,
        "best_type": state.best_type,
        "total_rounds": round_num,
        "total_steps": state.global_step,
        "type_summary": {
            k: {
                "total_steps": v.total_steps,
                "total_samples": v.total_samples,
                "best_accuracy": v.best_accuracy,
                "avg_improvement": v.avg_improvement,
                "is_exhausted": v.is_exhausted
            }
            for k, v in state.type_stats.items()
        },
        "history": state.history
    }

    with open(OUTPUT_DIR / "final_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nFinal report saved to: {OUTPUT_DIR / 'final_report.json'}")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    sorted_types = sorted(
        state.type_stats.items(),
        key=lambda x: x[1].avg_improvement if x[1].improvement_history else 0,
        reverse=True
    )

    print("\nType effectiveness ranking (by avg improvement):")
    for i, (type_name, stats) in enumerate(sorted_types, 1):
        avg = stats.avg_improvement if stats.improvement_history else 0
        print(f"  {i}. {type_name}: {avg:.2f}% avg improvement, {stats.total_steps} total steps")

    if sorted_types[0][1].avg_improvement > 0:
        best = sorted_types[0][0]
        print(f"\nMost effective for KorMedMCQA: {best}")
        print(f"  -> Focus training on {TYPE_NAMES[best]}")


if __name__ == "__main__":
    main()
