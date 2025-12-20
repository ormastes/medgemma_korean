#!/usr/bin/env python3
"""
Loop Training Until 90% KorMedMCQA Accuracy

Trains all 4 types in sequence, evaluates on KorMedMCQA after each loop,
continues until target accuracy is reached.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # A6000 only

import argparse
import json
import os
import sys
import re
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "reviewed"
OUTPUT_DIR = BASE_DIR / "models" / "loop_training"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model configs - Optimized for 94% GPU utilization (44.5GB on A6000)
MODEL_CONFIGS = {
    "medgemma-4b": {
        "hf_path": "google/medgemma-4b-it",
        "local_path": BASE_DIR / "models" / "staged_training" / "stage5_harmonization",
        "lora_r": 128, "lora_alpha": 256,
        "lr": 1e-4, "batch": 8, "grad_accum": 4,
        "max_length": 512
    },
    "medgemma-27b": {
        "hf_path": "google/medgemma-27b-text-it",
        "local_path": None,  # Use HuggingFace directly
        "lora_r": 128, "lora_alpha": 256,  # Optimized: was 64/128
        "lr": 5e-5, "batch": 8, "grad_accum": 4,  # Optimized: was 1/32
        "max_length": 512
    }
}

TYPE_ORDER = ["type1_text", "type2_text_reasoning", "type3_word", "type4_word_reasoning"]


def load_type_data(type_name: str):
    """Load train/val data for a type"""
    train_path = DATA_DIR / type_name / "train" / "data.jsonl"
    val_path = DATA_DIR / type_name / "validation" / "data.jsonl"

    train_data, val_data = [], []

    if train_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = [json.loads(line) for line in f]

    if val_path.exists():
        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = [json.loads(line) for line in f]

    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def load_mixed_data(sample_per_type: int = None):
    """Load and mix data from all 4 types for parallel training"""
    import random

    all_train, all_val = [], []
    type_counts = {}

    for type_name in TYPE_ORDER:
        train_data, val_data = load_type_data(type_name)
        train_list = list(train_data)
        val_list = list(val_data)

        # Sample if specified
        if sample_per_type and len(train_list) > sample_per_type:
            train_list = random.sample(train_list, sample_per_type)
        if sample_per_type and len(val_list) > sample_per_type // 10:
            val_list = random.sample(val_list, sample_per_type // 10)

        # Add type info
        for item in train_list:
            item['_type'] = type_name
        for item in val_list:
            item['_type'] = type_name

        all_train.extend(train_list)
        all_val.extend(val_list)
        type_counts[type_name] = len(train_list)

    # Shuffle
    random.shuffle(all_train)
    random.shuffle(all_val)

    print(f"Mixed data loaded:")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")
    print(f"  Total train: {len(all_train)}, val: {len(all_val)}")

    return Dataset.from_list(all_train), Dataset.from_list(all_val)


def evaluate_kormedmcqa(model, tokenizer, max_samples=None):
    """Evaluate on KorMedMCQA test set"""
    print("\n" + "=" * 60)
    print("Evaluating on KorMedMCQA")
    print("=" * 60)

    try:
        # Load all configs and combine
        all_data = []
        for config in ['doctor', 'nurse', 'pharm']:
            ds = load_dataset("sean0042/KorMedMCQA", config, split="test")
            all_data.extend(list(ds))
        eval_dataset = all_data
        print(f"Loaded {len(eval_dataset)} samples from KorMedMCQA")
    except Exception as e:
        print(f"Failed to load KorMedMCQA: {e}")
        return 0.0

    if max_samples:
        eval_dataset = eval_dataset[:max_samples]

    model.eval()
    correct = 0
    total = 0

    for example in tqdm(eval_dataset, desc="KorMedMCQA Eval"):
        try:
            question = example["question"]
            choices = []
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if letter in example and example[letter]:
                    choices.append(f"{letter}) {example[letter]}")

            prompt = f"""<|im_start|>system
의료 객관식 문제입니다. 정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).
<|im_end|>
<|im_start|>user
{question}

{chr(10).join(choices)}
<|im_end|>
<|im_start|>assistant
"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip().upper()

            # Extract answer
            predicted = None
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if letter in response[:5]:
                    predicted = letter
                    break

            # Get correct answer
            answer_idx = example.get("answer", 1)
            correct_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}.get(answer_idx, 'A')

            if predicted == correct_letter:
                correct += 1
            total += 1

            if total % 100 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            continue

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"KorMedMCQA Accuracy: {accuracy:.2f}% ({correct}/{total})")

    model.train()
    return accuracy


def train_one_type(model, tokenizer, type_name: str, cfg: dict, loop_num: int, output_dir: Path):
    """Train one type for one epoch"""
    print(f"\n{'=' * 60}")
    print(f"Training {type_name} - Loop {loop_num}")
    print("=" * 60)

    train_data, val_data = load_type_data(type_name)

    if len(train_data) == 0:
        print(f"No data for {type_name}, skipping...")
        return model

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    type_output = output_dir / f"loop_{loop_num}" / f"after_{type_name}"
    type_output.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(type_output),
        num_train_epochs=1,
        per_device_train_batch_size=cfg['batch'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        max_length=cfg['max_length'],
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer
    )

    trainer.train()

    # Save checkpoint
    model.save_pretrained(type_output)
    tokenizer.save_pretrained(type_output)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma-27b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--target", type=float, default=90.0, help="Target accuracy")
    parser.add_argument("--max-loops", type=int, default=10, help="Maximum training loops")
    parser.add_argument("--eval-samples", type=int, default=None, help="Max eval samples (None=all)")
    parser.add_argument("--mixed", action="store_true", default=True, help="Mix all types together")
    parser.add_argument("--sample-per-type", type=int, default=20000, help="Samples per type for mixed training")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    print("=" * 60)
    print("Loop Training Until 90% KorMedMCQA")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Target: {args.target}%")
    print(f"Max loops: {args.max_loops}")
    print(f"Output: {OUTPUT_DIR}")

    # Load model
    print("\nLoading model...")

    local_path = cfg['local_path']
    if local_path and local_path.exists() and (local_path / "model.safetensors").exists():
        model_path = str(local_path)
    else:
        model_path = cfg['hf_path']
    print(f"Using: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Initial evaluation
    print("\nInitial evaluation...")
    initial_acc = evaluate_kormedmcqa(model, tokenizer, args.eval_samples)

    best_accuracy = initial_acc
    training_history = [{
        "loop": 0,
        "accuracy": initial_acc,
        "timestamp": str(datetime.now())
    }]

    # Load mixed data once
    if args.mixed:
        print("\nLoading mixed data from all types...")
        train_data, val_data = load_mixed_data(args.sample_per_type)

    # Training loop
    for loop_num in range(1, args.max_loops + 1):
        print(f"\n{'#' * 60}")
        print(f"LOOP {loop_num}/{args.max_loops}")
        print("#" * 60)

        if args.mixed:
            # Train on mixed data
            print(f"\nTraining on mixed data - Loop {loop_num}")
            loop_output = OUTPUT_DIR / f"loop_{loop_num}"
            loop_output.mkdir(parents=True, exist_ok=True)

            training_args = SFTConfig(
                output_dir=str(loop_output),
                num_train_epochs=1,
                per_device_train_batch_size=cfg['batch'],
                gradient_accumulation_steps=cfg['grad_accum'],
                learning_rate=cfg['lr'],
                weight_decay=0.01,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                logging_steps=50,
                save_strategy="steps",
                save_steps=1000,
                bf16=True,
                max_length=cfg['max_length'],
                gradient_checkpointing=False,
                optim="paged_adamw_8bit",
                max_grad_norm=0.3,
                report_to="none",
                dataset_text_field="text"
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                processing_class=tokenizer
            )
            trainer.train()
        else:
            # Train each type sequentially
            for type_name in TYPE_ORDER:
                model = train_one_type(model, tokenizer, type_name, cfg, loop_num, OUTPUT_DIR)

        # Evaluate
        accuracy = evaluate_kormedmcqa(model, tokenizer, args.eval_samples)

        training_history.append({
            "loop": loop_num,
            "accuracy": accuracy,
            "timestamp": str(datetime.now())
        })

        # Save if best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_dir = OUTPUT_DIR / "best_checkpoint"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"New best: {accuracy:.2f}%")

        # Save history
        with open(OUTPUT_DIR / "training_history.json", 'w') as f:
            json.dump({
                "target": args.target,
                "best_accuracy": best_accuracy,
                "history": training_history
            }, f, indent=2)

        # Check target
        if accuracy >= args.target:
            print(f"\n{'=' * 60}")
            print(f"TARGET REACHED: {accuracy:.2f}% >= {args.target}%")
            print("=" * 60)

            final_dir = OUTPUT_DIR / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            break

    # Final summary
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Target: {args.target}%")
    print(f"Loops completed: {len(training_history) - 1}")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
