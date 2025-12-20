#!/usr/bin/env python3
"""
Parallel Training: All 4 Types Together

Trains on mixed data so model learns from system prompt:
- "답변하세요" → Direct answer (no reasoning)
- "추론한 후 답변하세요" → Use <R>...<R/> reasoning

This teaches the model to naturally skip reasoning for Type 1/3
while using reasoning for Type 2/4, based on prompt cues.
"""

import argparse
import json
import math
import re
import random
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
REFINED_DIR = BASE_DIR / "data" / "refined"
REVIEWED_DIR = BASE_DIR / "data" / "reviewed"
OUTPUT_DIR = BASE_DIR / "models" / "all_types_parallel"

# Special tokens
R_START = "<R>"
R_END = "<R/>"

def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB"""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return gpu_mem
    return 0


def get_optimal_config(model_name: str) -> dict:
    """Get optimal config based on GPU memory"""
    gpu_mem = get_gpu_memory_gb()
    print(f"Detected GPU memory: {gpu_mem:.1f} GB")

    # Base configs
    configs = {
        "medgemma-4b": {
            "path": "google/medgemma-4b-it",
            "lora_r": 64, "lora_alpha": 128,
            "lr": 5e-5
        },
        "medgemma-27b": {
            "path": "google/medgemma-27b-text-it",
            "lora_r": 128, "lora_alpha": 256,
            "lr": 2e-5
        }
    }

    cfg = configs.get(model_name, configs["medgemma-4b"]).copy()

    # Adjust based on GPU memory
    if model_name == "medgemma-27b":
        if gpu_mem >= 80:  # A100 80GB
            cfg.update({"batch": 2, "grad_accum": 16, "max_length": 1024, "grad_ckpt": True})
        elif gpu_mem >= 48:  # A6000 48GB
            cfg.update({"batch": 1, "grad_accum": 32, "max_length": 512, "grad_ckpt": True})
        elif gpu_mem >= 24:  # 3090/4090 24GB
            cfg.update({"batch": 1, "grad_accum": 64, "max_length": 256, "grad_ckpt": True})
        else:  # <24GB - may not fit
            cfg.update({"batch": 1, "grad_accum": 128, "max_length": 256, "grad_ckpt": True})
            print("WARNING: GPU memory may be insufficient for 27B model")
    else:  # medgemma-4b
        if gpu_mem >= 48:  # A6000
            cfg.update({"batch": 4, "grad_accum": 8, "max_length": 1024, "grad_ckpt": False})
        elif gpu_mem >= 24:  # 3090/4090
            cfg.update({"batch": 2, "grad_accum": 16, "max_length": 1024, "grad_ckpt": False})
        elif gpu_mem >= 16:  # 4080/V100
            cfg.update({"batch": 1, "grad_accum": 32, "max_length": 512, "grad_ckpt": True})
        elif gpu_mem >= 8:  # 3070/4070
            cfg.update({"batch": 1, "grad_accum": 64, "max_length": 512, "grad_ckpt": True})
        else:  # <8GB
            cfg.update({"batch": 1, "grad_accum": 128, "max_length": 256, "grad_ckpt": True})
            print("WARNING: GPU memory very limited")

    # Calculate effective batch size
    effective_batch = cfg["batch"] * cfg["grad_accum"]
    print(f"Config: batch={cfg['batch']}, grad_accum={cfg['grad_accum']}, "
          f"effective_batch={effective_batch}, max_length={cfg['max_length']}")

    return cfg


# Legacy configs (fallback)
MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 64, "lora_alpha": 128,
        "lr": 5e-5, "batch": 2, "grad_accum": 16,
        "max_length": 1024, "grad_ckpt": False
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 128, "lora_alpha": 256,
        "lr": 2e-5, "batch": 1, "grad_accum": 32,
        "max_length": 512, "grad_ckpt": True
    }
}

# Type names
TYPE_NAMES = ["type1_text", "type2_text_reasoning", "type3_word", "type4_word_reasoning"]


def get_type_dirs(source: str = "reviewed") -> dict:
    """Get type directories based on source (reviewed or refined)"""
    base_dir = REVIEWED_DIR if source == "reviewed" else REFINED_DIR
    return {name: base_dir / name for name in TYPE_NAMES}


def skip_reasoning_block(text: str) -> str:
    """Skip reasoning block if present"""
    if R_END in text:
        return text.split(R_END)[-1].strip()
    elif R_START in text:
        return text.split(R_START)[0].strip()
    return text.strip()


def load_all_types(source: str = "reviewed", sample_ratio: dict = None):
    """Load all 4 types and combine for parallel training"""
    all_train = []
    all_val = []

    default_ratio = {
        "type1_text": 1.0,
        "type2_text_reasoning": 1.0,
        "type3_word": 1.0,
        "type4_word_reasoning": 1.0
    }
    ratio = sample_ratio or default_ratio

    type_dirs = get_type_dirs(source)
    print(f"Loading from: {REVIEWED_DIR if source == 'reviewed' else REFINED_DIR}")

    for type_name, type_dir in type_dirs.items():
        train_file = type_dir / "train" / "data.jsonl"
        val_file = type_dir / "validation" / "data.jsonl"

        train_samples = []
        val_samples = []

        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line)
                    sample['data_type'] = type_name  # Tag with type
                    train_samples.append(sample)

        if val_file.exists():
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line)
                    sample['data_type'] = type_name
                    val_samples.append(sample)

        # Apply sampling ratio
        r = ratio.get(type_name, 1.0)
        if r < 1.0:
            random.shuffle(train_samples)
            train_samples = train_samples[:int(len(train_samples) * r)]
            random.shuffle(val_samples)
            val_samples = val_samples[:int(len(val_samples) * r)]

        all_train.extend(train_samples)
        all_val.extend(val_samples)

        print(f"  {type_name}: {len(train_samples)} train, {len(val_samples)} val")

    # Shuffle combined data
    random.shuffle(all_train)
    random.shuffle(all_val)

    print(f"\nTotal: {len(all_train)} train, {len(all_val)} val")
    return Dataset.from_list(all_train), Dataset.from_list(all_val)


class ParallelTypeCallback(TrainerCallback):
    """
    Evaluate all 4 types together.
    Model should learn:
    - Type 1/3: Answer directly (no reasoning tokens)
    - Type 2/4: Use <R>...<R/> reasoning tokens
    """

    def __init__(self, eval_dataset, tokenizer, model, eval_steps=500,
                 output_dir=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.eval_steps = eval_steps
        self.output_dir = output_dir
        self.best_score = 0.0
        self.best_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            score = self._evaluate(state.global_step)

            if score > self.best_score:
                self.best_score = score
                self.best_step = state.global_step
                self._save_checkpoint(state.global_step)

    def _evaluate(self, step: int) -> float:
        print(f"\n{'='*60}")
        print(f"Parallel Training Evaluation - Step {step}")
        print(f"{'='*60}")

        self.model.eval()

        # Track metrics per type
        metrics = {
            "type1_text": {"correct": 0, "total": 0, "no_reasoning": 0},
            "type2_text_reasoning": {"correct": 0, "total": 0, "has_reasoning": 0},
            "type3_word": {"correct": 0, "total": 0, "no_reasoning": 0},
            "type4_word_reasoning": {"correct": 0, "total": 0, "has_reasoning": 0}
        }

        # Sample from each type
        samples_by_type = {}
        for sample in self.eval_dataset:
            dtype = sample.get('data_type', 'unknown')
            if dtype not in samples_by_type:
                samples_by_type[dtype] = []
            samples_by_type[dtype].append(sample)

        with torch.no_grad():
            for dtype, samples in samples_by_type.items():
                # Evaluate up to 50 samples per type
                for sample in samples[:50]:
                    try:
                        result = self._evaluate_sample(sample, dtype)
                        metrics[dtype]["total"] += 1

                        if result["correct"]:
                            metrics[dtype]["correct"] += 1

                        if dtype in ["type1_text", "type3_word"]:
                            if not result["has_reasoning"]:
                                metrics[dtype]["no_reasoning"] += 1
                        else:
                            if result["has_reasoning"]:
                                metrics[dtype]["has_reasoning"] += 1

                    except:
                        continue

        # Print results
        print("\n  Per-Type Results:")
        print("  " + "-" * 55)

        total_score = 0
        type_scores = {}

        for dtype, m in metrics.items():
            if m["total"] == 0:
                continue

            acc = 100 * m["correct"] / m["total"]

            if dtype in ["type1_text", "type3_word"]:
                # Type 1/3: Want NO reasoning
                no_reason_pct = 100 * m["no_reasoning"] / m["total"]
                behavior_score = no_reason_pct / 100  # 1.0 if no reasoning
                print(f"  {dtype}:")
                print(f"    Accuracy: {acc:.1f}%, No reasoning: {no_reason_pct:.1f}%")
            else:
                # Type 2/4: Want HAS reasoning
                has_reason_pct = 100 * m["has_reasoning"] / m["total"]
                behavior_score = has_reason_pct / 100  # 1.0 if has reasoning
                print(f"  {dtype}:")
                print(f"    Accuracy: {acc:.1f}%, Has reasoning: {has_reason_pct:.1f}%")

            # Combined score: accuracy * behavior
            type_score = (acc / 100) * behavior_score
            type_scores[dtype] = type_score
            total_score += type_score

        # Average score across types
        avg_score = total_score / len(type_scores) if type_scores else 0

        print(f"\n  Overall Score: {avg_score:.3f} (best: {self.best_score:.3f})")
        print(f"  (Score = accuracy * correct_reasoning_behavior)")

        self.model.train()
        return avg_score

    def _evaluate_sample(self, sample: dict, dtype: str) -> dict:
        """Evaluate a single sample"""
        prompt = sample['prompt']
        expected = sample.get('answer', sample.get('completion', ''))

        inputs = self.tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=400).to(self.model.device)

        # Generate
        max_tokens = 50 if dtype in ["type3_word", "type4_word_reasoning"] else 200
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        ).strip()

        # Check for reasoning tokens
        has_reasoning = R_START in response and R_END in response

        # Extract answer
        if has_reasoning:
            answer_part = skip_reasoning_block(response)
        else:
            answer_part = response

        # Check correctness based on type
        correct = False
        if dtype in ["type3_word", "type4_word_reasoning"]:
            # Word/letter matching
            predicted = answer_part.split()[0] if answer_part.split() else ""
            predicted = re.sub(r'[^A-Za-z가-힣0-9]', '', predicted).upper()
            expected_clean = re.sub(r'[^A-Za-z가-힣0-9]', '', expected).upper()

            if len(expected_clean) == 1 and expected_clean in "ABCDE":
                correct = predicted and predicted[0] == expected_clean
            else:
                correct = predicted == expected_clean
        else:
            # Text matching - check if answer is reasonable length
            correct = len(answer_part) >= 20  # Has substantial content

        return {
            "correct": correct,
            "has_reasoning": has_reasoning,
            "response": response[:100]
        }

    def _save_checkpoint(self, step: int):
        if self.output_dir:
            ckpt = Path(self.output_dir) / "best_checkpoint"
            ckpt.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(ckpt)
            self.tokenizer.save_pretrained(ckpt)
            with open(ckpt / "info.json", 'w') as f:
                json.dump({"step": step, "score": self.best_score}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma-4b", choices=["medgemma-4b", "medgemma-27b"])
    parser.add_argument("--source", default="reviewed", choices=["reviewed", "refined"],
                        help="Data source: reviewed (DeepSeek fixed) or refined (rule-based)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--balance", action="store_true",
                        help="Balance types by downsampling larger datasets")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override auto batch size")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Override auto gradient accumulation")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Override auto max sequence length")
    args = parser.parse_args()

    # Get optimal config based on GPU memory
    cfg = get_optimal_config(args.model)

    # Override with command line args if provided
    if args.batch_size:
        cfg["batch"] = args.batch_size
    if args.grad_accum:
        cfg["grad_accum"] = args.grad_accum
    if args.max_length:
        cfg["max_length"] = args.max_length

    output_dir = str(OUTPUT_DIR / args.model)

    print("=" * 60)
    print("Parallel Training: All 4 Types Together")
    print("=" * 60)
    print(f"\nData source: {args.source.upper()}")
    print("Model will learn from system prompt:")
    print("  - '답변하세요' → No reasoning (Type 1, 3)")
    print("  - '추론한 후' → Use <R>...<R/> (Type 2, 4)")

    # Load all types
    print("\nLoading data...")

    if args.balance:
        # Balance by downsampling
        sample_ratio = {
            "type1_text": 0.3,  # Downsample large datasets
            "type2_text_reasoning": 1.0,
            "type3_word": 0.5,
            "type4_word_reasoning": 1.0
        }
    else:
        sample_ratio = None

    train_data, eval_data = load_all_types(source=args.source, sample_ratio=sample_ratio)

    if len(train_data) == 0:
        print("No data! Run refine_4types.py first.")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    tokenizer.add_special_tokens({"additional_special_tokens": [R_START, R_END]})

    # Load model
    print("\nLoading model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg['path'], quantization_config=bnb,
                                                  device_map="auto", trust_remote_code=True,
                                                  attn_implementation="eager")
    model.resize_token_embeddings(len(tokenizer))
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
        bf16=True, max_length=cfg['max_length'],
        gradient_checkpointing=cfg['grad_ckpt'],
        optim="paged_adamw_8bit", max_grad_norm=0.3, report_to="none"
    )

    callback = ParallelTypeCallback(eval_data, tokenizer, model,
                                    args.eval_steps, output_dir)

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_data,
                         processing_class=tokenizer, callbacks=[callback])

    print("\nStarting parallel training...")
    trainer.train()

    trainer.save_model(output_dir + "/final")
    tokenizer.save_pretrained(output_dir + "/final")
    print(f"\nDone! Best score: {callback.best_score:.3f}")


if __name__ == "__main__":
    main()
