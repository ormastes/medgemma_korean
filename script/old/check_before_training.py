#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-flight Check for Training Pipeline

Validates:
1. GPU availability and memory
2. Data files exist
3. Max length fits training data
4. Model configs are valid

Usage:
    python check_before_training.py --model medgemma-4b
    python check_before_training.py --model medgemma-27b --device cuda:1
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS, MAX_LENGTHS, TRAIN_02_CONFIGS


def check_gpu():
    """Check GPU availability and memory."""
    print("\n" + "=" * 60)
    print("GPU Check")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            return False

        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA available, {gpu_count} GPU(s) found")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
            free_gb = total_gb - allocated_gb

            print(f"\n  GPU {i}: {props.name}")
            print(f"    Total: {total_gb:.1f} GB")
            print(f"    Free:  {free_gb:.1f} GB")

            # Check if enough memory
            if free_gb < 20:
                print(f"    ⚠️ Low memory! May need 8-bit quantization")
            else:
                print(f"    ✓ Sufficient memory")

        return True

    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False


def check_data_files():
    """Check if all data files exist."""
    print("\n" + "=" * 60)
    print("Data Files Check")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "02_refined"

    required_files = {
        "train_00": [
            data_dir / "00_plain_text" / "train.jsonl",
        ],
        "train_01": [
            data_dir / "01_medical_dict.json",
            data_dir / "02_char_dict.json",
        ],
        "train_02": [
            data_dir / "02_kor_med_test" / "train.jsonl",
            data_dir / "02_kor_med_test" / "test.jsonl",
        ],
    }

    all_ok = True

    for stage, files in required_files.items():
        print(f"\n  {stage}:")
        for file_path in files:
            if file_path.exists():
                # Get file size and line count
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_path.suffix == ".jsonl":
                    with open(file_path, 'r') as f:
                        lines = sum(1 for _ in f)
                    print(f"    ✓ {file_path.name} ({lines} samples, {size_mb:.1f} MB)")
                else:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    count = len(data) if isinstance(data, list) else 1
                    print(f"    ✓ {file_path.name} ({count} entries, {size_mb:.1f} MB)")
            else:
                print(f"    ❌ {file_path.name} NOT FOUND!")
                all_ok = False

    return all_ok


def check_max_lengths(model: str):
    """Check max lengths for each training stage."""
    print("\n" + "=" * 60)
    print(f"Max Length Check: {model}")
    print("=" * 60)

    if model not in MAX_LENGTHS:
        print(f"❌ Unknown model: {model}")
        return False

    lengths = MAX_LENGTHS[model]

    print(f"\n  Configured max lengths:")
    for stage, max_len in lengths.items():
        print(f"    {stage}: {max_len} tokens")

    # Check if train_02 has enough for detailed prompts
    if lengths.get("train_02", 0) < 1500:
        print(f"\n  ⚠️ train_02 max_length may be too short for detailed prompts!")
        print(f"     Recommended: >= 2048")

    return True


def check_model_config(model: str):
    """Check model configuration."""
    print("\n" + "=" * 60)
    print(f"Model Config Check: {model}")
    print("=" * 60)

    if model not in MODEL_CONFIGS:
        print(f"❌ Unknown model: {model}")
        return False

    cfg = MODEL_CONFIGS[model]

    print(f"\n  Model path: {cfg['path']}")
    print(f"  LoRA rank: {cfg['lora_r']}")
    print(f"  LoRA alpha: {cfg['lora_alpha']}")
    print(f"  Learning rate: {cfg['lr']}")
    print(f"  Batch size: {cfg['batch']}")
    print(f"  Gradient accumulation: {cfg['grad_accum']}")
    print(f"  Effective batch: {cfg['batch'] * cfg['grad_accum']}")

    # Estimate memory
    if "27b" in model:
        est_mem_gb = 40  # With 8-bit
    else:
        est_mem_gb = 16  # With 8-bit

    print(f"\n  Estimated VRAM (8-bit): ~{est_mem_gb} GB")

    return True


def check_train_02_config(model: str):
    """Check train_02 specific config."""
    print("\n" + "=" * 60)
    print(f"Train 02 Config Check: {model}")
    print("=" * 60)

    if model not in TRAIN_02_CONFIGS:
        print(f"⚠️ No train_02 config for {model}, using defaults")
        return True

    cfg = TRAIN_02_CONFIGS[model]

    print(f"\n  Full mode samples: {cfg['full_samples']}")
    print(f"  Reasoning threshold: {cfg['reasoning_threshold']}")
    print(f"  Eval interval: {cfg['eval_interval']} steps")
    print(f"  Eval samples: {cfg['eval_samples']}")

    return True


def sample_data_validation(model: str):
    """Quick validation of sample data lengths."""
    print("\n" + "=" * 60)
    print(f"Sample Data Validation: {model}")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
        from data_validation import check_data_lengths

        cfg = MODEL_CONFIGS[model]
        tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)

        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data" / "02_refined"

        stages = {
            "train_00": (data_dir / "00_plain_text" / "train.jsonl", MAX_LENGTHS[model]["train_00"]),
            "train_02": (data_dir / "02_kor_med_test" / "train.jsonl", MAX_LENGTHS[model]["train_02"]),
        }

        for stage, (file_path, max_len) in stages.items():
            if not file_path.exists():
                continue

            print(f"\n  {stage} (max_length={max_len}):")

            # Load sample data
            samples = []
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 50:  # Check first 50 samples
                        break
                    samples.append(json.loads(line))

            result = check_data_lengths(samples, tokenizer, max_len)

            print(f"    Samples checked: {result['total_samples']}")
            print(f"    Token range: {result['min_tokens']} - {result['max_tokens']}")
            print(f"    Avg tokens: {result['avg_tokens']:.0f}")

            if result['overflow_count'] > 0:
                print(f"    ⚠️ Overflow: {result['overflow_count']} samples exceed max_length")
            else:
                print(f"    ✓ All samples fit within max_length")

    except Exception as e:
        print(f"  ⚠️ Could not validate sample data: {e}")
        print(f"     (This is OK - will validate at training start)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check for training")
    parser.add_argument("--model", default="medgemma-4b",
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-tokenizer", action="store_true",
                       help="Skip tokenizer-based validation")
    args = parser.parse_args()

    print("=" * 60)
    print("PRE-FLIGHT CHECK FOR TRAINING PIPELINE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    all_ok = True

    # Run checks
    all_ok &= check_gpu()
    all_ok &= check_data_files()
    all_ok &= check_max_lengths(args.model)
    all_ok &= check_model_config(args.model)
    all_ok &= check_train_02_config(args.model)

    if not args.skip_tokenizer:
        sample_data_validation(args.model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_ok:
        print("\n✓ All checks passed! Ready to train.")
        print("\nTo start training:")
        print(f"  ./run_full_pipeline.sh --model {args.model} --device {args.device}")
        print("\nTo run in background:")
        print(f"  ./run_full_pipeline.sh --model {args.model} --device {args.device} --background")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
