#!/usr/bin/env python3
"""
Prepare all model components for training.

Pipeline:
1. Download base model from HuggingFace
2. Prepare extended Korean tokenizer
3. Add LoRA adapter

Usage:
    python script/prepare/model/prepare_all.py --model medgemma-4b
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_script(script_name: str, args: list = None) -> bool:
    """Run a preparation script."""
    script_path = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare all model components")
    parser.add_argument("--model", choices=["medgemma-4b", "medgemma-27b"],
                       default="medgemma-4b", help="Model to prepare")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip base model download")
    parser.add_argument("--skip-tokenizer", action="store_true",
                       help="Skip tokenizer preparation")
    parser.add_argument("--skip-lora", action="store_true",
                       help="Skip LoRA adapter addition")
    parser.add_argument("--device", default="cuda:0",
                       help="Device for LoRA preparation")
    args = parser.parse_args()

    print("=" * 60)
    print("MedGemma Korean - Model Preparation Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")

    results = {}

    # Step 1: Download base model
    if not args.skip_download:
        results["download"] = run_script(
            "download_base_model.py",
            ["--model", args.model]
        )
    else:
        print("\n[SKIPPED] Base model download")
        results["download"] = None

    # Step 2: Prepare tokenizer
    if not args.skip_tokenizer:
        # Determine base model path
        base_model_arg = f"google/medgemma-4b-it" if args.model == "medgemma-4b" else f"google/medgemma-27b-text-it"
        results["tokenizer"] = run_script(
            "prepare_tokenizer.py",
            ["--base-model", base_model_arg]
        )
    else:
        print("\n[SKIPPED] Tokenizer preparation")
        results["tokenizer"] = None

    # Step 3: Add LoRA
    if not args.skip_lora:
        results["lora"] = run_script(
            "add_lora.py",
            ["--model", args.model, "--device", args.device]
        )
    else:
        print("\n[SKIPPED] LoRA addition")
        results["lora"] = None

    # Summary
    print("\n" + "=" * 60)
    print("Preparation Summary")
    print("=" * 60)

    for step, success in results.items():
        if success is None:
            status = "⊘ SKIPPED"
        elif success:
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"
        print(f"  {status}: {step}")

    # Check for failures
    failures = [k for k, v in results.items() if v is False]
    if failures:
        print(f"\nFailed steps: {', '.join(failures)}")
        sys.exit(1)

    print("\nModel preparation completed!")


if __name__ == "__main__":
    main()
