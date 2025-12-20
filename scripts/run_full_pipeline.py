#!/usr/bin/env python3
"""
Full Pipeline: Download → Refine → Review → Train

Runs all steps in sequence:
1. Download datasets from HuggingFace
2. Refine into 4 types (type1-4)
3. Review & fix with DeepSeek-R1
4. Train with reviewed data
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def run_step(name: str, cmd: list, skip_on_error: bool = False):
    """Run a pipeline step"""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=BASE_DIR)

    if result.returncode != 0:
        if skip_on_error:
            print(f"WARNING: {name} failed but continuing...")
        else:
            print(f"ERROR: {name} failed with code {result.returncode}")
            sys.exit(1)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full data pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download")
    parser.add_argument("--skip-refine", action="store_true",
                        help="Skip data refinement")
    parser.add_argument("--skip-review", action="store_true",
                        help="Skip DeepSeek review")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training")
    parser.add_argument("--review-max-samples", type=int, default=None,
                        help="Limit samples for review (for testing)")
    parser.add_argument("--model", default="medgemma-27b",
                        choices=["medgemma-4b", "medgemma-27b"])
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("MedGemma Korean - Full Pipeline")
    print("=" * 60)
    print("\nSteps:")
    print("  1. Download datasets from HuggingFace")
    print("  2. Refine into 4 types")
    print("  3. Review & fix with DeepSeek-R1")
    print("  4. Train with reviewed data")

    # Step 1: Download
    if not args.skip_download:
        run_step(
            "Download Datasets",
            [sys.executable, "scripts/download_all_datasets.py"],
            skip_on_error=True  # Continue even if some datasets fail
        )
    else:
        print("\n[Skipping download]")

    # Step 2: Refine
    if not args.skip_refine:
        run_step(
            "Refine into 4 Types",
            [sys.executable, "refine_scripts/refine_4types.py"]
        )
    else:
        print("\n[Skipping refine]")

    # Step 3: Review with DeepSeek
    if not args.skip_review:
        review_cmd = [sys.executable, "refine_scripts/review_with_deepseek.py", "--all"]
        if args.review_max_samples:
            review_cmd.extend(["--max-samples", str(args.review_max_samples)])

        run_step(
            "DeepSeek Review & Fix",
            review_cmd
        )
    else:
        print("\n[Skipping review]")

    # Step 4: Train
    if not args.skip_train:
        run_step(
            "Training with Reviewed Data",
            [sys.executable, "scripts/train_all_types_parallel.py",
             "--model", args.model,
             "--source", "reviewed",
             "--epochs", str(args.epochs)]
        )
    else:
        print("\n[Skipping training]")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
