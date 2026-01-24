#!/usr/bin/env python3
"""
Master script to run complete data and model preparation pipeline.

Pipeline:
1. Download data (automatic sources)
2. Validate manual data
3. Download and prepare model
4. Refine all data
5. (Optional) LLM enhancement

Usage:
    python script/prepare/prepare_all.py
    python script/prepare/prepare_all.py --model medgemma-4b --skip-download
    python script/prepare/prepare_all.py --with-llm --llm-device cuda:1
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_script(script_path: Path, args: list = None, description: str = "") -> bool:
    """Run a preparation script."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'=' * 70}")
    print(f"[{description}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete data and model preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full preparation
    python script/prepare/prepare_all.py

    # Skip data download (if already done)
    python script/prepare/prepare_all.py --skip-download

    # Prepare specific model
    python script/prepare/prepare_all.py --model medgemma-27b

    # Include LLM enhancement
    python script/prepare/prepare_all.py --with-llm --llm-device cuda:1
        """
    )

    parser.add_argument("--model", choices=["medgemma-4b", "medgemma-27b"],
                       default="medgemma-4b", help="Model to prepare")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip data download")
    parser.add_argument("--skip-model", action="store_true",
                       help="Skip model preparation")
    parser.add_argument("--skip-refine", action="store_true",
                       help="Skip data refinement")
    parser.add_argument("--with-llm", action="store_true",
                       help="Run LLM enhancement (requires GPU)")
    parser.add_argument("--llm-device", default="cuda:1",
                       help="Device for LLM enhancement")
    parser.add_argument("--device", default="cuda:0",
                       help="Device for model preparation")

    args = parser.parse_args()

    print("=" * 70)
    print("MedGemma Korean - Complete Preparation Pipeline")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    results = {}

    # Step 1: Download data
    if not args.skip_download:
        results["1_download_data"] = run_script(
            SCRIPT_DIR / "data" / "download_all.py",
            description="Step 1/5: Download Data"
        )
    else:
        print("\n[SKIPPED] Data download")
        results["1_download_data"] = None

    # Step 2: Validate manual data
    if not args.skip_download:
        results["2_validate_manual"] = run_script(
            SCRIPT_DIR / "data" / "validate_manual_data.py",
            description="Step 2/5: Validate Manual Data"
        )
        # Don't fail on validation - just warn
        if results["2_validate_manual"] is False:
            print("\n⚠ Manual data validation failed. Some files may be missing.")
            print("  See: script/prepare/data/MANUAL_DATA.md")
            results["2_validate_manual"] = "warning"
    else:
        print("\n[SKIPPED] Manual data validation")
        results["2_validate_manual"] = None

    # Step 3: Prepare model
    if not args.skip_model:
        results["3_prepare_model"] = run_script(
            SCRIPT_DIR / "model" / "prepare_all.py",
            ["--model", args.model, "--device", args.device],
            description="Step 3/5: Prepare Model"
        )
    else:
        print("\n[SKIPPED] Model preparation")
        results["3_prepare_model"] = None

    # Step 4: Refine data
    if not args.skip_refine:
        refine_args = []
        if args.with_llm:
            refine_args.extend(["--with-llm", "--llm-device", args.llm_device])

        results["4_refine_data"] = run_script(
            SCRIPT_DIR / "refine" / "refine_all.py",
            refine_args if refine_args else None,
            description="Step 4/5: Refine Data"
        )
    else:
        print("\n[SKIPPED] Data refinement")
        results["4_refine_data"] = None

    # Step 5: LLM Enhancement (if requested and not already done in refine)
    if args.with_llm and not args.skip_refine:
        # Already done in refine step
        results["5_llm_enhance"] = "included in step 4"
    elif args.with_llm:
        results["5_llm_enhance"] = run_script(
            SCRIPT_DIR / "refine" / "enhance_with_llm.py",
            ["--task", "all", "--device", args.llm_device],
            description="Step 5/5: LLM Enhancement"
        )
    else:
        results["5_llm_enhance"] = None

    # Summary
    print("\n" + "=" * 70)
    print("Preparation Summary")
    print("=" * 70)

    for step, status in results.items():
        if status is None:
            icon = "⊘"
            text = "SKIPPED"
        elif status is True:
            icon = "✓"
            text = "SUCCESS"
        elif status == "warning":
            icon = "⚠"
            text = "WARNING"
        elif isinstance(status, str):
            icon = "ℹ"
            text = status
        else:
            icon = "✗"
            text = "FAILED"
        print(f"  {icon} {step}: {text}")

    # Final status
    failures = [k for k, v in results.items() if v is False]
    warnings = [k for k, v in results.items() if v == "warning"]

    if failures:
        print(f"\n✗ Pipeline failed at: {', '.join(failures)}")
        sys.exit(1)
    elif warnings:
        print(f"\n⚠ Completed with warnings: {', '.join(warnings)}")
    else:
        print("\n✓ Pipeline completed successfully!")

    # Next steps
    print("\n" + "-" * 70)
    print("Next Steps:")
    print("-" * 70)
    print("  1. Start training:")
    print(f"     python script/train/train_00_plain_text.py --model {args.model}")
    print("  2. Or run full pipeline:")
    print(f"     ./run_full_pipeline.sh --model {args.model}")


if __name__ == "__main__":
    main()
