#!/usr/bin/env python3
"""
Run all data refinement scripts.

Pipeline:
1. Refine plain text (Korean Wikipedia, Namu Wiki, C4)
2. Refine KorMedMCQA (transform answer format)
3. Refine medical dictionary (merge sources)
4. Refine translation data (filter and format)
5. (Optional) Enhance with LLM

Usage:
    python script/prepare/refine/refine_all.py
    python script/prepare/refine/refine_all.py --skip-plain-text
    python script/prepare/refine/refine_all.py --with-llm
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_script(script_name: str, description: str) -> bool:
    """Run a refinement script."""
    script_path = SCRIPT_DIR / script_name

    print(f"\n{'=' * 60}")
    print(f"[{description}]")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all data refinement")
    parser.add_argument("--skip-plain-text", action="store_true",
                       help="Skip plain text refinement (slow)")
    parser.add_argument("--skip-translation", action="store_true",
                       help="Skip translation data refinement")
    parser.add_argument("--skip-mcq", action="store_true",
                       help="Skip KorMedMCQA refinement")
    parser.add_argument("--skip-dict", action="store_true",
                       help="Skip medical dictionary refinement")
    parser.add_argument("--with-llm", action="store_true",
                       help="Run LLM enhancement (requires GPU)")
    parser.add_argument("--llm-device", default="cuda:1",
                       help="Device for LLM enhancement")
    args = parser.parse_args()

    print("=" * 60)
    print("MedGemma Korean - Data Refinement Pipeline")
    print("=" * 60)

    results = {}

    # 1. Plain text
    if not args.skip_plain_text:
        results["plain_text"] = run_script(
            "refine_plain_text.py",
            "Step 1/4: Plain Text"
        )
    else:
        print("\n[SKIPPED] Plain text refinement")
        results["plain_text"] = None

    # 2. KorMedMCQA
    if not args.skip_mcq:
        results["kormedmcqa"] = run_script(
            "refine_kormedmcqa.py",
            "Step 2/4: KorMedMCQA"
        )
    else:
        print("\n[SKIPPED] KorMedMCQA refinement")
        results["kormedmcqa"] = None

    # 3. Medical dictionary
    if not args.skip_dict:
        results["medical_dict"] = run_script(
            "refine_medical_dict.py",
            "Step 3/4: Medical Dictionary"
        )
    else:
        print("\n[SKIPPED] Medical dictionary refinement")
        results["medical_dict"] = None

    # 4. Translation
    if not args.skip_translation:
        results["translation"] = run_script(
            "refine_translation.py",
            "Step 4/4: Translation"
        )
    else:
        print("\n[SKIPPED] Translation refinement")
        results["translation"] = None

    # 5. LLM Enhancement (optional)
    if args.with_llm:
        print(f"\n{'=' * 60}")
        print("[Optional: LLM Enhancement]")
        print(f"{'=' * 60}")

        script_path = SCRIPT_DIR / "enhance_with_llm.py"
        try:
            subprocess.run([
                sys.executable, str(script_path),
                "--task", "all",
                "--device", args.llm_device
            ], check=True)
            results["llm_enhance"] = True
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            results["llm_enhance"] = False

    # Summary
    print("\n" + "=" * 60)
    print("Refinement Summary")
    print("=" * 60)

    for step, success in results.items():
        if success is None:
            status = "⊘ SKIPPED"
        elif success:
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"
        print(f"  {status}: {step}")

    # Check output
    print("\n" + "-" * 60)
    print("Output Files:")
    print("-" * 60)

    output_dir = SCRIPT_DIR.parent.parent.parent / "data" / "02_refined"
    if output_dir.exists():
        for item in sorted(output_dir.rglob("*")):
            if item.is_file():
                size = item.stat().st_size / (1024 * 1024)  # MB
                rel_path = item.relative_to(output_dir)
                print(f"  {rel_path}: {size:.2f} MB")

    # Exit status
    failures = [k for k, v in results.items() if v is False]
    if failures:
        print(f"\nFailed: {', '.join(failures)}")
        sys.exit(1)

    print("\nRefinement completed!")


if __name__ == "__main__":
    main()
