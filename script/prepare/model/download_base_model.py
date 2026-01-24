#!/usr/bin/env python3
"""
Download MedGemma base models from HuggingFace.

Models:
- google/medgemma-4b-it: 4B parameter instruction-tuned model
- google/medgemma-27b-text-it: 27B parameter text instruction-tuned model

IMPORTANT: These models require HuggingFace authentication.
           You must accept the license agreement on HuggingFace first.
           See: script/prepare/model/MANUAL_MODEL.md

Output: model/raw/
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, login, HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: Required packages not installed")
    print("Run: pip install huggingface_hub transformers")
    sys.exit(1)

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "model" / "raw"

# Model configurations
MODELS = {
    "medgemma-4b": {
        "hf_id": "google/medgemma-4b-it",
        "description": "4B parameter instruction-tuned MedGemma",
        "size_gb": 8.5
    },
    "medgemma-27b": {
        "hf_id": "google/medgemma-27b-text-it",
        "description": "27B parameter text instruction-tuned MedGemma",
        "size_gb": 55.0
    }
}


def check_hf_login():
    """Check if user is logged into HuggingFace."""
    try:
        api = HfApi()
        user = api.whoami()
        print(f"Logged in as: {user.get('name', 'Unknown')}")
        return True
    except Exception:
        return False


def download_model(model_name: str, output_dir: Path, force: bool = False):
    """Download a model from HuggingFace."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODELS.keys())}")
        return False

    config = MODELS[model_name]
    model_path = output_dir / model_name

    print(f"\n{'=' * 60}")
    print(f"Downloading: {model_name}")
    print(f"  HuggingFace ID: {config['hf_id']}")
    print(f"  Size: ~{config['size_gb']} GB")
    print(f"  Output: {model_path}")
    print(f"{'=' * 60}")

    if model_path.exists() and not force:
        print(f"Already exists. Use --force to re-download.")
        return True

    try:
        print("\nDownloading model files...")
        snapshot_download(
            repo_id=config['hf_id'],
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"\n✓ Downloaded to: {model_path}")

        # Verify by loading tokenizer
        print("\nVerifying download...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        print(f"  Tokenizer vocab size: {len(tokenizer)}")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")

        if "401" in str(e) or "403" in str(e) or "access" in str(e).lower():
            print("\nAuthentication error. Please:")
            print("1. Run: huggingface-cli login")
            print("2. Accept model license at: https://huggingface.co/{config['hf_id']}")

        return False


def main():
    parser = argparse.ArgumentParser(description="Download MedGemma base models")
    parser.add_argument("--model", choices=["medgemma-4b", "medgemma-27b", "all"],
                       default="medgemma-4b", help="Model to download")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if exists")
    args = parser.parse_args()

    print("=" * 60)
    print("MedGemma Base Model Download")
    print("=" * 60)

    # Check login
    print("\nChecking HuggingFace authentication...")
    if not check_hf_login():
        print("\n⚠ Not logged into HuggingFace.")
        print("Run: huggingface-cli login")
        print("\nContinuing... (may fail for gated models)")

    # Create output directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download selected model(s)
    if args.model == "all":
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = [args.model]

    results = {}
    for model_name in models_to_download:
        results[model_name] = download_model(model_name, MODEL_DIR, args.force)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for model_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model_name}")

    if all(results.values()):
        print(f"\nModels saved to: {MODEL_DIR}")
    else:
        print("\nSome downloads failed. See errors above.")
        print("Check: script/prepare/model/MANUAL_MODEL.md")
        sys.exit(1)


if __name__ == "__main__":
    main()
