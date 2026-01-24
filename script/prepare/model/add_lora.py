#!/usr/bin/env python3
"""
Add LoRA adapter to base model for training.

This script:
1. Loads the base model with 8-bit quantization
2. Configures LoRA adapter with appropriate settings
3. Saves the model ready for training

Output: model/raw_lora_added/{model}/
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    print("Error: Required packages not installed")
    print("Run: pip install torch transformers peft bitsandbytes")
    sys.exit(1)

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "model"

# Add script/train to path for config
sys.path.insert(0, str(BASE_DIR / "script" / "train"))
try:
    from training_config import MODEL_CONFIGS, LORA_TARGET_MODULES, TRAINING_DEFAULTS
except ImportError:
    # Fallback config
    MODEL_CONFIGS = {
        "medgemma-4b": {
            "path": "google/medgemma-4b-it",
            "lora_r": 16,
            "lora_alpha": 32,
        },
        "medgemma-27b": {
            "path": "google/medgemma-27b-text-it",
            "lora_r": 16,
            "lora_alpha": 32,
        }
    }
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    TRAINING_DEFAULTS = {"lora_dropout": 0.05}


def add_lora_to_model(
    model_name: str,
    base_model_path: str = None,
    tokenizer_path: str = None,
    output_dir: Path = None,
    device: str = "cuda:0"
):
    """Add LoRA adapter to base model."""
    print("=" * 60)
    print(f"Adding LoRA to: {model_name}")
    print("=" * 60)

    config = MODEL_CONFIGS.get(model_name)
    if not config:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        return False

    # Resolve paths
    if base_model_path is None:
        base_model_path = MODEL_DIR / "raw" / model_name
        if not base_model_path.exists():
            base_model_path = config["path"]  # Use HuggingFace ID

    if tokenizer_path is None:
        tokenizer_path = MODEL_DIR / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = base_model_path

    if output_dir is None:
        output_dir = MODEL_DIR / "raw_lora_added" / model_name

    print(f"Base model: {base_model_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Output: {output_dir}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    print(f"Vocabulary size: {len(tokenizer):,}")

    # Configure 8-bit quantization
    print("\nConfiguring 8-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    print("\nLoading base model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Resize embeddings if tokenizer was extended
    if len(tokenizer) > model.config.vocab_size:
        print(f"\nResizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Prepare for k-bit training
    print("\nPreparing for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("\nConfiguring LoRA adapter...")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=TRAINING_DEFAULTS.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=config.get("use_rslora", True),
    )
    print(f"  r={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"  target_modules={lora_config.target_modules}")

    # Add LoRA adapter
    print("\nAdding LoRA adapter...")
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Save model
    print(f"\nSaving to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save info
    info = {
        "base_model": str(base_model_path),
        "model_name": model_name,
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "target_modules": LORA_TARGET_MODULES,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "vocab_size": len(tokenizer),
    }
    with open(output_dir / "lora_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print("\n✓ LoRA adapter added successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add LoRA adapter to base model")
    parser.add_argument("--model", choices=["medgemma-4b", "medgemma-27b"],
                       default="medgemma-4b", help="Model to add LoRA to")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Path to base model (default: model/raw/{model})")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Path to tokenizer (default: model/tokenizer)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output directory")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use")
    args = parser.parse_args()

    success = add_lora_to_model(
        args.model,
        args.base_model,
        args.tokenizer,
        args.output,
        args.device
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
