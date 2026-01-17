#!/usr/bin/env python3
"""
Merge Previous LoRAs and Add New Trainable LoRA

This script prevents catastrophic forgetting by:
1. Loading the base model
2. Merging all previous LoRA adapters into the base weights
3. Adding a new trainable LoRA adapter for the next phase

Usage:
    python merge_loras.py --previous-loras lora_0 lora_1 --output merged_model
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_config import MODEL_CONFIGS, LORA_TARGET_MODULES


def merge_previous_loras(
    base_model_path: str,
    previous_lora_paths: list,
    device: str = "cuda:0"
):
    """
    Load base model and merge all previous LoRA adapters.

    Args:
        base_model_path: Path to base model (e.g., "google/medgemma-4b-it")
        previous_lora_paths: List of LoRA adapter directory paths
        device: Device to use

    Returns:
        Model with all previous LoRAs merged (no trainable LoRA yet)
    """
    print("=" * 80)
    print("MERGING PREVIOUS LORA ADAPTERS")
    print("=" * 80)

    # Load base model
    print(f"\n[1/3] Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print(f"  Base model loaded successfully")
    print(f"  Device map: {model.hf_device_map}")

    # Merge each previous LoRA
    if not previous_lora_paths:
        print("\n[2/3] No previous LoRAs to merge")
        return model

    print(f"\n[2/3] Merging {len(previous_lora_paths)} previous LoRA adapter(s)")
    for i, lora_path in enumerate(previous_lora_paths):
        print(f"\n  LoRA {i}: {lora_path}")

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")

        # Load LoRA adapter (not trainable)
        print(f"    Loading adapter...")
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            is_trainable=False  # Load in inference mode
        )

        # Merge LoRA weights into base model
        print(f"    Merging adapter into base weights...")
        model = model.merge_and_unload()
        print(f"    ✓ LoRA {i} merged successfully")

    print(f"\n  All {len(previous_lora_paths)} LoRA adapter(s) merged!")

    return model


def add_new_lora(
    model,
    lora_config: dict,
    include_embeddings: bool = False
):
    """
    Add a new trainable LoRA adapter to the model.

    Args:
        model: Model with previous LoRAs merged
        lora_config: LoRA configuration dict (r, alpha, dropout, etc.)
        include_embeddings: Whether to include embeddings in trainable params

    Returns:
        Model with new trainable LoRA adapter
    """
    print("\n[3/3] Adding new trainable LoRA adapter")

    # Create LoRA config
    peft_config = LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("alpha", 128),
        target_modules=lora_config.get("target_modules", LORA_TARGET_MODULES),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=lora_config.get("use_rslora", True)
    )

    # Add embeddings to trainable params if requested
    if include_embeddings:
        peft_config.modules_to_save = ["embed_tokens", "lm_head"]
        print("  Including embeddings in trainable parameters")

    print(f"  LoRA rank: {peft_config.r}")
    print(f"  LoRA alpha: {peft_config.lora_alpha}")
    print(f"  Target modules: {peft_config.target_modules}")
    print(f"  Use rsLoRA: {peft_config.use_rslora}")

    # Add new LoRA
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    print("\n  Trainable parameters:")
    model.print_trainable_parameters()

    return model


def save_model(model, tokenizer, output_dir: str, save_merged: bool = False):
    """
    Save model with new LoRA adapter.

    Args:
        model: Model with new LoRA
        tokenizer: Tokenizer
        output_dir: Directory to save model
        save_merged: If True, also save fully merged model
    """
    print(f"\n[4/4] Saving model to: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save LoRA adapter only
    lora_dir = os.path.join(output_dir, "lora_adapter")
    print(f"  Saving LoRA adapter to: {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"  ✓ LoRA adapter saved")

    # Optionally save fully merged model
    if save_merged:
        merged_dir = os.path.join(output_dir, "merged")
        print(f"  Saving fully merged model to: {merged_dir}")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"  ✓ Merged model saved")

    print("\n✓ Model saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Merge previous LoRAs and add new trainable LoRA")

    # Input arguments
    parser.add_argument("--base-model", type=str, default="google/medgemma-4b-it",
                        help="Base model path or HuggingFace ID")
    parser.add_argument("--previous-loras", type=str, nargs="*", default=[],
                        help="List of previous LoRA adapter directories to merge")

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--use-rslora", action="store_true", default=True,
                        help="Use rsLoRA (rank-stabilized LoRA)")
    parser.add_argument("--include-embeddings", action="store_true",
                        help="Include embeddings in trainable parameters")

    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for model with new LoRA")
    parser.add_argument("--save-merged", action="store_true",
                        help="Also save fully merged model (not just LoRA)")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")

    args = parser.parse_args()

    # Print configuration
    print("\nConfiguration:")
    print(f"  Base model: {args.base_model}")
    print(f"  Previous LoRAs: {args.previous_loras if args.previous_loras else 'None'}")
    print(f"  Output: {args.output}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"  Include embeddings: {args.include_embeddings}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print("✓ Tokenizer loaded\n")

    # Step 1: Merge previous LoRAs
    model = merge_previous_loras(
        base_model_path=args.base_model,
        previous_lora_paths=args.previous_loras,
        device=args.device
    )

    # Step 2: Add new trainable LoRA
    lora_config = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "use_rslora": args.use_rslora,
        "target_modules": LORA_TARGET_MODULES
    }

    model = add_new_lora(
        model=model,
        lora_config=lora_config,
        include_embeddings=args.include_embeddings
    )

    # Step 3: Save model
    save_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output,
        save_merged=args.save_merged
    )

    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Train using this model as base: {args.output}/lora_adapter")
    print(f"  2. Previous knowledge is frozen (no catastrophic forgetting)")
    print(f"  3. Only new LoRA adapter will be trainable")
    print()


if __name__ == "__main__":
    main()
