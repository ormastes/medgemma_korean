#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared LoRA Management for Training Pipeline

Provides unified LoRA loading, adding, and training control for all training scripts.

Usage:
    # train_00/01: Single LoRA (add if missing, train it)
    model, tokenizer = load_model_with_lora(
        model_path="model/raw_lora_added/medgemma-4b",
        lora_count=1,
        lora_enable_count=1,
        device="cuda:0"
    )

    # train_02: Dual LoRA (add second if missing, train only second)
    model, tokenizer = load_model_with_lora(
        model_path="model/01_trained/medgemma-4b",
        lora_count=2,
        lora_enable_count=1,
        lora_enable_index=-1,  # -1 = last (lora_01)
        device="cuda:0"
    )

LoRA Naming Convention:
    - "default" or "lora_00": First LoRA (train_00/01)
    - "lora_01": Second LoRA (train_02)
"""

import gc
import json
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    PeftModel, PeftConfig, LoraConfig,
    get_peft_model, prepare_model_for_kbit_training
)

# Import configs
import sys
sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS, MEMORY_CONFIGS, LORA_TARGET_MODULES, TRAINING_DEFAULTS


# =============================================================================
# Constants
# =============================================================================

LORA_ADAPTER_NAMES = ["default", "lora_01"]  # Ordered list of adapter names


# =============================================================================
# Helper Functions
# =============================================================================

def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_lora_config(
    rank: int = 16,
    alpha: int = 32,
    include_embeddings: bool = False,
    use_rslora: bool = True,
    dropout: float = None
) -> LoraConfig:
    """
    Create LoRA configuration.

    Args:
        rank: LoRA rank (r)
        alpha: LoRA alpha scaling
        include_embeddings: Include embed_tokens and lm_head in modules_to_save
        use_rslora: Use rank-stabilized LoRA
        dropout: LoRA dropout (None = use default from TRAINING_DEFAULTS)

    Returns:
        LoraConfig instance
    """
    if dropout is None:
        dropout = TRAINING_DEFAULTS.get('lora_dropout', 0.05)

    modules_to_save = ["embed_tokens", "lm_head"] if include_embeddings else None

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES.copy(),
        modules_to_save=modules_to_save,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=use_rslora,
    )

    return config


# =============================================================================
# LoRA Detection Functions
# =============================================================================

def has_lora_adapter(model_path: str) -> bool:
    """Check if model path contains a LoRA adapter."""
    path = Path(model_path)
    adapter_config = path / "adapter_config.json"
    return adapter_config.exists()


def count_lora_adapters(model) -> int:
    """
    Count existing LoRA adapters in a PeftModel.

    Args:
        model: PeftModel instance

    Returns:
        Number of LoRA adapters (0 if not a PeftModel)
    """
    if not hasattr(model, 'peft_config'):
        return 0
    return len(model.peft_config)


def get_adapter_names(model) -> List[str]:
    """
    Get list of adapter names in a PeftModel.

    Args:
        model: PeftModel instance

    Returns:
        List of adapter names
    """
    if not hasattr(model, 'peft_config'):
        return []
    return list(model.peft_config.keys())


def get_active_adapter(model) -> Optional[str]:
    """Get currently active adapter name."""
    if hasattr(model, 'active_adapter'):
        return model.active_adapter
    return None


# =============================================================================
# LoRA Management Functions
# =============================================================================

def add_lora_adapter(
    model,
    adapter_name: str,
    lora_config: LoraConfig,
    set_trainable: bool = True
) -> Any:
    """
    Add a new LoRA adapter to model.

    Args:
        model: Base model or PeftModel
        adapter_name: Name for the new adapter
        lora_config: LoRA configuration
        set_trainable: Make this adapter trainable

    Returns:
        Model with new adapter
    """
    if hasattr(model, 'peft_config'):
        # Already a PeftModel - add new adapter
        model.add_adapter(adapter_name, lora_config)
        if set_trainable:
            model.set_adapter(adapter_name)
    else:
        # Base model - wrap with PeftModel
        model = get_peft_model(model, lora_config)

    return model


def set_trainable_adapters(
    model,
    adapter_names: List[str],
    freeze_others: bool = True
) -> None:
    """
    Set which LoRA adapters are trainable.

    Args:
        model: PeftModel instance
        adapter_names: List of adapter names to make trainable
        freeze_others: Freeze all other adapters
    """
    if not hasattr(model, 'peft_config'):
        return

    all_adapters = get_adapter_names(model)

    for name in all_adapters:
        if name in adapter_names:
            # Make trainable
            for param_name, param in model.named_parameters():
                if name in param_name:
                    param.requires_grad = True
        elif freeze_others:
            # Freeze
            for param_name, param in model.named_parameters():
                if name in param_name:
                    param.requires_grad = False

    # Set active adapter to first trainable one
    if adapter_names:
        model.set_adapter(adapter_names[0])


def freeze_all_lora(model) -> None:
    """Freeze all LoRA adapters."""
    if not hasattr(model, 'peft_config'):
        return

    for name, param in model.named_parameters():
        # LoRA params have patterns like:
        # .lora_A.default.weight, .lora_B.lora_01.weight
        if '.lora_A.' in name or '.lora_B.' in name:
            param.requires_grad = False


def unfreeze_lora(model, adapter_name: str) -> None:
    """
    Unfreeze a specific LoRA adapter.

    Args:
        model: PeftModel
        adapter_name: Adapter to unfreeze (e.g., "default", "lora_01")
    """
    if not hasattr(model, 'peft_config'):
        return

    # Pattern: .lora_A.{adapter_name}.weight or .lora_B.{adapter_name}.weight
    pattern = f'.{adapter_name}.'

    for name, param in model.named_parameters():
        if pattern in name and ('.lora_A.' in name or '.lora_B.' in name):
            param.requires_grad = True


# =============================================================================
# Main Loading Function
# =============================================================================

def load_model_with_lora(
    model_path: str,
    lora_count: Optional[int] = None,
    lora_enable_count: int = 1,
    lora_enable_index: int = 0,
    lora_config: Optional[Dict] = None,
    device: str = "cuda:0",
    model_name: Optional[str] = None,
    include_embeddings: Optional[bool] = None,
    verbose: bool = True
) -> Tuple[Any, Any]:
    """
    Load model with expected LoRA count and training configuration.

    This is the main function for loading models in train_00, train_01, train_02.
    It handles:
    - Loading base model with 8-bit quantization
    - Loading existing LoRA adapters
    - Adding new LoRA adapters if needed
    - Setting which LoRAs are trainable

    Args:
        model_path: Path to model (can be base model, raw_lora_added, or trained)
        lora_count: Minimum number of LoRA adapters required.
                    None = use existing count (don't add any)
                    1 = ensure at least 1 (add if 0)
                    2 = ensure at least 2 (add if < 2)
        lora_enable_count: Number of LoRAs to enable for training (usually 1)
        lora_enable_index: Which LoRA to start enabling (0 = first, -1 = last)
        lora_config: Dict with LoRA config (rank, alpha, etc.) or None for defaults
        device: CUDA device
        model_name: Model name for config lookup (e.g., "medgemma-4b")
        include_embeddings: Include embeddings in LoRA (None = auto from MEMORY_CONFIGS)
        verbose: Print progress messages

    Returns:
        Tuple of (model, tokenizer)

    Examples:
        # train_00/01: Use existing LoRA(s), train first one only
        model, tokenizer = load_model_with_lora(
            "model/00_trained/medgemma-4b",
            lora_count=None,  # Don't add any
            lora_enable_count=1,
            lora_enable_index=0  # Train first LoRA
        )

        # train_00 on fresh model: Add LoRA if missing
        model, tokenizer = load_model_with_lora(
            "model/raw_lora_added/medgemma-4b",
            lora_count=1,  # Ensure at least 1
            lora_enable_count=1
        )

        # train_02: Add second LoRA if needed, train only it
        model, tokenizer = load_model_with_lora(
            "model/01_trained/medgemma-4b",
            lora_count=2,  # Ensure at least 2
            lora_enable_count=1,
            lora_enable_index=-1  # Train last LoRA
        )

        # 2-LoRA model in train_00: Train only first LoRA
        model, tokenizer = load_model_with_lora(
            "model/with_2_loras/medgemma-4b",
            lora_count=None,  # Keep existing 2 LoRAs
            lora_enable_count=1,
            lora_enable_index=0  # Train only first
        )
    """
    def log(msg):
        if verbose:
            print(f"[LoRA] {msg}")

    model_path = str(model_path)
    path = Path(model_path)

    # Get model config
    if model_name is None:
        # Try to infer from path
        for name in MODEL_CONFIGS.keys():
            if name in model_path:
                model_name = name
                break

    cfg = MODEL_CONFIGS.get(model_name, {})
    mem_cfg = MEMORY_CONFIGS.get(model_name, {})

    # Determine include_embeddings
    if include_embeddings is None:
        include_embeddings = mem_cfg.get('train_embeddings', False)

    # Build LoRA config
    if lora_config is None:
        lora_config = {}

    # Note: final_lora_config is only used for adding the FIRST LoRA (index 0)
    # include_embeddings is only for first LoRA
    final_lora_config = get_lora_config(
        rank=lora_config.get('rank', cfg.get('lora_r', 16)),
        alpha=lora_config.get('alpha', cfg.get('lora_alpha', 32)),
        include_embeddings=include_embeddings,  # Only applies to first LoRA
        use_rslora=lora_config.get('use_rslora', cfg.get('use_rslora', True)),
        dropout=lora_config.get('dropout', None)
    )

    # Check if path has LoRA adapter
    has_lora = has_lora_adapter(model_path)

    log(f"Model path: {model_path}")
    log(f"Has LoRA: {has_lora}")
    log(f"Min LoRA count: {lora_count} (None=use existing)")
    log(f"Enable count: {lora_enable_count}, index: {lora_enable_index}")

    # =========================================================================
    # Step 1: Load tokenizer
    # =========================================================================
    log("Loading tokenizer...")
    if has_lora:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.get('path', model_path),
            trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Tokenizer vocab size: {len(tokenizer)}")

    # =========================================================================
    # Step 2: Load base model
    # =========================================================================
    log("Loading base model (8-bit)...")

    # Get base model path
    if has_lora:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path
    else:
        base_model_path = cfg.get('path', model_path)

    log(f"Base model: {base_model_path}")

    # Load with 8-bit quantization
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    use_gradient_checkpointing = mem_cfg.get('use_gradient_checkpointing', False)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=use_gradient_checkpointing
    )

    # Resize embeddings if needed
    base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    if tokenizer_vocab_size > base_vocab_size:
        log(f"Resizing embeddings: {base_vocab_size} -> {tokenizer_vocab_size}")
        base_model.resize_token_embeddings(tokenizer_vocab_size)

    # =========================================================================
    # Step 3: Load/Add LoRA adapters
    # =========================================================================
    current_lora_count = 0

    if has_lora:
        # Load existing LoRA
        log(f"Loading existing LoRA from: {model_path}")
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            is_trainable=True
        )
        # Count actual loaded adapters
        current_lora_count = count_lora_adapters(model)
        log(f"Loaded LoRA adapters: {get_adapter_names(model)} (count={current_lora_count})")
    else:
        model = base_model
        current_lora_count = 0

    # Determine target count
    target_lora_count = lora_count if lora_count is not None else current_lora_count
    if target_lora_count < 1 and current_lora_count == 0:
        # Must have at least 1 LoRA
        target_lora_count = 1
        log("No LoRA found and lora_count=None, will add 1 LoRA")

    log(f"Current LoRA count: {current_lora_count}, target: {target_lora_count}")

    # Add more LoRAs if needed
    while current_lora_count < target_lora_count:
        adapter_name = LORA_ADAPTER_NAMES[current_lora_count] if current_lora_count < len(LORA_ADAPTER_NAMES) else f"lora_{current_lora_count:02d}"

        log(f"Adding new LoRA adapter: {adapter_name}")

        # For second+ LoRA, don't include embeddings
        if current_lora_count > 0:
            add_config = get_lora_config(
                rank=lora_config.get('rank', cfg.get('lora_r', 16)),
                alpha=lora_config.get('alpha', cfg.get('lora_alpha', 32)),
                include_embeddings=False,  # Only first LoRA gets embeddings
                use_rslora=lora_config.get('use_rslora', cfg.get('use_rslora', True)),
            )
        else:
            add_config = final_lora_config

        model = add_lora_adapter(model, adapter_name, add_config, set_trainable=False)
        current_lora_count += 1
        log(f"Added LoRA: {adapter_name}")

    # =========================================================================
    # Step 4: Set trainable adapters
    # =========================================================================
    all_adapters = get_adapter_names(model)
    log(f"All adapters: {all_adapters}")

    # Determine which adapters to train
    if lora_enable_index < 0:
        # Negative index from end
        enable_start = max(0, len(all_adapters) + lora_enable_index - lora_enable_count + 1)
    else:
        enable_start = lora_enable_index

    enable_end = min(enable_start + lora_enable_count, len(all_adapters))
    trainable_adapters = all_adapters[enable_start:enable_end]

    log(f"Trainable adapters: {trainable_adapters}")

    # Freeze all first
    freeze_all_lora(model)

    # Then unfreeze selected
    for adapter in trainable_adapters:
        unfreeze_lora(model, adapter)

    # Set active adapter
    if trainable_adapters:
        model.set_adapter(trainable_adapters[-1])  # Set last trainable as active

    # =========================================================================
    # Step 5: Print summary
    # =========================================================================
    log("=" * 50)
    log("Model loaded successfully")
    log(f"  Adapters: {all_adapters}")
    log(f"  Trainable: {trainable_adapters}")
    model.print_trainable_parameters()
    log("=" * 50)

    return model, tokenizer


# =============================================================================
# Convenience Functions for Training Scripts
# =============================================================================

def load_for_train_00(
    model_path: str,
    device: str = "cuda:0",
    model_name: str = None,
    include_embeddings: bool = None,
    verbose: bool = True
) -> Tuple[Any, Any]:
    """
    Load model for train_00 (plain text pre-training).

    - lora_count=None: Use existing LoRAs (add 1 if none exist)
    - lora_enable_count=1: Train only first LoRA
    - lora_enable_index=0: Start from first LoRA

    Works with:
    - Fresh model (no LoRA): Adds 1 LoRA, trains it
    - 1-LoRA model: Trains that LoRA
    - 2-LoRA model: Trains only first LoRA, freezes second
    """
    return load_model_with_lora(
        model_path=model_path,
        lora_count=None,  # Use existing, add 1 if none
        lora_enable_count=1,
        lora_enable_index=0,  # Train first LoRA
        device=device,
        model_name=model_name,
        include_embeddings=include_embeddings,
        verbose=verbose
    )


def load_for_train_01(
    model_path: str,
    device: str = "cuda:0",
    model_name: str = None,
    verbose: bool = True
) -> Tuple[Any, Any]:
    """
    Load model for train_01 (mixed training).

    - lora_count=None: Use existing LoRAs (add 1 if none exist)
    - lora_enable_count=1: Train only first LoRA
    - lora_enable_index=0: Start from first LoRA

    Works with:
    - 1-LoRA model: Trains that LoRA
    - 2-LoRA model: Trains only first LoRA, freezes second
    """
    return load_model_with_lora(
        model_path=model_path,
        lora_count=None,  # Use existing, add 1 if none
        lora_enable_count=1,
        lora_enable_index=0,  # Train first LoRA
        device=device,
        model_name=model_name,
        include_embeddings=False,  # Embeddings already trained in train_00
        verbose=verbose
    )


def load_for_train_02(
    model_path: str,
    device: str = "cuda:0",
    model_name: str = None,
    verbose: bool = True
) -> Tuple[Any, Any]:
    """
    Load model for train_02 (MCQ with reasoning).

    - lora_count=2: Ensure at least 2 LoRAs (add if < 2)
    - lora_enable_count=1: Train only last LoRA
    - lora_enable_index=-1: Train last LoRA (freeze previous)

    Works with:
    - 1-LoRA model: Adds second LoRA, trains only it
    - 2-LoRA model: Trains only second LoRA, freezes first
    - 3-LoRA model: Trains only last LoRA, freezes first two
    """
    return load_model_with_lora(
        model_path=model_path,
        lora_count=2,  # Ensure at least 2
        lora_enable_count=1,
        lora_enable_index=-1,  # Train last LoRA
        device=device,
        model_name=model_name,
        include_embeddings=False,  # No embeddings for second LoRA
        verbose=verbose
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LoRA loading")
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument("--lora-count", type=int, default=1, help="Expected LoRA count")
    parser.add_argument("--lora-enable", type=int, default=1, help="LoRAs to enable")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    args = parser.parse_args()

    print(f"Testing load_model_with_lora:")
    print(f"  Path: {args.model_path}")
    print(f"  LoRA count: {args.lora_count}")
    print(f"  Enable: {args.lora_enable}")
    print()

    model, tokenizer = load_model_with_lora(
        model_path=args.model_path,
        lora_count=args.lora_count,
        lora_enable_count=args.lora_enable,
        device=args.device,
        verbose=True
    )

    print("\nAdapter info:")
    print(f"  Names: {get_adapter_names(model)}")
    print(f"  Active: {get_active_adapter(model)}")
    print(f"  Count: {count_lora_adapters(model)}")

    # Print trainable params
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print(f"\nParameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
