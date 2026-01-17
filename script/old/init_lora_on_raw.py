#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialize LoRA on raw HuggingFace model and save to raw_lora_added/

This creates a base model with LoRA adapter initialized (untrained).
This is the starting point for train_00.

Supports extended tokenizer with Korean vocabulary:
    python init_lora_on_raw.py --model medgemma-4b --extended-tokenizer

Usage:
    python init_lora_on_raw.py --model medgemma-27b
    python init_lora_on_raw.py --model medgemma-4b --extended-tokenizer
"""

import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from training_utils import load_tokenizer, load_model_8bit, create_lora_config
from training_config import MODEL_CONFIGS
from peft import get_peft_model

BASE_DIR = Path(__file__).parent.parent
RAW_MODEL_DIR = BASE_DIR / "model" / "raw"
OUTPUT_DIR = BASE_DIR / "model" / "raw_lora_added"
TOKENIZER_DIR = BASE_DIR / "model" / "tokenizer"


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def get_raw_model_path(model_name: str) -> str:
    """Get raw model path from config"""
    config_file = RAW_MODEL_DIR / "model_paths.json"

    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        if model_name in config:
            return config[model_name]["huggingface_id"]

    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]['path']

    raise ValueError(f"Unknown model: {model_name}")


def load_extended_tokenizer():
    """Load extended Korean tokenizer (shared between 4b and 27b)."""
    from transformers import AutoTokenizer

    # Shared tokenizer location
    tokenizer_path = TOKENIZER_DIR / "extended_tokenizer"

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Extended tokenizer not found: {tokenizer_path}\n"
            f"Run build_korean_tokenizer.py first."
        )

    log(f"Loading extended tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    # Load token mapping info
    mapping_file = TOKENIZER_DIR / "token_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        log(f"  New tokens: {len(mapping.get('new_tokens', []))}")
    else:
        mapping = {}

    return tokenizer, mapping


def resize_model_embeddings(model, tokenizer, token_mapping: dict):
    """Resize model embeddings and initialize new tokens."""
    # Handle Gemma3 config (vocab_size in text_config)
    if hasattr(model.config, 'vocab_size'):
        original_vocab_size = model.config.vocab_size
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'vocab_size'):
        original_vocab_size = model.config.text_config.vocab_size
    else:
        # Fallback: get from embeddings
        embed = model.get_input_embeddings()
        original_vocab_size = embed.weight.shape[0]

    new_vocab_size = len(tokenizer)

    if new_vocab_size <= original_vocab_size:
        log(f"No resize needed: {original_vocab_size} -> {new_vocab_size}")
        return model

    log(f"Resizing embeddings: {original_vocab_size} -> {new_vocab_size}")

    # Resize token embeddings
    model.resize_token_embeddings(new_vocab_size)

    # Initialize new embeddings with mean of existing embeddings
    new_token_count = new_vocab_size - original_vocab_size

    with torch.no_grad():
        # Get embedding layer - use get_input_embeddings (works for all model types)
        embed_tokens = model.get_input_embeddings()
        if embed_tokens is None:
            log("  Warning: Could not find embedding layer for initialization")
            return model

        # Calculate mean of existing embeddings
        existing_embeddings = embed_tokens.weight[:original_vocab_size]
        mean_embedding = existing_embeddings.mean(dim=0)

        # Initialize new embeddings with mean + small noise
        log(f"  Initializing {new_token_count} new embeddings...")
        for i in range(original_vocab_size, new_vocab_size):
            noise = torch.randn_like(mean_embedding) * 0.01
            embed_tokens.weight[i] = mean_embedding + noise

        # Also initialize lm_head if it exists
        if hasattr(model, 'lm_head'):
            lm_head = model.lm_head
            if hasattr(lm_head, 'weight') and lm_head.weight.shape[0] == new_vocab_size:
                log(f"  Initializing lm_head for new tokens...")
                for i in range(original_vocab_size, new_vocab_size):
                    noise = torch.randn(lm_head.weight.shape[1]) * 0.01
                    lm_head.weight[i] = noise

    log(f"  Embeddings initialized successfully")
    return model


def main():
    parser = argparse.ArgumentParser(description="Initialize LoRA on raw model")
    parser.add_argument("--model", default="medgemma-27b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extended-tokenizer", action="store_true",
                       help="Use extended Korean tokenizer (run build_korean_tokenizer.py first)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    output_dir = OUTPUT_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("Initialize LoRA on Raw Model")
    log("=" * 60)

    # Get raw model path
    model_path = get_raw_model_path(args.model)
    log(f"Raw model: {model_path}")
    log(f"Output: {output_dir}")
    log(f"Extended tokenizer: {args.extended_tokenizer}")

    # Load tokenizer
    token_mapping = {}
    if args.extended_tokenizer:
        log("Loading extended Korean tokenizer (shared)...")
        tokenizer, token_mapping = load_extended_tokenizer()
    else:
        log("Loading tokenizer...")
        tokenizer = load_tokenizer(model_path)

    # Load model with 8-bit quantization
    log("Loading model (8-bit)...")
    model = load_model_8bit(model_path, device=args.device)

    # Resize embeddings if using extended tokenizer
    if args.extended_tokenizer and token_mapping:
        log("Resizing model embeddings for extended tokenizer...")
        model = resize_model_embeddings(model, tokenizer, token_mapping)

    # Create LoRA config (include embeddings for continued pretraining)
    log("Creating LoRA config...")
    lora_config = create_lora_config(
        lora_r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        include_embeddings=True,
        use_rslora=True
    )

    # Add LoRA to model
    log("Adding LoRA adapter...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Save model and tokenizer
    log(f"Saving to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save info
    info = {
        "script": "init_lora_on_raw",
        "model": args.model,
        "base_model": model_path,
        "lora_r": cfg['lora_r'],
        "lora_alpha": cfg['lora_alpha'],
        "include_embeddings": True,
        "use_rslora": True,
        "extended_tokenizer": args.extended_tokenizer,
        "new_tokens_count": len(token_mapping.get('new_tokens', [])) if token_mapping else 0,
        "status": "initialized_not_trained"
    }

    with open(output_dir / "init_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    log("=" * 60)
    log("DONE!")
    log(f"Output: {output_dir}")
    if args.extended_tokenizer:
        log(f"Extended tokenizer with {info['new_tokens_count']} new Korean tokens")
    log("This model has LoRA initialized but NOT trained.")
    log("Use train_00_plain_text.py to train it.")
    log("=" * 60)


if __name__ == "__main__":
    main()
