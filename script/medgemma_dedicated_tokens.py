#!/usr/bin/env python3
"""
MedGemma Dedicated Tokens - Add special medical characters to tokenizer

This script demonstrates how to:
1. Load special characters from special_chars_report.json
2. Add them as dedicated tokens to MedGemma tokenizer
3. Resize model embeddings to accommodate new tokens
4. Initialize new token embeddings properly

Usage:
    python script/medgemma_dedicated_tokens.py --check-only  # Just check which tokens are new
    python script/medgemma_dedicated_tokens.py --output model/tokenizer/dedicated  # Add tokens and save
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_special_chars(report_path: str) -> List[Dict]:
    """Load special characters from report JSON."""
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('characters', [])


def extract_dedicated_tokens(special_chars: List[Dict]) -> List[str]:
    """Extract unique characters as potential dedicated tokens."""
    tokens = []
    for item in special_chars:
        char = item.get('char', '')
        if char:
            tokens.append(char)
    return tokens


def check_existing_tokens(tokenizer, tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Check which tokens already exist in tokenizer vocabulary.

    Returns:
        (existing_tokens, new_tokens)
    """
    existing = []
    new = []

    for token in tokens:
        # Check if token is in vocabulary (not split into multiple tokens)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)

        # If single token AND decodes back correctly, it exists
        if len(encoded) == 1 and decoded == token:
            existing.append(token)
        else:
            new.append(token)

    return existing, new


def add_dedicated_tokens(
    tokenizer,
    model,
    new_tokens: List[str],
    init_method: str = "mean"
) -> Tuple[int, int]:
    """
    Add dedicated tokens to tokenizer and resize model embeddings.

    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model (optional, can be None for tokenizer-only)
        new_tokens: List of new tokens to add
        init_method: How to initialize new embeddings ("mean", "random", "zero")

    Returns:
        (num_added, new_vocab_size)
    """
    # Add tokens to tokenizer
    num_added = tokenizer.add_tokens(new_tokens)
    new_vocab_size = len(tokenizer)

    print(f"Added {num_added} new tokens to tokenizer")
    print(f"New vocabulary size: {new_vocab_size}")

    if model is not None:
        # Get old embeddings
        old_embeddings = model.get_input_embeddings()
        old_vocab_size, embedding_dim = old_embeddings.weight.shape

        print(f"Old embedding shape: ({old_vocab_size}, {embedding_dim})")

        # Resize token embeddings
        model.resize_token_embeddings(new_vocab_size)

        # Get new embeddings
        new_embeddings = model.get_input_embeddings()

        # Initialize new token embeddings
        with torch.no_grad():
            if init_method == "mean":
                # Initialize with mean of existing embeddings
                mean_embedding = old_embeddings.weight[:old_vocab_size].mean(dim=0)
                for i in range(old_vocab_size, new_vocab_size):
                    new_embeddings.weight[i] = mean_embedding
            elif init_method == "random":
                # Random initialization (scaled)
                std = old_embeddings.weight[:old_vocab_size].std()
                for i in range(old_vocab_size, new_vocab_size):
                    new_embeddings.weight[i] = torch.randn(embedding_dim) * std * 0.02
            elif init_method == "zero":
                # Zero initialization
                for i in range(old_vocab_size, new_vocab_size):
                    new_embeddings.weight[i] = torch.zeros(embedding_dim)

        print(f"New embedding shape: {new_embeddings.weight.shape}")

        # Also resize output embeddings (lm_head) if it exists
        if hasattr(model, 'lm_head'):
            # lm_head is usually tied to embeddings, but verify
            pass

    return num_added, new_vocab_size


def create_token_mapping(tokenizer, added_tokens: List[str]) -> Dict:
    """Create a mapping of added tokens to their IDs."""
    mapping = {}
    for token in added_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        mapping[token] = {
            "id": token_id,
            "unicode": f"U+{ord(token):04X}" if len(token) == 1 else "multi-char",
            "code": ord(token) if len(token) == 1 else None
        }
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Add dedicated tokens to MedGemma tokenizer")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",  # Use smaller model for testing
        help="Base model to load tokenizer from"
    )
    parser.add_argument(
        "--special-chars",
        type=str,
        default="data/01_raw/02_kor_med_test/special_chars_report.json",
        help="Path to special characters report JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory to save modified tokenizer"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check which tokens are new, don't modify"
    )
    parser.add_argument(
        "--init-method",
        type=str,
        choices=["mean", "random", "zero"],
        default="mean",
        help="Method to initialize new token embeddings"
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Also load and resize the model (requires more memory)"
    )
    parser.add_argument(
        "--additional-tokens",
        type=str,
        nargs="+",
        default=[],
        help="Additional tokens to add (e.g., <reasoning> </reasoning>)"
    )

    args = parser.parse_args()

    # Load special characters
    print(f"\n{'='*60}")
    print("Loading special characters...")
    print(f"{'='*60}")

    special_chars = load_special_chars(args.special_chars)
    dedicated_tokens = extract_dedicated_tokens(special_chars)

    print(f"Found {len(dedicated_tokens)} special characters in report")

    # Add any additional tokens
    if args.additional_tokens:
        dedicated_tokens.extend(args.additional_tokens)
        print(f"Added {len(args.additional_tokens)} additional tokens")

    # Load tokenizer
    print(f"\n{'='*60}")
    print(f"Loading tokenizer from: {args.model}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")

    # Check which tokens are new
    print(f"\n{'='*60}")
    print("Checking token existence...")
    print(f"{'='*60}")

    existing_tokens, new_tokens = check_existing_tokens(tokenizer, dedicated_tokens)

    print(f"\nExisting tokens ({len(existing_tokens)}):")
    for token in existing_tokens[:10]:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        print(f"  '{token}' (U+{ord(token):04X}) -> ID {token_id}")
    if len(existing_tokens) > 10:
        print(f"  ... and {len(existing_tokens) - 10} more")

    print(f"\nNew tokens to add ({len(new_tokens)}):")
    for token in new_tokens:
        if len(token) == 1:
            print(f"  '{token}' (U+{ord(token):04X})")
        else:
            print(f"  '{token}'")

    if args.check_only:
        print(f"\n{'='*60}")
        print("Check-only mode - not modifying tokenizer")
        print(f"{'='*60}")
        return

    if len(new_tokens) == 0:
        print("\nNo new tokens to add!")
        return

    # Load model if requested
    model = None
    if args.load_model:
        print(f"\n{'='*60}")
        print("Loading model...")
        print(f"{'='*60}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    # Add tokens
    print(f"\n{'='*60}")
    print("Adding dedicated tokens...")
    print(f"{'='*60}")

    num_added, new_vocab_size = add_dedicated_tokens(
        tokenizer, model, new_tokens, init_method=args.init_method
    )

    # Create token mapping
    token_mapping = create_token_mapping(tokenizer, new_tokens)

    # Verify tokens were added
    print(f"\n{'='*60}")
    print("Verifying added tokens...")
    print(f"{'='*60}")

    for token in new_tokens[:5]:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        print(f"  '{token}' -> {encoded} -> '{decoded}'")

    # Save if output specified
    if args.output:
        print(f"\n{'='*60}")
        print(f"Saving to: {args.output}")
        print(f"{'='*60}")

        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        tokenizer.save_pretrained(output_path)
        print(f"Saved tokenizer to {output_path}")

        # Save token mapping
        mapping_path = output_path / "dedicated_token_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                "original_vocab_size": original_vocab_size,
                "new_vocab_size": new_vocab_size,
                "num_added": num_added,
                "tokens": token_mapping
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved token mapping to {mapping_path}")

        # Save model if loaded
        if model is not None:
            model.save_pretrained(output_path)
            print(f"Saved model to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Original vocab size: {original_vocab_size}")
    print(f"New vocab size: {new_vocab_size}")
    print(f"Tokens added: {num_added}")
    print(f"Existing (skipped): {len(existing_tokens)}")


if __name__ == "__main__":
    main()
