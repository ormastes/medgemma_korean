#!/usr/bin/env python3
"""
Prepare extended Korean tokenizer for MedGemma models.

This script:
1. Loads the base MedGemma tokenizer
2. Extracts Korean tokens from training data
3. Adds new Korean tokens to tokenizer
4. Saves the extended tokenizer

Input:
- Base model tokenizer
- Training data (for token extraction)
- Medical dictionary (for medical terms)

Output:
- model/tokenizer/: Extended tokenizer with Korean tokens
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"


def extract_korean_tokens(text: str) -> list[str]:
    """Extract Korean word tokens from text."""
    # Korean character ranges
    korean_pattern = r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]+'
    tokens = re.findall(korean_pattern, text)
    return tokens


def load_training_data(data_dir: Path, max_files: int = None) -> list[str]:
    """Load training data and extract text."""
    texts = []

    # Load plain text data
    plain_text_file = data_dir / "02_refined" / "00_plain_text" / "train.jsonl"
    if plain_text_file.exists():
        print(f"Loading: {plain_text_file}")
        with open(plain_text_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Plain text"):
                try:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        texts.append(data['text'])
                except json.JSONDecodeError:
                    continue

    # Load MCQ data
    mcq_file = data_dir / "02_refined" / "02_kor_med_test" / "train.jsonl"
    if mcq_file.exists():
        print(f"Loading: {mcq_file}")
        with open(mcq_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'question' in data:
                        texts.append(data['question'])
                    for key in ['A', 'B', 'C', 'D', 'E']:
                        if key in data:
                            texts.append(data[key])
                except json.JSONDecodeError:
                    continue

    return texts


def load_medical_dict(data_dir: Path) -> list[str]:
    """Load medical dictionary terms."""
    terms = []

    dict_file = data_dir / "02_refined" / "01_medical_dict.json"
    if dict_file.exists():
        print(f"Loading: {dict_file}")
        with open(dict_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                if 'term' in entry:
                    terms.append(entry['term'])

    return terms


def get_new_tokens(
    tokenizer,
    texts: list[str],
    medical_terms: list[str],
    min_freq: int = 10,
    min_length: int = 2
) -> list[str]:
    """Find new tokens not in tokenizer vocabulary."""
    print("\nExtracting Korean tokens...")

    # Count token frequencies
    token_counts = Counter()

    for text in tqdm(texts, desc="Processing texts"):
        tokens = extract_korean_tokens(text)
        token_counts.update(tokens)

    # Add medical terms with high frequency
    for term in medical_terms:
        token_counts[term] += 100  # Boost medical terms

    # Filter tokens
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = []

    for token, count in token_counts.items():
        # Skip short tokens
        if len(token) < min_length:
            continue

        # Skip low frequency tokens
        if count < min_freq:
            continue

        # Skip if already in vocabulary
        if token in existing_vocab:
            continue

        # Check if token is split into multiple pieces
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) > 1:  # Token is split
            new_tokens.append(token)

    # Sort by frequency
    new_tokens = sorted(new_tokens, key=lambda t: token_counts[t], reverse=True)

    print(f"Found {len(new_tokens):,} new tokens")
    return new_tokens


def prepare_tokenizer(
    base_model: str,
    output_dir: Path,
    max_new_tokens: int = 30000
):
    """Prepare extended tokenizer."""
    print("=" * 60)
    print("Preparing Extended Korean Tokenizer")
    print("=" * 60)

    # Load base tokenizer
    print(f"\nLoading base tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    print(f"Base vocabulary size: {len(tokenizer):,}")

    # Load training data
    print("\nLoading training data...")
    texts = load_training_data(DATA_DIR)
    medical_terms = load_medical_dict(DATA_DIR)
    print(f"Loaded {len(texts):,} text samples")
    print(f"Loaded {len(medical_terms):,} medical terms")

    # Get new tokens
    new_tokens = get_new_tokens(
        tokenizer, texts, medical_terms,
        min_freq=5, min_length=2
    )

    # Limit number of new tokens
    if len(new_tokens) > max_new_tokens:
        print(f"Limiting to top {max_new_tokens:,} tokens")
        new_tokens = new_tokens[:max_new_tokens]

    # Add new tokens
    print(f"\nAdding {len(new_tokens):,} new tokens...")
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Successfully added: {num_added:,} tokens")
    print(f"New vocabulary size: {len(tokenizer):,}")

    # Save tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))

    # Save token lists for reference
    with open(output_dir / "new_tokens.txt", 'w', encoding='utf-8') as f:
        for token in new_tokens:
            f.write(token + '\n')

    with open(output_dir / "dictionary_terms.txt", 'w', encoding='utf-8') as f:
        for term in medical_terms:
            f.write(term + '\n')

    # Save info
    info = {
        "base_model": base_model,
        "base_vocab_size": len(tokenizer) - num_added,
        "new_tokens_added": num_added,
        "final_vocab_size": len(tokenizer),
        "training_texts": len(texts),
        "medical_terms": len(medical_terms)
    }
    with open(output_dir / "tokenizer_info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Tokenizer saved to: {output_dir}")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare extended Korean tokenizer")
    parser.add_argument("--base-model", default="google/medgemma-4b-it",
                       help="Base model for tokenizer")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "tokenizer",
                       help="Output directory for tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=25000,
                       help="Maximum new tokens to add")
    args = parser.parse_args()

    prepare_tokenizer(args.base_model, args.output, args.max_new_tokens)


if __name__ == "__main__":
    main()
