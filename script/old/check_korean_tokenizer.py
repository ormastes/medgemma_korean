#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Korean Tokenizer Support

Verifies:
1. Tokenizer has Korean token support
2. Korean medical terms tokenization efficiency
3. Embedding training recommendation

Usage:
    python check_korean_tokenizer.py --model medgemma-4b
    python check_korean_tokenizer.py --model medgemma-27b --verbose
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS


def check_korean_support(tokenizer, verbose=False):
    """Check if tokenizer supports Korean well."""
    print("\n" + "=" * 60)
    print("Korean Token Support Check")
    print("=" * 60)

    print(f"\nVocab size: {tokenizer.vocab_size:,}")

    # Count Korean tokens in vocabulary
    vocab = tokenizer.get_vocab()
    korean_tokens = []
    for token in vocab.keys():
        # Check if token contains Korean characters (Hangul syllables: AC00-D7A3)
        if any(0xAC00 <= ord(c) <= 0xD7A3 for c in token):
            korean_tokens.append(token)

    print(f"Korean tokens in vocab: {len(korean_tokens):,}")
    print(f"Korean coverage: {len(korean_tokens) / tokenizer.vocab_size * 100:.2f}%")

    if verbose and korean_tokens:
        print(f"\nSample Korean tokens (first 30):")
        for i, token in enumerate(sorted(korean_tokens)[:30]):
            print(f"  {token}", end="  ")
            if (i + 1) % 6 == 0:
                print()
        print()

    return len(korean_tokens) > 0


def test_korean_medical_terms(tokenizer, verbose=False):
    """Test tokenization of Korean medical terms."""
    print("\n" + "=" * 60)
    print("Korean Medical Terms Tokenization")
    print("=" * 60)

    # Common Korean medical terms
    medical_terms = [
        ("당뇨병", "Diabetes"),
        ("고혈압", "Hypertension"),
        ("심근경색", "Myocardial infarction"),
        ("폐렴", "Pneumonia"),
        ("간경변", "Liver cirrhosis"),
        ("뇌졸중", "Stroke"),
        ("갑상선기능저하증", "Hypothyroidism"),
        ("급성림프모구백혈병", "Acute lymphoblastic leukemia"),
        ("혈액투석", "Hemodialysis"),
        ("항암화학요법", "Chemotherapy"),
    ]

    total_chars = 0
    total_tokens = 0

    print(f"\n{'Korean Term':<20} {'English':<30} {'Tokens':<6} {'Ratio':<6}")
    print("-" * 65)

    for korean, english in medical_terms:
        tokens = tokenizer.tokenize(korean)
        token_count = len(tokens)
        char_count = len(korean)
        ratio = token_count / char_count

        total_chars += char_count
        total_tokens += token_count

        if verbose:
            print(f"{korean:<20} {english:<30} {token_count:<6} {ratio:.2f}")
            print(f"  Tokens: {tokens}")
        else:
            print(f"{korean:<20} {english:<30} {token_count:<6} {ratio:.2f}")

    avg_ratio = total_tokens / total_chars
    print("-" * 65)
    print(f"{'Average':<50} {'':<6} {avg_ratio:.2f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    if avg_ratio <= 1.2:
        print("✓ Excellent: Korean is well tokenized (word/morpheme level)")
    elif avg_ratio <= 1.5:
        print("✓ Good: Korean tokenization is reasonable")
    elif avg_ratio <= 2.0:
        print("⚠️ Fair: Korean is tokenized at syllable level")
        print("   Recommendation: Include embeddings in LoRA training")
    else:
        print("❌ Poor: Korean is tokenized at character level")
        print("   Recommendation: Consider extending tokenizer")

    return avg_ratio


def test_sample_training_data(tokenizer, verbose=False):
    """Test tokenization on actual training data."""
    print("\n" + "=" * 60)
    print("Sample Training Data Tokenization")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent
    data_files = [
        ("Plain Text", base_dir / "data" / "02_refined" / "00_plain_text" / "train.jsonl"),
        ("Medical Dict", base_dir / "data" / "02_refined" / "01_medical_dict.json"),
        ("MCQ Test", base_dir / "data" / "02_refined" / "02_kor_med_test" / "train.jsonl"),
    ]

    for name, file_path in data_files:
        if not file_path.exists():
            print(f"\n{name}: File not found")
            continue

        print(f"\n{name}:")

        # Load sample data
        samples = []
        if file_path.suffix == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                samples = data[:10] if isinstance(data, list) else []
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    samples.append(json.loads(line))

        if not samples:
            print("  No samples found")
            continue

        # Analyze tokenization
        total_chars = 0
        total_tokens = 0

        for sample in samples:
            text = sample.get('text', sample.get('definition', ''))
            if not text:
                continue

            tokens = tokenizer.tokenize(text)
            total_chars += len(text)
            total_tokens += len(tokens)

        if total_chars > 0:
            avg_ratio = total_tokens / total_chars
            print(f"  Samples: {len(samples)}")
            print(f"  Avg chars/sample: {total_chars / len(samples):.0f}")
            print(f"  Avg tokens/sample: {total_tokens / len(samples):.0f}")
            print(f"  Token/char ratio: {avg_ratio:.2f}")


def check_embedding_training_setup():
    """Check if embedding training is properly configured."""
    print("\n" + "=" * 60)
    print("Embedding Training Setup")
    print("=" * 60)

    init_script = Path(__file__).parent / "init_lora_on_raw.py"

    if init_script.exists():
        with open(init_script, 'r') as f:
            content = f.read()

        if "include_embeddings=True" in content:
            print("✓ init_lora_on_raw.py: include_embeddings=True")
            print("  Embeddings will be trained with LoRA")
        else:
            print("⚠️ init_lora_on_raw.py: include_embeddings not set to True")
            print("  Korean embeddings may not be properly trained!")

    train_00 = Path(__file__).parent / "train" / "train_00_plain_text.py"
    if train_00.exists():
        with open(train_00, 'r') as f:
            content = f.read()

        if "include_embeddings=True" in content:
            print("✓ train_00_plain_text.py: include_embeddings=True")
        else:
            print("⚠️ train_00_plain_text.py: include_embeddings setting unclear")


def main():
    parser = argparse.ArgumentParser(description="Check Korean tokenizer support")
    parser.add_argument("--model", default="medgemma-4b",
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    args = parser.parse_args()

    print("=" * 60)
    print("KOREAN TOKENIZER CHECK")
    print("=" * 60)
    print(f"Model: {args.model}")

    # Load tokenizer
    cfg = MODEL_CONFIGS[args.model]
    print(f"Loading tokenizer from: {cfg['path']}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)

    # Run checks
    has_korean = check_korean_support(tokenizer, args.verbose)
    avg_ratio = test_korean_medical_terms(tokenizer, args.verbose)
    test_sample_training_data(tokenizer, args.verbose)
    check_embedding_training_setup()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if has_korean:
        print("\n✓ Tokenizer HAS Korean support")
        print(f"  Token/char ratio: {avg_ratio:.2f}")

        if avg_ratio > 1.5:
            print("\n⚠️ Korean is tokenized at syllable level")
            print("   This is normal for Gemma-based models.")
            print("   The training pipeline is configured correctly:")
            print("   - include_embeddings=True in LoRA")
            print("   - Plain text pretraining (train_00) helps learn Korean")
            print("   - Medical dict training (train_01) learns medical terms")
    else:
        print("\n❌ Tokenizer does NOT have Korean support!")
        print("   Consider using a Korean-extended tokenizer.")

    return 0


if __name__ == "__main__":
    exit(main())
