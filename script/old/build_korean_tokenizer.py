#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Extended Korean Medical Tokenizer

This script:
1. Extracts all terms from medical dictionary
2. Parses all Korean words from train_02 data
3. Uses old tokenizer corpus for additional Korean words
4. Adds all unique Korean words to MedGemma tokenizer
5. Saves extended tokenizer and training data

Usage:
    python build_korean_tokenizer.py --model medgemma-4b
    python build_korean_tokenizer.py --model medgemma-27b --min-freq 5
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS


def extract_korean_words(text: str) -> list:
    """Extract Korean words from text."""
    # Match sequences of Korean characters (Hangul syllables)
    # Range: AC00-D7A3 (가-힣)
    pattern = r'[가-힣]+'
    words = re.findall(pattern, text)
    return words


def load_medical_dictionary(dict_path: Path) -> list:
    """Load medical dictionary and extract all terms."""
    print(f"\n[1/4] Loading medical dictionary: {dict_path}")

    with open(dict_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    terms = set()
    for entry in data:
        term = entry.get('term', '')
        definition = entry.get('definition', '')

        # Add the term itself
        if term:
            terms.add(term)
            # Also extract individual words from compound terms
            terms.update(extract_korean_words(term))

        # Extract Korean words from definition if it's in Korean
        if definition:
            terms.update(extract_korean_words(definition))

    print(f"  Extracted {len(terms)} unique terms from dictionary")
    return list(terms)


def load_train02_words(train_path: Path) -> list:
    """Load train_02 data and extract all Korean words."""
    print(f"\n[2/4] Loading train_02 data: {train_path}")

    words = set()
    line_count = 0

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing train_02"):
            line_count += 1
            try:
                data = json.loads(line)

                # Extract from all text fields
                for key in ['question', 'A', 'B', 'C', 'D', 'E', 'answer', 'text']:
                    if key in data and data[key]:
                        words.update(extract_korean_words(str(data[key])))

            except json.JSONDecodeError:
                continue

    print(f"  Processed {line_count} samples")
    print(f"  Extracted {len(words)} unique Korean words")
    return list(words)


def load_old_corpus_words(corpus_path: Path, max_lines: int = 1000000) -> Counter:
    """Load old tokenizer corpus and count word frequencies."""
    print(f"\n[3/4] Loading old tokenizer corpus: {corpus_path}")

    if not corpus_path.exists():
        print(f"  Corpus not found, skipping...")
        return Counter()

    word_counts = Counter()

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Processing corpus", total=max_lines)):
            if i >= max_lines:
                break
            words = extract_korean_words(line)
            word_counts.update(words)

    print(f"  Processed {min(i+1, max_lines)} lines")
    print(f"  Found {len(word_counts)} unique Korean words")
    return word_counts


def filter_existing_tokens(words: list, tokenizer) -> list:
    """Filter out words that are already single tokens in the tokenizer."""
    print("\n  Filtering existing tokens...")

    vocab = tokenizer.get_vocab()
    new_words = []
    existing_count = 0

    for word in tqdm(words, desc="Checking vocab"):
        # Check if word exists as single token (with or without special prefix)
        if word in vocab:
            existing_count += 1
        elif f"▁{word}" in vocab:
            existing_count += 1
        else:
            # Check if word is tokenized as single token
            tokens = tokenizer.tokenize(word)
            if len(tokens) == 1:
                existing_count += 1
            else:
                new_words.append(word)

    print(f"  Already in vocab: {existing_count}")
    print(f"  New words to add: {len(new_words)}")
    return new_words


def build_extended_tokenizer(
    model_name: str,
    dict_terms: list,
    train02_words: list,
    corpus_counts: Counter,
    min_freq: int = 3,
    max_new_tokens: int = 20000,
):
    """Build extended tokenizer with Korean words."""
    print("\n[4/4] Building extended tokenizer...")

    from transformers import AutoTokenizer

    # Load base tokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"  Loading base tokenizer: {cfg['path']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)

    original_vocab_size = len(tokenizer)
    print(f"  Original vocab size: {original_vocab_size}")

    # Collect all candidate words with priority
    # Priority: 1. Dictionary terms (always add)
    #           2. Train02 words (always add)
    #           3. Corpus words (by frequency)

    priority_words = set(dict_terms) | set(train02_words)
    print(f"  Priority words (dict + train02): {len(priority_words)}")

    # Filter corpus words by frequency
    corpus_words = [word for word, count in corpus_counts.most_common()
                    if count >= min_freq and word not in priority_words]
    print(f"  Corpus words (freq >= {min_freq}): {len(corpus_words)}")

    # Combine all words
    all_words = list(priority_words) + corpus_words
    print(f"  Total candidate words: {len(all_words)}")

    # Filter out existing tokens
    new_words = filter_existing_tokens(all_words, tokenizer)

    # Limit to max_new_tokens
    if len(new_words) > max_new_tokens:
        print(f"  Limiting to {max_new_tokens} tokens...")
        # Keep all priority words, then most frequent corpus words
        priority_new = [w for w in new_words if w in priority_words]
        corpus_new = [w for w in new_words if w not in priority_words]

        remaining = max_new_tokens - len(priority_new)
        new_words = priority_new + corpus_new[:max(0, remaining)]

    print(f"  Adding {len(new_words)} new tokens...")

    # Add new tokens
    num_added = tokenizer.add_tokens(new_words)

    new_vocab_size = len(tokenizer)
    print(f"  New vocab size: {new_vocab_size}")
    print(f"  Tokens added: {num_added}")

    return tokenizer, new_words, {
        'original_vocab_size': original_vocab_size,
        'new_vocab_size': new_vocab_size,
        'tokens_added': num_added,
        'dict_terms_count': len(dict_terms),
        'train02_words_count': len(train02_words),
        'corpus_words_used': len([w for w in new_words if w not in priority_words]),
    }


def test_tokenization(tokenizer, original_tokenizer):
    """Test tokenization improvement."""
    print("\n" + "=" * 60)
    print("Tokenization Test")
    print("=" * 60)

    test_sentences = [
        "당뇨병은 혈당 조절에 문제가 생기는 대사 질환입니다.",
        "환자가 발열과 기침 증상을 호소합니다.",
        "고혈압 환자는 염분 섭취를 줄여야 합니다.",
        "MRI 검사 결과 뇌에 이상 소견이 발견되었습니다.",
        "심근경색의 초기 증상으로 흉통이 나타납니다.",
    ]

    print(f"\n{'Sentence':<40} | {'Original':>10} | {'Extended':>10} | {'Ratio':>8}")
    print("-" * 75)

    total_original = 0
    total_extended = 0

    for sentence in test_sentences:
        orig_tokens = len(original_tokenizer.tokenize(sentence))
        ext_tokens = len(tokenizer.tokenize(sentence))
        ratio = orig_tokens / ext_tokens if ext_tokens > 0 else 0

        total_original += orig_tokens
        total_extended += ext_tokens

        short_sent = sentence[:37] + "..." if len(sentence) > 40 else sentence
        print(f"{short_sent:<40} | {orig_tokens:>10} | {ext_tokens:>10} | {ratio:>7.2f}x")

    avg_ratio = total_original / total_extended if total_extended > 0 else 0
    print("-" * 75)
    print(f"{'Average':<40} | {'':<10} | {'':<10} | {avg_ratio:>7.2f}x")

    return avg_ratio


def save_tokenizer_data(
    output_dir: Path,
    tokenizer,
    new_words: list,
    stats: dict,
    dict_terms: list,
    train02_words: list,
):
    """Save tokenizer and training data."""
    print(f"\nSaving to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save extended tokenizer
    tokenizer_dir = output_dir / "extended_tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"  Tokenizer saved to: {tokenizer_dir}")

    # Save new tokens list
    tokens_file = output_dir / "new_tokens.txt"
    with open(tokens_file, 'w', encoding='utf-8') as f:
        for word in new_words:
            f.write(f"{word}\n")
    print(f"  New tokens list: {tokens_file}")

    # Save dictionary terms
    dict_file = output_dir / "dictionary_terms.txt"
    with open(dict_file, 'w', encoding='utf-8') as f:
        for term in sorted(set(dict_terms)):
            f.write(f"{term}\n")
    print(f"  Dictionary terms: {dict_file}")

    # Save train02 words
    train02_file = output_dir / "train02_words.txt"
    with open(train02_file, 'w', encoding='utf-8') as f:
        for word in sorted(set(train02_words)):
            f.write(f"{word}\n")
    print(f"  Train02 words: {train02_file}")

    # Save stats
    stats_file = output_dir / "tokenizer_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats: {stats_file}")

    # Save token mapping for embedding initialization
    mapping = {
        'new_tokens': new_words,
        'new_token_ids': {
            word: tokenizer.convert_tokens_to_ids(word)
            for word in new_words
        }
    }
    mapping_file = output_dir / "token_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  Token mapping: {mapping_file}")


def copy_corpus_data(src_corpus: Path, output_dir: Path):
    """Copy or link old corpus data."""
    if not src_corpus.exists():
        print(f"  Old corpus not found: {src_corpus}")
        return

    # Create symlink instead of copying (232MB file)
    dest = output_dir / "korean_corpus.txt"
    if dest.exists():
        dest.unlink()

    try:
        dest.symlink_to(src_corpus.resolve())
        print(f"  Linked corpus: {dest} -> {src_corpus}")
    except OSError:
        # If symlink fails, note the location
        ref_file = output_dir / "corpus_location.txt"
        with open(ref_file, 'w') as f:
            f.write(str(src_corpus.resolve()))
        print(f"  Corpus reference: {ref_file}")


def save_training_data(
    data_dir: Path,
    dict_terms: list,
    train02_words: list,
    corpus_path: Path,
):
    """Save training data to data/tokenizer/ for reference."""
    print(f"\nSaving training data to {data_dir}...")

    data_dir.mkdir(parents=True, exist_ok=True)

    # Save dictionary terms
    dict_file = data_dir / "dictionary_terms.txt"
    with open(dict_file, 'w', encoding='utf-8') as f:
        for term in sorted(set(dict_terms)):
            f.write(f"{term}\n")
    print(f"  Dictionary terms: {dict_file} ({len(dict_terms)} terms)")

    # Save train02 words
    train02_file = data_dir / "train02_words.txt"
    with open(train02_file, 'w', encoding='utf-8') as f:
        for word in sorted(set(train02_words)):
            f.write(f"{word}\n")
    print(f"  Train02 words: {train02_file} ({len(train02_words)} words)")

    # Link corpus
    if corpus_path.exists():
        corpus_link = data_dir / "korean_corpus.txt"
        if corpus_link.exists():
            corpus_link.unlink()
        try:
            corpus_link.symlink_to(corpus_path.resolve())
            print(f"  Corpus: {corpus_link} -> {corpus_path}")
        except OSError:
            ref_file = data_dir / "corpus_location.txt"
            with open(ref_file, 'w') as f:
                f.write(str(corpus_path.resolve()))
            print(f"  Corpus reference: {ref_file}")

    # Save README
    readme = data_dir / "README.md"
    with open(readme, 'w', encoding='utf-8') as f:
        f.write("""# Tokenizer Training Data

This directory contains the source data used for building the extended Korean tokenizer.

## Files

- `dictionary_terms.txt` - Medical terms from 01_medical_dict.json
- `train02_words.txt` - Korean words extracted from train_02 MCQ data
- `korean_corpus.txt` - Old tokenizer corpus (symlink to data/raw/)

## Usage

The extended tokenizer is built in `model/tokenizer/{model}/`.

To rebuild:
```bash
python script/build_korean_tokenizer.py --model medgemma-4b
```
""")
    print(f"  README: {readme}")


def main():
    parser = argparse.ArgumentParser(description="Build extended Korean medical tokenizer")
    parser.add_argument("--base-model", default="medgemma-4b",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Base model for tokenizer (4b and 27b share same tokenizer)")
    parser.add_argument("--min-freq", type=int, default=3,
                       help="Minimum frequency for corpus words")
    parser.add_argument("--max-tokens", type=int, default=20000,
                       help="Maximum new tokens to add")
    parser.add_argument("--max-corpus-lines", type=int, default=1000000,
                       help="Maximum corpus lines to process")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: model/tokenizer/)")
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent

    dict_path = base_dir / "data" / "02_refined" / "01_medical_dict.json"
    train02_path = base_dir / "data" / "02_refined" / "02_kor_med_test" / "train.jsonl"
    corpus_path = base_dir / "data" / "raw" / "korean_corpus_for_tokenizer.txt"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Shared tokenizer for all models (4b and 27b use same Gemma tokenizer)
        output_dir = base_dir / "model" / "tokenizer"

    print("=" * 60)
    print("BUILD EXTENDED KOREAN MEDICAL TOKENIZER")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Min frequency: {args.min_freq}")
    print(f"Max new tokens: {args.max_tokens}")
    print(f"Output: {output_dir}")
    print("Note: Tokenizer is shared between 4b and 27b models")

    # Step 1: Load medical dictionary
    dict_terms = load_medical_dictionary(dict_path)

    # Step 2: Load train02 words
    train02_words = load_train02_words(train02_path)

    # Step 3: Load old corpus words
    corpus_counts = load_old_corpus_words(corpus_path, args.max_corpus_lines)

    # Step 4: Build extended tokenizer
    from transformers import AutoTokenizer
    cfg = MODEL_CONFIGS[args.base_model]
    original_tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)

    tokenizer, new_words, stats = build_extended_tokenizer(
        args.base_model,
        dict_terms,
        train02_words,
        corpus_counts,
        min_freq=args.min_freq,
        max_new_tokens=args.max_tokens,
    )

    # Test tokenization
    avg_ratio = test_tokenization(tokenizer, original_tokenizer)
    stats['tokenization_improvement'] = avg_ratio

    # Save tokenizer to model/tokenizer/
    save_tokenizer_data(
        output_dir,
        tokenizer,
        new_words,
        stats,
        dict_terms,
        train02_words,
    )

    # Copy/link corpus data to model/tokenizer/
    copy_corpus_data(corpus_path, output_dir)

    # Save training data to data/tokenizer/
    data_dir = base_dir / "data" / "tokenizer"
    save_training_data(data_dir, dict_terms, train02_words, corpus_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original vocab: {stats['original_vocab_size']:,}")
    print(f"Extended vocab: {stats['new_vocab_size']:,}")
    print(f"Tokens added: {stats['tokens_added']:,}")
    print(f"  - Dictionary terms: {stats['dict_terms_count']:,}")
    print(f"  - Train02 words: {stats['train02_words_count']:,}")
    print(f"  - Corpus words: {stats['corpus_words_used']:,}")
    print(f"Tokenization improvement: {avg_ratio:.2f}x")
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Run init_lora_on_raw.py with --extended-tokenizer")
    print("  2. This will resize embeddings and initialize new tokens")

    return 0


if __name__ == "__main__":
    exit(main())
