#!/usr/bin/env python3
"""
Merge English-Korean Parallel Datasets

Combines all downloaded parallel datasets into a single refined dataset
with normalized "english" and "korean" fields.

Output: data/02_refined/01_english_korean/train.jsonl
"""

import os
import json
import gzip
from pathlib import Path
from typing import Iterator, Dict, Tuple

# Paths
RAW_DIR = Path("/home/ormastes/dev/pub/medgemma_korean/data/01_raw/03_korean_english")
OUTPUT_DIR = Path("/home/ormastes/dev/pub/medgemma_korean/data/02_refined/01_english_korean")


def load_korean_parallel_corpora() -> Iterator[Dict[str, str]]:
    """Load Moo/korean-parallel-corpora dataset"""
    data_dir = RAW_DIR / "korean_parallel_corpora"
    if not data_dir.exists():
        return

    print("Loading: korean_parallel_corpora")
    count = 0
    for split_file in data_dir.glob("*.jsonl"):
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Fields: ko, en
                en = item.get('en', '').strip()
                ko = item.get('ko', '').strip()
                if en and ko:
                    yield {"english": en, "korean": ko, "source": "korean_parallel_corpora"}
                    count += 1
    print(f"  Loaded: {count:,} pairs")


def load_ted_talks() -> Iterator[Dict[str, str]]:
    """Load TED Talks Korean-English dataset"""
    data_dir = RAW_DIR / "ted_talks_ko_en"
    if not data_dir.exists():
        return

    print("Loading: ted_talks_ko_en")
    count = 0
    for split_file in data_dir.glob("*.jsonl"):
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Fields: korean, english
                en = item.get('english', '').strip()
                ko = item.get('korean', '').strip()
                if en and ko:
                    yield {"english": en, "korean": ko, "source": "ted_talks"}
                    count += 1
    print(f"  Loaded: {count:,} pairs")


def load_opus_tatoeba() -> Iterator[Dict[str, str]]:
    """Load OPUS Tatoeba Korean-English dataset"""
    data_dir = RAW_DIR / "opus_tatoeba_ko_en" / "data" / "release" / "v2023-09-26" / "eng-kor"
    if not data_dir.exists():
        print(f"Tatoeba dir not found: {data_dir}")
        return

    print("Loading: opus_tatoeba_ko_en")
    count = 0

    # Process each split (train, dev, test)
    for split in ['train', 'dev', 'test']:
        src_file = data_dir / f"{split}.src"
        trg_file = data_dir / f"{split}.trg"
        src_gz = data_dir / f"{split}.src.gz"
        trg_gz = data_dir / f"{split}.trg.gz"

        # Handle both compressed and uncompressed
        if src_gz.exists() and trg_gz.exists():
            with gzip.open(src_gz, 'rt', encoding='utf-8') as src_f, \
                 gzip.open(trg_gz, 'rt', encoding='utf-8') as trg_f:
                for src_line, trg_line in zip(src_f, trg_f):
                    en = src_line.strip()
                    ko = trg_line.strip()
                    if en and ko:
                        yield {"english": en, "korean": ko, "source": "opus_tatoeba"}
                        count += 1
        elif src_file.exists() and trg_file.exists():
            with open(src_file, 'r', encoding='utf-8') as src_f, \
                 open(trg_file, 'r', encoding='utf-8') as trg_f:
                for src_line, trg_line in zip(src_f, trg_f):
                    en = src_line.strip()
                    ko = trg_line.strip()
                    if en and ko:
                        yield {"english": en, "korean": ko, "source": "opus_tatoeba"}
                        count += 1

    print(f"  Loaded: {count:,} pairs")


def deduplicate_and_filter(pairs: Iterator[Dict[str, str]], min_length: int = 5) -> Iterator[Dict[str, str]]:
    """Remove duplicates and filter too-short pairs"""
    seen = set()
    dup_count = 0
    short_count = 0

    for pair in pairs:
        en = pair['english']
        ko = pair['korean']

        # Skip too short
        if len(en) < min_length or len(ko) < min_length:
            short_count += 1
            continue

        # Skip duplicates
        key = (en.lower(), ko)
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)

        yield pair

    print(f"  Filtered: {dup_count:,} duplicates, {short_count:,} too short")


def main():
    """Merge all datasets and save to refined directory"""
    print("="*60)
    print("Merging English-Korean Parallel Datasets")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all pairs
    all_pairs = []

    # Load each dataset
    for loader in [load_korean_parallel_corpora, load_ted_talks, load_opus_tatoeba]:
        for pair in loader():
            all_pairs.append(pair)

    print(f"\nTotal before dedup: {len(all_pairs):,}")

    # Deduplicate and filter
    print("\nDeduplicating and filtering...")
    unique_pairs = list(deduplicate_and_filter(iter(all_pairs)))
    print(f"Total after dedup: {len(unique_pairs):,}")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(unique_pairs)

    # Split: 95% train, 5% validation
    split_idx = int(len(unique_pairs) * 0.95)
    train_pairs = unique_pairs[:split_idx]
    val_pairs = unique_pairs[split_idx:]

    # Save train
    train_path = OUTPUT_DIR / "train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"\nSaved train: {len(train_pairs):,} pairs to {train_path}")

    # Save validation
    val_path = OUTPUT_DIR / "validation.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for pair in val_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"Saved validation: {len(val_pairs):,} pairs to {val_path}")

    # Save stats
    source_counts = {}
    for pair in unique_pairs:
        src = pair.get('source', 'unknown')
        source_counts[src] = source_counts.get(src, 0) + 1

    stats = {
        "total_pairs": len(unique_pairs),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "source_distribution": source_counts
    }

    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved stats: {stats_path}")

    print("\n" + "="*60)
    print("Source Distribution:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt:,}")
    print("="*60)


if __name__ == "__main__":
    main()
