#!/usr/bin/env python3
"""
Refine Korean-English translation data for training.

Input: data/01_raw/03_korean_english/
Output: data/02_refined/01_english_korean/

Processing:
1. Load parallel corpus
2. Filter by length and quality
3. Create bidirectional training pairs

Output formats:
- en_to_ko.jsonl: {"en": "...", "ko": "..."} for English->Korean training
- ko_to_en.jsonl: {"ko": "...", "en": "..."} for Korean->English training
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "01_raw" / "03_korean_english"
OUTPUT_DIR = BASE_DIR / "data" / "02_refined" / "01_english_korean"


def has_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    return bool(re.search(r'[\uac00-\ud7af]', text))


def is_valid_pair(en: str, ko: str, min_length: int = 5, max_length: int = 500) -> bool:
    """Check if translation pair is valid for training."""
    # Check lengths
    if len(en) < min_length or len(ko) < min_length:
        return False
    if len(en) > max_length or len(ko) > max_length:
        return False

    # Korean text should have Korean characters
    if not has_korean(ko):
        return False

    # English text should have mostly ASCII
    ascii_ratio = sum(1 for c in en if ord(c) < 128) / len(en)
    if ascii_ratio < 0.8:
        return False

    return True


def load_opus_tatoeba(input_dir: Path) -> list[dict]:
    """Load OPUS Tatoeba data."""
    pairs = []
    data_path = input_dir / "opus_tatoeba_ko_en" / "train.jsonl"

    if not data_path.exists():
        print(f"  File not found: {data_path}")
        return pairs

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Tatoeba"):
                data = json.loads(line.strip())
                en = data.get('en', '')
                ko = data.get('ko', '')

                if is_valid_pair(en, ko):
                    pairs.append({"en": en, "ko": ko})

        print(f"  Loaded {len(pairs)} pairs from Tatoeba")
    except Exception as e:
        print(f"  Error: {e}")

    return pairs


def load_opus_books(input_dir: Path) -> list[dict]:
    """Load OPUS Books data."""
    pairs = []
    data_path = input_dir / "opus_books_ko_en" / "train.jsonl"

    if not data_path.exists():
        print(f"  File not found: {data_path}")
        return pairs

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Books"):
                data = json.loads(line.strip())
                en = data.get('en', '')
                ko = data.get('ko', '')

                if is_valid_pair(en, ko):
                    pairs.append({"en": en, "ko": ko})

        print(f"  Loaded {len(pairs)} pairs from Books")
    except Exception as e:
        print(f"  Error: {e}")

    return pairs


def save_training_data(pairs: list[dict], output_dir: Path):
    """Save training data in both directions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # English to Korean
    en_to_ko_path = output_dir / "en_to_ko.jsonl"
    with open(en_to_ko_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"  Saved {len(pairs)} pairs to {en_to_ko_path}")

    # Korean to English
    ko_to_en_path = output_dir / "ko_to_en.jsonl"
    with open(ko_to_en_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            # Swap order for ko->en training
            swapped = {"ko": pair["ko"], "en": pair["en"]}
            f.write(json.dumps(swapped, ensure_ascii=False) + '\n')
    print(f"  Saved {len(pairs)} pairs to {ko_to_en_path}")


def main():
    print("=" * 60)
    print("Refine Translation Data")
    print("=" * 60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    all_pairs = []

    # Load from all sources
    print("\nLoading sources...")
    all_pairs.extend(load_opus_tatoeba(INPUT_DIR))
    all_pairs.extend(load_opus_books(INPUT_DIR))

    if not all_pairs:
        print("\nNo translation data found!")
        print("Run: python script/prepare/data/download_translation.py")
        return

    print(f"\nTotal pairs: {len(all_pairs)}")

    # Remove duplicates
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = (pair['en'], pair['ko'])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"After dedup: {len(unique_pairs)}")

    # Save
    save_training_data(unique_pairs, OUTPUT_DIR)

    # Show samples
    print("\nSample pairs:")
    for pair in unique_pairs[:3]:
        print(f"  EN: {pair['en'][:60]}...")
        print(f"  KO: {pair['ko'][:60]}...")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
