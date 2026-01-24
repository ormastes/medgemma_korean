#!/usr/bin/env python3
"""
Refine Korean plain text data for continued pretraining.

Input: data/01_raw/00_korean/ (Namu Wiki, Wikipedia, C4)
Output: data/02_refined/00_plain_text/train.jsonl

Processing:
1. Clean wiki markup
2. Filter by Korean ratio
3. Filter adult content
4. Validate minimum length

Format: {"text": "plain text content"}
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_from_disk
except ImportError:
    print("Error: datasets package not installed")
    print("Run: pip install datasets")
    exit(1)

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "01_raw" / "00_korean"
OUTPUT_DIR = BASE_DIR / "data" / "02_refined" / "00_plain_text"
OUTPUT_FILE = OUTPUT_DIR / "train.jsonl"

# Wiki markup patterns to remove
WIKI_PATTERNS = [
    (r'\[\[파일:[^\]]+\]\]', ''),           # File links
    (r'\[\[분류:[^\]]+\]\]', ''),           # Category links
    (r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1'),  # [[link|text]] -> text
    (r'\[\[([^\]]+)\]\]', r'\1'),           # [[link]] -> link
    (r'\{\{[^}]+\}\}', ''),                 # Templates {{}}
    (r'\{{{[^}]+\}}}', ''),                 # Namu wiki special {{{}}}
    (r"'''([^']+)'''", r'\1'),              # Bold '''text''' -> text
    (r"''([^']+)''", r'\1'),                # Italic ''text'' -> text
    (r'==+\s*([^=]+)\s*==+', r'\n\1\n'),    # Headers == text == -> text
    (r'\[목차\]', ''),                       # Table of contents
    (r'\[각주\]', ''),                       # Footnotes marker
    (r'\[\*[^\]]*\]', ''),                  # Footnote references [* ...]
    (r'<ref[^>]*>.*?</ref>', ''),           # <ref> tags
    (r'<[^>]+>', ''),                       # Any remaining HTML tags
    (r'\|[^\n]+', ''),                      # Table rows
    (r'\n{3,}', '\n\n'),                    # Multiple newlines
    (r'  +', ' '),                          # Multiple spaces
]

# Adult content filter
ADULT_KEYWORDS = [
    '19금', '성인', '야동', '섹스', '포르노', '음란',
    '노출', '19이상', '후방주의', '딸치', '자위',
]


def clean_wiki_text(text: str) -> str:
    """Remove wiki markup from text."""
    for pattern, replacement in WIKI_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def is_valid_text(text: str, min_length: int = 100) -> bool:
    """Check if text is valid for training."""
    if len(text) < min_length:
        return False

    # Check Korean ratio (should have some Korean)
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
    total_chars = len(text)
    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0

    # At least 10% Korean for general content
    return korean_ratio >= 0.1


def filter_adult_content(text: str) -> bool:
    """Filter out adult content. Returns True if text is clean."""
    text_lower = text.lower()
    for keyword in ADULT_KEYWORDS:
        if keyword in text or keyword.lower() in text_lower:
            return False
    return True


def process_namu_wiki(output_file, max_samples: int = None):
    """Process Namu Wiki dataset."""
    print("\n[1/3] Processing Namu Wiki...")
    ds_path = INPUT_DIR / "namu_wiki"

    if not ds_path.exists():
        print(f"  Not found: {ds_path}")
        return 0

    try:
        ds = load_from_disk(str(ds_path))
        count = 0

        for item in tqdm(ds['train'], desc="Namu Wiki"):
            if max_samples and count >= max_samples:
                break

            text = clean_wiki_text(item['text'])

            if is_valid_text(text) and filter_adult_content(text):
                output_file.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                count += 1

        print(f"  Added {count:,} samples")
        return count
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def process_wikipedia(output_file, max_samples: int = None):
    """Process Wikipedia Korean dataset."""
    print("\n[2/3] Processing Wikipedia Korean...")
    ds_path = INPUT_DIR / "wikipedia-korean"

    if not ds_path.exists():
        print(f"  Not found: {ds_path}")
        return 0

    try:
        ds = load_from_disk(str(ds_path))
        count = 0

        for item in tqdm(ds, desc="Wikipedia Korean"):
            if max_samples and count >= max_samples:
                break

            text = clean_wiki_text(item['text'])

            if is_valid_text(text):
                output_file.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                count += 1

        print(f"  Added {count:,} samples")
        return count
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def process_c4_korean(output_file, max_samples: int = None):
    """Process C4 Korean dataset."""
    print("\n[3/3] Processing C4 Korean...")
    ds_path = INPUT_DIR / "c4_korean"

    if not ds_path.exists():
        print(f"  Not found: {ds_path}")
        return 0

    try:
        ds = load_from_disk(str(ds_path))
        count = 0
        filtered = 0

        for item in tqdm(ds, desc="C4 Korean"):
            if max_samples and count >= max_samples:
                break

            text = item['text'].strip()

            if not filter_adult_content(text):
                filtered += 1
                continue

            if is_valid_text(text):
                output_file.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                count += 1

        print(f"  Added {count:,} samples (filtered {filtered} for adult content)")
        return count
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def main():
    print("=" * 60)
    print("Refine Plain Text for Continued Pretraining")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_FILE}")

    total_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        total_count += process_namu_wiki(f)
        total_count += process_wikipedia(f)
        total_count += process_c4_korean(f)

    print("\n" + "=" * 60)
    print(f"Total: {total_count:,} samples")
    print(f"Output: {OUTPUT_FILE}")

    # Show sample
    if total_count > 0:
        print("\nSample output:")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
            text = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
            print(f"  {text}")

    print("\nDone!")


if __name__ == "__main__":
    main()
