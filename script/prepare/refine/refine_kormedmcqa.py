#!/usr/bin/env python3
"""
Refine KorMedMCQA data for training.

Input: data/01_raw/02_kor_med_test/
Output: data/02_refined/02_kor_med_test/

Processing:
1. Transform answer format (1-5 -> A-E)
2. Validate question/answer format
3. Extract special characters for tokenizer

Format:
{
    "question": "...",
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "...",
    "E": "...",
    "answer": "A/B/C/D/E"
}
"""

import json
import re
from pathlib import Path
from collections import Counter

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "01_raw" / "02_kor_med_test"
OUTPUT_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"

ANSWER_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


def transform_sample(sample: dict) -> dict:
    """Transform a single sample."""
    # Handle both raw (1-5) and already transformed (A-E) answers
    answer = sample["answer"]
    if isinstance(answer, int):
        answer = ANSWER_MAP[answer]
    elif isinstance(answer, str) and answer.isdigit():
        answer = ANSWER_MAP[int(answer)]

    return {
        "question": sample["question"],
        "A": sample["A"],
        "B": sample["B"],
        "C": sample["C"],
        "D": sample["D"],
        "E": sample["E"],
        "answer": answer
    }


def find_special_chars(text: str) -> list[str]:
    """Find special characters not Korean or basic ASCII."""
    korean_pattern = re.compile(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]')
    simple_ascii = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789 \t\n\r.,;:!?\'"-()/[]{}@#$%^&*+=_~`|\\'
    )

    special = []
    for char in text:
        if not korean_pattern.match(char) and char not in simple_ascii:
            special.append(char)
    return special


def process_file(input_path: Path, output_path: Path) -> tuple[int, Counter]:
    """Process a JSONL file."""
    samples = []
    char_counter = Counter()

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            transformed = transform_sample(sample)
            samples.append(transformed)

            # Collect special characters
            for field in ['question', 'A', 'B', 'C', 'D', 'E']:
                chars = find_special_chars(transformed[field])
                char_counter.update(chars)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    return len(samples), char_counter


def create_char_dict(char_counter: Counter, output_path: Path):
    """Create character dictionary for special characters."""
    char_meanings = {
        '↑': 'increased, elevated, or upward trend',
        '↓': 'decreased, reduced, or downward trend',
        '→': 'leads to, results in, or right direction',
        '←': 'from, originates from, or left direction',
        '↔': 'bidirectional, reversible',
        '≥': 'greater than or equal to',
        '≤': 'less than or equal to',
        '±': 'plus or minus, approximate range',
        '×': 'multiplication, times',
        '÷': 'division, divided by',
        '√': 'square root',
        'α': 'alpha, first type or receptor',
        'β': 'beta, second type or receptor',
        'γ': 'gamma, third type',
        'δ': 'delta, change or fourth type',
        'θ': 'theta, angle',
        'μ': 'micro, one millionth',
        '㎖': 'milliliter',
        '㎎': 'milligram',
        '㎍': 'microgram',
        '℃': 'degrees Celsius',
        '°': 'degree',
        '–': 'en dash, range',
        '—': 'em dash, pause',
        '″': 'inch or seconds',
        '′': 'foot or minutes',
        '∞': 'infinity',
        '≈': 'approximately equal',
        '≠': 'not equal',
    }

    char_list = []
    for char, count in char_counter.most_common():
        meaning = char_meanings.get(char, f"special character (code: U+{ord(char):04X})")
        char_list.append({
            "term": char,
            "definition": meaning,
            "count": count
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(char_list, f, ensure_ascii=False, indent=2)

    return len(char_list)


def main():
    print("=" * 60)
    print("Refine KorMedMCQA Data")
    print("=" * 60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chars = Counter()

    for filename in ['train.jsonl', 'test.jsonl']:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename

        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue

        count, char_counter = process_file(input_path, output_path)
        all_chars.update(char_counter)
        print(f"Transformed {filename}: {count} samples")

    # Create character dictionary
    if all_chars:
        char_dict_path = BASE_DIR / "data" / "02_refined" / "02_char_dict.json"
        num_chars = create_char_dict(all_chars, char_dict_path)
        print(f"\nCreated character dictionary: {num_chars} special characters")
        print(f"  Path: {char_dict_path}")

    # Show sample
    sample_path = OUTPUT_DIR / "test.jsonl"
    if sample_path.exists():
        print("\nSample output:")
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
            for key, value in sample.items():
                val_str = str(value)[:60] + "..." if len(str(value)) > 60 else value
                print(f"  {key}: {val_str}")

    print("\nDone!")


if __name__ == "__main__":
    main()
