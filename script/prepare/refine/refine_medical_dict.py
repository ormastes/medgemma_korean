#!/usr/bin/env python3
"""
Refine medical dictionary data from multiple sources.

Input: data/01_raw/01_medical_dict/
Output: data/02_refined/01_medical_dict.json

Sources:
1. bilingual_medical_dict.json - {"english": "korean"}
2. bilingual_medical_dict_categorized.json - {"category": {"english": "korean"}}
3. bilingual_medical_dict_ko_en.json - {"korean": "english"}
4. korean_medical_dict.jsonl - {"term": "korean", "definition": "..."}

Output format:
[{"term": "korean", "definition": "english"}]
"""

import json
import re
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "01_raw" / "01_medical_dict"
OUTPUT_DIR = BASE_DIR / "data" / "02_refined"
OUTPUT_FILE = OUTPUT_DIR / "01_medical_dict.json"


def has_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    return bool(re.search(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]', text))


def load_bilingual_dict(filepath: Path) -> list[dict]:
    """Load bilingual_medical_dict.json: {"english": "korean"}"""
    results = []
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return results

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for english, korean in data.items():
            if korean and english:
                results.append({
                    "term": korean,
                    "definition": english
                })
        print(f"  Loaded {len(results)} from bilingual_medical_dict.json")
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")

    return results


def load_bilingual_categorized(filepath: Path) -> list[dict]:
    """Load bilingual_medical_dict_categorized.json: {"category": {"english": "korean"}}"""
    results = []
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return results

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for category, terms in data.items():
            if isinstance(terms, dict):
                for english, korean in terms.items():
                    if korean and english:
                        results.append({
                            "term": korean,
                            "definition": english
                        })
        print(f"  Loaded {len(results)} from bilingual_medical_dict_categorized.json")
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")

    return results


def load_bilingual_ko_en(filepath: Path) -> list[dict]:
    """Load bilingual_medical_dict_ko_en.json: {"korean": "english"}"""
    results = []
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return results

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for korean, english in data.items():
            if korean and english:
                results.append({
                    "term": korean,
                    "definition": english
                })
        print(f"  Loaded {len(results)} from bilingual_medical_dict_ko_en.json")
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")

    return results


def load_korean_medical_dict(filepath: Path) -> list[dict]:
    """Load korean_medical_dict.jsonl: {"term": "...", "definition": "..."}"""
    results = []
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return results

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                term = data.get("term", "")
                definition = data.get("definition", "")

                # Get first line of definition
                first_line = definition.split('\n')[0].strip()

                # Skip if first line contains Korean (want English definition)
                if has_korean(first_line):
                    continue

                if term and first_line:
                    results.append({
                        "term": term,
                        "definition": first_line
                    })
        print(f"  Loaded {len(results)} from korean_medical_dict.jsonl")
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")

    return results


def deduplicate(entries: list[dict]) -> list[dict]:
    """Remove duplicates based on term."""
    seen = set()
    unique = []
    for entry in entries:
        term = entry["term"]
        if term not in seen:
            seen.add(term)
            unique.append(entry)
    return unique


def main():
    print("=" * 60)
    print("Refine Medical Dictionary Data")
    print("=" * 60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_entries = []

    # Load from all sources
    print("\nLoading sources...")
    all_entries.extend(load_bilingual_dict(INPUT_DIR / "bilingual_medical_dict.json"))
    all_entries.extend(load_bilingual_categorized(INPUT_DIR / "bilingual_medical_dict_categorized.json"))
    all_entries.extend(load_bilingual_ko_en(INPUT_DIR / "bilingual_medical_dict_ko_en.json"))
    all_entries.extend(load_korean_medical_dict(INPUT_DIR / "korean_medical_dict.jsonl"))

    print(f"\nTotal before dedup: {len(all_entries)}")

    # Deduplicate
    unique_entries = deduplicate(all_entries)
    print(f"Total after dedup: {len(unique_entries)}")

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(unique_entries, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")

    # Show samples
    print("\nSample entries:")
    for entry in unique_entries[:5]:
        print(f"  term: {entry['term']}")
        print(f"  definition: {entry['definition'][:60]}...")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
