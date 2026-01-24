#!/usr/bin/env python3
"""
Create Raw Text Dataset with Korean Book Corpus + Format QA Examples

This script converts all data sources INCLUDING the 17GB Korean book corpus
to plain text format and creates a mixed dataset with 5x random sampling.

Data Sources:
1. Korean Plain Text (Namu Wiki, Wikipedia, C4)
2. Korean Book Corpus (17GB - 029.대규모_구매도서_기반_한국어_말뭉치_데이터)
3. KorMedMCQA (MCQ format)
4. Medical Dictionary (term-definition pairs)
5. Character Dictionary (medical symbols)
6. Format QA Examples (from research/___fine_tune_qa.md) - MEDICAL 5x
7. Full Format Examples (from research/___format.md) - MEDICAL 5x

Output: data/02_refined/00_plain_text/train.jsonl
Format: {"text": "plain text content"}
"""

import json
import random
import re
import zipfile
import tempfile
from pathlib import Path
from typing import Iterator, Dict, Any, List
from datasets import load_from_disk
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
RAW_KOREAN_DIR = BASE_DIR / "data" / "01_raw" / "00_korean"
BOOK_CORPUS_DIR = RAW_KOREAN_DIR / "korean" / "029.대규모_구매도서_기반_한국어_말뭉치_데이터"
RAW_MCQ_FILE = BASE_DIR / "data" / "01_raw" / "02_kor_med_test" / "train.jsonl"
MEDICAL_DICT_FILE = BASE_DIR / "data" / "02_refined" / "01_medical_dict.json"
CHAR_DICT_FILE = BASE_DIR / "data" / "02_refined" / "02_char_dict.json"

# Format QA files (research markdown files)
FORMAT_QA_FILE = BASE_DIR / "research" / "___fine_tune_qa.md"
FORMAT_FULL_FILE = BASE_DIR / "research" / "___format.md"

# Output to 02_refined for training
OUTPUT_DIR = BASE_DIR / "data" / "02_refined" / "00_plain_text"
OUTPUT_FILE = OUTPUT_DIR / "train.jsonl"
OUTPUT_FILE_5K = OUTPUT_DIR / "train_5k.jsonl"

# Sampling multiplier (5x = each sample appears ~5 times on average)
SAMPLING_MULTIPLIER = 5

# Limit for Korean datasets (to manage memory)
MAX_KOREAN_DATASET_SAMPLES = 100000

# Book corpus - process ALL data
BOOK_CORPUS_ENABLED = True


# ============================================================================
# Format Conversion Functions
# ============================================================================

def format_korean_plain_text(text: str) -> str:
    """Clean Korean plain text from wiki markup."""
    patterns = [
        (r'\[목차\]', ''),
        (r'\[\[파일:[^\]]+\]\]', ''),
        (r'\[\[분류:[^\]]+\]\]', ''),
        (r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1'),
        (r'\[\[([^\]]+)\]\]', r'\1'),
        (r'\{\{[^}]+\}\}', ''),
        (r'\{{{[^}]+\}}}', ''),
        (r"'''([^']+)'''", r'\1'),
        (r"''([^']+)''", r'\1'),
        (r'==+\s*([^=]+)\s*==+', r'\n\1\n'),
        (r'\[\*[^\]]*\]', ''),
        (r'<ref[^>]*>.*?</ref>', ''),
        (r'<[^>]+>', ''),
        (r'\n{3,}', '\n\n'),
        (r'  +', ' '),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    return text.strip()


def format_mcq_to_text(item: Dict[str, Any]) -> str:
    """Convert MCQ to Q&A text format."""
    question = item['question'].strip()
    choices = [item.get(k, '') for k in ['A', 'B', 'C', 'D', 'E'] if item.get(k)]
    answer_idx = item['answer'] - 1 if isinstance(item['answer'], int) else ord(item['answer'].upper()) - ord('A')
    answer_text = choices[answer_idx] if 0 <= answer_idx < len(choices) else ''

    answer_main = answer_text.split('(')[0].strip()
    choices_text = ', '.join(choices)

    text = f"{question}\n"
    text += f"선택지: {choices_text}\n"
    text += f"정답: {answer_main}\n\n"

    if '(' in answer_text and ')' in answer_text:
        english = answer_text[answer_text.find('(')+1:answer_text.find(')')]
        text += f"{answer_main}({english})은 의학적으로 중요한 개념이다."

    return text.strip()


def format_medical_dict_to_text(item: Dict[str, str]) -> str:
    """Convert medical dictionary to definition format."""
    term = item['term'].strip()
    definition = item['definition'].strip()

    if '.' in definition:
        parts = definition.split('.', 1)
        english = parts[0].strip()
        korean_def = parts[1].strip() if len(parts) > 1 else definition
    else:
        english = definition
        korean_def = ''

    if korean_def:
        text = f"{term}({english})은 {korean_def}"
        if not korean_def.endswith('.'):
            text += '.'
    else:
        text = f"{term}은 {english}을 의미하는 의학 용어이다."

    return text.strip()


def format_char_dict_to_text(item: Dict[str, str]) -> str:
    """Convert character dictionary to explanation format."""
    symbol = item['term'].strip()
    definition = item['definition'].strip()
    text = f"의학 기호 {symbol}는 {definition}를 나타낸다."
    return text.strip()


def format_book_corpus_text(sentences: List[Dict[str, Any]]) -> str:
    """
    Convert book corpus sentences to plain text.

    Args:
        sentences: List of sentence dicts with 'text' field

    Returns:
        Plain text string
    """
    texts = []
    for sent in sentences:
        if 'text' in sent:
            text = sent['text'].strip()
            if text:
                texts.append(text)

    return ' '.join(texts)


def parse_format_qa_md(file_path: Path) -> List[str]:
    """
    Parse ___fine_tune_qa.md and extract QA examples as plain text.

    Each example in the file looks like:
    ### N
    ```
    question:\\n
    ...
    short_answer:\\n
    ...
    ```

    Returns list of formatted QA texts.
    """
    if not file_path.exists():
        print(f"  Warning: Format QA file not found at {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract code blocks - the file uses literal \n (backslash-n) in section names
    examples = []
    # Pattern: find code blocks that contain question:\n ... short_answer:\n
    # In the file, it appears as question:\\n (literal backslash-n)
    pattern = r'`{3}\n(question:\\n\n.*?short_answer:\\n\n.*?)`{3}'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        # Keep the text as-is
        text = match.strip()
        if text:
            examples.append(text)

    return examples


def parse_format_full_md(file_path: Path) -> List[str]:
    """
    Parse ___format.md and extract full format examples as plain text.

    Each example contains sections like:
    question:\\n
    translate:\\n
    answer_type:\\n
    remind:\\n
    reasoning:\\n
    facts:\\n
    candidates:\\n
    criteria:\\n
    analysis:\\n
    evaluation:\\n
    answer:\\n

    Returns list of formatted full QA texts.
    """
    if not file_path.exists():
        print(f"  Warning: Format full file not found at {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract code blocks that contain full format examples
    examples = []
    # Pattern: find code blocks that contain question:\n ... answer:\n at the end
    pattern = r'`{3}\n(question:\\n\n.*?answer:\\n\n[^\n]+)\n`{3}'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        # Keep the text as-is
        text = match.strip()
        if text and 'answer_type:\\n' in text:  # Full format has answer_type
            examples.append(text)

    return examples


# ============================================================================
# Data Iterators
# ============================================================================

def iter_korean_plain_text(max_samples: int = None) -> Iterator[str]:
    """Iterate Korean plain text from Namu Wiki, Wikipedia, C4."""
    sources = [
        ("namu_wiki", RAW_KOREAN_DIR / "namu_wiki"),
        ("wikipedia-korean", RAW_KOREAN_DIR / "wikipedia-korean"),
        ("c4_korean", RAW_KOREAN_DIR / "c4_korean"),
    ]

    total_count = 0

    for source_name, source_path in sources:
        if not source_path.exists():
            print(f"  Warning: {source_name} not found at {source_path}")
            continue

        try:
            ds = load_from_disk(str(source_path))
            data = ds['train'] if 'train' in ds else ds

            print(f"  Loading {source_name}: {len(data):,} samples")

            for item in data:
                if max_samples and total_count >= max_samples:
                    return

                text = item.get('text', '')
                if text:
                    cleaned = format_korean_plain_text(text)
                    if len(cleaned) >= 100:
                        korean_ratio = len(re.findall(r'[\uac00-\ud7af]', cleaned)) / len(cleaned)
                        if korean_ratio >= 0.1:
                            yield cleaned
                            total_count += 1

        except Exception as e:
            print(f"  Error loading {source_name}: {e}")


def iter_book_corpus_text(max_samples: int = None) -> Iterator[str]:
    """
    Iterate Korean book corpus from zip files.

    Extracts and parses all book corpus data from the 17GB dataset.
    """
    if not BOOK_CORPUS_ENABLED:
        return

    if not BOOK_CORPUS_DIR.exists():
        print(f"  Warning: Book corpus not found at {BOOK_CORPUS_DIR}")
        return

    # Find all zip files
    training_dir = BOOK_CORPUS_DIR / "01.데이터" / "1.Training" / "라벨링데이터"
    validation_dir = BOOK_CORPUS_DIR / "01.데이터" / "2.Validation" / "라벨링데이터"

    zip_files = []
    if training_dir.exists():
        zip_files.extend(sorted(training_dir.glob("*.zip.part0")))
    if validation_dir.exists():
        zip_files.extend(sorted(validation_dir.glob("*.zip.part0")))

    print(f"  Found {len(zip_files)} book corpus zip files")

    count = 0

    for zip_path in tqdm(zip_files, desc="Processing book corpus"):
        if max_samples and count >= max_samples:
            return

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Get all TEXT json files (not INFO)
                text_files = [name for name in zf.namelist() if 'TEXT' in name and name.endswith('.json')]

                for text_file in text_files:
                    if max_samples and count >= max_samples:
                        return

                    try:
                        # Extract and parse JSON
                        with zf.open(text_file) as f:
                            data = json.load(f)

                        # Extract paragraphs
                        paragraphs = data.get('paragraphs', [])

                        for para in paragraphs:
                            if max_samples and count >= max_samples:
                                return

                            sentences = para.get('sentences', [])
                            if sentences:
                                text = format_book_corpus_text(sentences)
                                if text and len(text) >= 50:  # Minimum length filter
                                    yield text
                                    count += 1

                    except Exception as e:
                        print(f"    Error processing {text_file}: {e}")
                        continue

        except Exception as e:
            print(f"  Error processing {zip_path.name}: {e}")
            continue


def iter_mcq_text(max_samples: int = None) -> Iterator[str]:
    """Iterate KorMedMCQA as plain text."""
    if not RAW_MCQ_FILE.exists():
        print(f"  Warning: MCQ file not found at {RAW_MCQ_FILE}")
        return

    count = 0
    with open(RAW_MCQ_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and count >= max_samples:
                return

            item = json.loads(line)
            text = format_mcq_to_text(item)
            if text:
                yield text
                count += 1


def iter_medical_dict_text(max_samples: int = None) -> Iterator[str]:
    """Iterate medical dictionary as plain text."""
    if not MEDICAL_DICT_FILE.exists():
        print(f"  Warning: Medical dict not found at {MEDICAL_DICT_FILE}")
        return

    with open(MEDICAL_DICT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    count = 0
    for item in data:
        if max_samples and count >= max_samples:
            return

        text = format_medical_dict_to_text(item)
        if text:
            yield text
            count += 1


def iter_char_dict_text(max_samples: int = None) -> Iterator[str]:
    """Iterate character dictionary as plain text."""
    if not CHAR_DICT_FILE.exists():
        print(f"  Warning: Char dict not found at {CHAR_DICT_FILE}")
        return

    with open(CHAR_DICT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    count = 0
    for item in data:
        if max_samples and count >= max_samples:
            return

        text = format_char_dict_to_text(item)
        if text:
            yield text
            count += 1


def iter_format_qa_text(max_samples: int = None) -> Iterator[str]:
    """Iterate format QA examples from ___fine_tune_qa.md."""
    examples = parse_format_qa_md(FORMAT_QA_FILE)
    print(f"  Parsed {len(examples)} format QA examples")

    count = 0
    for text in examples:
        if max_samples and count >= max_samples:
            return
        yield text
        count += 1


def iter_format_full_text(max_samples: int = None) -> Iterator[str]:
    """Iterate full format examples from ___format.md."""
    examples = parse_format_full_md(FORMAT_FULL_FILE)
    print(f"  Parsed {len(examples)} full format examples")

    count = 0
    for text in examples:
        if max_samples and count >= max_samples:
            return
        yield text
        count += 1


# ============================================================================
# Main Processing
# ============================================================================

def collect_all_texts(show_samples: bool = True) -> Dict[str, list]:
    """Collect all texts from all sources including book corpus and format QA."""
    print("\n" + "=" * 70)
    print("Collecting texts from all sources (INCLUDING FORMAT QA EXAMPLES)...")
    print("=" * 70)

    texts = {
        'korean_plain': [],
        'book_corpus': [],
        'mcq': [],
        'medical_dict': [],
        'char_dict': [],
        'format_qa': [],      # NEW: Format QA examples (MEDICAL)
        'format_full': [],    # NEW: Full format examples (MEDICAL)
    }

    # Korean plain text (limit to 100K for memory)
    print("\n[1/7] Korean Plain Text (Namu Wiki, Wikipedia, C4)")
    for text in tqdm(iter_korean_plain_text(max_samples=MAX_KOREAN_DATASET_SAMPLES), desc="Plain text"):
        texts['korean_plain'].append(text)
    print(f"  Collected: {len(texts['korean_plain']):,} samples")

    # Book corpus (process ALL - this is the 17GB data)
    print("\n[2/7] Korean Book Corpus (17GB)")
    for text in iter_book_corpus_text():
        texts['book_corpus'].append(text)
    print(f"  Collected: {len(texts['book_corpus']):,} samples")

    # MCQ
    print("\n[3/7] KorMedMCQA")
    for text in tqdm(iter_mcq_text(), desc="MCQ"):
        texts['mcq'].append(text)
    print(f"  Collected: {len(texts['mcq']):,} samples")

    # Medical dict
    print("\n[4/7] Medical Dictionary")
    for text in tqdm(iter_medical_dict_text(), desc="Med dict"):
        texts['medical_dict'].append(text)
    print(f"  Collected: {len(texts['medical_dict']):,} samples")

    # Char dict
    print("\n[5/7] Character Dictionary")
    for text in tqdm(iter_char_dict_text(), desc="Char dict"):
        texts['char_dict'].append(text)
    print(f"  Collected: {len(texts['char_dict']):,} samples")

    # Format QA (from ___fine_tune_qa.md) - MEDICAL
    print("\n[6/7] Format QA Examples (___fine_tune_qa.md)")
    for text in iter_format_qa_text():
        texts['format_qa'].append(text)
    print(f"  Collected: {len(texts['format_qa']):,} samples")

    # Full Format Examples (from ___format.md) - MEDICAL
    print("\n[7/7] Full Format Examples (___format.md)")
    for text in iter_format_full_text():
        texts['format_full'].append(text)
    print(f"  Collected: {len(texts['format_full']):,} samples")

    # Show samples
    if show_samples:
        print("\n" + "=" * 70)
        print("Sample Outputs")
        print("=" * 70)

        for source_name, source_texts in texts.items():
            if source_texts:
                print(f"\n{source_name.upper()}:")
                print("-" * 70)
                sample = source_texts[0][:200] + "..." if len(source_texts[0]) > 200 else source_texts[0]
                print(sample)

    return texts


def create_mixed_dataset(texts: Dict[str, list], multiplier: int = 5) -> None:
    """
    Create mixed dataset with selective multiplication.

    Medical data (MCQ, dicts, format QA): 5x multiplication
    General Korean data: 1x (no multiplication)
    """
    print("\n" + "=" * 70)
    print(f"Creating mixed dataset (medical data ×{multiplier})...")
    print("=" * 70)

    # Define which sources are medical (get 5x multiplier)
    medical_sources = {'mcq', 'medical_dict', 'char_dict', 'format_qa', 'format_full'}
    general_sources = {'korean_plain', 'book_corpus'}

    # Apply different multipliers
    pool = []

    print("\nApplying multipliers:")
    for source_name, source_texts in texts.items():
        if source_name in medical_sources:
            mult = multiplier
            count = len(source_texts) * mult
            print(f"  {source_name:20s}: {len(source_texts):6,} × {mult} = {count:8,} (MEDICAL)")

            # Add medical data with multiplier
            for _ in range(mult):
                for text in source_texts:
                    pool.append((source_name, text))
        else:
            mult = 1
            count = len(source_texts)
            print(f"  {source_name:20s}: {len(source_texts):6,} × {mult} = {count:8,} (GENERAL)")

            # Add general data once
            for text in source_texts:
                pool.append((source_name, text))

    # Shuffle pool
    random.shuffle(pool)
    total_mixed = len(pool)

    print(f"\nTotal mixed samples: {total_mixed:,}")

    # Count final distribution
    distribution = {k: 0 for k in texts.keys()}
    for source_name, _ in pool:
        distribution[source_name] += 1

    print("\nFinal distribution:")
    for source_name, count in distribution.items():
        pct = 100 * count / total_mixed if total_mixed > 0 else 0
        marker = "MEDICAL" if source_name in medical_sources else "GENERAL"
        print(f"  {source_name:20s}: {count:8,} ({pct:5.1f}%) [{marker}]")

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for source_name, text in tqdm(pool, desc="Writing"):
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

    # Show file size
    file_size = OUTPUT_FILE.stat().st_size
    size_gb = file_size / (1024 * 1024 * 1024)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Size: {size_gb:.2f} GB")
    print(f"Total samples: {total_mixed:,}")

    # Also create 5K sample file for quick testing
    print(f"\nCreating 5K sample file: {OUTPUT_FILE_5K}")
    sample_5k = random.sample(pool, min(5000, len(pool)))
    with open(OUTPUT_FILE_5K, 'w', encoding='utf-8') as f:
        for source_name, text in sample_5k:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    print(f"  Written {len(sample_5k):,} samples to {OUTPUT_FILE_5K}")


def main():
    print("=" * 70)
    print("CREATE RAW TEXT DATASET WITH FORMAT QA EXAMPLES")
    print("=" * 70)
    print("\nThis script processes ALL data sources including:")
    print("  - Korean plain text (Wiki, C4)")
    print("  - Korean book corpus (17GB)")
    print("  - Medical MCQ, dictionaries")
    print("  - Format QA examples (from research/___fine_tune_qa.md) [MEDICAL 5x]")
    print("  - Full format examples (from research/___format.md) [MEDICAL 5x]")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"        {OUTPUT_FILE_5K} (5K sample)")
    print("Medical data gets 5x multiplier for emphasis.")

    # Collect texts
    texts = collect_all_texts(show_samples=True)

    # Create mixed dataset
    create_mixed_dataset(texts, multiplier=SAMPLING_MULTIPLIER)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Full: {OUTPUT_FILE}")
    print(f"  5K sample: {OUTPUT_FILE_5K}")
    print("\nNext steps:")
    print("1. Inspect output: head -n 10 " + str(OUTPUT_FILE))
    print("2. Train: python script/train/train_00_plain_text.py")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
