#!/usr/bin/env python3
"""
Refine MCQ data - rule-based filtering and formatting
"""

import json
import re
from pathlib import Path
from datasets import load_from_disk, Dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "by_source"
REFINED_DIR = DATA_DIR / "refined" / "mcq"

REFINED_DIR.mkdir(parents=True, exist_ok=True)


def extract_answer_letter(text):
    """Extract A-E answer from text."""
    if not text:
        return None

    text = str(text).upper().strip()

    # Direct letter
    if text in ['A', 'B', 'C', 'D', 'E']:
        return text

    # Patterns
    patterns = [
        r'정답[은는이가]?\s*[:：]?\s*\**([A-E])',
        r'답[은는이가]?\s*[:：]?\s*\**([A-E])',
        r'([A-E])\s*[)）]?\s*입니다',
        r'정답\s*[:：]?\s*\(?([A-E])\)?',
        r'^([A-E])[)）.\s]',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # First letter if short
    if len(text) <= 5:
        for char in text:
            if char in 'ABCDE':
                return char

    return None


def format_mcq_to_chat(item):
    """Format MCQ item to chat format."""
    question = item.get("question", "")

    # Build options string
    options_text = ""
    if "options" in item:
        opts = item["options"]
        if isinstance(opts, dict):
            for key, val in sorted(opts.items()):
                options_text += f"{key}) {val}\n"
        elif isinstance(opts, list):
            for i, opt in enumerate(opts):
                options_text += f"{chr(65+i)}) {opt}\n"
    elif "option_a" in item:
        for letter in ['A', 'B', 'C', 'D', 'E']:
            opt_key = f"option_{letter.lower()}"
            if opt_key in item and item[opt_key]:
                options_text += f"{letter}) {item[opt_key]}\n"

    # Get answer
    answer = extract_answer_letter(item.get("answer", item.get("correct_answer", "")))
    if not answer:
        return None

    # Format as chat
    text = f"""<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다. 정확하고 도움이 되는 의료 정보를 제공하세요.
<|im_end|>
<|im_start|>user
{question}

{options_text.strip()}
<|im_end|>
<|im_start|>assistant
정답은 {answer}입니다.
<|im_end|>"""

    return {
        "text": text,
        "question": question,
        "correct_answer": answer,
        "source": item.get("source", "unknown"),
        "type": "mcq"
    }


def refine_mcq_dataset(dataset_name, min_question_len=20):
    """Refine a single MCQ dataset."""
    print(f"\nRefining: {dataset_name}")

    # Find dataset splits
    dataset_path = RAW_DIR / dataset_name
    if not dataset_path.exists():
        print(f"  Not found: {dataset_path}")
        return 0

    total_kept = 0

    for split in ["train", "test", "validation"]:
        split_path = dataset_path / split
        if not split_path.exists():
            continue

        print(f"  Processing {split}...")
        ds = load_from_disk(str(split_path))

        refined = []
        for item in tqdm(ds, desc=f"  {split}", leave=False):
            result = format_mcq_to_chat(item)

            if result is None:
                continue

            # Quality checks
            if len(result["question"]) < min_question_len:
                continue

            result["source"] = dataset_name
            refined.append(result)

        if refined:
            output_path = REFINED_DIR / dataset_name / split
            output_path.mkdir(parents=True, exist_ok=True)

            refined_ds = Dataset.from_list(refined)
            refined_ds.save_to_disk(str(output_path))

            print(f"    {split}: {len(refined)}/{len(ds)} samples kept")
            total_kept += len(refined)

    return total_kept


def main():
    print("=" * 60)
    print("MCQ Data Refinement")
    print("=" * 60)

    # Load config
    config_path = DATA_DIR / "data_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"datasets": {}}

    # Find MCQ datasets
    mcq_datasets = [name for name, cfg in config["datasets"].items() if cfg["type"] == "mcq"]

    print(f"MCQ datasets: {mcq_datasets}")

    results = {}
    for name in mcq_datasets:
        kept = refine_mcq_dataset(name)
        results[name] = kept

    # Save summary
    summary = {
        "type": "mcq",
        "datasets": results,
        "total_kept": sum(results.values())
    }

    with open(REFINED_DIR / "mcq_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal MCQ samples: {summary['total_kept']}")


if __name__ == "__main__":
    main()
