#!/usr/bin/env python3
"""
Refine all data types - master script
"""

import json
import re
from pathlib import Path
from datasets import load_from_disk, Dataset, concatenate_datasets
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "by_source"
REFINED_DIR = DATA_DIR / "refined"

REFINED_DIR.mkdir(parents=True, exist_ok=True)


def extract_answer_letter(text):
    """Extract A-E answer from text."""
    if not text:
        return None
    text = str(text).upper().strip()
    if text in ['A', 'B', 'C', 'D', 'E']:
        return text
    patterns = [
        r'정답[은는이가]?\s*[:：]?\s*\**([A-E])',
        r'답[은는이가]?\s*[:：]?\s*\**([A-E])',
        r'([A-E])\s*[)）]?\s*입니다',
        r'^([A-E])[)）.\s]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    if len(text) <= 5:
        for char in text:
            if char in 'ABCDE':
                return char
    return None


def format_mcq(item, source_name):
    """Format MCQ to chat."""
    question = item.get("question", "")
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

    answer = extract_answer_letter(item.get("answer", item.get("correct_answer", "")))
    if not answer or len(question) < 20:
        return None

    text = f"""<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
{question}

{options_text.strip()}
<|im_end|>
<|im_start|>assistant
정답은 {answer}입니다.
<|im_end|>"""

    return {"text": text, "question": question, "correct_answer": answer, "source": source_name, "type": "mcq"}


def format_qa(item, source_name):
    """Format QA to chat."""
    question = item.get("question", "")
    answer = item.get("answer", item.get("response", ""))

    if len(question) < 10 or len(answer) < 20:
        return None

    text = f"""<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>"""

    return {"text": text, "question": question, "answer": answer, "source": source_name, "type": "qa"}


def format_instruction(item, source_name):
    """Format instruction to chat."""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", item.get("response", ""))

    if len(instruction) < 10 or len(output) < 20:
        return None

    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"

    text = f"""<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
{user_content}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""

    return {"text": text, "instruction": instruction, "output": output, "source": source_name, "type": "instruction"}


def format_reasoning(item, source_name):
    """Format reasoning to chat with CoT."""
    question = item.get("question", "")
    reasoning = item.get("reasoning", item.get("chain", ""))
    answer = item.get("answer", item.get("diagnosis", ""))

    if len(question) < 10 or len(reasoning) < 30:
        return None

    response = reasoning
    if answer:
        response = f"{reasoning}\n\n따라서, {answer}"

    text = f"""<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다. 단계별로 추론하여 답변하세요.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>"""

    return {"text": text, "question": question, "reasoning": reasoning, "source": source_name, "type": "reasoning"}


FORMAT_FUNCTIONS = {
    "mcq": format_mcq,
    "qa": format_qa,
    "instruction": format_instruction,
    "reasoning": format_reasoning,
}


def refine_dataset(name, data_type):
    """Refine a single dataset."""
    dataset_path = RAW_DIR / name
    if not dataset_path.exists():
        return []

    format_fn = FORMAT_FUNCTIONS.get(data_type)
    if not format_fn:
        return []

    all_refined = []

    for split in ["train", "test", "validation"]:
        split_path = dataset_path / split
        if not split_path.exists():
            continue

        ds = load_from_disk(str(split_path))

        for item in ds:
            result = format_fn(item, name)
            if result:
                all_refined.append(result)

    return all_refined


def main():
    print("=" * 60)
    print("Refining All Datasets")
    print("=" * 60)

    # Load config
    config_path = DATA_DIR / "data_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        print("No data_config.json found!")
        return

    # Refine by type
    results = {"mcq": [], "qa": [], "instruction": [], "reasoning": []}

    for name, ds_config in tqdm(config["datasets"].items(), desc="Datasets"):
        data_type = ds_config["type"]
        refined = refine_dataset(name, data_type)

        if refined:
            results[data_type].extend(refined)
            print(f"  {name}: {len(refined)} samples")

    # Save by type
    for data_type, samples in results.items():
        if not samples:
            continue

        type_dir = REFINED_DIR / data_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # Split train/val
        train_size = int(len(samples) * 0.9)
        train_samples = samples[:train_size]
        val_samples = samples[train_size:]

        if train_samples:
            train_ds = Dataset.from_list(train_samples)
            train_ds.save_to_disk(str(type_dir / "train"))
            print(f"{data_type} train: {len(train_samples)}")

        if val_samples:
            val_ds = Dataset.from_list(val_samples)
            val_ds.save_to_disk(str(type_dir / "validation"))
            print(f"{data_type} validation: {len(val_samples)}")

    # Create combined dataset
    print("\nCreating combined dataset...")
    all_samples = []
    for samples in results.values():
        all_samples.extend(samples)

    if all_samples:
        combined_dir = REFINED_DIR / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        train_size = int(len(all_samples) * 0.9)
        train_ds = Dataset.from_list(all_samples[:train_size])
        val_ds = Dataset.from_list(all_samples[train_size:])

        train_ds.save_to_disk(str(combined_dir / "train"))
        val_ds.save_to_disk(str(combined_dir / "validation"))

        print(f"Combined: {train_size} train, {len(all_samples) - train_size} validation")

    # Save summary
    summary = {
        "mcq": len(results["mcq"]),
        "qa": len(results["qa"]),
        "instruction": len(results["instruction"]),
        "reasoning": len(results["reasoning"]),
        "total": len(all_samples)
    }

    with open(REFINED_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal refined: {summary['total']}")


if __name__ == "__main__":
    main()
