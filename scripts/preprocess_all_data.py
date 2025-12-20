#!/usr/bin/env python3
"""
Preprocess all downloaded Korean datasets for MedGemma training.

This script processes:
1. Medical-specific datasets (KorMedMCQA, KorMedLawQA, etc.) - for instruction tuning
2. General Korean corpora (Wikipedia, NamuWiki, C4, etc.) - for language modeling
3. DPO preference data (ko_ultrafeedback) - for alignment

Output:
- data/processed/stage1_5_lm/: Language modeling data for embedding training
- data/processed/stage6_7_instruction/: Instruction tuning data
- data/processed/stage_dpo/: Preference data for DPO alignment
- data/processed/evaluation/: Held-out evaluation data (including KorMedMCQA 2024)
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "korean_datasets"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# System prompts
SYSTEM_PROMPT_KO = """당신은 한국어 의료 전문 AI 어시스턴트입니다. 정확하고 도움이 되는 의료 정보를 제공하세요."""

SYSTEM_PROMPT_KO_DETAILED = """당신은 한국어 의료 전문 AI 어시스턴트입니다. 정확하고 도움이 되는 의료 정보를 제공하세요. 의료 질문에 대해 전문적이고 이해하기 쉬운 답변을 제공합니다."""


def clean_text(text):
    """Clean and normalize text"""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_arrow_dataset(path):
    """Load dataset saved in Arrow format"""
    try:
        ds = load_from_disk(str(path))
        return ds
    except Exception as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def load_parquet_dataset(path):
    """Load dataset from parquet files"""
    try:
        parquet_files = list(Path(path).rglob("*.parquet"))
        if not parquet_files:
            return None
        dfs = [pd.read_parquet(f) for f in parquet_files]
        combined = pd.concat(dfs, ignore_index=True)
        return Dataset.from_pandas(combined)
    except Exception as e:
        print(f"  Warning: Could not load parquet from {path}: {e}")
        return None


def load_jsonl_dataset(path):
    """Load dataset from JSONL files"""
    try:
        jsonl_files = list(Path(path).rglob("*.jsonl"))
        if not jsonl_files:
            return None
        examples = []
        for f in jsonl_files:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        examples.append(json.loads(line))
                    except:
                        continue
        return Dataset.from_list(examples)
    except Exception as e:
        print(f"  Warning: Could not load JSONL from {path}: {e}")
        return None


def format_kormedmcqa(example):
    """Format KorMedMCQA example for instruction tuning"""
    question = example.get("question", "")
    choices = []
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if letter in example and example[letter]:
            choices.append(f"{letter}. {example[letter]}")

    answer_idx = example.get("answer", 1)
    answer_letter = chr(ord('A') + answer_idx - 1) if isinstance(answer_idx, int) else answer_idx

    user_msg = f"{question}\n\n" + "\n".join(choices)
    assistant_msg = f"정답은 {answer_letter}입니다."

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_KO},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ],
        "text": f"질문: {question}\n선택지:\n" + "\n".join(choices) + f"\n정답: {answer_letter}",
        "year": example.get("year", 0)
    }


def format_medical_reasoning(example):
    """Format medical reasoning example with CoT"""
    question = example.get("question", "")
    thinking = example.get("thinking", "")
    response = example.get("response", "")

    if not question:
        return None

    # Build choices if available
    choices = []
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if letter in example and example[letter]:
            choices.append(f"{letter}. {example[letter]}")

    user_msg = question
    if choices:
        user_msg += "\n\n" + "\n".join(choices)
    user_msg += "\n\n단계별로 생각하며 답변해주세요."

    assistant_msg = response if response else thinking

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_KO_DETAILED},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ],
        "text": f"질문: {question}\n응답: {assistant_msg}"
    }


def format_kormedlawqa(example):
    """Format KorMedLawQA example"""
    question = example.get("question", "")
    options = example.get("options", [])
    answer = example.get("answer", "")
    reasoning = example.get("reasoning", "")

    if not question:
        return None

    user_msg = f"{question}\n\n" + "\n".join(options) if options else question
    assistant_msg = f"정답은 {answer}입니다.\n\n{reasoning}" if reasoning else f"정답은 {answer}입니다."

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_KO},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ],
        "text": f"질문: {question}\n응답: {assistant_msg}"
    }


def format_instruction(example):
    """Format general instruction example"""
    # Handle different field names
    instruction = example.get("instruction") or example.get("input") or example.get("question") or ""
    output = example.get("output") or example.get("response") or example.get("answer") or ""

    if not instruction or not output:
        return None

    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ],
        "text": f"질문: {instruction}\n응답: {output}"
    }


def process_lm_dataset(dataset, name):
    """Process dataset for language modeling"""
    texts = []
    text_field = None

    # Find text field
    if hasattr(dataset, 'column_names'):
        columns = dataset.column_names
        if isinstance(columns, dict):
            columns = list(columns.values())[0] if columns else []
        for field in ['text', 'content', 'document', 'article']:
            if field in columns:
                text_field = field
                break

    if not text_field:
        print(f"  Warning: No text field found in {name}")
        return []

    for example in tqdm(dataset, desc=f"Processing {name}"):
        text = clean_text(example.get(text_field, ""))
        if len(text) > 100:
            texts.append({"text": text})

    return texts


def main():
    print("=" * 60)
    print("Korean MedGemma Data Preprocessing")
    print("=" * 60)
    print(f"Raw data: {RAW_DIR}")
    print(f"Output: {PROCESSED_DIR}")
    print()

    # Create output directories
    for subdir in ["stage1_5_lm", "stage6_7_instruction", "stage_dpo", "evaluation"]:
        (PROCESSED_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # ========== 1. Process Medical Instruction Datasets ==========
    print("\n" + "=" * 60)
    print("1. Processing Medical Instruction Datasets")
    print("=" * 60)

    instruction_data = []
    eval_data_2024 = []  # KorMedMCQA 2024 held-out

    # 1.1 KorMedMCQA (all configs)
    kormedmcqa_dir = RAW_DIR / "kormedmcqa"
    if kormedmcqa_dir.exists():
        print("\nProcessing KorMedMCQA...")
        for config in ['doctor', 'dentist', 'nurse', 'pharm']:
            config_path = kormedmcqa_dir / config
            if config_path.exists():
                ds = load_arrow_dataset(config_path)
                if ds:
                    for split_name in ds.keys() if hasattr(ds, 'keys') else ['train']:
                        split_data = ds[split_name] if hasattr(ds, 'keys') else ds
                        for ex in tqdm(split_data, desc=f"  {config}/{split_name}"):
                            formatted = format_kormedmcqa(ex)
                            if formatted:
                                year = formatted.get("year", 0)
                                if year == 2024:
                                    eval_data_2024.append(formatted)
                                else:
                                    instruction_data.append(formatted)
        print(f"  KorMedMCQA training: {len(instruction_data)} examples")
        print(f"  KorMedMCQA 2024 eval (held-out): {len(eval_data_2024)} examples")

    # 1.2 Medical Reasoning KorMedMCQA
    med_reasoning_dir = RAW_DIR / "medical_reasoning_kormedmcqa"
    if med_reasoning_dir.exists():
        print("\nProcessing Medical Reasoning KorMedMCQA...")
        ds = load_parquet_dataset(med_reasoning_dir / "data")
        if ds:
            count = 0
            for ex in tqdm(ds, desc="  Processing"):
                formatted = format_medical_reasoning(ex)
                if formatted:
                    instruction_data.append(formatted)
                    count += 1
            print(f"  Added {count} medical reasoning examples")

    # 1.3 Medical O1 Reasoning Korean
    med_o1_dir = RAW_DIR / "medical_o1_reasoning_ko"
    if med_o1_dir.exists():
        print("\nProcessing Medical O1 Reasoning Korean...")
        ds = load_arrow_dataset(med_o1_dir)
        if ds:
            count = 0
            split_data = ds['train'] if hasattr(ds, 'keys') and 'train' in ds else ds
            for ex in tqdm(split_data, desc="  Processing"):
                formatted = format_instruction(ex)
                if formatted:
                    instruction_data.append(formatted)
                    count += 1
            print(f"  Added {count} O1 reasoning examples")

    # 1.4 KorMedLawQA
    kormedlawqa_dir = RAW_DIR / "kormedlawqa"
    if kormedlawqa_dir.exists():
        print("\nProcessing KorMedLawQA...")
        ds = load_jsonl_dataset(kormedlawqa_dir / "data")
        if ds:
            count = 0
            for ex in tqdm(ds, desc="  Processing"):
                formatted = format_kormedlawqa(ex)
                if formatted:
                    instruction_data.append(formatted)
                    count += 1
            print(f"  Added {count} medical law examples")

    # 1.5 KMMLU Medical subjects
    kmmlu_dir = RAW_DIR / "kmmlu_medical"
    if kmmlu_dir.exists():
        print("\nProcessing KMMLU Medical...")
        count = 0
        for subject_dir in kmmlu_dir.iterdir():
            if subject_dir.is_dir():
                ds = load_arrow_dataset(subject_dir)
                if ds:
                    for split_name in ds.keys() if hasattr(ds, 'keys') else ['train']:
                        split_data = ds[split_name] if hasattr(ds, 'keys') else ds
                        for ex in split_data:
                            formatted = format_instruction(ex)
                            if formatted:
                                instruction_data.append(formatted)
                                count += 1
        print(f"  Added {count} KMMLU medical examples")

    # 1.6 General Korean Instructions (KoAlpaca, KULLM, Open-Korean-Instructions)
    print("\nProcessing General Korean Instructions...")
    for dataset_name in ['ko_alpaca_data', 'koalpaca-v1.1a', 'kullm_v2', 'open-korean-instructions']:
        dataset_dir = RAW_DIR / dataset_name
        if dataset_dir.exists():
            ds = load_arrow_dataset(dataset_dir)
            if ds is None:
                ds = load_parquet_dataset(dataset_dir)
            if ds is None:
                ds = load_jsonl_dataset(dataset_dir)

            if ds:
                count = 0
                data_iter = ds['train'] if hasattr(ds, 'keys') and 'train' in ds else ds
                for ex in tqdm(data_iter, desc=f"  {dataset_name}"):
                    formatted = format_instruction(ex)
                    if formatted:
                        instruction_data.append(formatted)
                        count += 1
                print(f"  {dataset_name}: {count} examples")

    print(f"\nTotal instruction examples: {len(instruction_data)}")

    # ========== 2. Process Language Modeling Datasets ==========
    print("\n" + "=" * 60)
    print("2. Processing Language Modeling Datasets")
    print("=" * 60)

    lm_texts = []

    # 2.1 Wikipedia Korean
    wiki_dir = RAW_DIR / "wikipedia-korean"
    if wiki_dir.exists():
        print("\nProcessing Wikipedia Korean...")
        ds = load_arrow_dataset(wiki_dir)
        if ds:
            texts = process_lm_dataset(ds, "wikipedia")
            lm_texts.extend(texts)
            print(f"  Wikipedia: {len(texts)} documents")

    # 2.2 C4 Korean
    c4_dir = RAW_DIR / "c4_korean"
    if c4_dir.exists():
        print("\nProcessing C4 Korean...")
        ds = load_arrow_dataset(c4_dir)
        if ds:
            texts = process_lm_dataset(ds, "c4_korean")
            lm_texts.extend(texts)
            print(f"  C4 Korean: {len(texts)} documents")

    # 2.3 Korean Textbooks
    textbooks_dir = RAW_DIR / "korean_textbooks"
    if textbooks_dir.exists():
        print("\nProcessing Korean Textbooks...")
        ds = load_parquet_dataset(textbooks_dir)
        if ds:
            texts = process_lm_dataset(ds, "textbooks")
            # Limit to avoid overwhelming the dataset
            if len(texts) > 1000000:
                random.shuffle(texts)
                texts = texts[:1000000]
            lm_texts.extend(texts)
            print(f"  Textbooks: {len(texts)} documents")

    # 2.4 NamuWiki (sample - too large to use all)
    namu_dir = RAW_DIR / "namu_wiki"
    if namu_dir.exists():
        print("\nProcessing NamuWiki (sampling 500K)...")
        ds = load_arrow_dataset(namu_dir)
        if ds:
            texts = process_lm_dataset(ds, "namuwiki")
            if len(texts) > 500000:
                random.shuffle(texts)
                texts = texts[:500000]
            lm_texts.extend(texts)
            print(f"  NamuWiki: {len(texts)} documents")

    print(f"\nTotal LM documents: {len(lm_texts)}")

    # ========== 3. Process DPO Preference Data ==========
    print("\n" + "=" * 60)
    print("3. Processing DPO Preference Data")
    print("=" * 60)

    dpo_data = []

    # Korean Ultrafeedback
    ultrafeedback_dir = RAW_DIR / "ko_ultrafeedback"
    if ultrafeedback_dir.exists():
        print("\nProcessing Korean Ultrafeedback...")
        ds = load_parquet_dataset(ultrafeedback_dir / "data")
        if ds:
            for ex in tqdm(ds, desc="  Processing"):
                prompt = ex.get("prompt", "")
                chosen = ex.get("chosen", "")
                rejected = ex.get("rejected", "")
                if prompt and chosen and rejected:
                    dpo_data.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected
                    })
            print(f"  Korean Ultrafeedback: {len(dpo_data)} preference pairs")

    # ========== 4. Save Processed Data ==========
    print("\n" + "=" * 60)
    print("4. Saving Processed Data")
    print("=" * 60)

    # 4.1 Save instruction data
    if instruction_data:
        random.shuffle(instruction_data)
        split_idx = int(len(instruction_data) * 0.95)

        train_data = instruction_data[:split_idx]
        val_data = instruction_data[split_idx:]

        instruction_ds = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data)
        })

        save_path = PROCESSED_DIR / "stage6_7_instruction"
        instruction_ds.save_to_disk(str(save_path))
        print(f"\nInstruction data saved to {save_path}")
        print(f"  Train: {len(train_data)}, Validation: {len(val_data)}")

    # 4.2 Save LM data
    if lm_texts:
        random.shuffle(lm_texts)
        split_idx = int(len(lm_texts) * 0.98)

        train_texts = lm_texts[:split_idx]
        val_texts = lm_texts[split_idx:]

        lm_ds = DatasetDict({
            "train": Dataset.from_list(train_texts),
            "validation": Dataset.from_list(val_texts)
        })

        save_path = PROCESSED_DIR / "stage1_5_lm"
        lm_ds.save_to_disk(str(save_path))
        print(f"\nLM data saved to {save_path}")
        print(f"  Train: {len(train_texts)}, Validation: {len(val_texts)}")

    # 4.3 Save DPO data
    if dpo_data:
        random.shuffle(dpo_data)
        split_idx = int(len(dpo_data) * 0.95)

        dpo_ds = DatasetDict({
            "train": Dataset.from_list(dpo_data[:split_idx]),
            "validation": Dataset.from_list(dpo_data[split_idx:])
        })

        save_path = PROCESSED_DIR / "stage_dpo"
        dpo_ds.save_to_disk(str(save_path))
        print(f"\nDPO data saved to {save_path}")
        print(f"  Train: {split_idx}, Validation: {len(dpo_data) - split_idx}")

    # 4.4 Save evaluation data (KorMedMCQA 2024)
    if eval_data_2024:
        eval_ds = Dataset.from_list(eval_data_2024)
        save_path = PROCESSED_DIR / "evaluation" / "kormedmcqa_2024"
        eval_ds.save_to_disk(str(save_path))
        print(f"\nEvaluation data (KorMedMCQA 2024) saved to {save_path}")
        print(f"  Examples: {len(eval_data_2024)}")

    # 4.5 Save processing summary
    summary = {
        "processing_date": str(datetime.now()),
        "datasets": {
            "stage1_5_lm": {
                "path": str(PROCESSED_DIR / "stage1_5_lm"),
                "train_size": len(train_texts) if lm_texts else 0,
                "validation_size": len(val_texts) if lm_texts else 0,
                "use": "Stage 1-5 (Embedding training / Language Modeling)"
            },
            "stage6_7_instruction": {
                "path": str(PROCESSED_DIR / "stage6_7_instruction"),
                "train_size": len(train_data) if instruction_data else 0,
                "validation_size": len(val_data) if instruction_data else 0,
                "use": "Stage 6-7 (Instruction tuning)"
            },
            "stage_dpo": {
                "path": str(PROCESSED_DIR / "stage_dpo"),
                "train_size": split_idx if dpo_data else 0,
                "validation_size": len(dpo_data) - split_idx if dpo_data else 0,
                "use": "DPO Alignment Training"
            },
            "evaluation": {
                "path": str(PROCESSED_DIR / "evaluation"),
                "kormedmcqa_2024_size": len(eval_data_2024),
                "use": "Held-out evaluation (DO NOT use in training)"
            }
        }
    }

    with open(PROCESSED_DIR / "processing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
