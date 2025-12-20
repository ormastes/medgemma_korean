#!/usr/bin/env python3
"""
Refine data by category:
- Category A: Token Answer (MCQ → single letter)
- Category B: Generation (QA, Instruction, Reasoning → full text)
"""

import json
import re
from pathlib import Path
from datasets import load_from_disk, Dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "by_source"
REFINED_DIR = DATA_DIR / "refined"

# =============================================================================
# Category A: Token Answer Format
# =============================================================================

SYSTEM_PROMPT_MCQ = "의료 객관식 문제입니다. 정답 알파벳만 답하세요 (A, B, C, D, E 중 하나)."
SYSTEM_PROMPT_YESNO = "의료 질문입니다. 예 또는 아니오로만 답하세요."
SYSTEM_PROMPT_WORD = "의료 질문입니다. 한 단어로만 답하세요."


def extract_answer_letter(text):
    """Extract A-E answer letter."""
    if not text:
        return None
    text = str(text).upper().strip()
    if text in "ABCDE":
        return text
    patterns = [
        r'정답[은는이가]?\s*[:：]?\s*\**([A-E])',
        r'([A-E])\s*[)）]?\s*입니다',
        r'^([A-E])[)）.\s]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    for char in text[:5]:
        if char in "ABCDE":
            return char
    return None


def format_mcq_token(item, source_name):
    """
    Format MCQ for token answer training.
    Output: Single letter (A-E)
    """
    question = item.get("question", "")
    if len(question) < 10:
        return None

    # Build options
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
        for letter in "ABCDE":
            opt_key = f"option_{letter.lower()}"
            if opt_key in item and item[opt_key]:
                options_text += f"{letter}) {item[opt_key]}\n"

    if not options_text:
        return None

    # Get answer letter
    answer = extract_answer_letter(item.get("answer", item.get("correct_answer", "")))
    if not answer:
        return None

    # Format: prompt + completion (letter only)
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT_MCQ}
<|im_end|>
<|im_start|>user
{question}

{options_text.strip()}
<|im_end|>
<|im_start|>assistant
"""

    return {
        "prompt": prompt,
        "completion": answer,  # Just the letter!
        "text": prompt + answer,  # For SFT trainer
        "answer": answer,
        "source": source_name,
        "category": "A",
        "type": "mcq_token"
    }


# =============================================================================
# Category B: Generation Format
# =============================================================================

SYSTEM_PROMPT_QA = "당신은 한국어 의료 전문 AI입니다. 정확하고 상세하게 답변하세요."
SYSTEM_PROMPT_INSTRUCTION = "당신은 한국어 의료 전문 AI입니다. 지시에 따라 답변하세요."
SYSTEM_PROMPT_REASONING = "당신은 의료 전문 AI입니다. 단계별로 추론하여 답변하세요."


def format_qa_generation(item, source_name):
    """
    Format QA for generation training.
    Output: Full text response
    """
    question = item.get("question", "")
    answer = item.get("answer", item.get("response", ""))

    if len(question) < 10 or len(answer) < 20:
        return None

    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT_QA}
<|im_end|>
<|im_start|>user
{question}
<|im_end|}
<|im_start|>assistant
"""

    return {
        "prompt": prompt,
        "completion": answer,
        "text": prompt + answer + "\n<|im_end|>",
        "source": source_name,
        "category": "B",
        "type": "qa_generation"
    }


def format_instruction_generation(item, source_name):
    """
    Format instruction for generation training.
    Output: Full text response
    """
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", item.get("response", ""))

    if len(instruction) < 10 or len(output) < 20:
        return None

    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n입력: {input_text}"

    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT_INSTRUCTION}
<|im_end|>
<|im_start|>user
{user_content}
<|im_end|>
<|im_start|>assistant
"""

    return {
        "prompt": prompt,
        "completion": output,
        "text": prompt + output + "\n<|im_end|>",
        "source": source_name,
        "category": "B",
        "type": "instruction_generation"
    }


def format_reasoning_generation(item, source_name):
    """
    Format reasoning for CoT generation training.
    Output: Reasoning steps + conclusion
    """
    question = item.get("question", "")
    reasoning = item.get("reasoning", item.get("chain", ""))
    answer = item.get("answer", item.get("diagnosis", ""))

    if len(question) < 10 or len(reasoning) < 30:
        return None

    # Combine reasoning + answer
    completion = reasoning
    if answer and answer not in reasoning:
        completion = f"{reasoning}\n\n결론: {answer}"

    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT_REASONING}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion + "\n<|im_end|>",
        "source": source_name,
        "category": "B",
        "type": "reasoning_generation"
    }


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_CONFIG = {
    # Category A: Token Answer
    "kormedmcqa": {"category": "A", "format_fn": format_mcq_token},
    "medqa_korean": {"category": "A", "format_fn": format_mcq_token},
    "medqa_evol_korean": {"category": "A", "format_fn": format_mcq_token},
    "kormedlawqa": {"category": "A", "format_fn": format_mcq_token},

    # Category B: Generation - QA
    "asan_amc_healthinfo": {"category": "B", "format_fn": format_qa_generation},
    "healthsearchqa_ko": {"category": "B", "format_fn": format_qa_generation},
    "ai_healthcare_qa": {"category": "B", "format_fn": format_qa_generation},
    "kormedconceptsqa": {"category": "B", "format_fn": format_qa_generation},

    # Category B: Generation - Instruction
    "komedinstruct_52k": {"category": "B", "format_fn": format_instruction_generation},
    "genmedgpt_5k_ko": {"category": "B", "format_fn": format_instruction_generation},

    # Category B: Generation - Reasoning
    "medical_o1_reasoning_ko": {"category": "B", "format_fn": format_reasoning_generation},
    "medical_reasoning_kormedmcqa": {"category": "B", "format_fn": format_reasoning_generation},
    "chainofdiagnosis_ko": {"category": "B", "format_fn": format_reasoning_generation},
}


def refine_dataset(name, config):
    """Refine a single dataset."""
    dataset_path = RAW_DIR / name
    if not dataset_path.exists():
        return []

    format_fn = config["format_fn"]
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
    print("Refining Data by Category")
    print("=" * 60)
    print("Category A: Token Answer (MCQ → letter)")
    print("Category B: Generation (QA/Instruction/Reasoning → text)")
    print("=" * 60)

    # Collect by category
    category_a = []  # Token answers
    category_b = []  # Generation

    for name, config in tqdm(DATASET_CONFIG.items(), desc="Processing"):
        refined = refine_dataset(name, config)

        if not refined:
            continue

        if config["category"] == "A":
            category_a.extend(refined)
        else:
            category_b.extend(refined)

        print(f"  {name}: {len(refined)} samples (Category {config['category']})")

    # Save Category A
    if category_a:
        cat_a_dir = REFINED_DIR / "category_a_token"
        cat_a_dir.mkdir(parents=True, exist_ok=True)

        train_size = int(len(category_a) * 0.9)
        train_ds = Dataset.from_list(category_a[:train_size])
        val_ds = Dataset.from_list(category_a[train_size:])

        train_ds.save_to_disk(str(cat_a_dir / "train"))
        val_ds.save_to_disk(str(cat_a_dir / "validation"))

        print(f"\nCategory A (Token): {len(category_a)} total")
        print(f"  Train: {train_size}, Validation: {len(category_a) - train_size}")

    # Save Category B
    if category_b:
        cat_b_dir = REFINED_DIR / "category_b_generation"
        cat_b_dir.mkdir(parents=True, exist_ok=True)

        train_size = int(len(category_b) * 0.9)
        train_ds = Dataset.from_list(category_b[:train_size])
        val_ds = Dataset.from_list(category_b[train_size:])

        train_ds.save_to_disk(str(cat_b_dir / "train"))
        val_ds.save_to_disk(str(cat_b_dir / "validation"))

        print(f"\nCategory B (Generation): {len(category_b)} total")
        print(f"  Train: {train_size}, Validation: {len(category_b) - train_size}")

    # Save summary
    summary = {
        "category_a_token": {
            "total": len(category_a),
            "description": "MCQ with single letter answer (A-E)",
            "training_method": "token_classification",
            "output_format": "Single letter",
            "evaluation": "Accuracy"
        },
        "category_b_generation": {
            "total": len(category_b),
            "description": "QA/Instruction/Reasoning with full text",
            "training_method": "sequence_generation",
            "output_format": "Full text response",
            "evaluation": "Perplexity"
        }
    }

    with open(REFINED_DIR / "category_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {REFINED_DIR / 'category_summary.json'}")


if __name__ == "__main__":
    main()
