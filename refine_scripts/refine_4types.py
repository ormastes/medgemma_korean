#!/usr/bin/env python3
"""
Refine all data into exactly 4 training types:

Type 1: TEXT
    - Full text answer, NO reasoning tokens
    - For: QA, Instruction
    - Output: "Full text response..."

Type 2: TEXT_REASONING
    - Full text answer WITH reasoning tokens
    - For: Complex QA, Reasoning with explanation
    - Output: "<R>reasoning<R/>Full text response..."

Type 3: WORD
    - Single word/letter answer, NO reasoning tokens
    - For: MCQ (letter), Simple diagnosis
    - Output: "B" or "갑상선염"

Type 4: WORD_REASONING
    - Single word answer WITH reasoning tokens
    - For: MCQ with reasoning, Diagnosis with reasoning
    - Output: "<R>reasoning<R/>B" or "<R>reasoning<R/>갑상선염"

Special Tokens:
    <R>  = Reasoning start
    <R/> = Reasoning end
"""

import json
import os
import re
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from typing import Dict, List, Optional, Tuple

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "by_source"
REFINED_DIR = BASE_DIR / "data" / "refined"


def load_local_dataset(name: str) -> Optional[Dict]:
    """Load dataset from local disk (data/raw/by_source/{name}/)"""
    local_path = RAW_DIR / name
    if not local_path.exists():
        return None

    result = {}
    # Load all splits from local directory
    for split_dir in local_path.iterdir():
        if split_dir.is_dir() and not split_dir.name.startswith('.'):
            try:
                ds = load_from_disk(str(split_dir))
                # Handle multi-config splits (e.g., train_doctor, train_nurse)
                if '_' in split_dir.name:
                    base_split = split_dir.name.rsplit('_', 1)[0]
                    if base_split in result:
                        # Concatenate with existing
                        result[base_split] = concatenate_datasets([result[base_split], ds])
                    else:
                        result[base_split] = ds
                else:
                    result[split_dir.name] = ds
            except Exception as e:
                print(f"    Warning: Could not load {split_dir}: {e}")
                continue

    return result if result else None

# Special tokens
R_START = "<R>"
R_END = "<R/>"

# System prompts for each type
SYSTEM_PROMPTS = {
    "type1_text": "당신은 한국어 의료 전문 AI입니다. 정확하고 상세하게 답변하세요.",
    "type2_text_reasoning": "당신은 한국어 의료 전문 AI입니다. 먼저 단계별로 추론한 후 상세하게 답변하세요.",
    "type3_word": "의료 질문입니다. 정답만 간단히 답하세요.",
    "type4_word_reasoning": "의료 질문입니다. 먼저 단계별로 추론한 후 정답만 답하세요."
}

# MCQ specific prompt
SYSTEM_PROMPT_MCQ = "의료 객관식 문제입니다. 정답 알파벳만 답하세요 (A, B, C, D, E 중 하나)."
SYSTEM_PROMPT_MCQ_REASONING = "의료 객관식 문제입니다. 먼저 추론한 후 정답 알파벳만 답하세요."

# Dataset to Type mapping
DATASET_TYPE_MAP = {
    # Type 1: TEXT (full text, no reasoning)
    "asan_amc_healthinfo": "type1_text",
    "healthsearchqa_ko": "type1_text",
    "ai_healthcare_qa": "type1_text",
    "kormedconceptsqa": "type1_text",
    "komedinstruct_52k": "type1_text",
    "genmedgpt_5k_ko": "type1_text",

    # Type 2: TEXT_REASONING (full text with reasoning)
    "medical_o1_reasoning_ko_descriptive": "type2_text_reasoning",
    "chainofdiagnosis_ko_descriptive": "type2_text_reasoning",

    # Type 3: WORD (letter/word, no reasoning)
    "kormedmcqa": "type3_word",
    "medqa_korean": "type3_word",
    "medqa_evol_korean": "type3_word",
    "kormedlawqa": "type3_word",

    # Type 4: WORD_REASONING (word with reasoning)
    "medical_o1_reasoning_ko_word": "type4_word_reasoning",
    "chainofdiagnosis_ko_word": "type4_word_reasoning",
}

# Dataset configurations - field names match actual downloaded data schemas
DATASET_CONFIGS = {
    # QA/Instruction datasets (Type 1) - use instruction/input/output format
    "asan_amc_healthinfo": {
        "hf_id": "ChuGyouk/Asan-AMC-Healthinfo",
        "instruction": "instruction",  # actual field name
        "input": "input",
        "output": "output"
    },
    "healthsearchqa_ko": {
        "hf_id": "ChuGyouk/HealthSearchQA-ko",
        "question": "question_ko",  # actual field: question_ko
        "answer": "answer_ko"       # actual field: answer_ko
    },
    "ai_healthcare_qa": {
        "hf_id": "ChuGyouk/AI_healthcare_QA",
        "question": "question",
        "answer": "answer"
    },
    "kormedconceptsqa": {
        "hf_id": "ChuGyouk/KorMedConceptsQA",
        "question": "question",
        "answer": "answer"
    },
    # Instruction datasets (Type 1)
    "komedinstruct_52k": {
        "hf_id": "ChuGyouk/KoMedInstruct-52k",
        "instruction": "instruction",
        "input": "input",
        "output": "output"
    },
    "genmedgpt_5k_ko": {
        "hf_id": "ChuGyouk/GenMedGPT-5k-ko",
        "instruction": "instruction",
        "input": "input",
        "output": "output"
    },
    # MCQ datasets (Type 3)
    "kormedmcqa": {
        "hf_id": "sean0042/KorMedMCQA",
        "question": "question",
        "options": ["A", "B", "C", "D", "E"],  # actual field names
        "answer": "answer"
    },
    "medqa_korean": {
        "hf_id": "ChuGyouk/MedQA",
        "question": "question_ko",  # Korean question field
        "options": ["A_ko", "B_ko", "C_ko", "D_ko"],  # Korean option fields (4 options)
        "answer": "answer_idx"      # answer index field
    },
    "medqa_evol_korean": {
        "hf_id": "ChuGyouk/MedQA-Evol-Korean",
        "instruction": "input",     # uses input/output format
        "output": "output"
    },
    "kormedlawqa": {
        "hf_id": "snuh/KorMedLawQA",
        "question": "question",
        "options": "options",
        "answer": "answer",
        "skip": True  # causes SIGSEGV
    },
    # Reasoning datasets (Type 2 or 4 based on answer length)
    "medical_o1_reasoning_ko": {
        "hf_id": "ChuGyouk/medical-o1-reasoning-SFT-Ko",
        "question": "Question",         # capital Q
        "reasoning": "Complex_Cot",      # actual field name
        "answer": "Response"             # full response as answer
    },
    "medical_reasoning_kormedmcqa": {
        "hf_id": "ChuGyouk/Medical-Reasoning-KorMedMCQA",
        "question": "question",
        "reasoning": "thinking",         # actual field name
        "answer": "answer",              # int (1-5) -> convert to letter
        "options": ["A", "B", "C", "D", "E"],  # MCQ format
        "response": "response"           # full response available
    },
    "chainofdiagnosis_ko": {
        "hf_id": "ChuGyouk/ChainofDiagnosis-Ko",
        "question": "question",
        "reasoning": "chain",
        "answer": "diagnosis"
    },
}


def is_word_answer(answer: str) -> bool:
    """Check if answer is a short word (not full text)"""
    if not answer:
        return False

    # Remove spaces
    answer_clean = answer.replace(" ", "").strip()

    # Single letter A-E
    if len(answer_clean) == 1 and answer_clean.upper() in "ABCDE":
        return True

    # Short Korean word (medical term)
    words = answer.strip().split()
    if len(words) <= 2 and len(answer_clean) <= 20:
        return True

    return False


def normalize_word(answer: str) -> str:
    """Normalize word answer: remove spaces and prefixes"""
    answer = answer.replace(" ", "").strip()

    prefixes = ["진단:", "결론:", "답:", "정답:", "최종진단:"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]

    return answer.strip()


def format_options(item: Dict, config: Dict) -> str:
    """Format MCQ options as string"""
    options_config = config.get("options")

    if isinstance(options_config, list):
        # List of field names
        options = []
        for i, field in enumerate(options_config):
            opt = item.get(field, "")
            if opt:
                options.append(f"{chr(65+i)}) {opt}")
        return "\n".join(options)
    elif isinstance(options_config, str):
        # Single field containing dict or list
        opts = item.get(options_config, {})
        if isinstance(opts, dict):
            return "\n".join([f"{k}) {v}" for k, v in opts.items()])
        elif isinstance(opts, list):
            return "\n".join([f"{chr(65+i)}) {v}" for i, v in enumerate(opts)])

    return ""


def format_type1_text(question: str, answer: str, source: str) -> Dict:
    """Type 1: TEXT - Full text, no reasoning"""
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPTS['type1_text']}\n<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"

    return {
        "prompt": prompt,
        "completion": answer,
        "text": prompt + answer + "\n<|im_end|>",
        "source": source,
        "type": "type1_text",
        "has_reasoning": False
    }


def format_type2_text_reasoning(question: str, reasoning: str, answer: str, source: str) -> Dict:
    """Type 2: TEXT_REASONING - Full text with reasoning tokens"""
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPTS['type2_text_reasoning']}\n<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"

    completion = f"{R_START}{reasoning}{R_END}{answer}"

    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion + "\n<|im_end|>",
        "reasoning": reasoning,
        "answer": answer,
        "source": source,
        "type": "type2_text_reasoning",
        "has_reasoning": True
    }


def format_type3_word(question: str, answer: str, source: str,
                       options: str = None, is_mcq: bool = False) -> Dict:
    """Type 3: WORD - Single word/letter, no reasoning"""
    if is_mcq:
        system = SYSTEM_PROMPT_MCQ
        prompt = f"<|im_start|>system\n{system}\n<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}\n\n{options}\n<|im_end|>\n"
    else:
        system = SYSTEM_PROMPTS['type3_word']
        prompt = f"<|im_start|>system\n{system}\n<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}\n<|im_end|>\n"

    prompt += f"<|im_start|>assistant\n"

    answer_clean = normalize_word(answer)

    return {
        "prompt": prompt,
        "completion": answer_clean,
        "text": prompt + answer_clean,
        "answer": answer_clean,
        "source": source,
        "type": "type3_word",
        "is_mcq": is_mcq,
        "has_reasoning": False
    }


def format_type4_word_reasoning(question: str, reasoning: str, answer: str,
                                 source: str, options: str = None,
                                 is_mcq: bool = False) -> Dict:
    """Type 4: WORD_REASONING - Single word with reasoning tokens"""
    if is_mcq:
        system = SYSTEM_PROMPT_MCQ_REASONING
        prompt = f"<|im_start|>system\n{system}\n<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}\n\n{options}\n<|im_end|>\n"
    else:
        system = SYSTEM_PROMPTS['type4_word_reasoning']
        prompt = f"<|im_start|>system\n{system}\n<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}\n<|im_end|>\n"

    prompt += f"<|im_start|>assistant\n"

    answer_clean = normalize_word(answer)
    completion = f"{R_START}{reasoning}{R_END}{answer_clean}"

    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "reasoning": reasoning,
        "answer": answer_clean,
        "source": source,
        "type": "type4_word_reasoning",
        "is_mcq": is_mcq,
        "has_reasoning": True
    }


def process_qa_dataset(name: str, config: Dict) -> List[Dict]:
    """Process QA/Instruction datasets -> Type 1"""
    samples = []

    try:
        # Try local first, then HuggingFace
        dataset = load_local_dataset(name)
        if dataset is None:
            print(f"    (Loading from HuggingFace...)")
            dataset = load_dataset(config['hf_id'], trust_remote_code=True)

        for split in dataset.keys():
            for item in dataset[split]:
                if 'instruction' in config:
                    # Instruction format
                    instruction = item.get(config['instruction'], '')
                    input_text = item.get(config.get('input', ''), '')
                    output = item.get(config['output'], '')

                    question = instruction
                    if input_text:
                        question += f"\n\n{input_text}"
                    answer = output
                else:
                    # QA format
                    question = item.get(config['question'], '')
                    answer = item.get(config['answer'], '')

                if question and answer:
                    samples.append(format_type1_text(question, answer, name))

    except Exception as e:
        print(f"  Error: {e}")

    return samples


def process_mcq_dataset(name: str, config: Dict) -> List[Dict]:
    """Process MCQ datasets -> Type 3"""
    samples = []

    try:
        # Try local first, then HuggingFace
        dataset = load_local_dataset(name)
        if dataset is None:
            print(f"    (Loading from HuggingFace...)")
            dataset = load_dataset(config['hf_id'], trust_remote_code=True)

        for split in dataset.keys():
            for item in dataset[split]:
                question = item.get(config['question'], '')
                answer = item.get(config['answer'], '')
                options = format_options(item, config)

                if question and options:
                    # Handle different answer formats
                    answer_letter = None

                    # Integer answer (0-indexed or 1-indexed)
                    if isinstance(answer, int):
                        # KorMedMCQA uses 1-5, MedQA uses 0-4
                        if answer >= 1 and answer <= 5:
                            answer_letter = chr(64 + answer)  # 1->A, 2->B, etc.
                        elif answer >= 0 and answer <= 4:
                            answer_letter = chr(65 + answer)  # 0->A, 1->B, etc.
                    elif isinstance(answer, str):
                        answer_str = answer.strip().upper()
                        if len(answer_str) == 1 and answer_str in "ABCDE":
                            answer_letter = answer_str
                        else:
                            match = re.search(r'([A-E])', answer_str)
                            if match:
                                answer_letter = match.group(1)

                    if answer_letter and answer_letter in "ABCDE":
                        samples.append(format_type3_word(
                            question, answer_letter, name,
                            options=options, is_mcq=True
                        ))

    except Exception as e:
        print(f"  Error: {e}")

    return samples


def process_reasoning_dataset(name: str, config: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Process reasoning datasets -> Type 2 (descriptive) or Type 4 (word)"""
    type2_samples = []  # Text with reasoning
    type4_samples = []  # Word with reasoning

    try:
        # Try local first, then HuggingFace
        dataset = load_local_dataset(name)
        if dataset is None:
            print(f"    (Loading from HuggingFace...)")
            dataset = load_dataset(config['hf_id'], trust_remote_code=True)

        for split in dataset.keys():
            for item in dataset[split]:
                question = item.get(config['question'], '')
                reasoning = item.get(config['reasoning'], '')
                answer = item.get(config['answer'], '')

                if not question:
                    continue

                # Convert int answer to letter
                if isinstance(answer, int):
                    if 1 <= answer <= 5:
                        answer = chr(64 + answer)  # 1->A, 2->B, etc.
                    elif 0 <= answer <= 4:
                        answer = chr(65 + answer)  # 0->A, 1->B, etc.

                if not answer:
                    continue

                # Convert answer to string for is_word_answer check
                answer_str = str(answer)

                if is_word_answer(answer_str):
                    # Short answer -> Type 4
                    type4_samples.append(format_type4_word_reasoning(
                        question, reasoning, answer_str, name
                    ))
                else:
                    # Long answer -> Type 2
                    type2_samples.append(format_type2_text_reasoning(
                        question, reasoning, answer_str, name
                    ))

    except Exception as e:
        print(f"  Error: {e}")

    return type2_samples, type4_samples


def save_by_type(samples: List[Dict], type_name: str, split_ratio: float = 0.9):
    """Save samples to type directory with train/val split"""
    if not samples:
        return 0, 0

    import random
    random.shuffle(samples)

    split_idx = int(len(samples) * split_ratio)
    train = samples[:split_idx]
    val = samples[split_idx:]

    type_dir = REFINED_DIR / type_name
    type_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_dir = type_dir / "train"
    train_dir.mkdir(exist_ok=True)
    with open(train_dir / "data.jsonl", 'w', encoding='utf-8') as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    # Validation
    val_dir = type_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    with open(val_dir / "data.jsonl", 'w', encoding='utf-8') as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    return len(train), len(val)


def main():
    print("=" * 70)
    print("Refining Data into 4 Training Types")
    print("=" * 70)
    print(f"\nType 1: TEXT           - Full text, NO reasoning")
    print(f"Type 2: TEXT_REASONING - Full text WITH {R_START}...{R_END}")
    print(f"Type 3: WORD           - Single word/letter, NO reasoning")
    print(f"Type 4: WORD_REASONING - Single word WITH {R_START}...{R_END}")

    # Collect samples by type
    type1_samples = []  # TEXT
    type2_samples = []  # TEXT_REASONING
    type3_samples = []  # WORD
    type4_samples = []  # WORD_REASONING

    # Process QA datasets -> Type 1
    print("\n" + "-" * 50)
    print("Processing QA/Instruction datasets (Type 1: TEXT)")
    print("-" * 50)

    # Note: medqa_evol_korean uses instruction format (input/output), so include in QA
    qa_datasets = ["asan_amc_healthinfo", "healthsearchqa_ko", "ai_healthcare_qa",
                   "kormedconceptsqa", "komedinstruct_52k", "genmedgpt_5k_ko",
                   "medqa_evol_korean"]  # instruction format

    for name in qa_datasets:
        if name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[name]
            if config.get("skip"):
                print(f"\n  {name}... SKIPPED")
                continue
            print(f"\n  {name}...")
            samples = process_qa_dataset(name, config)
            type1_samples.extend(samples)
            print(f"    -> {len(samples)} samples")

    # Process MCQ datasets -> Type 3
    print("\n" + "-" * 50)
    print("Processing MCQ datasets (Type 3: WORD)")
    print("-" * 50)

    # kormedlawqa excluded (causes SIGSEGV)
    mcq_datasets = ["kormedmcqa", "medqa_korean"]

    for name in mcq_datasets:
        if name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[name]
            if config.get("skip"):
                print(f"\n  {name}... SKIPPED")
                continue
            print(f"\n  {name}...")
            samples = process_mcq_dataset(name, config)
            type3_samples.extend(samples)
            print(f"    -> {len(samples)} samples")

    # Process Reasoning datasets -> Type 2 (descriptive) or Type 4 (word)
    print("\n" + "-" * 50)
    print("Processing Reasoning datasets (Type 2/4)")
    print("-" * 50)

    reasoning_datasets = ["medical_o1_reasoning_ko", "medical_reasoning_kormedmcqa", "chainofdiagnosis_ko"]

    for name in reasoning_datasets:
        if name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[name]
            if config.get("skip"):
                print(f"\n  {name}... SKIPPED")
                continue
            print(f"\n  {name}...")
            t2, t4 = process_reasoning_dataset(name, DATASET_CONFIGS[name])
            type2_samples.extend(t2)
            type4_samples.extend(t4)
            print(f"    -> Type 2 (text): {len(t2)}, Type 4 (word): {len(t4)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nType 1 (TEXT):           {len(type1_samples):,} samples")
    print(f"Type 2 (TEXT_REASONING): {len(type2_samples):,} samples")
    print(f"Type 3 (WORD):           {len(type3_samples):,} samples")
    print(f"Type 4 (WORD_REASONING): {len(type4_samples):,} samples")
    print(f"{'='*40}")
    print(f"TOTAL:                   {len(type1_samples)+len(type2_samples)+len(type3_samples)+len(type4_samples):,} samples")

    # Save by type
    print("\n" + "-" * 50)
    print("Saving refined data...")
    print("-" * 50)

    t1_train, t1_val = save_by_type(type1_samples, "type1_text")
    t2_train, t2_val = save_by_type(type2_samples, "type2_text_reasoning")
    t3_train, t3_val = save_by_type(type3_samples, "type3_word")
    t4_train, t4_val = save_by_type(type4_samples, "type4_word_reasoning")

    print(f"\nType 1: {t1_train} train, {t1_val} val -> {REFINED_DIR}/type1_text/")
    print(f"Type 2: {t2_train} train, {t2_val} val -> {REFINED_DIR}/type2_text_reasoning/")
    print(f"Type 3: {t3_train} train, {t3_val} val -> {REFINED_DIR}/type3_word/")
    print(f"Type 4: {t4_train} train, {t4_val} val -> {REFINED_DIR}/type4_word_reasoning/")

    # Save summary
    summary = {
        "special_tokens": {"R_START": R_START, "R_END": R_END},
        "types": {
            "type1_text": {
                "description": "Full text answer, NO reasoning tokens",
                "datasets": qa_datasets,
                "count": len(type1_samples),
                "train": t1_train,
                "val": t1_val,
                "evaluation": "perplexity"
            },
            "type2_text_reasoning": {
                "description": f"Full text with {R_START}reasoning{R_END}",
                "datasets": [f"{d}_descriptive" for d in reasoning_datasets],
                "count": len(type2_samples),
                "train": t2_train,
                "val": t2_val,
                "evaluation": "perplexity"
            },
            "type3_word": {
                "description": "Single word/letter, NO reasoning tokens",
                "datasets": mcq_datasets,
                "count": len(type3_samples),
                "train": t3_train,
                "val": t3_val,
                "evaluation": "exact_match"
            },
            "type4_word_reasoning": {
                "description": f"Single word with {R_START}reasoning{R_END}",
                "datasets": [f"{d}_word" for d in reasoning_datasets],
                "count": len(type4_samples),
                "train": t4_train,
                "val": t4_val,
                "evaluation": "exact_match + reasoning_score"
            }
        }
    }

    with open(REFINED_DIR / "types_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSummary saved to {REFINED_DIR}/types_summary.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
