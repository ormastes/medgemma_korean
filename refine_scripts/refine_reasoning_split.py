#!/usr/bin/env python3
"""
Refine reasoning data into 2 types:
- Type 1: Word Answer (diagnosis, disease name, etc.) - short answer without spaces
- Type 2: Descriptive Answer (full explanation) - long answer

Adds special tokens:
- <reasoning_start> : Start of reasoning process
- <reasoning_end>   : End of reasoning process

Format for Word Answer:
    System: 단계별로 추론하고 최종 진단명을 한 단어로 답하세요.
    User: [question]
    Assistant: <reasoning_start>[reasoning]<reasoning_end>[answer_word]

Format for Descriptive Answer:
    System: 단계별로 추론하고 상세하게 답변하세요.
    User: [question]
    Assistant: <reasoning_start>[reasoning]<reasoning_end>[descriptive_answer]
"""

import json
import os
import re
from pathlib import Path
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple, Optional

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "by_source"
REFINED_DIR = BASE_DIR / "data" / "refined"

# Special tokens
REASONING_START = "<reasoning_start>"
REASONING_END = "<reasoning_end>"

# System prompts
SYSTEM_PROMPT_WORD = "당신은 의료 전문 AI입니다. 단계별로 추론하고 최종 진단명을 한 단어로 답하세요. 답변에 공백을 포함하지 마세요."
SYSTEM_PROMPT_DESCRIPTIVE = "당신은 의료 전문 AI입니다. 단계별로 추론하고 상세하게 답변하세요."

# Reasoning datasets configuration
REASONING_DATASETS = {
    "medical_o1_reasoning_ko": {
        "hf_id": "ChuGyouk/medical-o1-reasoning-SFT-Ko",
        "question_field": "question",
        "reasoning_field": "reasoning",
        "answer_field": "answer"
    },
    "chainofdiagnosis_ko": {
        "hf_id": "ChuGyouk/ChainofDiagnosis-Ko",
        "question_field": "question",
        "reasoning_field": "chain",
        "answer_field": "diagnosis"
    },
    "med_reasoning_ko": {
        "hf_id": "ChuGyouk/MedReasoning-Ko",
        "question_field": "question",
        "reasoning_field": "reasoning",
        "answer_field": "conclusion"
    }
}

# Patterns to identify word-type answers (diagnoses, disease names, etc.)
WORD_ANSWER_PATTERNS = [
    r'^[가-힣]+증$',           # 갑상선기능저하증, 고혈압증
    r'^[가-힣]+염$',           # 폐렴, 간염
    r'^[가-힣]+암$',           # 폐암, 간암
    r'^[가-힣]+병$',           # 당뇨병, 파킨슨병
    r'^[가-힣]+균$',           # 포도상구균, 대장균
    r'^[가-힣]+종$',           # 선종, 육종
    r'^[가-힣]+경$',           # 백내장, 녹내장
    r'^[가-힣]+증후군$',       # 다운증후군
    r'^[가-힣]+질환$',         # 심혈관질환
    r'^[가-힣]+장애$',         # 불안장애
    r'^[가-힣]+감염$',         # 요로감염
    r'^[가-힣]+출혈$',         # 뇌출혈
    r'^[가-힣]+경색$',         # 심근경색
    r'^[가-힣]+파열$',         # 동맥류파열
    r'^[가-힣A-Za-z0-9]+$',    # Single word without spaces
]

# Max word count for word-type answer
MAX_WORD_COUNT_FOR_WORD_TYPE = 3


def is_word_answer(answer: str) -> bool:
    """Check if answer is word-type (short, no spaces, medical term)"""
    if not answer:
        return False

    # Remove spaces and check
    answer_clean = answer.replace(" ", "").strip()

    # Check word count
    words = answer.strip().split()
    if len(words) > MAX_WORD_COUNT_FOR_WORD_TYPE:
        return False

    # Check if matches word patterns
    for pattern in WORD_ANSWER_PATTERNS:
        if re.match(pattern, answer_clean):
            return True

    # Check length - very short answers are likely word-type
    if len(answer_clean) <= 15 and len(words) <= 2:
        return True

    return False


def normalize_word_answer(answer: str) -> str:
    """Normalize word answer: remove spaces, standardize format"""
    if not answer:
        return ""

    # Remove all spaces
    answer = answer.replace(" ", "")

    # Remove common prefixes/suffixes
    prefixes = ["진단:", "결론:", "답:", "최종진단:"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]

    return answer.strip()


def count_reasoning_tokens(reasoning: str) -> int:
    """Count approximate tokens in reasoning (simple word-based)"""
    if not reasoning:
        return 0
    # Simple approximation: Korean characters + English words
    korean_chars = len(re.findall(r'[가-힣]', reasoning))
    english_words = len(re.findall(r'[A-Za-z]+', reasoning))
    numbers = len(re.findall(r'\d+', reasoning))
    # Rough estimate: 2 Korean chars ≈ 1 token
    return (korean_chars // 2) + english_words + numbers


def extract_key_terms(text: str) -> set:
    """Extract key medical terms from text"""
    terms = set()

    # Korean medical terms
    patterns = [
        r'[가-힣]+증',
        r'[가-힣]+염',
        r'[가-힣]+암',
        r'[가-힣]+병',
        r'[가-힣]+균',
        r'[가-힣]+선',
        r'[가-힣]+막',
        r'[가-힣]+관',
        r'[가-힣]+종',
        r'[가-힣]+경',
        r'[가-힣]+술',
        r'[가-힣]+제',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        terms.update(matches)

    # Also extract any quoted terms
    quoted = re.findall(r'[\'"]([가-힣A-Za-z]+)[\'"]', text)
    terms.update(quoted)

    return terms


def format_reasoning_word(question: str, reasoning: str, answer: str, source: str) -> Dict:
    """Format reasoning data with word answer"""
    normalized_answer = normalize_word_answer(answer)
    reasoning_tokens = count_reasoning_tokens(reasoning)
    key_terms = extract_key_terms(reasoning)

    # Check if answer term appears in reasoning
    answer_in_reasoning = normalized_answer in reasoning.replace(" ", "") if reasoning else False

    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT_WORD}\n<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"

    completion = f"{REASONING_START}{reasoning}{REASONING_END}{normalized_answer}"

    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion + "\n<|im_end|}",
        "question": question,
        "reasoning": reasoning,
        "answer": normalized_answer,
        "answer_original": answer,
        "source": source,
        "category": "C",  # New category for reasoning with word answer
        "type": "reasoning_word",
        "reasoning_tokens": reasoning_tokens,
        "key_terms": list(key_terms),
        "answer_in_reasoning": answer_in_reasoning,
        "has_min_reasoning": reasoning_tokens >= 10
    }


def format_reasoning_descriptive(question: str, reasoning: str, answer: str, source: str) -> Dict:
    """Format reasoning data with descriptive answer"""
    reasoning_tokens = count_reasoning_tokens(reasoning)
    key_terms = extract_key_terms(reasoning)

    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT_DESCRIPTIVE}\n<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"

    completion = f"{REASONING_START}{reasoning}{REASONING_END}{answer}"

    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion + "\n<|im_end|}",
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        "source": source,
        "category": "B",  # Category B for descriptive
        "type": "reasoning_descriptive",
        "reasoning_tokens": reasoning_tokens,
        "key_terms": list(key_terms),
        "has_min_reasoning": reasoning_tokens >= 10
    }


def process_dataset(name: str, config: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Process a single reasoning dataset, split into word and descriptive"""
    print(f"\nProcessing {name}...")

    word_samples = []
    descriptive_samples = []

    try:
        # Try loading from raw directory first
        raw_path = RAW_DIR / name
        if raw_path.exists():
            # Load from local files
            for split_dir in raw_path.iterdir():
                if split_dir.is_dir():
                    for file in split_dir.glob("*.json*"):
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                if file.suffix == '.jsonl':
                                    data = [json.loads(line) for line in f]
                                else:
                                    data = json.load(f)
                                    if isinstance(data, dict):
                                        data = [data]
                        except:
                            continue

                        for item in data:
                            question = item.get(config['question_field'], '')
                            reasoning = item.get(config['reasoning_field'], '')
                            answer = item.get(config['answer_field'], '')

                            if not question or not answer:
                                continue

                            if is_word_answer(answer):
                                word_samples.append(
                                    format_reasoning_word(question, reasoning, answer, name)
                                )
                            else:
                                descriptive_samples.append(
                                    format_reasoning_descriptive(question, reasoning, answer, name)
                                )
        else:
            # Load from HuggingFace
            dataset = load_dataset(config['hf_id'], trust_remote_code=True)

            for split in dataset.keys():
                for item in dataset[split]:
                    question = item.get(config['question_field'], '')
                    reasoning = item.get(config['reasoning_field'], '')
                    answer = item.get(config['answer_field'], '')

                    if not question or not answer:
                        continue

                    if is_word_answer(answer):
                        word_samples.append(
                            format_reasoning_word(question, reasoning, answer, name)
                        )
                    else:
                        descriptive_samples.append(
                            format_reasoning_descriptive(question, reasoning, answer, name)
                        )

    except Exception as e:
        print(f"  Error processing {name}: {e}")

    print(f"  Word answers: {len(word_samples)}")
    print(f"  Descriptive answers: {len(descriptive_samples)}")

    return word_samples, descriptive_samples


def save_refined_data(samples: List[Dict], output_dir: Path, split_ratio: float = 0.9):
    """Save refined data with train/validation split"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    import random
    random.shuffle(samples)

    split_idx = int(len(samples) * split_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Save train
    train_dir = output_dir / "train"
    train_dir.mkdir(exist_ok=True)
    with open(train_dir / "data.jsonl", 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Save validation
    val_dir = output_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    with open(val_dir / "data.jsonl", 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"  Saved: {len(train_samples)} train, {len(val_samples)} validation")

    return len(train_samples), len(val_samples)


def main():
    print("=" * 60)
    print("Reasoning Data Refinement - Split into Word vs Descriptive")
    print("=" * 60)

    # Collect all samples
    all_word_samples = []
    all_descriptive_samples = []

    for name, config in REASONING_DATASETS.items():
        word_samples, descriptive_samples = process_dataset(name, config)
        all_word_samples.extend(word_samples)
        all_descriptive_samples.extend(descriptive_samples)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Word Answer samples: {len(all_word_samples)}")
    print(f"Total Descriptive samples: {len(all_descriptive_samples)}")

    # Analyze word answers
    if all_word_samples:
        avg_reasoning_tokens = sum(s['reasoning_tokens'] for s in all_word_samples) / len(all_word_samples)
        samples_with_min_reasoning = sum(1 for s in all_word_samples if s['has_min_reasoning'])
        samples_with_answer_in_reasoning = sum(1 for s in all_word_samples if s['answer_in_reasoning'])

        print(f"\nWord Answer Analysis:")
        print(f"  Avg reasoning tokens: {avg_reasoning_tokens:.1f}")
        print(f"  Samples with >=10 reasoning tokens: {samples_with_min_reasoning} ({100*samples_with_min_reasoning/len(all_word_samples):.1f}%)")
        print(f"  Samples with answer in reasoning: {samples_with_answer_in_reasoning} ({100*samples_with_answer_in_reasoning/len(all_word_samples):.1f}%)")

    # Save refined data
    print("\n" + "=" * 60)
    print("Saving refined data...")
    print("=" * 60)

    # Save word answer type (Category C)
    word_dir = REFINED_DIR / "category_c_reasoning_word"
    print(f"\nSaving word answer data to {word_dir}")
    save_refined_data(all_word_samples, word_dir)

    # Save descriptive type (kept in Category B)
    descriptive_dir = REFINED_DIR / "reasoning_descriptive"
    print(f"\nSaving descriptive data to {descriptive_dir}")
    save_refined_data(all_descriptive_samples, descriptive_dir)

    # Save summary
    summary = {
        "special_tokens": {
            "reasoning_start": REASONING_START,
            "reasoning_end": REASONING_END
        },
        "categories": {
            "C_reasoning_word": {
                "count": len(all_word_samples),
                "path": str(word_dir),
                "system_prompt": SYSTEM_PROMPT_WORD,
                "output_type": "word (no spaces)",
                "evaluation": "exact_match"
            },
            "B_reasoning_descriptive": {
                "count": len(all_descriptive_samples),
                "path": str(descriptive_dir),
                "system_prompt": SYSTEM_PROMPT_DESCRIPTIVE,
                "output_type": "full text",
                "evaluation": "perplexity"
            }
        },
        "scoring": {
            "exact_match": "1.0 if answer matches exactly",
            "reasoning_length_bonus": "+0.1 if reasoning >= 10 tokens",
            "key_term_bonus": "+0.05 per matching key term (max 0.3)"
        }
    }

    with open(REFINED_DIR / "reasoning_split_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSummary saved to {REFINED_DIR / 'reasoning_split_summary.json'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
