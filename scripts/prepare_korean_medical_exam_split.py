#!/usr/bin/env python3
"""
Prepare Korean Medical Exam Data with 75/25 Train/Verification Split

This script:
1. Collects all Korean medical exam/MCQ data
2. Splits into 75% training and 25% verification
3. Formats data for instruction tuning
4. Saves both splits for training and evaluation
"""

import os
import sys
import json
import random
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm

# Configuration
SEED = 42
TRAIN_RATIO = 0.75  # 75% for training
VERIFICATION_RATIO = 0.25  # 25% for verification

random.seed(SEED)

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "korean_datasets"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "korean_medical_exam_75_25_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# System prompt for medical QA
SYSTEM_PROMPT = "당신은 한국어 의료 전문 AI 어시스턴트입니다. 의료 문제에 대해 정확하게 답변하고 설명을 제공하세요."

print("=" * 60)
print("Preparing Korean Medical Exam Data (75/25 Split)")
print("=" * 60)
print(f"Train ratio: {TRAIN_RATIO * 100}%")
print(f"Verification ratio: {VERIFICATION_RATIO * 100}%")
print(f"Random seed: {SEED}")


def format_mcqa_with_explanation(question, choices, correct_answer, explanation=None):
    """Format MCQ with ChatML format"""
    choices_text = "\n".join(choices)

    user_content = f"""다음 의료 관련 질문에 답하세요.

질문: {question}

선택지:
{choices_text}

정답과 그 이유를 설명해주세요."""

    if explanation:
        assistant_content = f"정답은 {correct_answer}입니다.\n\n{explanation}"
    else:
        assistant_content = f"정답은 {correct_answer}입니다.\n\n이 답이 정답인 이유는 해당 의료 지식과 임상적 판단에 기반합니다."

    text = f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{user_content}
<|im_end|>
<|im_start|>assistant
{assistant_content}
<|im_end|>"""

    return {
        "text": text,
        "question": question,
        "correct_answer": correct_answer,
        "source": "korean_medical_exam"
    }


def format_qa(question, answer, system_prompt=None):
    """Format general QA with ChatML format"""
    sp = system_prompt or SYSTEM_PROMPT

    text = f"""<|im_start|>system
{sp}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>"""

    return {
        "text": text,
        "question": question,
        "correct_answer": answer[:100],  # First 100 chars as reference
        "source": "korean_medical_qa"
    }


# =============================================================================
# Collect Medical Exam Data
# =============================================================================
print("\n" + "=" * 60)
print("Collecting Medical Exam Data")
print("=" * 60)

all_exam_data = []

# 1. KorMedMCQA (main exam dataset)
print("\n[1] Loading KorMedMCQA...")
try:
    # Load from local if available
    kormedmcqa_path = BASE_DIR / "data" / "raw" / "kormedmcqa"
    if kormedmcqa_path.exists():
        ds = load_from_disk(str(kormedmcqa_path))
    else:
        ds = load_dataset("sean0042/KorMedMCQA")

    count = 0
    for split in ds:
        for example in tqdm(ds[split], desc=f"  KorMedMCQA {split}"):
            question = example.get("question", "")
            choices = []
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if letter in example and example[letter]:
                    choices.append(f"{letter}. {example[letter]}")

            answer_idx = example.get("answer", 1)
            if isinstance(answer_idx, int):
                correct_letter = chr(ord('A') + answer_idx - 1)
            else:
                correct_letter = str(answer_idx)

            if question and choices:
                formatted = format_mcqa_with_explanation(question, choices, correct_letter)
                formatted["source"] = "KorMedMCQA"
                all_exam_data.append(formatted)
                count += 1

    print(f"  Added {count} KorMedMCQA examples")
except Exception as e:
    print(f"  Error: {e}")

# 2. Medical Reasoning KorMedMCQA (with explanations)
print("\n[2] Loading Medical Reasoning KorMedMCQA...")
try:
    med_reasoning_path = RAW_DIR / "medical_reasoning_kormedmcqa"
    if med_reasoning_path.exists():
        ds = load_from_disk(str(med_reasoning_path))
    else:
        ds = load_dataset("ChuGyouk/medical-reasoning-train-kormedmcqa")

    count = 0
    data = ds["train"] if "train" in ds else ds
    for example in tqdm(data, desc="  Medical Reasoning"):
        question = example.get("question", "")
        choices = []
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if letter in example and example[letter]:
                choices.append(f"{letter}. {example[letter]}")

        answer_idx = example.get("answer", 1)
        if isinstance(answer_idx, int):
            correct_letter = chr(ord('A') + answer_idx - 1)
        else:
            correct_letter = str(answer_idx)

        thinking = example.get("thinking", "")
        response = example.get("response", "")
        explanation = response if response else thinking

        if question and choices:
            formatted = format_mcqa_with_explanation(question, choices, correct_letter, explanation)
            formatted["source"] = "Medical_Reasoning_KorMedMCQA"
            all_exam_data.append(formatted)
            count += 1

    print(f"  Added {count} Medical Reasoning examples")
except Exception as e:
    print(f"  Error: {e}")

# 3. KMMLU Medical subjects
print("\n[3] Loading KMMLU Medical...")
try:
    kmmlu_dir = RAW_DIR / "kmmlu_medical"
    if kmmlu_dir.exists():
        count = 0
        for subject_dir in kmmlu_dir.iterdir():
            if subject_dir.is_dir():
                try:
                    ds = load_from_disk(str(subject_dir))
                    for split in ds:
                        for example in ds[split]:
                            question = example.get("question", "")
                            choices = []
                            for i in range(1, 5):
                                opt = example.get(f"option_{i}", "")
                                if opt:
                                    letter = chr(ord('A') + i - 1)
                                    choices.append(f"{letter}. {opt}")

                            answer = example.get("answer", "A")

                            if question and choices:
                                formatted = format_mcqa_with_explanation(question, choices, answer)
                                formatted["source"] = f"KMMLU_{subject_dir.name}"
                                all_exam_data.append(formatted)
                                count += 1
                except Exception as e:
                    print(f"    Could not load {subject_dir.name}: {e}")
        print(f"  Added {count} KMMLU examples")
except Exception as e:
    print(f"  Error: {e}")

# 4. MedQA (translated Korean)
print("\n[4] Loading MedQA Korean...")
try:
    medqa_path = RAW_DIR / "chugyouk_medqa"
    if medqa_path.exists():
        ds = load_from_disk(str(medqa_path))
    else:
        try:
            ds = load_dataset("ChuGyouk/MedQA", trust_remote_code=True)
        except:
            ds = None

    if ds:
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  MedQA"):
            question = example.get("question", example.get("instruction", ""))
            answer = example.get("answer", example.get("output", ""))

            # Check if it has options
            options = example.get("options", {})
            if options and isinstance(options, dict):
                choices = [f"{k}. {v}" for k, v in options.items()]
                correct = example.get("answer_idx", "A")
                if question and choices:
                    formatted = format_mcqa_with_explanation(question, choices, correct)
                    formatted["source"] = "MedQA_Korean"
                    all_exam_data.append(formatted)
                    count += 1
            elif question and answer:
                formatted = format_qa(question, answer)
                formatted["source"] = "MedQA_Korean"
                all_exam_data.append(formatted)
                count += 1

        print(f"  Added {count} MedQA examples")
except Exception as e:
    print(f"  Error: {e}")

# 5. KorMedConceptsQA
print("\n[5] Loading KorMedConceptsQA...")
try:
    concepts_path = RAW_DIR / "chugyouk_kormedconceptsqa"
    if concepts_path.exists():
        ds = load_from_disk(str(concepts_path))
    else:
        try:
            ds = load_dataset("ChuGyouk/KorMedConceptsQA", trust_remote_code=True)
        except:
            ds = None

    if ds:
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  KorMedConceptsQA"):
            question = example.get("question", example.get("instruction", ""))
            answer = example.get("answer", example.get("output", ""))

            if question and answer:
                formatted = format_qa(question, answer)
                formatted["source"] = "KorMedConceptsQA"
                all_exam_data.append(formatted)
                count += 1

        print(f"  Added {count} KorMedConceptsQA examples")
except Exception as e:
    print(f"  Error: {e}")

# 6. Medical O1 Reasoning
print("\n[6] Loading Medical O1 Reasoning...")
try:
    o1_path = RAW_DIR / "medical_o1_reasoning_ko"
    if o1_path.exists():
        ds = load_from_disk(str(o1_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  O1 Reasoning"):
            # Field names: Question, Complex_Cot, Response (capital letters)
            question = example.get("Question", example.get("instruction", example.get("question", "")))
            answer = example.get("Response", example.get("output", example.get("response", "")))

            if question and answer:
                formatted = format_qa(question, answer, "당신은 한국어 의료 전문 AI 어시스턴트입니다. 단계별로 추론하여 답변하세요.")
                formatted["source"] = "Medical_O1_Reasoning_Ko"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} O1 Reasoning examples")
except Exception as e:
    print(f"  Error: {e}")

# 7. KoMedInstruct-52k
print("\n[7] Loading KoMedInstruct-52k...")
try:
    instruct_path = RAW_DIR / "chugyouk_komedinstruct_52k"
    if instruct_path.exists():
        ds = load_from_disk(str(instruct_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  KoMedInstruct"):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")

            question = f"{instruction}\n{input_text}".strip() if input_text else instruction

            if question and output:
                formatted = format_qa(question, output)
                formatted["source"] = "KoMedInstruct_52k"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} KoMedInstruct examples")
except Exception as e:
    print(f"  Error: {e}")

# 8. ChainOfDiagnosis-Ko (gated, may not be available)
print("\n[8] Loading ChainOfDiagnosis-Ko...")
try:
    chain_path = RAW_DIR / "chugyouk_chainofdiagnosis_ko"
    if chain_path.exists():
        ds = load_from_disk(str(chain_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  ChainOfDiagnosis"):
            question = example.get("question", example.get("instruction", ""))
            answer = example.get("answer", example.get("output", ""))

            if question and answer:
                formatted = format_qa(question, answer, "당신은 한국어 의료 진단 전문 AI 어시스턴트입니다. 체계적인 진단 과정을 통해 답변하세요.")
                formatted["source"] = "ChainOfDiagnosis_Ko"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} ChainOfDiagnosis examples")
    else:
        print("  ChainOfDiagnosis-Ko not found (gated dataset)")
except Exception as e:
    print(f"  Error: {e}")

# 9. Asan-AMC-Healthinfo
print("\n[9] Loading Asan-AMC-Healthinfo...")
try:
    asan_path = RAW_DIR / "chugyouk_asan_amc_healthinfo"
    if asan_path.exists():
        ds = load_from_disk(str(asan_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  Asan AMC"):
            question = example.get("question", example.get("instruction", ""))
            answer = example.get("answer", example.get("output", ""))

            if question and answer:
                formatted = format_qa(question, answer)
                formatted["source"] = "Asan_AMC_Healthinfo"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} Asan AMC examples")
except Exception as e:
    print(f"  Error: {e}")

# 10. KorMedLawQA
print("\n[10] Loading KorMedLawQA...")
try:
    lawqa_path = RAW_DIR / "kormedlawqa"
    if lawqa_path.exists():
        ds = load_from_disk(str(lawqa_path))
    else:
        try:
            ds = load_dataset("snuh/KorMedLawQA", trust_remote_code=True)
        except:
            ds = None

    if ds:
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  KorMedLawQA"):
            question = example.get("question", "")
            answer = example.get("answer", "")
            reasoning = example.get("reasoning", "")

            full_answer = f"{answer}\n\n{reasoning}" if reasoning else answer

            if question and full_answer:
                formatted = format_qa(question, full_answer, "당신은 한국어 의료법 전문 AI 어시스턴트입니다.")
                formatted["source"] = "KorMedLawQA"
                all_exam_data.append(formatted)
                count += 1

        print(f"  Added {count} KorMedLawQA examples")
except Exception as e:
    print(f"  Error: {e}")

# 11. KorMedConceptsQA (multiple configs)
print("\n[11] Loading KorMedConceptsQA...")
try:
    concepts_configs = ['kcd8_easy', 'kcd8_medium', 'kcd8_hard', 'kcd8_merged', 'atc_easy', 'atc_medium', 'atc_hard']
    count = 0
    for config in concepts_configs:
        config_path = RAW_DIR / f"kormedconceptsqa_{config}"
        if config_path.exists():
            ds = load_from_disk(str(config_path))
            for split in ds:
                for example in ds[split]:
                    question = example.get("question", "")
                    answer = example.get("answer", "")

                    if question and answer:
                        formatted = format_qa(question, answer)
                        formatted["source"] = f"KorMedConceptsQA_{config}"
                        all_exam_data.append(formatted)
                        count += 1
    print(f"  Added {count} KorMedConceptsQA examples")
except Exception as e:
    print(f"  Error: {e}")

# 12. MedQA-Evol-Korean
print("\n[12] Loading MedQA-Evol-Korean...")
try:
    evol_path = RAW_DIR / "chugyouk_medqa_evol_korean"
    if evol_path.exists():
        ds = load_from_disk(str(evol_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  MedQA-Evol"):
            # Field names: input, output, conversations
            question = example.get("input", "")
            answer = example.get("output", "")

            if question and answer:
                formatted = format_qa(question, answer)
                formatted["source"] = "MedQA_Evol_Korean"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} MedQA-Evol examples")
except Exception as e:
    print(f"  Error: {e}")

# 13. MedQA Korean
print("\n[13] Loading MedQA Korean...")
try:
    medqa_ko_path = RAW_DIR / "chugyouk_medqa_ko"
    if medqa_ko_path.exists():
        ds = load_from_disk(str(medqa_ko_path))
        count = 0
        for split in ds:
            for example in ds[split]:
                # Field names: question_ko, A_ko, B_ko, C_ko, D_ko, answer_idx
                question = example.get("question_ko", "")
                answer_idx = example.get("answer_idx", "A")

                choices = []
                for letter in ['A', 'B', 'C', 'D']:
                    opt = example.get(f"{letter}_ko", "")
                    if opt:
                        choices.append(f"{letter}. {opt}")

                if question and choices:
                    formatted = format_mcqa_with_explanation(question, choices, answer_idx)
                    formatted["source"] = "MedQA_Korean"
                    all_exam_data.append(formatted)
                    count += 1
        print(f"  Added {count} MedQA Korean examples")
except Exception as e:
    print(f"  Error: {e}")

# 14. HealthSearchQA-ko
print("\n[14] Loading HealthSearchQA-ko...")
try:
    health_search_path = RAW_DIR / "chugyouk_healthsearchqa_ko"
    if health_search_path.exists():
        ds = load_from_disk(str(health_search_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  HealthSearchQA"):
            # Field names: question_ko, answer_ko
            question = example.get("question_ko", "")
            answer = example.get("answer_ko", "")

            if question and answer:
                formatted = format_qa(question, answer)
                formatted["source"] = "HealthSearchQA_ko"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} HealthSearchQA examples")
except Exception as e:
    print(f"  Error: {e}")

# 15. GenMedGPT-5k-ko
print("\n[15] Loading GenMedGPT-5k-ko...")
try:
    genmed_path = RAW_DIR / "chugyouk_genmedgpt_5k_ko"
    if genmed_path.exists():
        ds = load_from_disk(str(genmed_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  GenMedGPT"):
            question = example.get("question", example.get("instruction", ""))
            answer = example.get("answer", example.get("output", ""))

            if question and answer:
                formatted = format_qa(question, answer)
                formatted["source"] = "GenMedGPT_5k_ko"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} GenMedGPT examples")
except Exception as e:
    print(f"  Error: {e}")

# 16. MedExpQA-Kor
print("\n[16] Loading MedExpQA-Kor...")
try:
    medexp_path = RAW_DIR / "chugyouk_medexpqa_kor"
    if medexp_path.exists():
        ds = load_from_disk(str(medexp_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  MedExpQA"):
            question = example.get("question", example.get("instruction", ""))
            answer = example.get("answer", example.get("output", ""))

            if question and answer:
                formatted = format_qa(question, answer)
                formatted["source"] = "MedExpQA_Kor"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} MedExpQA examples")
except Exception as e:
    print(f"  Error: {e}")

# 17. Medical O1 Reasoning SFT Ko (additional download)
print("\n[17] Loading Medical O1 Reasoning SFT Ko (additional)...")
try:
    o1_sft_path = RAW_DIR / "chugyouk_medical_o1_reasoning_sft_ko"
    if o1_sft_path.exists():
        ds = load_from_disk(str(o1_sft_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  O1 Reasoning SFT"):
            # Field names: Question, Complex_Cot, Response
            question = example.get("Question", "")
            answer = example.get("Response", "")

            if question and answer:
                formatted = format_qa(question, answer, "당신은 한국어 의료 전문 AI 어시스턴트입니다. 단계별로 추론하여 답변하세요.")
                formatted["source"] = "Medical_O1_Reasoning_SFT_Ko"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} O1 Reasoning SFT examples")
except Exception as e:
    print(f"  Error: {e}")

# 19. PubMedQA-test-Ko
print("\n[19] Loading PubMedQA-test-Ko...")
try:
    pubmed_path = RAW_DIR / "chugyouk_pubmedqa_test_ko"
    if pubmed_path.exists():
        ds = load_from_disk(str(pubmed_path))
        count = 0
        for split in ds:
            for example in ds[split]:
                question = example.get("question", example.get("instruction", ""))
                answer = example.get("answer", example.get("output", example.get("final_decision", "")))

                if question and answer:
                    formatted = format_qa(question, str(answer))
                    formatted["source"] = "PubMedQA_test_Ko"
                    all_exam_data.append(formatted)
                    count += 1
        print(f"  Added {count} PubMedQA examples")
except Exception as e:
    print(f"  Error: {e}")

# 20. Medical Question Pairs Ko
print("\n[20] Loading Medical Question Pairs Ko...")
try:
    pairs_path = RAW_DIR / "chugyouk_medical_questions_pairs_ko"
    if pairs_path.exists():
        ds = load_from_disk(str(pairs_path))
        count = 0
        data = ds["train"] if "train" in ds else ds
        for example in tqdm(data, desc="  Medical Pairs"):
            q1 = example.get("question1", "")
            q2 = example.get("question2", "")

            if q1 and q2:
                # Create a QA pair about question similarity
                formatted = format_qa(
                    f"다음 두 의료 질문이 유사한지 판단해주세요:\n질문1: {q1}\n질문2: {q2}",
                    "두 질문은 유사한 의료 주제에 대해 묻고 있습니다."
                )
                formatted["source"] = "Medical_Question_Pairs_Ko"
                all_exam_data.append(formatted)
                count += 1
        print(f"  Added {count} Medical Pairs examples")
except Exception as e:
    print(f"  Error: {e}")


# =============================================================================
# Split into 75% Train / 25% Verification
# =============================================================================
print("\n" + "=" * 60)
print("Creating 75/25 Train/Verification Split")
print("=" * 60)

# Remove duplicates based on question text
print("\nRemoving duplicates...")
seen_questions = set()
unique_data = []
for item in all_exam_data:
    q = item["question"]
    if q not in seen_questions:
        seen_questions.add(q)
        unique_data.append(item)

print(f"Original: {len(all_exam_data)}, After dedup: {len(unique_data)}")

# Shuffle
random.shuffle(unique_data)

# Split
split_idx = int(len(unique_data) * TRAIN_RATIO)
train_data = unique_data[:split_idx]
verification_data = unique_data[split_idx:]

print(f"\nTotal unique samples: {len(unique_data)}")
print(f"Training samples (75%): {len(train_data)}")
print(f"Verification samples (25%): {len(verification_data)}")

# Show source distribution
print("\nSource distribution in training set:")
train_sources = {}
for item in train_data:
    src = item.get("source", "unknown")
    train_sources[src] = train_sources.get(src, 0) + 1
for src, count in sorted(train_sources.items(), key=lambda x: -x[1]):
    print(f"  {src}: {count}")

print("\nSource distribution in verification set:")
verification_sources = {}
for item in verification_data:
    src = item.get("source", "unknown")
    verification_sources[src] = verification_sources.get(src, 0) + 1
for src, count in sorted(verification_sources.items(), key=lambda x: -x[1]):
    print(f"  {src}: {count}")

# =============================================================================
# Save Datasets
# =============================================================================
print("\n" + "=" * 60)
print("Saving Datasets")
print("=" * 60)

# Create datasets
train_dataset = Dataset.from_list(train_data)
verification_dataset = Dataset.from_list(verification_data)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "verification": verification_dataset
})

# Save
dataset_dict.save_to_disk(str(OUTPUT_DIR))
print(f"Dataset saved to {OUTPUT_DIR}")

# Save summary
summary = {
    "total_samples": len(unique_data),
    "train_samples": len(train_data),
    "verification_samples": len(verification_data),
    "train_ratio": TRAIN_RATIO,
    "verification_ratio": VERIFICATION_RATIO,
    "seed": SEED,
    "train_sources": train_sources,
    "verification_sources": verification_sources,
}

with open(OUTPUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"Summary saved to {OUTPUT_DIR / 'split_summary.json'}")

# Also save verification set as separate file for easy evaluation
verification_only = DatasetDict({
    "test": verification_dataset
})
verification_dir = BASE_DIR / "data" / "processed" / "korean_medical_verification_25"
verification_dir.mkdir(parents=True, exist_ok=True)
verification_only.save_to_disk(str(verification_dir))
print(f"Verification set also saved to {verification_dir}")

print("\n" + "=" * 60)
print("Data Preparation Complete!")
print("=" * 60)
print(f"\nTraining data: {len(train_data)} samples (75%)")
print(f"Verification data: {len(verification_data)} samples (25%)")
print(f"\nNext step: Run training with scripts/train_korean_medical_90_accuracy.py")
