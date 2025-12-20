#!/usr/bin/env python3
"""
Filter training data to only include valid MCQ questions with A-E answers.
Removes "정답은" prefix, strips, and takes first letter.
Reports any malformed answers for review.
"""

import re
from datasets import load_from_disk, Dataset
from pathlib import Path

DATA_DIR = Path("data/processed/korean_medical_exam_75_25_split")
OUTPUT_DIR = Path("data/processed/korean_medical_mcq_filtered")

def extract_answer_letter(correct_answer: str, text: str = "") -> tuple[str | None, str | None]:
    """
    Extract answer letter from correct_answer field or from text (assistant response).
    Returns (letter, error_message) - if letter is None, error_message explains why.
    """
    # Try to find answer in correct_answer field first, then in text
    sources_to_check = [correct_answer]

    # Also extract assistant response from text if available
    if text and "<|im_start|>assistant" in text:
        assistant_start = text.find("<|im_start|>assistant") + len("<|im_start|>assistant")
        assistant_end = text.find("<|im_end|>", assistant_start)
        if assistant_end > assistant_start:
            assistant_response = text[assistant_start:assistant_end].strip()
            sources_to_check.append(assistant_response)

    for source in sources_to_check:
        if not source:
            continue

        original = source
        answer = source.strip()

        # Pattern 1: Look for "정답은 X" or "답은 X" patterns anywhere in text
        import re
        patterns = [
            r'정답은\s*\**([A-Ea-e])',  # 정답은 A or 정답은 **A
            r'정답:\s*([A-Ea-e])',   # 정답: A
            r'답은\s*([A-Ea-e])',    # 답은 A
            r'답:\s*([A-Ea-e])',     # 답: A
            r'따라서[,\s]+정답은\s*([A-Ea-e])',  # 따라서, 정답은 A
            r'따라서[,\s]+([A-Ea-e])\s*입니다',  # 따라서 A 입니다
            r'따라서[,\s]+([A-Ea-e])[).\s]',     # 따라서 A) or A.
            r'그러므로[,\s]+([A-Ea-e])',         # 그러므로 A
            r'Answer[:\s]+([A-Ea-e])',           # Answer: A
            r'answer is\s+([A-Ea-e])',           # answer is A
            r'\(([A-Ea-e])\)\s*입니다',          # (A) 입니다
            r'([A-Ea-e])\s*번입니다',            # A 번입니다
            r'([A-Ea-e])\)\s*입니다',            # A) 입니다
            r'^([A-Ea-e])[).\s]',                # Starts with A) or A.
            r'^([A-Ea-e])$',                     # Just the letter
            # Additional patterns for common formats
            r'가장.*(?:진단|설명|원인|치료|약물)은?\s*([A-Ea-e])[).\s]',  # 가장 가능성이 높은 진단은 C)
            r'옵션\s*\(?([A-Ea-e])\)?',          # 옵션 (B) or 옵션 B
            r'선택지\s*\(?([A-Ea-e])\)?',        # 선택지 (B)
            r'([A-Ea-e])\)\s*[가-힣]+입니다',    # A) 폐입니다
            r'([A-Ea-e])\.\s*[가-힣]+입니다',    # A. 폐입니다
            r'정답은\s*\*+([A-Ea-e])\.',         # 정답은 **A.
            r'정답은\s*\*+([A-Ea-e])\*',         # 정답은 **A**
            # More patterns
            r'과정은\s*([A-Ea-e])\)',            # 과정은 A)
            r'([A-Ea-e])\s*번\)',                # A 번) - with closing paren
            r'\(([A-Ea-e])번\)',                 # (A번)
            r'균은\s*([A-Ea-e])\.',              # 균은 A.
            r'정답은\s*([A-Ea-e])입니다',        # 정답은 A입니다 (no space)
            r'정답은\s*([A-Ea-e])\s*[).]',       # 정답은 A) or 정답은 A.
            r'정답은\s*\*{0,2}([A-Ea-e])\s*\*{0,2}입니다',  # 정답은 **A** 입니다
            r'정답은?\s*\**\s*([A-Ea-e])\s*\**\s*[입\.)]', # flexible 정답은 A
            r'설명은\s*([A-Ea-e])[).\s]',        # 설명은 B)
        ]

        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                return match.group(1).upper(), None

        # Pattern 2: If text is short (< 50 chars), look for standalone letter
        if len(answer) < 50:
            # Remove common prefixes
            prefixes_to_remove = [
                "정답은", "정답:", "정답 :", "답:", "답은",
                "Answer:", "Answer is", "The answer is",
                "따라서,", "따라서 ", "그러므로,", "그러므로 "
            ]

            temp = answer
            for prefix in prefixes_to_remove:
                if temp.startswith(prefix):
                    temp = temp[len(prefix):].strip()

            # Remove trailing punctuation
            temp = temp.rstrip(".)!?입니다이다 ")
            temp = temp.strip()

            if temp and temp[0].upper() in ['A', 'B', 'C', 'D', 'E']:
                return temp[0].upper(), None

    # No valid answer found
    preview = correct_answer[:100] if correct_answer else "(empty)"
    return None, f"No answer pattern found in: '{preview}'"


def has_mcq_options(text: str) -> bool:
    """Check if the text contains MCQ options (A, B, C, D pattern)"""
    # Look for patterns like "A)", "A.", "A:", or "A)" in the question
    patterns = [
        r'\bA\s*[).\]:]\s*\S',  # A) or A. or A: followed by text
        r'\b[Aa]\s*[).\]:]\s*\S',
        r'선택지|옵션|보기',  # Korean words for "options/choices"
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def process_dataset(split_name: str):
    """Process a dataset split and filter to valid MCQ only"""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}")

    ds = load_from_disk(str(DATA_DIR / split_name))
    print(f"Original samples: {len(ds)}")

    valid_samples = []
    error_samples = []
    no_mcq_options = 0

    for idx, example in enumerate(ds):
        text = example.get("text", "")
        correct_answer = example.get("correct_answer", "")

        # Check if it has MCQ options in the question
        if not has_mcq_options(text):
            no_mcq_options += 1
            continue

        # Try to extract answer letter (from correct_answer field or text)
        letter, error = extract_answer_letter(correct_answer, text)

        if letter:
            # Valid MCQ sample
            valid_samples.append({
                "text": text,
                "question": example.get("question", ""),
                "correct_answer": letter,  # Store just the letter
                "source": example.get("source", ""),
                "original_answer": correct_answer  # Keep original for reference
            })
        else:
            # Invalid answer format
            error_samples.append({
                "idx": idx,
                "error": error,
                "correct_answer": correct_answer,
                "text_preview": text[:200] if text else ""
            })

    print(f"\nResults:")
    print(f"  - Valid MCQ samples: {len(valid_samples)}")
    print(f"  - No MCQ options (skipped): {no_mcq_options}")
    print(f"  - Invalid answers: {len(error_samples)}")

    # Print error samples for review
    if error_samples:
        print(f"\n--- Invalid Answer Samples ({len(error_samples)}) ---")
        for i, err in enumerate(error_samples[:50]):  # Show first 50
            print(f"\n[{i+1}] Index {err['idx']}: {err['error']}")
            if err['correct_answer']:
                print(f"    Full answer: '{err['correct_answer'][:200]}'")

    return valid_samples, error_samples


def main():
    print("="*60)
    print("MCQ Data Filter - Extract Valid A-E Answers Only")
    print("="*60)

    # Process train and verification splits
    train_valid, train_errors = process_dataset("train")
    verify_valid, verify_errors = process_dataset("verification")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Train: {len(train_valid)} valid MCQ samples")
    print(f"Verification: {len(verify_valid)} valid MCQ samples")
    print(f"Total errors: {len(train_errors) + len(verify_errors)}")

    # Ask user to review errors before saving
    total_errors = len(train_errors) + len(verify_errors)
    if total_errors > 0:
        print(f"\n⚠️  Found {total_errors} samples with invalid answer formats.")
        print("Review the errors above. These samples will be excluded.")

        # Save error log for review
        error_log_path = OUTPUT_DIR / "filter_errors.txt"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(error_log_path, "w") as f:
            f.write("=== Train Errors ===\n")
            for err in train_errors:
                f.write(f"Index {err['idx']}: {err['error']}\n")
                f.write(f"  Answer: {err['correct_answer'][:200] if err['correct_answer'] else 'None'}\n\n")

            f.write("\n=== Verification Errors ===\n")
            for err in verify_errors:
                f.write(f"Index {err['idx']}: {err['error']}\n")
                f.write(f"  Answer: {err['correct_answer'][:200] if err['correct_answer'] else 'None'}\n\n")

        print(f"Error log saved to: {error_log_path}")

    # Save filtered datasets
    if len(train_valid) > 0 and len(verify_valid) > 0:
        print(f"\nSaving filtered datasets to {OUTPUT_DIR}...")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        train_ds = Dataset.from_list(train_valid)
        verify_ds = Dataset.from_list(verify_valid)

        train_ds.save_to_disk(str(OUTPUT_DIR / "train"))
        verify_ds.save_to_disk(str(OUTPUT_DIR / "verification"))

        # Save summary
        summary = {
            "original_train": len(load_from_disk(str(DATA_DIR / "train"))),
            "original_verification": len(load_from_disk(str(DATA_DIR / "verification"))),
            "filtered_train": len(train_valid),
            "filtered_verification": len(verify_valid),
            "train_errors": len(train_errors),
            "verification_errors": len(verify_errors),
        }

        import json
        with open(OUTPUT_DIR / "filter_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Filtered datasets saved!")
        print(f"   Train: {len(train_valid)} samples")
        print(f"   Verification: {len(verify_valid)} samples")
    else:
        print("\n❌ No valid samples found. Check your data.")


if __name__ == "__main__":
    main()
