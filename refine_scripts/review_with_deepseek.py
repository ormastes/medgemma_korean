#!/usr/bin/env python3
"""
Review and FIX refined data with DeepSeek-R1-Distill-32B

Uses local LLM on A6000 (4-bit quantization) to:
1. Check each sample for quality issues
2. FIX malformed data (grammar, format, accuracy)
3. FILTER OUT test data contamination (samples similar to benchmark)
4. Save corrected data to reviewed/ folder

For each sample:
- If good: keep as-is
- If fixable: correct and save fixed version
- If contaminated: discard (too similar to test set)
- If unfixable: discard
"""

import argparse
import json
import os
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Paths
BASE_DIR = Path(__file__).parent.parent
REFINED_DIR = BASE_DIR / "data" / "refined"
REVIEWED_DIR = BASE_DIR / "data" / "reviewed"

# Test set for contamination check
TEST_DATASET = "sean0042/KorMedMCQA"

# DeepSeek model
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Special tokens
R_START = "<R>"
R_END = "<R/>"

# Review and Fix prompts by type
REVIEW_FIX_PROMPTS = {
    "type1_text": """당신은 의료 데이터 품질 전문가입니다.
다음 의료 QA 데이터를 검토하고 필요하면 수정해주세요.

질문: {question}
답변: {answer}

검토 항목:
1. 의학적 정확성
2. 한국어 문법
3. 답변 완전성

JSON 형식으로 답변하세요:
{{
  "status": "good" / "fixed" / "discard",
  "fixed_answer": "수정된 답변 (status가 fixed인 경우만)",
  "reason": "수정 이유 또는 폐기 이유"
}}

수정이 필요하면 fixed_answer에 올바른 답변을 작성하세요.
문법만 수정하거나 의학적 오류를 바로잡아주세요.""",

    "type2_text_reasoning": """당신은 의료 데이터 품질 전문가입니다.
다음 의료 추론 데이터를 검토하고 필요하면 수정해주세요.

질문: {question}
추론: {reasoning}
답변: {answer}

검토 항목:
1. 추론 논리성
2. 의학적 정확성
3. 한국어 문법

JSON 형식으로 답변하세요:
{{
  "status": "good" / "fixed" / "discard",
  "fixed_reasoning": "수정된 추론 (필요시)",
  "fixed_answer": "수정된 답변 (필요시)",
  "reason": "수정 이유"
}}""",

    "type3_word": """당신은 의료 데이터 품질 전문가입니다.
다음 의료 객관식 문제를 검토하고 필요하면 수정해주세요.

질문: {question}
정답: {answer}

검토 항목:
1. 정답 정확성 (A-E 중 하나여야 함)
2. 문제 명확성
3. 한국어 문법

JSON 형식으로 답변하세요:
{{
  "status": "good" / "fixed" / "discard",
  "fixed_question": "수정된 질문 (필요시)",
  "fixed_answer": "수정된 정답 (A-E 중 하나)",
  "reason": "수정 이유"
}}

정답은 반드시 A, B, C, D, E 중 하나여야 합니다.""",

    "type4_word_reasoning": """당신은 의료 데이터 품질 전문가입니다.
다음 의료 추론 + 진단 데이터를 검토하고 필요하면 수정해주세요.

질문: {question}
추론: {reasoning}
정답: {answer}

검토 항목:
1. 추론-정답 일치성
2. 의학적 정확성
3. 정답 형식 (공백 없는 단어)

JSON 형식으로 답변하세요:
{{
  "status": "good" / "fixed" / "discard",
  "fixed_reasoning": "수정된 추론 (필요시)",
  "fixed_answer": "수정된 정답 (공백 없는 단어)",
  "reason": "수정 이유"
}}

정답은 공백 없이 한 단어로 작성하세요."""
}


class ContaminationChecker:
    """Check if training data is contaminated with test set questions"""

    def __init__(self, similarity_threshold: float = 0.8):
        self.threshold = similarity_threshold
        self.test_questions: Set[str] = set()
        self.test_ngrams: List[Set[str]] = []
        self._load_test_set()

    def _load_test_set(self):
        """Load KorMedMCQA test set questions from local data"""
        print("Loading test set for contamination check...")
        try:
            # Try local data first (data/raw/by_source/kormedmcqa/)
            local_path = BASE_DIR / "data" / "raw" / "by_source" / "kormedmcqa"
            if local_path.exists():
                test_datasets = []
                for split_dir in local_path.iterdir():
                    if split_dir.is_dir() and split_dir.name.startswith("test_"):
                        ds = load_from_disk(str(split_dir))
                        test_datasets.append(ds)
                if test_datasets:
                    combined = concatenate_datasets(test_datasets)
                    for item in combined:
                        question = item.get("question", "")
                        if question:
                            normalized = self._normalize(question)
                            self.test_questions.add(normalized)
                            self.test_ngrams.append(self._get_ngrams(normalized, n=3))
                    print(f"Loaded {len(self.test_questions)} test questions from local data")
                    return
            # Fallback to HuggingFace (load all configs and filter test split)
            for config in ["doctor", "nurse", "pharm", "dentist"]:
                dataset = load_dataset(TEST_DATASET, config, split="test")
                for item in dataset:
                    question = item.get("question", "")
                    if question:
                        normalized = self._normalize(question)
                        self.test_questions.add(normalized)
                        self.test_ngrams.append(self._get_ngrams(normalized, n=3))
            print(f"Loaded {len(self.test_questions)} test questions from HuggingFace")
        except Exception as e:
            print(f"Warning: Could not load test set: {e}")
            print("Contamination check will be skipped")

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove whitespace, punctuation, lowercase
        text = re.sub(r'\s+', '', text.lower())
        text = re.sub(r'[^\w가-힣]', '', text)
        return text

    def _get_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Extract character n-grams"""
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def is_contaminated(self, text: str) -> Tuple[bool, float]:
        """
        Check if text is too similar to any test question.
        Returns: (is_contaminated, max_similarity)
        """
        if not self.test_questions:
            return False, 0.0

        normalized = self._normalize(text)

        # Exact match check
        if normalized in self.test_questions:
            return True, 1.0

        # N-gram similarity check
        text_ngrams = self._get_ngrams(normalized, n=3)
        max_sim = 0.0

        for test_ngrams in self.test_ngrams:
            sim = self._jaccard_similarity(text_ngrams, test_ngrams)
            max_sim = max(max_sim, sim)
            if sim >= self.threshold:
                return True, sim

        return False, max_sim

    def check_sample(self, sample: Dict) -> Tuple[bool, float, str]:
        """
        Check sample for contamination.
        Returns: (is_contaminated, similarity, field_name)
        """
        # Check prompt/question
        prompt = sample.get("prompt", "")
        # Extract question from ChatML format
        if "<|im_start|>user" in prompt:
            user_part = prompt.split("<|im_start|>user")[-1]
            if "<|im_end|>" in user_part:
                question = user_part.split("<|im_end|>")[0].strip()
            else:
                question = user_part.strip()
        else:
            question = prompt

        is_cont, sim = self.is_contaminated(question)
        if is_cont:
            return True, sim, "question"

        return False, sim, ""


class DeepSeekReviewer:
    def __init__(self, device: str = "cuda"):
        print(f"Loading DeepSeek-R1-Distill-32B (4-bit)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            DEEPSEEK_MODEL,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            DEEPSEEK_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    def review_and_fix(self, sample: Dict, data_type: str) -> Tuple[str, Dict]:
        """
        Review and fix a sample.
        Returns: (status, fixed_sample)
        - status: "good", "fixed", "discard"
        - fixed_sample: corrected sample (or original if good)
        """
        prompt_template = REVIEW_FIX_PROMPTS.get(data_type, REVIEW_FIX_PROMPTS["type1_text"])

        # Extract fields
        question = self._extract_question(sample.get("prompt", ""))
        completion = sample.get("completion", "")

        if data_type in ["type2_text_reasoning", "type4_word_reasoning"]:
            if R_START in completion and R_END in completion:
                reasoning = completion.split(R_START)[1].split(R_END)[0]
                answer = completion.split(R_END)[-1].strip()
            else:
                reasoning = ""
                answer = completion.strip()

            review_prompt = prompt_template.format(
                question=question,
                reasoning=reasoning,
                answer=answer
            )
        else:
            answer = sample.get("answer", completion).strip()
            review_prompt = prompt_template.format(
                question=question,
                answer=answer
            )

        # Generate review
        try:
            inputs = self.tokenizer(
                review_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Parse response
            result = self._parse_fix_response(response)
            status = result.get("status", "good")

            if status == "discard":
                return "discard", None

            elif status == "fixed":
                # Apply fixes
                fixed_sample = self._apply_fixes(sample, result, data_type)
                fixed_sample["review_status"] = "fixed"
                fixed_sample["review_reason"] = result.get("reason", "")
                return "fixed", fixed_sample

            else:
                # Good as-is
                sample["review_status"] = "good"
                return "good", sample

        except Exception as e:
            print(f"  Error: {e}")
            # Keep original on error
            sample["review_status"] = "error"
            return "good", sample

    def _extract_question(self, prompt: str) -> str:
        """Extract question from ChatML prompt"""
        if "<|im_start|>user" in prompt:
            parts = prompt.split("<|im_start|>user")
            if len(parts) > 1:
                user_part = parts[-1].split("<|im_end|>")[0]
                return user_part.strip()
        return prompt[:500]

    def _parse_fix_response(self, response: str) -> Dict:
        """Parse JSON fix response"""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        return {"status": "good"}

    def _apply_fixes(self, sample: Dict, fixes: Dict, data_type: str) -> Dict:
        """Apply fixes to sample"""
        fixed = sample.copy()

        if data_type in ["type2_text_reasoning", "type4_word_reasoning"]:
            # Rebuild completion with reasoning tokens
            reasoning = fixes.get("fixed_reasoning", "")
            answer = fixes.get("fixed_answer", "")

            if not reasoning:
                # Extract original reasoning
                orig = sample.get("completion", "")
                if R_START in orig and R_END in orig:
                    reasoning = orig.split(R_START)[1].split(R_END)[0]

            if not answer:
                # Extract original answer
                orig = sample.get("completion", "")
                if R_END in orig:
                    answer = orig.split(R_END)[-1].strip()
                else:
                    answer = sample.get("answer", orig)

            # Remove spaces from word answer (type4)
            if data_type == "type4_word_reasoning":
                answer = answer.replace(" ", "")

            fixed["completion"] = f"{R_START}{reasoning}{R_END}{answer}"
            fixed["answer"] = answer

        elif data_type == "type3_word":
            # MCQ - fix answer letter
            if "fixed_answer" in fixes:
                answer = fixes["fixed_answer"].strip().upper()
                if len(answer) >= 1 and answer[0] in "ABCDE":
                    answer = answer[0]
                fixed["completion"] = answer
                fixed["answer"] = answer

            if "fixed_question" in fixes:
                # Rebuild prompt with fixed question
                fixed_q = fixes["fixed_question"]
                fixed["prompt"] = self._rebuild_prompt(sample["prompt"], fixed_q, data_type)

        else:  # type1_text
            if "fixed_answer" in fixes:
                fixed["completion"] = fixes["fixed_answer"]

        # Rebuild text field
        fixed["text"] = fixed["prompt"] + fixed["completion"]
        if not fixed["text"].endswith("<|im_end|>"):
            fixed["text"] += "\n<|im_end|>"

        return fixed

    def _rebuild_prompt(self, original_prompt: str, new_question: str, data_type: str) -> str:
        """Rebuild prompt with new question"""
        # Keep system part, replace user part
        if "<|im_start|>system" in original_prompt:
            system_part = original_prompt.split("<|im_end|>")[0] + "<|im_end|>\n"
        else:
            system_part = ""

        return f"{system_part}<|im_start|>user\n{new_question}\n<|im_end|>\n<|im_start|>assistant\n"


def review_type(reviewer: DeepSeekReviewer, data_type: str,
                contamination_checker: ContaminationChecker = None,
                max_samples: int = None, save_interval: int = 100):
    """Review and fix all samples of a type"""

    input_dir = REFINED_DIR / data_type
    output_dir = REVIEWED_DIR / data_type

    if not input_dir.exists():
        print(f"  {data_type}: No data found at {input_dir}")
        return {"good": 0, "fixed": 0, "discard": 0, "contaminated": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"good": 0, "fixed": 0, "discard": 0, "contaminated": 0}

    for split in ["train", "validation"]:
        input_file = input_dir / split / "data.jsonl"
        if not input_file.exists():
            continue

        output_split_dir = output_dir / split
        output_split_dir.mkdir(exist_ok=True)
        output_file = output_split_dir / "data.jsonl"

        # Load samples
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))

        if max_samples:
            samples = samples[:max_samples]

        print(f"\n  {split}: {len(samples)} samples")

        reviewed_samples = []

        for i, sample in enumerate(tqdm(samples, desc=f"  Reviewing {split}")):
            # Check contamination FIRST (before LLM review)
            if contamination_checker:
                is_contaminated, sim, field = contamination_checker.check_sample(sample)
                if is_contaminated:
                    stats["contaminated"] += 1
                    tqdm.write(f"    [CONTAMINATED] sim={sim:.2f} in {field}")
                    continue  # Skip this sample entirely

            status, fixed_sample = reviewer.review_and_fix(sample, data_type)

            stats[status] += 1

            if fixed_sample:
                reviewed_samples.append(fixed_sample)

            # Save periodically
            if (i + 1) % save_interval == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for s in reviewed_samples:
                        f.write(json.dumps(s, ensure_ascii=False) + '\n')
                print(f"    Saved {len(reviewed_samples)} (good:{stats['good']}, fixed:{stats['fixed']}, discard:{stats['discard']}, contaminated:{stats['contaminated']})")

        # Final save
        with open(output_file, 'w', encoding='utf-8') as f:
            for s in reviewed_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

        print(f"    Result: {len(reviewed_samples)} samples (good:{stats['good']}, fixed:{stats['fixed']}, discard:{stats['discard']}, contaminated:{stats['contaminated']})")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Review and fix data with DeepSeek-R1")
    parser.add_argument("--type", type=str, default=None,
                        choices=["type1_text", "type2_text_reasoning",
                                 "type3_word", "type4_word_reasoning"],
                        help="Specific type to review")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split (for testing)")
    parser.add_argument("--all", action="store_true",
                        help="Review all types")
    args = parser.parse_args()

    print("=" * 60)
    print("DeepSeek-R1 Data Review & Fix")
    print("=" * 60)
    print(f"Input: {REFINED_DIR}")
    print(f"Output: {REVIEWED_DIR}")
    print("\nProcess:")
    print("  - good: Keep as-is")
    print("  - fixed: Correct errors and save")
    print("  - contaminated: Filter out (too similar to test set)")
    print("  - discard: Remove unfixable samples")

    # Initialize contamination checker (loads test set)
    print("\n" + "-" * 40)
    contamination_checker = ContaminationChecker(similarity_threshold=0.8)

    # Initialize reviewer
    print("\n" + "-" * 40)
    reviewer = DeepSeekReviewer()

    # Determine types
    if args.type:
        types_to_review = [args.type]
    elif args.all:
        types_to_review = ["type1_text", "type2_text_reasoning",
                          "type3_word", "type4_word_reasoning"]
    else:
        print("\nSpecify --type TYPE or --all")
        return

    # Review each type
    total_stats = {"good": 0, "fixed": 0, "discard": 0, "contaminated": 0}

    for data_type in types_to_review:
        print(f"\n{'='*50}")
        print(f"Reviewing: {data_type}")
        print(f"{'='*50}")

        stats = review_type(
            reviewer, data_type,
            contamination_checker=contamination_checker,
            max_samples=args.max_samples
        )

        for k, v in stats.items():
            total_stats[k] += v

    # Summary
    print(f"\n{'='*60}")
    print("REVIEW SUMMARY")
    print(f"{'='*60}")
    total = sum(total_stats.values())
    print(f"Total processed: {total}")
    if total > 0:
        print(f"  Good (kept as-is):    {total_stats['good']:>6} ({100*total_stats['good']/total:.1f}%)")
        print(f"  Fixed (corrected):    {total_stats['fixed']:>6} ({100*total_stats['fixed']/total:.1f}%)")
        print(f"  Contaminated (test):  {total_stats['contaminated']:>6} ({100*total_stats['contaminated']/total:.1f}%)")
        print(f"  Discard (removed):    {total_stats['discard']:>6} ({100*total_stats['discard']/total:.1f}%)")

    kept = total_stats['good'] + total_stats['fixed']
    print(f"\nKept: {kept} samples ({100*kept/total:.1f}%)" if total > 0 else "")
    print(f"Reviewed data saved to: {REVIEWED_DIR}")


if __name__ == "__main__":
    main()
