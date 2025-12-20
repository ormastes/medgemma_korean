#!/usr/bin/env python3
"""
Verify refined data formats are correct.

Checks:
- Type 1: TEXT - NO reasoning tokens, has full text answer
- Type 2: TEXT_REASONING - HAS <R>...<R/>, has full text after
- Type 3: WORD - NO reasoning tokens, short answer (letter or word)
- Type 4: WORD_REASONING - HAS <R>...<R/>, short answer after

Reports format violations and statistics.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent.parent
REFINED_DIR = BASE_DIR / "data" / "refined"

# Special tokens
R_START = "<R>"
R_END = "<R/>"

# Format rules
FORMAT_RULES = {
    "type1_text": {
        "has_reasoning": False,
        "answer_type": "text",  # Full text (long)
        "min_answer_len": 20,
        "description": "Full text, NO reasoning"
    },
    "type2_text_reasoning": {
        "has_reasoning": True,
        "answer_type": "text",
        "min_answer_len": 10,
        "description": "Full text WITH reasoning"
    },
    "type3_word": {
        "has_reasoning": False,
        "answer_type": "word",  # Short (letter or word)
        "max_answer_len": 30,
        "description": "Word/letter, NO reasoning"
    },
    "type4_word_reasoning": {
        "has_reasoning": True,
        "answer_type": "word",
        "max_answer_len": 30,
        "description": "Word WITH reasoning"
    }
}


class FormatVerifier:
    def __init__(self):
        self.errors = defaultdict(list)
        self.stats = defaultdict(lambda: defaultdict(int))

    def has_reasoning_tokens(self, text: str) -> bool:
        """Check if text contains reasoning tokens"""
        return R_START in text and R_END in text

    def extract_after_reasoning(self, text: str) -> str:
        """Extract text after reasoning end token"""
        if R_END in text:
            return text.split(R_END)[-1].strip()
        return text.strip()

    def extract_reasoning(self, text: str) -> str:
        """Extract reasoning content between tokens"""
        if R_START in text and R_END in text:
            start = text.find(R_START) + len(R_START)
            end = text.find(R_END)
            return text[start:end].strip()
        return ""

    def is_word_answer(self, answer: str) -> bool:
        """Check if answer is word-type (short)"""
        answer = answer.strip()
        # Single letter
        if len(answer) == 1 and answer.upper() in "ABCDE":
            return True
        # Short word (no spaces, <= 30 chars)
        if " " not in answer and len(answer) <= 30:
            return True
        # Few words
        if len(answer.split()) <= 3 and len(answer) <= 50:
            return True
        return False

    def verify_sample(self, sample: Dict, type_name: str, idx: int) -> List[str]:
        """Verify a single sample, return list of errors"""
        errors = []
        rules = FORMAT_RULES.get(type_name, {})

        completion = sample.get("completion", "")
        text = sample.get("text", "")

        # Check reasoning tokens
        has_reasoning = self.has_reasoning_tokens(completion)
        should_have_reasoning = rules.get("has_reasoning", False)

        if should_have_reasoning and not has_reasoning:
            errors.append(f"Missing reasoning tokens {R_START}...{R_END}")
        elif not should_have_reasoning and has_reasoning:
            errors.append(f"Should NOT have reasoning tokens but found them")

        # Extract answer
        if has_reasoning:
            answer = self.extract_after_reasoning(completion)
            reasoning = self.extract_reasoning(completion)

            # Check reasoning is not empty
            if should_have_reasoning and len(reasoning) < 10:
                errors.append(f"Reasoning too short: {len(reasoning)} chars")
        else:
            answer = completion.strip()
            reasoning = ""

        # Check answer type
        answer_type = rules.get("answer_type", "text")

        if answer_type == "word":
            max_len = rules.get("max_answer_len", 30)
            if len(answer) > max_len:
                errors.append(f"Answer too long for word type: {len(answer)} chars (max {max_len})")

            # Check no spaces in word answer
            if " " in answer.strip():
                self.stats[type_name]["has_space_in_answer"] += 1

        elif answer_type == "text":
            min_len = rules.get("min_answer_len", 20)
            if len(answer) < min_len:
                errors.append(f"Answer too short for text type: {len(answer)} chars (min {min_len})")

        # Check ChatML format
        if "<|im_start|>" not in text:
            errors.append("Missing ChatML format (<|im_start|>)")

        # Stats
        self.stats[type_name]["total"] += 1
        if has_reasoning:
            self.stats[type_name]["has_reasoning"] += 1
        if self.is_word_answer(answer):
            self.stats[type_name]["word_answer"] += 1
        else:
            self.stats[type_name]["text_answer"] += 1

        return errors

    def verify_type(self, type_name: str) -> Tuple[int, int, List[Dict]]:
        """Verify all samples for a type, return (valid, invalid, error_samples)"""
        type_dir = REFINED_DIR / type_name
        if not type_dir.exists():
            print(f"  Directory not found: {type_dir}")
            return 0, 0, []

        valid = 0
        invalid = 0
        error_samples = []

        for split in ["train", "validation"]:
            split_dir = type_dir / split
            data_file = split_dir / "data.jsonl"

            if not data_file.exists():
                continue

            with open(data_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    try:
                        sample = json.loads(line)
                        errors = self.verify_sample(sample, type_name, idx)

                        if errors:
                            invalid += 1
                            if len(error_samples) < 5:  # Keep first 5 errors
                                error_samples.append({
                                    "split": split,
                                    "idx": idx,
                                    "errors": errors,
                                    "completion": sample.get("completion", "")[:100] + "..."
                                })
                        else:
                            valid += 1

                    except json.JSONDecodeError:
                        invalid += 1

        return valid, invalid, error_samples

    def print_sample_examples(self, type_name: str, count: int = 3):
        """Print example samples for a type"""
        type_dir = REFINED_DIR / type_name / "train"
        data_file = type_dir / "data.jsonl"

        if not data_file.exists():
            return

        print(f"\n  Sample examples from {type_name}:")
        print("  " + "-" * 60)

        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= count:
                    break

                sample = json.loads(line)
                completion = sample.get("completion", "")

                # Truncate for display
                if len(completion) > 150:
                    display = completion[:150] + "..."
                else:
                    display = completion

                print(f"\n  [{idx+1}] {display}")

    def run(self):
        """Run full verification"""
        print("=" * 70)
        print("Verifying Refined Data Formats")
        print("=" * 70)

        results = {}

        for type_name, rules in FORMAT_RULES.items():
            print(f"\n{'-'*50}")
            print(f"Checking {type_name}")
            print(f"  Rule: {rules['description']}")
            print(f"  Reasoning: {'REQUIRED' if rules['has_reasoning'] else 'FORBIDDEN'}")
            print(f"  Answer: {rules['answer_type'].upper()}")
            print(f"{'-'*50}")

            valid, invalid, error_samples = self.verify_type(type_name)

            if valid + invalid == 0:
                print(f"  No data found")
                continue

            pct_valid = 100 * valid / (valid + invalid) if (valid + invalid) > 0 else 0

            print(f"\n  Results: {valid:,} valid, {invalid:,} invalid ({pct_valid:.1f}% valid)")

            # Show stats
            stats = self.stats[type_name]
            print(f"\n  Stats:")
            print(f"    Total samples: {stats['total']:,}")
            print(f"    With reasoning: {stats['has_reasoning']:,}")
            print(f"    Word answers: {stats['word_answer']:,}")
            print(f"    Text answers: {stats['text_answer']:,}")
            if stats.get('has_space_in_answer', 0) > 0:
                print(f"    ⚠️  Answers with spaces: {stats['has_space_in_answer']}")

            # Show errors
            if error_samples:
                print(f"\n  ❌ Error examples:")
                for err in error_samples[:3]:
                    print(f"    [{err['split']}:{err['idx']}] {err['errors']}")
                    print(f"      Completion: {err['completion']}")

            # Show examples
            self.print_sample_examples(type_name)

            results[type_name] = {
                "valid": valid,
                "invalid": invalid,
                "pct_valid": pct_valid,
                "stats": dict(stats)
            }

        # Final summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)

        total_valid = sum(r["valid"] for r in results.values())
        total_invalid = sum(r["invalid"] for r in results.values())
        total = total_valid + total_invalid

        print(f"\n{'Type':<25} {'Valid':>10} {'Invalid':>10} {'%':>8}")
        print("-" * 55)

        for type_name, r in results.items():
            print(f"{type_name:<25} {r['valid']:>10,} {r['invalid']:>10,} {r['pct_valid']:>7.1f}%")

        print("-" * 55)
        pct = 100 * total_valid / total if total > 0 else 0
        print(f"{'TOTAL':<25} {total_valid:>10,} {total_invalid:>10,} {pct:>7.1f}%")

        # Recommendations
        print("\n" + "-" * 50)
        print("Format Requirements Check:")
        print("-" * 50)

        checks_passed = True

        for type_name, rules in FORMAT_RULES.items():
            stats = self.stats[type_name]
            if stats['total'] == 0:
                print(f"  ⚠️  {type_name}: No data")
                continue

            # Check reasoning requirement
            if rules['has_reasoning']:
                pct_reasoning = 100 * stats['has_reasoning'] / stats['total']
                if pct_reasoning < 95:
                    print(f"  ❌ {type_name}: Only {pct_reasoning:.1f}% have reasoning (should be ~100%)")
                    checks_passed = False
                else:
                    print(f"  ✅ {type_name}: {pct_reasoning:.1f}% have reasoning")
            else:
                pct_no_reasoning = 100 * (stats['total'] - stats['has_reasoning']) / stats['total']
                if pct_no_reasoning < 95:
                    print(f"  ❌ {type_name}: {100-pct_no_reasoning:.1f}% have reasoning (should be 0%)")
                    checks_passed = False
                else:
                    print(f"  ✅ {type_name}: {pct_no_reasoning:.1f}% have no reasoning")

            # Check answer type
            if rules['answer_type'] == 'word':
                pct_word = 100 * stats['word_answer'] / stats['total']
                if pct_word < 90:
                    print(f"  ❌ {type_name}: Only {pct_word:.1f}% are word answers (should be ~100%)")
                    checks_passed = False
            else:
                pct_text = 100 * stats['text_answer'] / stats['total']
                if pct_text < 80:
                    print(f"  ⚠️  {type_name}: Only {pct_text:.1f}% are text answers")

        print("\n" + "-" * 50)
        if checks_passed:
            print("✅ All format checks PASSED!")
        else:
            print("❌ Some format checks FAILED - review the data")

        return results


def main():
    verifier = FormatVerifier()
    verifier.run()


if __name__ == "__main__":
    main()
