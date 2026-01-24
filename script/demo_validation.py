#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of train_01's validation process.
Shows actual prompts, model responses, and scoring.
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM
import sys
import random

# Add script path for imports
sys.path.append(str(Path(__file__).parent))

BASE_DIR = Path(__file__).parent.parent
VALIDATION_DATA_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"

def log(msg: str, level: str = "INFO"):
    """Simple logging function."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    print(log_msg)

# MCQ validation prompt (what train_01 uses)
VALIDATION_PROMPT_TEMPLATE = """<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

{question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}

<end_of_turn>
<start_of_turn>model
"""

def load_validation_data(max_samples=None):
    """Load KorMedMCQA test data."""
    data = []
    test_file = VALIDATION_DATA_DIR / "test.jsonl"

    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line))
        log(f"Loaded {len(data)} validation samples", "INFO")
    else:
        log(f"Validation file not found: {test_file}", "ERROR")

    return data

def truncate_at_end_of_turn(text):
    """Truncate response at <end_of_turn> token."""
    if "<end_of_turn>" in text:
        return text.split("<end_of_turn>")[0].strip()
    return text.strip()

def calc_score(response, expected):
    """
    Calculate if MCQ answer is correct.
    Looks for expected letter (A/B/C/D/E) in response.
    """
    response_clean = response.strip().upper()
    expected_clean = expected.strip().upper()

    # Check if expected answer appears in response
    is_correct = expected_clean in response_clean

    return {
        'is_correct': is_correct,
        'response': response_clean,
        'expected': expected_clean
    }

def run_validation_demo(model_path, num_samples=5, device="cuda:0"):
    """Run validation and show detailed logs."""

    log(f"Loading model from: {model_path}", "INFO")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Setup termination tokens
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id == tokenizer.unk_token_id:
        end_of_turn_id = None

    terminators = [tokenizer.eos_token_id]
    if end_of_turn_id is not None:
        terminators.append(end_of_turn_id)

    # Load validation data
    validation_data = load_validation_data()

    # Sample random questions
    indices = random.sample(range(len(validation_data)), min(num_samples, len(validation_data)))

    log(f"\n{'='*80}", "INFO")
    log(f"VALIDATION DEMONSTRATION - {num_samples} Samples", "INFO")
    log(f"{'='*80}\n", "INFO")

    correct = 0
    total = 0

    for i, idx in enumerate(indices, 1):
        sample = validation_data[idx]

        # Build MCQ prompt
        prompt = VALIDATION_PROMPT_TEMPLATE.format(
            question=sample['question'],
            A=sample['A'],
            B=sample['B'],
            C=sample['C'],
            D=sample['D'],
            E=sample['E']
        )

        log(f"\n{'─'*80}", "INFO")
        log(f"SAMPLE {i}/{num_samples} (idx={idx})", "INFO")
        log(f"{'─'*80}", "INFO")

        log(f"\n[PROMPT]", "INFO")
        log(prompt, "INFO")

        log(f"\n[EXPECTED ANSWER]: {sample['answer']}", "INFO")

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
            )

        # Decode response (only new tokens)
        input_len = inputs['input_ids'].shape[1]
        generated = outputs[0][input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=False)
        response = truncate_at_end_of_turn(response)

        log(f"\n[MODEL RESPONSE]", "INFO")
        log(response, "INFO")

        # Score
        score = calc_score(response, sample['answer'])

        log(f"\n[RESULT]", "INFO")
        log(f"  Expected: {score['expected']}", "INFO")
        log(f"  Got: {score['response']}", "INFO")
        log(f"  Correct: {'✓' if score['is_correct'] else '✗'}", "INFO")

        if score['is_correct']:
            correct += 1
        total += 1

    # Final summary
    accuracy = (correct / total * 100) if total > 0 else 0

    log(f"\n{'='*80}", "INFO")
    log(f"VALIDATION SUMMARY", "INFO")
    log(f"{'='*80}", "INFO")
    log(f"Accuracy: {accuracy:.1f}% ({correct}/{total})", "INFO")
    log(f"{'='*80}\n", "INFO")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demonstrate train_01 validation")
    parser.add_argument("--model", type=str, default="model/00_trained/medgemma-4b",
                       help="Path to trained model")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of validation samples to show")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")

    args = parser.parse_args()

    run_validation_demo(
        model_path=args.model,
        num_samples=args.samples,
        device=args.device
    )
