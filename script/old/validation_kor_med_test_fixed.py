#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KorMedMCQA Validation (Fixed for Extended Embeddings)

IMPORTANT: Does NOT use 8-bit quantization due to incompatibility with extended embeddings.
Requires ~16-20GB VRAM for medgemma-4b in bfloat16.

Usage:
    python validation_kor_med_test_fixed.py --model model/02_mixed/medgemma-4b/final --max-samples 50
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = Path(__file__).parent.parent
TEST_FILE = BASE_DIR / "data" / "02_refined" / "02_kor_med_test" / "test.jsonl"
OUTPUT_DIR = BASE_DIR / "results" / "kor_med_test"


def load_test_data(filepath):
    """Load test data from JSONL file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def format_prompt(sample):
    """Format MCQ question as prompt (matches training format)."""
    prompt = (
        "<start_of_turn>user\n"
        "Reasoning 후 정답 알파벳 하나만 답하세요.\n\n"
        f"{sample['question']}\n"
        f"A) {sample['A']}\n"
        f"B) {sample['B']}\n"
        f"C) {sample['C']}\n"
        f"D) {sample['D']}\n"
        f"E) {sample['E']}\n\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    return prompt


def extract_answer(response):
    """Extract answer letter from model response (handles reasoning format)."""
    response = response.strip()

    # The model outputs: <reasoning>...</reasoning>X where X is the answer
    if '</reasoning>' in response:
        after_reasoning = response.split('</reasoning>')[-1].strip().upper()
        for char in after_reasoning:
            if char in 'ABCDE':
                return char

    # Fallback: look for first A, B, C, D, or E
    response_upper = response.upper()
    for char in response_upper:
        if char in 'ABCDE':
            return char

    return ""


def evaluate_model(model, tokenizer, test_data, device, max_samples=None):
    """Evaluate model on test data."""
    model.eval()

    correct = 0
    total = 0
    results = []

    if max_samples:
        test_data = test_data[:max_samples]

    with torch.no_grad():
        for sample in tqdm(test_data, desc="Evaluating"):
            prompt = format_prompt(sample)
            expected = sample['answer']

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted = extract_answer(response)

            is_correct = predicted == expected
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": sample['question'][:100] + "...",
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "response": response[:200],
            })

    accuracy = correct / total * 100 if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KorMedMCQA Validation (Fixed - No 8-bit Quantization)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    # Check if model is adapter
    model_path_obj = Path(args.model)
    is_adapter = (model_path_obj / "adapter_config.json").exists()

    if is_adapter:
        # Load adapter
        with open(model_path_obj / "adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "google/medgemma-4b-it")

        print(f"\nDetected PEFT adapter")
        print(f"Base model: {base_model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model (NO 8-bit quantization!)
        print(f"Loading base model in bfloat16...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="cpu",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        # Resize embeddings
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        # Load adapter
        print(f"Loading PEFT adapter...")
        model = PeftModel.from_pretrained(model, args.model)
        model = model.to(args.device)

    else:
        # Base model
        print(f"Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=args.device,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

    # Load test data
    test_data = load_test_data(TEST_FILE)
    print(f"Loaded {len(test_data)} test samples")

    # Evaluate
    print("\nEvaluating...")
    eval_results = evaluate_model(model, tokenizer, test_data, args.device, args.max_samples)

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Accuracy: {eval_results['accuracy']:.2f}%")
    print(f"Correct: {eval_results['correct']}/{eval_results['total']}")

    target = 90.0
    if eval_results['accuracy'] >= target:
        print(f"\n✓ TARGET MET: {eval_results['accuracy']:.2f}% >= {target}%")
    else:
        gap = target - eval_results['accuracy']
        print(f"\n✗ TARGET NOT MET: {eval_results['accuracy']:.2f}% < {target}% (gap: {gap:.2f}%)")

    # Show sample results
    wrong = [r for r in eval_results['results'] if not r['correct']]
    correct = [r for r in eval_results['results'] if r['correct']]

    if correct:
        print("\nSample CORRECT answers:")
        for r in correct[:3]:
            print(f"  Q: {r['question']}")
            print(f"  Answer: {r['expected']} ✓")
            print()

    if wrong:
        print("Sample WRONG answers:")
        for r in wrong[:3]:
            print(f"  Q: {r['question']}")
            print(f"  Expected: {r['expected']}, Predicted: {r['predicted']}")
            print()

    # Save results
    output_file = args.output or (OUTPUT_DIR / f"eval_{model_path_obj.name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model,
            "accuracy": eval_results['accuracy'],
            "correct": eval_results['correct'],
            "total": eval_results['total'],
            "target": target,
            "target_met": eval_results['accuracy'] >= target,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
