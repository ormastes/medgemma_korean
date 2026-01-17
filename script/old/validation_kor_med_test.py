#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation on KorMedMCQA Test Set

Evaluates model accuracy on Korean medical MCQ questions.
This is the primary benchmark for Korean medical understanding.

Target: >= 90% accuracy on 604 test samples

Usage:
    python validation_kor_med_test.py --model path/to/model
    python validation_kor_med_test.py --model medgemma-4b  # Use base model
"""

import argparse
import json
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS

BASE_DIR = Path(__file__).parent.parent
TEST_FILE = BASE_DIR / "data" / "02_refined" / "02_kor_med_test" / "test.jsonl"
OUTPUT_DIR = BASE_DIR / "results" / "kor_med_test"


def load_test_data(filepath: Path) -> list:
    """Load test data from JSONL file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def format_prompt(sample: dict) -> str:
    """Format MCQ question as prompt (matches training format)."""
    question = sample['question']

    # Match the exact format used during training (train_02_mixed_data.py)
    prompt = (
        "<start_of_turn>user\n"
        "Reasoning 후 정답 알파벳 하나만 답하세요.\n\n"
        f"{question}\n"
        f"A) {sample['A']}\n"
        f"B) {sample['B']}\n"
        f"C) {sample['C']}\n"
        f"D) {sample['D']}\n"
        f"E) {sample['E']}\n\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    return prompt


def extract_answer(response: str) -> str:
    """Extract answer letter from model response (handles reasoning format)."""
    response = response.strip()

    # The model outputs: <reasoning>...</reasoning>X where X is the answer
    # Extract text after </reasoning> if present
    if '</reasoning>' in response:
        # Get text after </reasoning>
        after_reasoning = response.split('</reasoning>')[-1].strip().upper()
        # Look for first A, B, C, D, or E
        for char in after_reasoning:
            if char in 'ABCDE':
                return char

    # Fallback: look for first A, B, C, D, or E in entire response
    response_upper = response.upper()
    for char in response_upper:
        if char in 'ABCDE':
            return char

    return ""


def evaluate_model(model, tokenizer, test_data: list, device: str, max_samples: int = None) -> dict:
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

            # Generate (allow longer output for reasoning format)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # Need space for <reasoning>...</reasoning>ANSWER
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode response (only new tokens)
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
                "response": response[:50],
            })

    accuracy = correct / total * 100 if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on KorMedMCQA Test Set")
    parser.add_argument("--model", required=True, help="Model path or name (e.g., medgemma-4b)")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KorMedMCQA Validation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Adapter: {args.adapter or 'None'}")
    print(f"Test file: {TEST_FILE}")

    # Determine if model is adapter path or base model
    model_path_obj = Path(args.model)
    is_adapter = (model_path_obj / "adapter_config.json").exists()

    if is_adapter:
        # Model is a PEFT adapter
        adapter_path = args.model

        # Load adapter config to get base model
        with open(model_path_obj / "adapter_config.json", 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "google/medgemma-4b-it")

        print(f"Detected PEFT adapter: {adapter_path}")
        print(f"Base model: {base_model_name}")

        # Load tokenizer from adapter (may have extended vocabulary)
        print(f"\nLoading tokenizer from adapter: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        print(f"Loading base model: {base_model_name}")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=args.device,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # Resize embeddings if needed
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        # Load adapter
        print(f"Loading PEFT adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    else:
        # Model is a base model path or name
        if args.model in MODEL_CONFIGS:
            model_path = MODEL_CONFIGS[args.model]['path']
        else:
            model_path = args.model

        print(f"Loading base model: {model_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=args.device,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # Load adapter if specified
        if args.adapter:
            print(f"Loading adapter: {args.adapter}")
            model = PeftModel.from_pretrained(model, args.adapter)

    # Load test data
    if not TEST_FILE.exists():
        print(f"\nTest file not found: {TEST_FILE}")
        print("Run transform_kormedmcqa.py first!")
        return

    test_data = load_test_data(TEST_FILE)
    print(f"Loaded {len(test_data)} test samples")

    # Evaluate
    print("\nEvaluating...")
    eval_results = evaluate_model(
        model, tokenizer, test_data,
        device=args.device,
        max_samples=args.max_samples
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Accuracy: {eval_results['accuracy']:.2f}%")
    print(f"Correct: {eval_results['correct']}/{eval_results['total']}")

    target = 90.0
    if eval_results['accuracy'] >= target:
        print(f"\n✓ TARGET MET: {eval_results['accuracy']:.2f}% >= {target}%")
    else:
        gap = target - eval_results['accuracy']
        print(f"\n✗ TARGET NOT MET: {eval_results['accuracy']:.2f}% < {target}% (gap: {gap:.2f}%)")

    # Show some wrong answers
    wrong = [r for r in eval_results['results'] if not r['correct']]
    if wrong:
        print("\nSample wrong answers:")
        for r in wrong[:5]:
            print(f"  Q: {r['question']}")
            print(f"  Expected: {r['expected']}, Predicted: {r['predicted']}")
            print()

    # Save results
    output_file = args.output or (OUTPUT_DIR / f"eval_{Path(args.model).name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model,
            "adapter": args.adapter,
            "accuracy": eval_results['accuracy'],
            "correct": eval_results['correct'],
            "total": eval_results['total'],
            "target": target,
            "target_met": eval_results['accuracy'] >= target,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
