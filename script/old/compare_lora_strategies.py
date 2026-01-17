#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Single-LoRA vs Dual/Triple-LoRA Strategies

Tests:
1. Korean language preservation (perplexity on Korean text)
2. Medical vocabulary retention
3. MCQ accuracy on KorMedMCQA

Usage:
    python script/compare_lora_strategies.py
"""

import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


def load_model_and_tokenizer(model_path, device="cuda:0"):
    """Load model with 8-bit quantization"""
    print(f"Loading model: {model_path}")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-4b-it",
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def calculate_perplexity(model, tokenizer, texts, max_samples=100):
    """Calculate perplexity on a set of texts"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts[:max_samples], desc="Calculating perplexity"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            num_tokens = inputs["input_ids"].size(1)

            total_loss += loss * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def test_korean_preservation(model, tokenizer):
    """Test Korean language preservation using Korean Wikipedia"""
    print("\n" + "=" * 70)
    print("Test 1: Korean Language Preservation")
    print("=" * 70)

    # Load Korean text samples
    try:
        dataset = load_dataset("wikipedia", "20220301.ko", split="train", streaming=True)
        korean_texts = [item['text'] for item in list(dataset.take(100))]
    except:
        # Fallback: use local Korean text if available
        korean_texts = [
            "한국은 동아시아에 위치한 나라입니다.",
            "서울은 한국의 수도이며 가장 큰 도시입니다.",
            "한국어는 한글로 표기되는 언어입니다.",
            "김치는 한국의 대표적인 전통 음식입니다.",
            "한국의 역사는 수천 년을 거슬러 올라갑니다.",
        ] * 20

    perplexity = calculate_perplexity(model, tokenizer, korean_texts)

    print(f"Korean Text Perplexity: {perplexity:.2f}")
    print(f"  Lower is better (good: < 3.0, excellent: < 2.5)")

    return perplexity


def test_medical_vocabulary(model, tokenizer):
    """Test medical vocabulary retention"""
    print("\n" + "=" * 70)
    print("Test 2: Medical Vocabulary Retention")
    print("=" * 70)

    # Load medical dictionary
    dict_file = Path("data/02_refined/01_medical_dict.json")
    with open(dict_file, 'r', encoding='utf-8') as f:
        medical_terms = json.load(f)

    # Test on sample terms
    test_terms = medical_terms[:50]  # First 50 terms

    model.eval()
    correct = 0

    with torch.no_grad():
        for entry in tqdm(test_terms, desc="Testing medical terms"):
            term = entry['term']
            definition = entry['definition']

            # Test if model can complete the definition
            prompt = f"<start_of_turn>user\nMeaning of word {term}:<end_of_turn>\n<start_of_turn>model\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check if key terms from definition appear in response
            key_terms = definition.split()[:5]  # First 5 words
            matches = sum(1 for term in key_terms if term.lower() in response.lower())

            if matches >= 2:  # At least 2/5 key terms match
                correct += 1

    accuracy = 100 * correct / len(test_terms)
    print(f"Medical Vocabulary Accuracy: {accuracy:.1f}% ({correct}/{len(test_terms)})")

    return accuracy


def test_mcq_accuracy(model, tokenizer):
    """Test MCQ accuracy on KorMedMCQA"""
    print("\n" + "=" * 70)
    print("Test 3: MCQ Accuracy (KorMedMCQA)")
    print("=" * 70)

    # Load test data
    test_file = Path("data/02_refined/02_kor_med_test/test.jsonl")
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    model.eval()
    correct = 0

    with torch.no_grad():
        for sample in tqdm(test_data[:100], desc="Testing MCQ"):  # First 100 samples
            question = sample['question']
            choices = [sample[c] for c in ['A', 'B', 'C', 'D', 'E']]
            answer = sample['answer']

            # Format prompt
            prompt = (
                "<start_of_turn>user\n"
                "정답 알파벳 하나만 답하세요.\n\n"
                f"{question}\n"
                f"A) {choices[0]}\n"
                f"B) {choices[1]}\n"
                f"C) {choices[2]}\n"
                f"D) {choices[3]}\n"
                f"E) {choices[4]}\n\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract predicted answer
            predicted = None
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if letter in response.split("<start_of_turn>model")[-1][:10]:
                    predicted = letter
                    break

            if predicted == answer:
                correct += 1

    accuracy = 100 * correct / min(100, len(test_data))
    print(f"MCQ Accuracy: {accuracy:.1f}% ({correct}/100)")

    return accuracy


def compare_models(single_lora_path, multi_lora_path, device="cuda:0"):
    """Compare single-LoRA vs multi-LoRA strategies"""
    print("=" * 70)
    print("Comparing LoRA Strategies")
    print("=" * 70)

    results = {}

    # Test Single-LoRA (Phase 1)
    if Path(single_lora_path).exists():
        print(f"\nTesting Single-LoRA: {single_lora_path}")
        print("-" * 70)

        model, tokenizer = load_model_and_tokenizer(single_lora_path, device)

        korean_ppl = test_korean_preservation(model, tokenizer)
        medical_acc = test_medical_vocabulary(model, tokenizer)
        # mcq_acc = test_mcq_accuracy(model, tokenizer)  # Skip if not Phase 2

        results['single_lora'] = {
            'korean_perplexity': korean_ppl,
            'medical_accuracy': medical_acc,
            # 'mcq_accuracy': mcq_acc
        }

        del model
        torch.cuda.empty_cache()

    # Test Multi-LoRA (Phase 1 Dual-LoRA)
    if Path(multi_lora_path).exists():
        print(f"\n\nTesting Multi-LoRA: {multi_lora_path}")
        print("-" * 70)

        model, tokenizer = load_model_and_tokenizer(multi_lora_path, device)

        korean_ppl = test_korean_preservation(model, tokenizer)
        medical_acc = test_medical_vocabulary(model, tokenizer)
        # mcq_acc = test_mcq_accuracy(model, tokenizer)

        results['multi_lora'] = {
            'korean_perplexity': korean_ppl,
            'medical_accuracy': medical_acc,
            # 'mcq_accuracy': mcq_acc
        }

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    if 'single_lora' in results and 'multi_lora' in results:
        single = results['single_lora']
        multi = results['multi_lora']

        print(f"\n{'Metric':<30} {'Single-LoRA':>15} {'Multi-LoRA':>15} {'Winner':>10}")
        print("-" * 72)

        # Korean perplexity (lower is better)
        korean_winner = "Multi" if multi['korean_perplexity'] < single['korean_perplexity'] else "Single"
        print(f"{'Korean Perplexity':<30} {single['korean_perplexity']:>15.2f} {multi['korean_perplexity']:>15.2f} {korean_winner:>10}")

        # Medical accuracy (higher is better)
        medical_winner = "Multi" if multi['medical_accuracy'] > single['medical_accuracy'] else "Single"
        print(f"{'Medical Accuracy (%)':<30} {single['medical_accuracy']:>15.1f} {multi['medical_accuracy']:>15.1f} {medical_winner:>10}")

        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)

        if korean_winner == "Multi" and multi['medical_accuracy'] >= single['medical_accuracy'] * 0.9:
            print("✅ Multi-LoRA strategy is BETTER!")
            print("   - Preserves Korean language better")
            print("   - Maintains medical vocabulary")
            print("   - Recommended for continued training")
        elif korean_winner == "Single" and single['medical_accuracy'] > multi['medical_accuracy'] * 1.1:
            print("⚠️  Single-LoRA has better medical accuracy but worse Korean")
            print("   - Trade-off: medical vs Korean preservation")
        else:
            print("📊 Results are mixed - both strategies have merits")

    # Save results
    output_file = Path("comparison_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare LoRA strategies")
    parser.add_argument("--single", default="model/01_trained/medgemma-4b",
                       help="Path to single-LoRA model (Phase 1)")
    parser.add_argument("--multi", default="model/01_dual_lora/medgemma-4b/final",
                       help="Path to multi-LoRA model (Phase 1 dual-LoRA)")
    parser.add_argument("--device", default="cuda:0", help="Device")

    args = parser.parse_args()

    compare_models(args.single, args.multi, args.device)


if __name__ == "__main__":
    main()
