#!/usr/bin/env python3
"""
Enhance training data using LLM (DeepSeek or similar).

Enhancements:
1. Generate reasoning chains for MCQ answers
2. Translate English medical terms to Korean
3. Add explanations to medical dictionary entries

IMPORTANT: This script requires a local LLM running on GPU.
           Tested with DeepSeek-7B-Chat on TITAN RTX (24GB).

Usage:
    python script/prepare/refine/enhance_with_llm.py --task mcq-reasoning
    python script/prepare/refine/enhance_with_llm.py --task dict-explain
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers package not installed")
    print("Run: pip install transformers torch")
    exit(1)

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "02_refined"

# LLM configuration
DEFAULT_MODEL = "deepseek-ai/deepseek-llm-7b-chat"
DEFAULT_DEVICE = "cuda:1"  # TITAN RTX


class LLMEnhancer:
    """LLM-based data enhancer."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE):
        self.device = device
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response
        response = response[len(prompt):].strip()
        return response


def enhance_mcq_reasoning(enhancer: LLMEnhancer, input_file: Path, output_file: Path, max_samples: int = None):
    """Add reasoning chains to MCQ answers."""
    print(f"\nEnhancing MCQ with reasoning...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    if max_samples:
        samples = samples[:max_samples]

    enhanced = []
    for sample in tqdm(samples, desc="Generating reasoning"):
        prompt = f"""다음 의료 객관식 문제의 정답에 대한 추론 과정을 한국어로 설명하세요.

문제: {sample['question']}
A) {sample['A']}
B) {sample['B']}
C) {sample['C']}
D) {sample['D']}
E) {sample['E']}

정답: {sample['answer']}

추론 과정:"""

        try:
            reasoning = enhancer.generate(prompt, max_length=256)
            sample['reasoning'] = reasoning
        except Exception as e:
            print(f"Error: {e}")
            sample['reasoning'] = ""

        enhanced.append(sample)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in enhanced:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Enhanced {len(enhanced)} samples")


def enhance_dict_explain(enhancer: LLMEnhancer, input_file: Path, output_file: Path, max_samples: int = None):
    """Add Korean explanations to medical dictionary entries."""
    print(f"\nEnhancing medical dictionary with explanations...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    if max_samples:
        entries = entries[:max_samples]

    enhanced = []
    for entry in tqdm(entries, desc="Generating explanations"):
        prompt = f"""다음 의학 용어를 한국어로 쉽게 설명하세요.

용어: {entry['term']}
영문: {entry['definition']}

한국어 설명:"""

        try:
            explanation = enhancer.generate(prompt, max_length=128)
            entry['korean_explanation'] = explanation
        except Exception as e:
            print(f"Error: {e}")
            entry['korean_explanation'] = ""

        enhanced.append(entry)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced, f, ensure_ascii=False, indent=2)

    print(f"Enhanced {len(enhanced)} entries")


def main():
    parser = argparse.ArgumentParser(description="Enhance data with LLM")
    parser.add_argument("--task", choices=["mcq-reasoning", "dict-explain", "all"],
                       default="mcq-reasoning", help="Enhancement task")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help="LLM model to use")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                       help="Device for LLM")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to process (for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("LLM Data Enhancement")
    print("=" * 60)

    # Initialize enhancer
    enhancer = LLMEnhancer(args.model, args.device)

    if args.task in ["mcq-reasoning", "all"]:
        input_file = DATA_DIR / "02_kor_med_test" / "train.jsonl"
        output_file = DATA_DIR / "02_kor_med_test" / "train_with_reasoning.jsonl"

        if input_file.exists():
            enhance_mcq_reasoning(enhancer, input_file, output_file, args.max_samples)
        else:
            print(f"File not found: {input_file}")

    if args.task in ["dict-explain", "all"]:
        input_file = DATA_DIR / "01_medical_dict.json"
        output_file = DATA_DIR / "01_medical_dict_explained.json"

        if input_file.exists():
            enhance_dict_explain(enhancer, input_file, output_file, args.max_samples)
        else:
            print(f"File not found: {input_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
