#!/usr/bin/env python3
"""Quick check to see what model outputs during evaluation."""
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_PATH = Path(__file__).parent.parent / "model" / "02_alternating" / "medgemma-4b" / "latest"
TEST_FILE = Path(__file__).parent.parent / "data" / "02_refined" / "02_kor_med_test" / "test.jsonl"

def main():
    print(f"Loading model from: {MODEL_PATH}")

    peft_config = PeftConfig.from_pretrained(MODEL_PATH)

    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()

    # Load test samples
    with open(TEST_FILE, 'r') as f:
        test_data = [json.loads(line) for line in f][:3]  # Just 3 samples

    print("\n" + "="*70)
    print("TESTING MODEL OUTPUTS")
    print("="*70)

    for i, sample in enumerate(test_data):
        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Expected: {sample['answer']}")

        # Test with training-style prompt
        prompt = f"""<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

{sample['question']}

A) {sample['A']}
B) {sample['B']}
C) {sample['C']}
D) {sample['D']}
E) {sample['E']}
<end_of_turn>
<start_of_turn>model
"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Model output:\n{response[:500]}")
        print("-"*50)

if __name__ == "__main__":
    main()
