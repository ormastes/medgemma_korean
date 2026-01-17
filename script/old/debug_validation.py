#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug validation - show actual model outputs"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

model_path = "model/02_mixed/medgemma-4b/final"
test_file = Path("data/02_refined/02_kor_med_test/test.jsonl")

# Load test data
test_data = []
with open(test_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:  # Only first 3 samples
            break
        test_data.append(json.loads(line))

print("Loading model...")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

# Resize and load adapter
print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, model_path)
model.eval()

print("\n" + "="*70)
print("TESTING 3 SAMPLES")
print("="*70)

for i, sample in enumerate(test_data):
    print(f"\n{'='*70}")
    print(f"Sample {i+1}/{len(test_data)}")
    print(f"{'='*70}")

    # Format prompt
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

    print(f"Question: {sample['question'][:100]}...")
    print(f"Expected answer: {sample['answer']}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Decode only new tokens
    new_tokens = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

    print(f"\nGenerated tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
    print(f"\nFull output (last 500 chars):\n{full_output[-500:]}")
    print(f"\nNew tokens only:\n{new_tokens}")
    print(f"\nNew tokens (no special):\n{tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)}")

    # Extract answer
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.strip()

    if '</reasoning>' in response:
        after_reasoning = response.split('</reasoning>')[-1].strip().upper()
        print(f"\nAfter </reasoning>: '{after_reasoning}'")
        for char in after_reasoning:
            if char in 'ABCDE':
                predicted = char
                break
        else:
            predicted = ""
    else:
        print(f"\nNo </reasoning> tag found")
        response_upper = response.upper()
        for char in response_upper:
            if char in 'ABCDE':
                predicted = char
                break
        else:
            predicted = ""

    print(f"\nExtracted answer: '{predicted}'")
    print(f"Correct: {predicted == sample['answer']}")

print("\n" + "="*70)
print("DEBUG COMPLETE")
print("="*70)
