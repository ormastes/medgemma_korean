#!/usr/bin/env python3
"""Test Phase 1 model generation"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("Testing Phase 1 model...")

phase1_path = "model/01_mixed/medgemma-4b/final"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(phase1_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer vocab size: {len(tokenizer)}")

# Load base model
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

# Resize embeddings
print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))

# Load Phase 1 adapter
model = PeftModel.from_pretrained(model, phase1_path)
model.eval()

# Test medical dictionary format
prompt = """<start_of_turn>user
Meaning of word 고혈압:<end_of_turn>
<start_of_turn>model
"""

print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

print(f"Input tokens: {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

new_tokens = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
print(f"\nGenerated ({outputs.shape[1] - inputs['input_ids'].shape[1]} tokens):")
print(new_tokens)
print(f"\nWithout special tokens:")
print(tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
