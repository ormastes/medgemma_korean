#!/usr/bin/env python3
"""Test if base model can generate text"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("Testing base model generation...")

tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
model.eval()

prompt = """<start_of_turn>user
What is diabetes?<end_of_turn>
<start_of_turn>model
"""

print(f"Prompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

print(f"Input tokens: {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

new_tokens = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
print(f"\nGenerated ({outputs.shape[1] - inputs['input_ids'].shape[1]} tokens):")
print(new_tokens)
print(f"\nWithout special tokens:")
print(tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
