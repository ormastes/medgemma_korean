#!/usr/bin/env python3
"""Test CORRECT loading without resize"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("Testing CORRECT loading (no resize before adapter load)...")

phase1_path = "model/01_mixed/medgemma-4b/final"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(phase1_path, trust_remote_code=True)
print(f"Tokenizer vocab: {len(tokenizer)}")

# Load base model - DO NOT resize yet!
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

print(f"Base model embeddings: {model.get_input_embeddings().weight.shape[0]}")

# Load adapter - this should load the extended embeddings
print("Loading adapter with extended embeddings...")
model = PeftModel.from_pretrained(model, phase1_path)
model.eval()

print(f"After adapter load: {model.get_input_embeddings().weight.shape[0]}")

# Test forward pass
text = "<start_of_turn>user\n고혈압<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

print(f"\nInput: {text}")
print(f"Input IDs: {inputs['input_ids'][0].tolist()}")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
next_token_logits = logits[0, -1, :]

# Get top 10 predictions
top_k = torch.topk(next_token_logits, 10)
print(f"\nTop 10 next token predictions:")
has_nan = False
for i, (score, token_id) in enumerate(zip(top_k.values, top_k.indices)):
    token = tokenizer.decode([token_id.item()])
    score_val = score.item()
    if torch.isnan(score):
        has_nan = True
    print(f"  {i+1}. Token {token_id.item()}: '{token}' (score: {score_val:.2f})")

if has_nan:
    print("\n⚠️  NaN detected in logits!")
else:
    print("\n✓ No NaN - logits are valid!")

# Try generation
print("\n" + "="*70)
print("Testing generation...")
print("="*70)

with torch.no_grad():
    gen_outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
print(f"Generated:\n{generated}")
print(f"\nWithout special tokens:\n{tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)}")
