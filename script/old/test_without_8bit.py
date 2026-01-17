#!/usr/bin/env python3
"""Test without 8-bit quantization"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Testing WITHOUT 8-bit quantization...")

phase1_path = "model/01_mixed/medgemma-4b/final"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(phase1_path, trust_remote_code=True)
print(f"Tokenizer vocab: {len(tokenizer)}")

# Load base model WITHOUT quantization
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    device_map="cpu",  # Load to CPU first (less VRAM)
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print(f"Base model embeddings: {model.get_input_embeddings().weight.shape[0]}")

# Resize embeddings
print(f"Resizing to {len(tokenizer)}...")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

# Load adapter
print("Loading adapter...")
model = PeftModel.from_pretrained(model, phase1_path)
model = model.to("cuda:0")  # Move to GPU
model.eval()

# Test forward pass
text = "<start_of_turn>user\nMeaning of word 고혈압:<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

print(f"\nTesting forward pass...")

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
    print(f"  {i+1}. Token {token_id.item()}: '{token}' (score: {score_val:.4f})")

if has_nan:
    print("\n⚠️  NaN detected in logits!")
else:
    print("\n✓ No NaN - logits are valid!")

    # Try generation if no NaN
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

    generated = tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Generated:\n{generated}")
