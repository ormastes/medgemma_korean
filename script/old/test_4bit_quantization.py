#!/usr/bin/env python3
"""Test if 4-bit quantization works with extended embeddings"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("Testing 4-bit quantization with extended embeddings...")

phase1_path = "model/01_mixed/medgemma-4b/final"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(phase1_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer vocab: {len(tokenizer)}")

# Load base model with 4-bit quantization
print("Loading base model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

print(f"Base model embeddings: {model.get_input_embeddings().weight.shape[0]}")

# Resize embeddings
print(f"Resizing to {len(tokenizer)}...")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

# Load adapter
print("Loading adapter...")
model = PeftModel.from_pretrained(model, phase1_path)
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
    print("\n❌ 4-bit quantization FAILED - NaN detected in logits!")
    print("Will need to use bfloat16 (no quantization) for training.")
    exit(1)
else:
    print("\n✓ 4-bit quantization WORKS - No NaN!")

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

    generated = tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Generated:\n{generated}")

    if generated.strip() and generated.strip() != "" and "<pad>" not in generated:
        print("\n✅ 4-BIT QUANTIZATION WORKS WITH EXTENDED EMBEDDINGS!")
        print("Can use 4-bit for retraining to save VRAM.")
        exit(0)
    else:
        print("\n❌ Generated output is empty or invalid")
        exit(1)
