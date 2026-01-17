#!/usr/bin/env python3
"""Test forward pass without generation"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("Testing forward pass...")

phase1_path = "model/01_mixed/medgemma-4b/final"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(phase1_path, trust_remote_code=True)
print(f"Tokenizer vocab: {len(tokenizer)}")

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
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, phase1_path)
model.eval()

# Simple text
text = "<start_of_turn>user\n안녕하세요<end_of_turn>\n<start_of_turn>model\n"
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
for i, (score, token_id) in enumerate(zip(top_k.values, top_k.indices)):
    token = tokenizer.decode([token_id.item()])
    print(f"  {i+1}. Token {token_id.item()}: '{token}' (score: {score.item():.2f})")

# Check if pad token is the top prediction
if top_k.indices[0].item() == tokenizer.pad_token_id:
    print("\n⚠️  WARNING: Pad token is the top prediction!")
else:
    print(f"\n✓ Top prediction is not pad token")
