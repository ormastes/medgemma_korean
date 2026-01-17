#!/usr/bin/env python3
"""Check if adapter embeddings contain NaN"""

import torch
from safetensors import safe_open

adapter_path = "model/01_mixed/medgemma-4b/final/adapter_model.safetensors"

print("Checking adapter embeddings for NaN...")

with safe_open(adapter_path, framework='pt') as f:
    # Check embed_tokens
    embed_key = "base_model.model.model.language_model.embed_tokens.weight"
    if embed_key in f.keys():
        embeddings = f.get_tensor(embed_key)
        print(f"\nEmbed tokens shape: {embeddings.shape}")
        print(f"Embed tokens dtype: {embeddings.dtype}")

        has_nan = torch.isnan(embeddings).any()
        has_inf = torch.isinf(embeddings).any()

        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")

        if has_nan:
            nan_count = torch.isnan(embeddings).sum().item()
            print(f"Number of NaN values: {nan_count} / {embeddings.numel()} ({100*nan_count/embeddings.numel():.2f}%)")

        # Check statistics
        valid_values = embeddings[~torch.isnan(embeddings)]
        if len(valid_values) > 0:
            print(f"\nValid values statistics:")
            print(f"  Min: {valid_values.min().item():.6f}")
            print(f"  Max: {valid_values.max().item():.6f}")
            print(f"  Mean: {valid_values.mean().item():.6f}")
            print(f"  Std: {valid_values.std().item():.6f}")

    # Check lm_head
    lm_head_key = "base_model.model.lm_head.weight"
    if lm_head_key in f.keys():
        lm_head = f.get_tensor(lm_head_key)
        print(f"\nLM head shape: {lm_head.shape}")
        print(f"LM head dtype: {lm_head.dtype}")

        has_nan = torch.isnan(lm_head).any()
        has_inf = torch.isinf(lm_head).any()

        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")

        if has_nan:
            nan_count = torch.isnan(lm_head).sum().item()
            print(f"Number of NaN values: {nan_count} / {lm_head.numel()} ({100*nan_count/lm_head.numel():.2f}%)")
