#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared training configurations for all model types
"""

MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 64,
        "lora_alpha": 128,
        "lr": 1e-4,
        "batch": 4,
        "grad_accum": 8,
        "max_length": 512,
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 64,
        "lora_alpha": 128,
        "lr": 1e-4,
        "batch": 2,
        "grad_accum": 16,
        "max_length": 512,
    }
}

# LoRA target modules (common for all models)
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj", 
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

# Training defaults
TRAINING_DEFAULTS = {
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "optim": "paged_adamw_8bit",
    "max_grad_norm": 0.3,
    "lora_dropout": 0.05,
}
