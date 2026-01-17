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
        "batch": 2,          # 32-bit training on A6000 (48GB)
        "grad_accum": 16,    # Effective batch size = 32
        "max_length": 512,   # Default, overridden per training type
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 64,
        "lora_alpha": 128,
        "lr": 1e-4,
        "batch": 2,
        "grad_accum": 16,
        "max_length": 512,  # Default, overridden per training type
    }
}

# Max lengths per training type
# train_02: FULL prompt max=633, NORMAL max=485, + ~300 response = ~933 max
MAX_LENGTHS = {
    "medgemma-4b": {
        "train_00": 512,   # Plain text
        "train_01": 256,   # Dictionary entries (short)
        "train_02": 1024,  # MCQ with reasoning (FULL: 633+300, NORMAL: 485+300)
    },
    "medgemma-27b": {
        "train_00": 512,   # Plain text
        "train_01": 256,   # Dictionary entries (short)
        "train_02": 1024,  # MCQ with reasoning (memory optimized)
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

# Memory optimization settings per model
# Measured at max_len=1024 on RTX A6000 (47.4 GB)
MEMORY_CONFIGS = {
    "medgemma-4b": {
        "use_gradient_checkpointing": True,   # Required with extended tokenizer embeddings
        "train_embeddings": True,              # Train extended Korean embeddings
        "peak_memory_gb": 44.0,               # With extended embeddings (1.59B trainable)
        "remaining_gb": 3.4,                  # Tight, use gradient checkpointing
    },
    "medgemma-27b": {
        "use_gradient_checkpointing": True,   # Required for training
        "train_embeddings": False,             # OOM with embedding training
        "peak_memory_gb": 37.0,               # Measured at 1024 seq
        "remaining_gb": 10.4,                 # 22.0% headroom
    }
}

# Train 02 specific configs
TRAIN_02_CONFIGS = {
    "medgemma-4b": {
        "full_samples": 500,          # Samples for FULL mode
        "reasoning_threshold": 0.7,   # Switch to NORMAL when score >= this
        "eval_interval": 50,          # Evaluate every N steps
        "eval_samples": 10,           # Samples per evaluation
    },
    "medgemma-27b": {
        "full_samples": 300,          # Fewer samples for larger model
        "reasoning_threshold": 0.7,
        "eval_interval": 30,          # More frequent eval for larger model
        "eval_samples": 10,
    }
}

# Train 01 with 00 monitor configs
TRAIN_01_MONITOR_CONFIGS = {
    "medgemma-4b": {
        "upper_bound": 3.0,       # Switch to train_00 when loss > this
        "lower_bound": 2.0,       # Switch back when loss < this
        "check_interval": 100,    # Check every N steps
        "recovery_steps": 50,     # Train_00 steps per recovery
    },
    "medgemma-27b": {
        "upper_bound": 3.0,
        "lower_bound": 2.0,
        "check_interval": 50,     # More frequent checks
        "recovery_steps": 30,
    }
}
