#!/usr/bin/env python3
"""
GPU Memory Auto-Configuration

Detects GPU memory and returns optimal training config.
Used by all training scripts.
"""

import torch


def get_gpu_memory_gb() -> float:
    """Get total GPU memory in GB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0


def get_free_gpu_memory_gb() -> float:
    """Get free GPU memory in GB"""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        return free / (1024**3)
    return 0


def get_optimal_config(model_name: str, verbose: bool = True) -> dict:
    """
    Get optimal training config based on GPU memory.

    Returns dict with:
        - path: model path
        - lora_r, lora_alpha: LoRA config
        - lr: learning rate
        - batch: per-device batch size
        - grad_accum: gradient accumulation steps
        - max_length: max sequence length
        - grad_ckpt: use gradient checkpointing
    """
    gpu_mem = get_gpu_memory_gb()

    if verbose:
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        print(f"Total memory: {gpu_mem:.1f} GB")

    # Base configs
    base_configs = {
        "medgemma-4b": {
            "path": "google/medgemma-4b-it",
            "lora_r": 64,
            "lora_alpha": 128,
            "lr": 5e-5
        },
        "medgemma-27b": {
            "path": "google/medgemma-27b-text-it",
            "lora_r": 128,
            "lora_alpha": 256,
            "lr": 2e-5
        },
        "stage5": {
            "path": "models/staged_training/stage5_harmonization",
            "lora_r": 64,
            "lora_alpha": 128,
            "lr": 1e-4
        },
        "stage6": {
            "path": "models/staged_training/stage6_hybrid_expansion",
            "lora_r": 64,
            "lora_alpha": 128,
            "lr": 1e-4
        }
    }

    cfg = base_configs.get(model_name, base_configs["medgemma-4b"]).copy()

    # Memory-based config table
    # Format: (min_gb, batch, grad_accum, max_length, grad_ckpt)
    if "27b" in model_name:
        memory_configs = [
            (80, 2, 16, 1024, True),   # A100 80GB
            (48, 1, 32, 512, True),    # A6000 48GB
            (24, 1, 64, 256, True),    # 3090/4090 24GB
            (16, 1, 128, 256, True),   # V100 16GB
            (0, 1, 256, 128, True),    # Fallback
        ]
    else:  # 4B models
        memory_configs = [
            (48, 4, 8, 1024, False),   # A6000 48GB
            (24, 2, 16, 1024, False),  # 3090/4090 24GB
            (16, 1, 32, 512, True),    # V100/4080 16GB
            (8, 1, 64, 512, True),     # 3070/4070 8GB
            (0, 1, 128, 256, True),    # Fallback
        ]

    # Find matching config
    for min_gb, batch, grad_accum, max_length, grad_ckpt in memory_configs:
        if gpu_mem >= min_gb:
            cfg.update({
                "batch": batch,
                "grad_accum": grad_accum,
                "max_length": max_length,
                "grad_ckpt": grad_ckpt
            })
            break

    effective_batch = cfg["batch"] * cfg["grad_accum"]

    if verbose:
        print(f"Config: batch={cfg['batch']}, grad_accum={cfg['grad_accum']}, "
              f"effective={effective_batch}, max_len={cfg['max_length']}, "
              f"grad_ckpt={cfg['grad_ckpt']}")

    return cfg


def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")


# Quick test
if __name__ == "__main__":
    print("Testing GPU config detection...\n")

    for model in ["medgemma-4b", "medgemma-27b"]:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")
        cfg = get_optimal_config(model)
        print(f"\nFull config: {cfg}")
