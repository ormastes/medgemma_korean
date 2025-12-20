"""GPU Setup Helper for Korean MedGemma Training"""

import os
import json
import torch

def setup_gpu(config_path="../config/gpu_config.json"):
    """Load GPU config and set as default"""

    # Load config if exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        gpu_idx = config.get("default_gpu_idx", 0)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    # Verify CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    device = torch.device("cuda:0")

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    return device

def get_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved,
        }
    return None

def print_memory_usage():
    """Print current GPU memory usage"""
    info = get_memory_info()
    if info:
        print(f"GPU Memory: {info['allocated_gb']:.2f} GB allocated, "
              f"{info['free_gb']:.2f} GB free / {info['total_gb']:.1f} GB total")

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory cache cleared")
