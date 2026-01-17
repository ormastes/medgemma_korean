#!/usr/bin/env python3
"""
GPU Memory Checker
Check available GPU memory before training
"""

import subprocess
import sys


def get_gpu_memory():
    """Get GPU memory info for all GPUs"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'total_mb': int(parts[2]),
                    'used_mb': int(parts[3]),
                    'free_mb': int(parts[4])
                })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []


def recommend_gpu(min_memory_gb=10):
    """Recommend best GPU for training"""
    gpus = get_gpu_memory()
    
    if not gpus:
        print("❌ No GPUs found!")
        return None
    
    print("=" * 70)
    print("GPU Memory Status")
    print("=" * 70)
    
    best_gpu = None
    max_free = 0
    
    for gpu in gpus:
        free_gb = gpu['free_mb'] / 1024
        total_gb = gpu['total_mb'] / 1024
        used_gb = gpu['used_mb'] / 1024
        usage_pct = (gpu['used_mb'] / gpu['total_mb']) * 100
        
        status = "✅" if free_gb >= min_memory_gb else "⚠️"
        
        print(f"\n{status} GPU {gpu['index']}: {gpu['name']}")
        print(f"   Total:  {total_gb:.1f} GB")
        print(f"   Used:   {used_gb:.1f} GB ({usage_pct:.1f}%)")
        print(f"   Free:   {free_gb:.1f} GB")
        
        if free_gb > max_free:
            max_free = free_gb
            best_gpu = gpu['index']
    
    print("\n" + "=" * 70)
    
    if best_gpu is not None and max_free >= min_memory_gb:
        print(f"✅ Recommended: Use GPU {best_gpu} (cuda:{best_gpu})")
        print(f"   Free memory: {max_free:.1f} GB")
        return best_gpu
    else:
        print(f"⚠️  Warning: No GPU with {min_memory_gb}GB+ free memory")
        if best_gpu is not None:
            print(f"   Best available: GPU {best_gpu} with {max_free:.1f} GB free")
        return best_gpu


def check_model_requirements(model_name):
    """Estimate memory requirements for different models"""
    requirements = {
        'medgemma-4b': {
            'model_8bit': 5.0,  # GB for model in 8-bit
            'training_overhead': 3.0,  # Optimizer states, gradients
            'batch_overhead': 2.0,  # Per-batch activation memory
            'total_min': 10.0
        },
        'medgemma-27b': {
            'model_8bit': 18.0,
            'training_overhead': 8.0,
            'batch_overhead': 4.0,
            'total_min': 30.0
        }
    }
    
    if model_name in requirements:
        req = requirements[model_name]
        print("\n" + "=" * 70)
        print(f"Memory Requirements: {model_name}")
        print("=" * 70)
        print(f"Model (8-bit):        ~{req['model_8bit']:.1f} GB")
        print(f"Training overhead:    ~{req['training_overhead']:.1f} GB")
        print(f"Batch overhead:       ~{req['batch_overhead']:.1f} GB")
        print(f"{'='*70}")
        print(f"Minimum required:     ~{req['total_min']:.1f} GB")
        return req['total_min']
    
    return 10.0  # Default


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check GPU memory availability")
    parser.add_argument("--model", default="medgemma-4b", 
                       choices=["medgemma-4b", "medgemma-27b"],
                       help="Model to check requirements for")
    parser.add_argument("--min-memory", type=float, default=None,
                       help="Minimum memory in GB (default: auto based on model)")
    args = parser.parse_args()
    
    # Get model requirements
    min_memory = args.min_memory if args.min_memory else check_model_requirements(args.model)
    
    # Recommend GPU
    best_gpu = recommend_gpu(min_memory_gb=min_memory)
    
    if best_gpu is None:
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Usage:")
    print("=" * 70)
    print(f"python train_script.py --model {args.model} --device cuda:{best_gpu}")
    print("=" * 70)
