#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check GPU Memory Usage for Model Training

Measures:
1. Model loading memory (8-bit quantized)
2. Forward pass with max sequence length
3. Estimated training memory

Usage:
    python check_memory.py --model medgemma-4b --device cuda:0
    python check_memory.py --model medgemma-27b --device cuda:0
"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS, MEMORY_CONFIGS


def get_gpu_memory(device_id: int = 0):
    """Get current and peak GPU memory usage."""
    if not torch.cuda.is_available():
        return 0, 0, 0

    current = torch.cuda.memory_allocated(device_id) / 1024**3
    peak = torch.cuda.max_memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    return current, peak, reserved


def clear_gpu(device_id: int = 0):
    """Clear GPU memory and reset stats."""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_id)


def check_memory_for_model(model_name: str, device: str, max_length: int = 2048):
    """Check memory usage for a specific model."""
    device_id = int(device.split(':')[1]) if ':' in device else 0

    print(f"\n{'='*60}")
    print(f"Memory Check: {model_name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Max sequence length: {max_length}")

    # Check available GPU memory
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_id)
        total_mem = props.total_memory / 1024**3
        print(f"GPU: {props.name}")
        print(f"Total GPU memory: {total_mem:.1f} GB")
    else:
        print("WARNING: CUDA not available!")
        return

    # Get paths
    base_dir = Path(__file__).parent.parent
    raw_lora_dir = base_dir / "model" / "raw_lora_added" / model_name
    tokenizer_dir = base_dir / "model" / "tokenizer" / "extended_tokenizer"

    print(f"\n--- Loading Model ---")
    clear_gpu(device_id)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Check if extended tokenizer is available
    use_extended = tokenizer_dir.exists()
    if use_extended:
        print(f"Using extended tokenizer: {tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        new_vocab_size = len(tokenizer)
        print(f"Vocab size: {new_vocab_size}")
    else:
        cfg = MODEL_CONFIGS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)
        new_vocab_size = None

    # Load model with 8-bit quantization
    cfg = MODEL_CONFIGS[model_name]
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False
    )

    print(f"Loading from: {cfg['path']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg['path'],
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Resize embeddings if extended tokenizer
    if new_vocab_size is not None:
        old_vocab = model.get_input_embeddings().weight.shape[0]
        if new_vocab_size > old_vocab:
            print(f"Resizing embeddings: {old_vocab} -> {new_vocab_size}")
            model.resize_token_embeddings(new_vocab_size)

    current, peak, reserved = get_gpu_memory(device_id)
    print(f"\nAfter loading model:")
    print(f"  Current: {current:.2f} GB")
    print(f"  Peak:    {peak:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")

    # Get memory config for this model
    mem_cfg = MEMORY_CONFIGS.get(model_name, {})
    use_gradient_checkpointing = mem_cfg.get('use_gradient_checkpointing', False)
    train_embeddings = mem_cfg.get('train_embeddings', True)

    # Apply gradient checkpointing if configured
    if use_gradient_checkpointing:
        print(f"Enabling gradient checkpointing (memory optimization for {model_name})")
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Add LoRA adapter
    print(f"\n--- Adding LoRA Adapter ---")
    from peft import LoraConfig, get_peft_model
    from training_config import LORA_TARGET_MODULES, TRAINING_DEFAULTS

    lora_r = cfg['lora_r']
    lora_alpha = cfg['lora_alpha']

    # Only include embeddings if configured (4b: yes, 27b: no due to memory)
    modules_to_save = ['embed_tokens', 'lm_head'] if (use_extended and train_embeddings) else None
    if not train_embeddings:
        print(f"  Embedding training: DISABLED (memory optimization for {model_name})")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=TRAINING_DEFAULTS['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    current, peak, reserved = get_gpu_memory(device_id)
    print(f"\nAfter adding LoRA:")
    print(f"  Current: {current:.2f} GB")
    print(f"  Peak:    {peak:.2f} GB")

    # Forward pass with max length
    print(f"\n--- Forward Pass (seq_len={max_length}) ---")

    try:
        # Create dummy input
        input_ids = torch.randint(0, len(tokenizer), (1, max_length), device=device)
        attention_mask = torch.ones_like(input_ids)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        current, peak, reserved = get_gpu_memory(device_id)
        print(f"After forward pass:")
        print(f"  Current: {current:.2f} GB")
        print(f"  Peak:    {peak:.2f} GB")

        forward_ok = True
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ❌ OOM at seq_len={max_length}")
            forward_ok = False
        else:
            raise e

    # Training simulation (backward pass)
    print(f"\n--- Training Simulation ---")
    clear_gpu(device_id)

    try:
        model.train()
        input_ids = torch.randint(0, len(tokenizer), (1, max_length), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        current, peak, reserved = get_gpu_memory(device_id)
        print(f"After backward pass:")
        print(f"  Current: {current:.2f} GB")
        print(f"  Peak:    {peak:.2f} GB")

        train_ok = True
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ❌ OOM during training at seq_len={max_length}")
            train_ok = False
        else:
            raise e

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")

    final_current, final_peak, _ = get_gpu_memory(device_id)
    remaining = total_mem - final_peak

    print(f"\nPeak memory used: {final_peak:.2f} GB / {total_mem:.1f} GB")
    print(f"Remaining: {remaining:.2f} GB ({remaining/total_mem*100:.1f}%)")

    if remaining > 10:
        print(f"\n✓ Good headroom - can increase batch size or sequence length")
    elif remaining > 5:
        print(f"\n✓ Adequate headroom - training should work")
    elif remaining > 2:
        print(f"\n⚠️ Low headroom - may need to reduce batch size")
    else:
        print(f"\n❌ Very low headroom - reduce max_length or batch size")

    print(f"\nForward pass (seq_len={max_length}): {'✓ OK' if forward_ok else '❌ FAILED'}")
    print(f"Training (seq_len={max_length}):      {'✓ OK' if train_ok else '❌ FAILED'}")

    # Cleanup
    del model
    clear_gpu(device_id)

    return {
        'model': model_name,
        'peak_memory_gb': final_peak,
        'total_memory_gb': total_mem,
        'remaining_gb': remaining,
        'forward_ok': forward_ok,
        'train_ok': train_ok
    }


def main():
    parser = argparse.ArgumentParser(description="Check GPU memory usage for models")
    parser.add_argument("--model", default="medgemma-4b",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to check")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--max-length", type=int, default=1024,
                       help="Max sequence length to test (default: 1024 for train_02)")
    parser.add_argument("--all", action="store_true",
                       help="Check all models")
    args = parser.parse_args()

    results = []

    if args.all:
        for model_name in MODEL_CONFIGS.keys():
            try:
                result = check_memory_for_model(model_name, args.device, args.max_length)
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error checking {model_name}: {e}")
    else:
        result = check_memory_for_model(args.model, args.device, args.max_length)
        results.append(result)

    # Final summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"\n{'Model':<15} {'Peak':<10} {'Remaining':<12} {'Status'}")
        print("-" * 50)
        for r in results:
            status = "✓ OK" if r['train_ok'] else "❌ OOM"
            print(f"{r['model']:<15} {r['peak_memory_gb']:.1f} GB    "
                  f"{r['remaining_gb']:.1f} GB      {status}")


if __name__ == "__main__":
    main()
