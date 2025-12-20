#!/usr/bin/env python3
"""
Find optimal training configuration to maximize GPU utilization (>40GB on A6000)
"""

import torch
import gc
import json
from pathlib import Path

# Test configurations
CONFIGS = [
    # batch, grad_accum, max_len, lora_r, description
    {"batch": 1, "grad_accum": 32, "max_len": 512, "lora_r": 64, "desc": "baseline"},
    {"batch": 2, "grad_accum": 16, "max_len": 512, "lora_r": 64, "desc": "batch=2"},
    {"batch": 4, "grad_accum": 8, "max_len": 512, "lora_r": 64, "desc": "batch=4"},
    {"batch": 8, "grad_accum": 4, "max_len": 512, "lora_r": 64, "desc": "batch=8"},
    {"batch": 4, "grad_accum": 8, "max_len": 1024, "lora_r": 64, "desc": "batch=4,len=1024"},
    {"batch": 8, "grad_accum": 4, "max_len": 1024, "lora_r": 64, "desc": "batch=8,len=1024"},
    {"batch": 4, "grad_accum": 8, "max_len": 512, "lora_r": 128, "desc": "batch=4,lora=128"},
    {"batch": 8, "grad_accum": 4, "max_len": 512, "lora_r": 128, "desc": "batch=8,lora=128"},
    {"batch": 16, "grad_accum": 2, "max_len": 512, "lora_r": 64, "desc": "batch=16"},
    {"batch": 12, "grad_accum": 2, "max_len": 512, "lora_r": 64, "desc": "batch=12"},
    {"batch": 8, "grad_accum": 4, "max_len": 768, "lora_r": 128, "desc": "batch=8,len=768,lora=128"},
]

def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return allocated, reserved, total
    return 0, 0, 0

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_config(cfg):
    """Test a single configuration and return memory usage"""
    clear_gpu()

    print(f"\n{'='*60}")
    print(f"Testing: {cfg['desc']}")
    print(f"  batch={cfg['batch']}, grad_accum={cfg['grad_accum']}, max_len={cfg['max_len']}, lora_r={cfg['lora_r']}")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Load model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            "google/medgemma-27b-text-it",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )

        tokenizer = AutoTokenizer.from_pretrained("google/medgemma-27b-text-it", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        lora_config = LoraConfig(
            r=cfg['lora_r'],
            lora_alpha=cfg['lora_r'] * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        alloc1, res1, total = get_gpu_memory()
        print(f"After model load: {alloc1:.1f}GB allocated, {res1:.1f}GB reserved")

        # Create dummy batch
        dummy_text = "한국어 의료 테스트 " * (cfg['max_len'] // 10)
        inputs = tokenizer(
            [dummy_text] * cfg['batch'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg['max_len']
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        alloc2, res2, _ = get_gpu_memory()
        print(f"After batch creation: {alloc2:.1f}GB allocated, {res2:.1f}GB reserved")

        # Forward pass
        model.train()
        outputs = model(**inputs)
        loss = outputs.loss

        alloc3, res3, _ = get_gpu_memory()
        print(f"After forward: {alloc3:.1f}GB allocated, {res3:.1f}GB reserved")

        # Backward pass
        loss.backward()

        alloc4, res4, _ = get_gpu_memory()
        print(f"After backward: {alloc4:.1f}GB allocated, {res4:.1f}GB reserved")

        result = {
            "config": cfg,
            "success": True,
            "memory_after_load": res1,
            "memory_after_forward": res3,
            "memory_after_backward": res4,
            "peak_memory": res4,
            "total_gpu": total
        }

        print(f"\n>>> PEAK MEMORY: {res4:.1f}GB / {total:.1f}GB ({100*res4/total:.0f}%)")

        # Cleanup
        del model, inputs, outputs, loss
        clear_gpu()

        return result

    except Exception as e:
        print(f"FAILED: {e}")
        clear_gpu()
        return {
            "config": cfg,
            "success": False,
            "error": str(e)
        }

def main():
    print("="*60)
    print("Finding Optimal GPU Configuration")
    print("Target: >40GB usage on A6000 (48GB)")
    print("="*60)

    _, _, total = get_gpu_memory()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total:.1f}GB")

    results = []

    for cfg in CONFIGS:
        result = test_config(cfg)
        results.append(result)

        # Save intermediate results
        with open("config_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Config':<35} {'Peak GB':>10} {'Status':>10}")
    print("-"*60)

    successful = []
    for r in results:
        if r["success"]:
            peak = r["peak_memory"]
            status = "OK" if peak < 45 else "TIGHT"
            print(f"{r['config']['desc']:<35} {peak:>10.1f} {status:>10}")
            successful.append(r)
        else:
            print(f"{r['config']['desc']:<35} {'FAIL':>10} {'OOM':>10}")

    # Find best config (closest to 40-45GB)
    print("\n" + "="*60)
    print("RECOMMENDED CONFIGS (>40GB, <46GB)")
    print("="*60)

    good_configs = [r for r in successful if 40 < r["peak_memory"] < 46]
    if good_configs:
        for r in sorted(good_configs, key=lambda x: x["peak_memory"], reverse=True):
            c = r["config"]
            print(f"  batch={c['batch']}, grad_accum={c['grad_accum']}, max_len={c['max_len']}, lora_r={c['lora_r']}")
            print(f"    -> Peak: {r['peak_memory']:.1f}GB")
    else:
        print("No config found in target range. Best options:")
        for r in sorted(successful, key=lambda x: x["peak_memory"], reverse=True)[:3]:
            c = r["config"]
            print(f"  {c['desc']}: {r['peak_memory']:.1f}GB")

if __name__ == "__main__":
    main()
