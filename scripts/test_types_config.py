#!/usr/bin/env python3
"""
Test each type trains well with optimized config on A6000 only
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # A6000 only

import torch
import gc
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "reviewed"

# Optimized config for A6000 (44.5GB, 94% utilization)
CONFIG = {
    "batch": 8,
    "grad_accum": 4,
    "max_length": 512,
    "lora_r": 128,
    "lora_alpha": 256,
    "lr": 5e-5
}

TYPE_ORDER = ["type1_text", "type2_text_reasoning", "type3_word", "type4_word_reasoning"]


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


def load_type_samples(type_name: str, n_samples: int = 100):
    """Load a few samples from each type"""
    train_path = DATA_DIR / type_name / "train" / "data.jsonl"
    samples = []
    if train_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n_samples:
                    break
                samples.append(json.loads(line))
    return samples


def test_type(type_name: str, model, tokenizer, cfg):
    """Test training on one type with a few steps"""
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print(f"\n{'='*60}")
    print(f"Testing: {type_name}")
    print("="*60)

    samples = load_type_samples(type_name, n_samples=100)
    if not samples:
        print(f"  No data for {type_name}")
        return False, 0

    print(f"  Loaded {len(samples)} samples")

    # Check sample format
    sample = samples[0]
    print(f"  Sample keys: {list(sample.keys())}")
    if 'text' in sample:
        print(f"  Text preview: {sample['text'][:100]}...")

    train_data = Dataset.from_list(samples)

    # Quick training test (just 10 steps)
    training_args = SFTConfig(
        output_dir="/tmp/test_type",
        max_steps=10,
        per_device_train_batch_size=cfg['batch'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        bf16=True,
        max_length=cfg['max_length'],
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text"
    )

    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer
        )

        # Run a few steps
        trainer.train()

        alloc, res, total = get_gpu_memory()
        print(f"  GPU Memory: {res:.1f}GB / {total:.1f}GB ({100*res/total:.0f}%)")
        print(f"  SUCCESS")
        return True, res

    except Exception as e:
        print(f"  FAILED: {e}")
        return False, 0


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print("="*60)
    print("Testing Each Type with Optimized Config")
    print("GPU: A6000 only (CUDA_VISIBLE_DEVICES=0)")
    print("="*60)
    print(f"Config: batch={CONFIG['batch']}, lora_r={CONFIG['lora_r']}")

    # Load model once
    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-27b-text-it",
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-27b-text-it", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    alloc, res, total = get_gpu_memory()
    print(f"After model load: {res:.1f}GB / {total:.1f}GB")

    # Test each type
    results = {}
    for type_name in TYPE_ORDER:
        success, mem = test_type(type_name, model, tokenizer, CONFIG)
        results[type_name] = {"success": success, "memory_gb": mem}
        clear_gpu()

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    all_ok = True
    for t, r in results.items():
        status = "OK" if r["success"] else "FAIL"
        mem = f"{r['memory_gb']:.1f}GB" if r["memory_gb"] > 0 else "-"
        print(f"  {t:<25} {status:<6} {mem}")
        if not r["success"]:
            all_ok = False

    if all_ok:
        print("\nAll types OK! Ready to start training.")
    else:
        print("\nSome types failed. Check config.")

    return all_ok


if __name__ == "__main__":
    main()
