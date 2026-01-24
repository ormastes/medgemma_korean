#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Training Pipeline - Verify all train scripts work correctly

Tests:
1. train_00: Load raw_lora_added, check LoRA structure, verify loss decreases
2. train_01: Load 00_trained, check LoRA trainable (not add new), verify loss decreases
3. train_02: Load 01_trained, add new LoRA, verify loss decreases

Uses temp folders and minimal samples for quick testing.
"""

import sys
import json
import torch
import gc
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).parent.parent.parent
TEMP_DIR = BASE_DIR / "model" / "_temp_test"


def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def clear_gpu(device_id: int = 1):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_model_structure(model, name: str = "Model"):
    """Print model structure showing LoRA layers and trainable status."""
    log(f"\n{'='*60}")
    log(f"Model Structure: {name}")
    log(f"{'='*60}")

    trainable_count = 0
    frozen_count = 0
    lora_modules = []
    frozen_modules = []
    trainable_modules = []

    for name_param, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += param.numel()
            if 'lora' in name_param.lower():
                lora_modules.append(name_param)
            else:
                trainable_modules.append(name_param)
        else:
            frozen_count += param.numel()
            frozen_modules.append(name_param)

    log(f"Trainable params: {trainable_count:,} ({trainable_count/1e6:.2f}M)")
    log(f"Frozen params: {frozen_count:,} ({frozen_count/1e6:.2f}M)")
    log(f"Trainable ratio: {trainable_count/(trainable_count+frozen_count)*100:.2f}%")

    log(f"\nLoRA modules ({len(lora_modules)}):")
    for m in lora_modules[:10]:
        log(f"  - {m}")
    if len(lora_modules) > 10:
        log(f"  ... and {len(lora_modules)-10} more")

    log(f"\nOther trainable modules ({len(trainable_modules)}):")
    for m in trainable_modules[:5]:
        log(f"  - {m}")
    if len(trainable_modules) > 5:
        log(f"  ... and {len(trainable_modules)-5} more")

    log(f"\nFrozen modules (sample):")
    for m in frozen_modules[:3]:
        log(f"  - {m}")
    log(f"  ... and {len(frozen_modules)-3} more frozen")

    return {
        'trainable': trainable_count,
        'frozen': frozen_count,
        'lora_modules': len(lora_modules),
        'trainable_modules': len(trainable_modules),
    }


def verify_loss_decreases(model, tokenizer, device, max_steps=5):
    """Run a few training steps and verify loss decreases."""
    from torch.optim import AdamW

    log("\nVerifying loss decreases...")
    model.train()

    # Simple optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    losses = []
    for step in range(max_steps):
        # Create dummy batch
        text = f"의료 테스트 문장입니다. 스텝 {step}."
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()

        # Forward + backward
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        log(f"  Step {step+1}: loss = {loss.item():.4f}")

    # Check if loss decreased
    first_loss = losses[0]
    last_loss = losses[-1]
    decreased = last_loss < first_loss

    log(f"\nLoss change: {first_loss:.4f} -> {last_loss:.4f} ({'✓ DECREASED' if decreased else '✗ NOT decreased'})")

    return {
        'first_loss': first_loss,
        'last_loss': last_loss,
        'decreased': decreased,
        'losses': losses,
    }


def test_train_00(device: str = "cuda:1"):
    """Test train_00: Load raw_lora_added, verify LoRA structure."""
    log("\n" + "="*70)
    log("TEST: train_00 (Plain Text Pre-training)")
    log("="*70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training

    lora_path = BASE_DIR / "model" / "raw_lora_added" / "medgemma-4b"

    if not lora_path.exists():
        log(f"ERROR: {lora_path} not found", "ERROR")
        return None

    log(f"Loading from: {lora_path}")

    # Load config
    peft_config = PeftConfig.from_pretrained(str(lora_path))
    log(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
    log(f"Target modules: {peft_config.target_modules}")
    log(f"Modules to save: {peft_config.modules_to_save}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(lora_path), trust_remote_code=True)
    log(f"Tokenizer vocab: {len(tokenizer)}")

    # Load base model
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # Resize embeddings
    base_vocab = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > base_vocab:
        log(f"Resizing embeddings: {base_vocab} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))

    # Load LoRA (trainable)
    model = PeftModel.from_pretrained(base_model, str(lora_path), is_trainable=True)

    # Print structure
    structure = print_model_structure(model, "train_00 (raw_lora_added)")

    # Verify loss decreases
    loss_result = verify_loss_decreases(model, tokenizer, device, max_steps=3)

    # Save test output to temp
    test_output = TEMP_DIR / "train_00_test"
    test_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(test_output))
    tokenizer.save_pretrained(str(test_output))
    log(f"Saved test output to: {test_output}")

    # Cleanup
    del model, base_model
    clear_gpu(int(device.split(':')[1]))

    return {
        'structure': structure,
        'loss': loss_result,
        'output': str(test_output),
    }


def test_train_01(device: str = "cuda:1"):
    """Test train_01: Load 00_trained, update existing LoRA (no new LoRA)."""
    log("\n" + "="*70)
    log("TEST: train_01 (Medical Dict / Mixed Training)")
    log("="*70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training

    # Use train_00 test output as input
    lora_path = TEMP_DIR / "train_00_test"

    if not lora_path.exists():
        log(f"ERROR: {lora_path} not found. Run test_train_00 first.", "ERROR")
        return None

    log(f"Loading from: {lora_path}")

    # Load config
    peft_config = PeftConfig.from_pretrained(str(lora_path))
    log(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(lora_path), trust_remote_code=True)

    # Load base model
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # Resize embeddings
    if len(tokenizer) > base_model.get_input_embeddings().weight.shape[0]:
        base_model.resize_token_embeddings(len(tokenizer))

    # Load existing LoRA (trainable) - NO new LoRA added
    log("Loading existing LoRA (is_trainable=True) - NOT adding new LoRA")
    model = PeftModel.from_pretrained(base_model, str(lora_path), is_trainable=True)

    # Print structure
    structure = print_model_structure(model, "train_01 (existing LoRA, trainable)")

    # Verify loss decreases
    loss_result = verify_loss_decreases(model, tokenizer, device, max_steps=3)

    # Save test output
    test_output = TEMP_DIR / "train_01_test"
    test_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(test_output))
    tokenizer.save_pretrained(str(test_output))
    log(f"Saved test output to: {test_output}")

    # Cleanup
    del model, base_model
    clear_gpu(int(device.split(':')[1]))

    return {
        'structure': structure,
        'loss': loss_result,
        'output': str(test_output),
    }


def test_train_02(device: str = "cuda:1"):
    """Test train_02: Merge existing LoRA, add NEW LoRA, train."""
    log("\n" + "="*70)
    log("TEST: train_02 (MCQ with Progressive LoRA)")
    log("="*70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training, get_peft_model, LoraConfig

    # Use train_01 test output as input
    lora_path = TEMP_DIR / "train_01_test"

    if not lora_path.exists():
        log(f"ERROR: {lora_path} not found. Run test_train_01 first.", "ERROR")
        return None

    log(f"Loading from: {lora_path}")

    # Load config
    peft_config = PeftConfig.from_pretrained(str(lora_path))
    log(f"Existing LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(lora_path), trust_remote_code=True)

    # Load base model
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # Resize embeddings
    if len(tokenizer) > base_model.get_input_embeddings().weight.shape[0]:
        base_model.resize_token_embeddings(len(tokenizer))

    # Load existing LoRA (not trainable - for merging)
    log("Step 1: Load existing LoRA (is_trainable=False)")
    model = PeftModel.from_pretrained(base_model, str(lora_path), is_trainable=False)

    # Merge existing LoRA into base
    log("Step 2: Merge existing LoRA into base model")
    model = model.merge_and_unload()
    log("  Merged! Previous LoRA knowledge is now frozen in base weights.")

    # Add NEW LoRA for train_02
    log("Step 3: Add NEW LoRA adapter")
    new_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    model = get_peft_model(model, new_lora_config)

    # Print structure
    structure = print_model_structure(model, "train_02 (NEW LoRA added after merge)")

    # Verify loss decreases
    loss_result = verify_loss_decreases(model, tokenizer, device, max_steps=3)

    # Save test output
    test_output = TEMP_DIR / "train_02_test"
    test_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(test_output))
    tokenizer.save_pretrained(str(test_output))
    log(f"Saved test output to: {test_output}")

    # Cleanup
    del model, base_model
    clear_gpu(int(device.split(':')[1]))

    return {
        'structure': structure,
        'loss': loss_result,
        'output': str(test_output),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test training pipeline")
    parser.add_argument("--device", default="cuda:1", help="GPU device")
    parser.add_argument("--clean", action="store_true", help="Clean temp directory before test")
    parser.add_argument("--skip-00", action="store_true", help="Skip train_00 test")
    parser.add_argument("--skip-01", action="store_true", help="Skip train_01 test")
    parser.add_argument("--skip-02", action="store_true", help="Skip train_02 test")
    args = parser.parse_args()

    log("="*70)
    log("TRAINING PIPELINE TEST")
    log("="*70)
    log(f"Device: {args.device}")
    log(f"Temp dir: {TEMP_DIR}")

    # Check GPU
    if torch.cuda.is_available():
        gpu_idx = int(args.device.split(':')[1])
        props = torch.cuda.get_device_properties(gpu_idx)
        log(f"GPU {gpu_idx}: {props.name}, {props.total_memory/1024**3:.1f} GB")

    # Clean temp if requested
    if args.clean and TEMP_DIR.exists():
        log(f"Cleaning temp directory: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Test train_00
    if not args.skip_00:
        results['train_00'] = test_train_00(args.device)
        clear_gpu(int(args.device.split(':')[1]))

    # Test train_01
    if not args.skip_01:
        results['train_01'] = test_train_01(args.device)
        clear_gpu(int(args.device.split(':')[1]))

    # Test train_02
    if not args.skip_02:
        results['train_02'] = test_train_02(args.device)
        clear_gpu(int(args.device.split(':')[1]))

    # Summary
    log("\n" + "="*70)
    log("TEST SUMMARY")
    log("="*70)

    all_passed = True
    for name, result in results.items():
        if result is None:
            log(f"{name}: SKIPPED/FAILED", "ERROR")
            all_passed = False
        else:
            loss_ok = result['loss']['decreased']
            status = "✓ PASSED" if loss_ok else "✗ FAILED"
            log(f"{name}: {status} (loss: {result['loss']['first_loss']:.4f} -> {result['loss']['last_loss']:.4f})")
            if not loss_ok:
                all_passed = False

    log("\n" + "="*70)
    if all_passed:
        log("ALL TESTS PASSED!")
    else:
        log("SOME TESTS FAILED!", "ERROR")
    log("="*70)

    # Save results
    results_file = TEMP_DIR / "test_results.json"
    with open(results_file, 'w') as f:
        # Convert to serializable format
        serializable = {}
        for k, v in results.items():
            if v:
                serializable[k] = {
                    'loss_decreased': v['loss']['decreased'],
                    'first_loss': v['loss']['first_loss'],
                    'last_loss': v['loss']['last_loss'],
                    'trainable_params': v['structure']['trainable'],
                    'lora_modules': v['structure']['lora_modules'],
                }
        json.dump(serializable, f, indent=2)
    log(f"Results saved to: {results_file}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
