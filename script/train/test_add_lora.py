#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test _add_lora.py on TITAN RTX (cuda:1)

Tests:
1. Load model with lora_count=1 (train_00/01 scenario)
2. Load model with lora_count=2 (train_02 scenario)
3. Verify trainable parameters are correct
4. Quick forward pass to verify model works
"""

import sys
import torch
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _add_lora import (
    load_model_with_lora,
    load_for_train_00,
    load_for_train_01,
    load_for_train_02,
    count_lora_adapters,
    get_adapter_names,
    has_lora_adapter,
)

BASE_DIR = Path(__file__).parent.parent.parent


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_trainable_breakdown(model):
    """Print breakdown of trainable vs frozen parameters."""
    trainable_lora = 0
    frozen_lora = 0
    trainable_other = 0
    frozen_other = 0

    for name, param in model.named_parameters():
        is_lora = 'lora_' in name.lower()
        if param.requires_grad:
            if is_lora:
                trainable_lora += param.numel()
            else:
                trainable_other += param.numel()
        else:
            if is_lora:
                frozen_lora += param.numel()
            else:
                frozen_other += param.numel()

    total = trainable_lora + frozen_lora + trainable_other + frozen_other
    print("  LoRA trainable:   {:,} params".format(trainable_lora))
    print("  LoRA frozen:      {:,} params".format(frozen_lora))
    print("  Other trainable:  {:,} params".format(trainable_other))
    print("  Other frozen:     {:,} params".format(frozen_other))
    print("  Total:            {:,} params".format(total))

    return {
        'trainable_lora': trainable_lora,
        'frozen_lora': frozen_lora,
        'trainable_other': trainable_other,
        'frozen_other': frozen_other,
    }


def test_forward_pass(model, tokenizer, device):
    """Quick forward pass test."""
    print("  Testing forward pass...")

    model.eval()
    test_input = "의료 AI 테스트입니다."

    with torch.no_grad():
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        outputs = model(**inputs)

    print(f"  ✓ Forward pass OK, logits shape: {outputs.logits.shape}")
    return True


def test_train_00_scenario(device="cuda:1"):
    """Test train_00 scenario: lora_count=1, add if missing."""
    print("\n" + "=" * 70)
    print("TEST 1: train_00 scenario (lora_count=1, enable=1)")
    print("=" * 70)

    model_path = BASE_DIR / "model" / "raw_lora_added" / "medgemma-4b"

    if not model_path.exists():
        print(f"  SKIP: {model_path} not found")
        return None

    print(f"  Model path: {model_path}")
    print(f"  Has LoRA: {has_lora_adapter(str(model_path))}")

    model, tokenizer = load_for_train_00(
        model_path=str(model_path),
        device=device,
        model_name="medgemma-4b",
        include_embeddings=False,  # Disable for TITAN RTX
        verbose=True
    )

    print("\nParameter breakdown:")
    stats = print_trainable_breakdown(model)

    print("\nAdapter info:")
    print(f"  Adapter names: {get_adapter_names(model)}")
    print(f"  Adapter count: {count_lora_adapters(model)}")

    test_forward_pass(model, tokenizer, device)

    # Verify: should have trainable LoRA params
    assert stats['trainable_lora'] > 0, "train_00 should have trainable LoRA params"
    print("\n✓ TEST 1 PASSED")

    del model, tokenizer
    clear_gpu()
    return stats


def test_train_01_scenario(device="cuda:1"):
    """Test train_01 scenario: lora_count=1, use existing."""
    print("\n" + "=" * 70)
    print("TEST 2: train_01 scenario (lora_count=1, use existing)")
    print("=" * 70)

    model_path = BASE_DIR / "model" / "00_trained" / "medgemma-4b"

    if not model_path.exists():
        print(f"  SKIP: {model_path} not found")
        print("  (Run train_00 first)")
        return None

    print(f"  Model path: {model_path}")
    print(f"  Has LoRA: {has_lora_adapter(str(model_path))}")

    model, tokenizer = load_for_train_01(
        model_path=str(model_path),
        device=device,
        model_name="medgemma-4b",
        verbose=True
    )

    print("\nParameter breakdown:")
    stats = print_trainable_breakdown(model)

    print("\nAdapter info:")
    print(f"  Adapter names: {get_adapter_names(model)}")
    print(f"  Adapter count: {count_lora_adapters(model)}")

    test_forward_pass(model, tokenizer, device)

    # Verify: should have trainable LoRA params
    assert stats['trainable_lora'] > 0, "train_01 should have trainable LoRA params"
    print("\n✓ TEST 2 PASSED")

    del model, tokenizer
    clear_gpu()
    return stats


def test_train_02_scenario(device="cuda:1"):
    """Test train_02 scenario: lora_count=2, add second, train only second."""
    print("\n" + "=" * 70)
    print("TEST 3: train_02 scenario (lora_count=2, enable=1, train last)")
    print("=" * 70)

    # Try different input paths
    possible_paths = [
        BASE_DIR / "model" / "01_mixed" / "medgemma-4b" / "final",
        BASE_DIR / "model" / "01_trained" / "medgemma-4b",
        BASE_DIR / "model" / "00_trained" / "medgemma-4b",
    ]

    model_path = None
    for p in possible_paths:
        if p.exists() and has_lora_adapter(str(p)):
            model_path = p
            break

    if model_path is None:
        print(f"  SKIP: No trained model found")
        print("  (Run train_00 or train_01 first)")
        return None

    print(f"  Model path: {model_path}")
    print(f"  Has LoRA: {has_lora_adapter(str(model_path))}")

    model, tokenizer = load_for_train_02(
        model_path=str(model_path),
        device=device,
        model_name="medgemma-4b",
        verbose=True
    )

    print("\nParameter breakdown:")
    stats = print_trainable_breakdown(model)

    print("\nAdapter info:")
    adapters = get_adapter_names(model)
    print(f"  Adapter names: {adapters}")
    print(f"  Adapter count: {count_lora_adapters(model)}")

    test_forward_pass(model, tokenizer, device)

    # Verify: should have 2 adapters, one frozen + one trainable
    assert count_lora_adapters(model) == 2, f"train_02 should have 2 adapters, got {count_lora_adapters(model)}"
    assert stats['trainable_lora'] > 0, "train_02 should have trainable LoRA params"
    assert stats['frozen_lora'] > 0, "train_02 should have frozen LoRA params (first adapter)"
    print("\n✓ TEST 3 PASSED")

    del model, tokenizer
    clear_gpu()
    return stats


def test_2lora_train_first_only(device="cuda:1"):
    """Test 2-LoRA model training only first adapter (train_00/01 scenario)."""
    print("\n" + "=" * 70)
    print("TEST 4: 2-LoRA model, train only first (lora_count=None, enable_index=0)")
    print("=" * 70)

    # Find a model with 2 LoRAs
    possible_paths = [
        BASE_DIR / "model" / "02_trained" / "medgemma-4b",
        BASE_DIR / "model" / "01_another_lora_added" / "medgemma-4b",
    ]

    model_path = None
    for p in possible_paths:
        if p.exists() and has_lora_adapter(str(p)):
            model_path = p
            break

    if model_path is None:
        # Create 2-LoRA model from 1-LoRA model for testing
        one_lora_path = BASE_DIR / "model" / "00_trained" / "medgemma-4b"
        if not one_lora_path.exists():
            print(f"  SKIP: No suitable model found")
            return None
        model_path = one_lora_path
        print(f"  Using 1-LoRA model, will add second: {model_path}")
        lora_count_for_test = 2  # Add second LoRA
    else:
        print(f"  Found multi-LoRA model: {model_path}")
        lora_count_for_test = None  # Use existing

    print(f"  Model path: {model_path}")
    print("  Testing: lora_count=None (or 2), enable_count=1, enable_index=0")
    print("  Expected: Train ONLY first LoRA, freeze others")

    model, tokenizer = load_model_with_lora(
        model_path=str(model_path),
        lora_count=lora_count_for_test,
        lora_enable_count=1,  # Train only 1
        lora_enable_index=0,  # First LoRA
        device=device,
        model_name="medgemma-4b",
        verbose=True
    )

    print("\nParameter breakdown:")
    stats = print_trainable_breakdown(model)

    print("\nAdapter info:")
    adapters = get_adapter_names(model)
    print(f"  Adapter names: {adapters}")
    print(f"  Adapter count: {count_lora_adapters(model)}")

    # Verify: should have trainable LoRA (first) and frozen LoRA (second)
    assert stats['trainable_lora'] > 0, "First LoRA should be trainable"
    if count_lora_adapters(model) > 1:
        assert stats['frozen_lora'] > 0, "Second LoRA should be frozen"
        print(f"  ✓ First LoRA trainable, others frozen")
    print("\n✓ TEST 4 PASSED")

    del model, tokenizer
    clear_gpu()
    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test _add_lora.py")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device (default: cuda:1 for TITAN RTX)")
    parser.add_argument("--test", type=int, default=0, help="Run specific test (0=all)")
    args = parser.parse_args()

    print("=" * 70)
    print("Testing _add_lora.py")
    print("=" * 70)
    print(f"Device: {args.device}")

    if torch.cuda.is_available():
        gpu_id = int(args.device.split(":")[-1])
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print()

    results = {}

    tests = {
        1: ("train_00 scenario", test_train_00_scenario),
        2: ("train_01 scenario", test_train_01_scenario),
        3: ("train_02 scenario", test_train_02_scenario),
        4: ("2-LoRA train first only", test_2lora_train_first_only),
    }

    if args.test > 0:
        # Run specific test
        if args.test in tests:
            name, func = tests[args.test]
            results[name] = func(args.device)
        else:
            print(f"Unknown test: {args.test}")
    else:
        # Run all tests
        for test_id, (name, func) in tests.items():
            try:
                results[name] = func(args.device)
            except Exception as e:
                print(f"\n✗ TEST {test_id} FAILED: {e}")
                import traceback
                traceback.print_exc()
                results[name] = None

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, stats in results.items():
        if stats is None:
            print(f"  {name}: SKIPPED or FAILED")
        else:
            print(f"  {name}: PASSED (trainable LoRA: {stats['trainable_lora']:,})")

    print("\nDone!")


if __name__ == "__main__":
    main()
