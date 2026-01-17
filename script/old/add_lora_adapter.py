#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add LoRA Adapter Layer after train_00_plain_text.py
Purpose: Prevent catastrophic forgetting when training on medical data
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from training_config import LORA_TARGET_MODULES, TRAINING_DEFAULTS


def add_lora_adapter(base_model_path: str, output_path: str, adapter_name: str = "medical"):
    """
    Add a new LoRA adapter to a trained model
    
    Args:
        base_model_path: Path to trained model (from train_00)
        output_path: Where to save model with new adapter
        adapter_name: Name of the new adapter layer
    """
    print("=" * 60)
    print("Adding LoRA Adapter Layer")
    print("=" * 60)
    print(f"Base model: {base_model_path}")
    print(f"Output: {output_path}")
    print(f"Adapter name: {adapter_name}")
    
    # Load the trained model
    print("\nLoading trained model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Check if model is already a PEFT model
    try:
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ),
            base_model_path
        )
        print("✓ Loaded existing PEFT model")
        
        # Add new adapter
        print(f"\nAdding new adapter: {adapter_name}")
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=TRAINING_DEFAULTS['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model.add_adapter(adapter_name, lora_config)
        model.set_adapter(adapter_name)
        
    except Exception as e:
        print(f"Note: {e}")
        print("Loading as regular model and adding first adapter...")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=TRAINING_DEFAULTS['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
    
    # Save model with new adapter
    print(f"\nSaving model to: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save adapter info
    adapter_info = {
        "base_model": base_model_path,
        "adapter_name": adapter_name,
        "lora_r": 64,
        "lora_alpha": 128,
        "target_modules": LORA_TARGET_MODULES,
        "purpose": "Prevent catastrophic forgetting for medical training"
    }
    
    import json
    with open(output_dir / "adapter_info.json", 'w') as f:
        json.dump(adapter_info, f, indent=2)
    
    print("\n✓ Successfully added LoRA adapter!")
    print(f"  Active adapter: {adapter_name}")
    print(f"  Trainable params info saved")
    
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Add LoRA adapter after plain text training")
    parser.add_argument("--base-model", required=True, help="Path to trained model from train_00")
    parser.add_argument("--output", required=True, help="Output directory for model with adapter")
    parser.add_argument("--adapter-name", default="medical", help="Name of adapter (default: medical)")
    args = parser.parse_args()
    
    try:
        output_path = add_lora_adapter(args.base_model, args.output, args.adapter_name)
        print(f"\n{'='*60}")
        print(f"Model ready for medical training: {output_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Failed to add adapter: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
