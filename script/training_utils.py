#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Training Utilities
Common functions used across training scripts to eliminate code duplication
"""

import argparse
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from training_config import MODEL_CONFIGS, LORA_TARGET_MODULES, TRAINING_DEFAULTS, MEMORY_CONFIGS


def create_base_parser(description: str):
    """
    Create argument parser with common training arguments
    
    Args:
        description: Script description
    
    Returns:
        ArgumentParser with common arguments
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="medgemma-4b", choices=list(MODEL_CONFIGS.keys()),
                       help="Model to train")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max-samples", type=int, default=None, 
                       help="Limit training samples (for testing)")
    parser.add_argument("--base-model", default=None, 
                       help="Base model path (default: use model from config)")
    parser.add_argument("--output", default=None, 
                       help="Output directory (default: models/<script_name>/<model>)")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    return parser


def load_jsonl_data(data_dir: Path, max_samples: int = None):
    """
    Load training and validation data from JSONL files
    
    Args:
        data_dir: Directory containing train/ and validation/ subdirectories
                  OR directory containing train.jsonl and validation.jsonl
        max_samples: Maximum samples to load (None = load all)
    
    Returns:
        (train_dataset, val_dataset) as HuggingFace Dataset objects
    """
    train_data = []
    val_data = []
    
    for split, data_list in [("train", train_data), ("validation", val_data)]:
        # Try multiple file locations
        file_paths = [
            data_dir / split / "data.jsonl",  # Standard: train/data.jsonl
            data_dir / f"{split}.jsonl",      # Flat: train.jsonl
            data_dir / "train" / f"{split}.jsonl",  # Alternative
        ]
        
        file_path = None
        for path in file_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path and file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    data_list.append(json.loads(line))
    
    print(f"Loaded {len(train_data)} train, {len(val_data)} validation samples")
    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def load_tokenizer(model_path: str):
    """
    Load tokenizer with standard settings
    
    Args:
        model_path: Path to model or HuggingFace model ID
    
    Returns:
        AutoTokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_8bit(model_path: str, device: str = "cuda:0",
                    model_name: str = None, use_gradient_checkpointing: bool = None):
    """
    Load model with 8-bit quantization

    Args:
        model_path: Path to model or HuggingFace model ID
        device: Target device (default: cuda:0)
        model_name: Model name for memory config lookup (optional)
        use_gradient_checkpointing: Override gradient checkpointing (None = use config)

    Returns:
        Model ready for LoRA training
    """
    print(f"Loading model: {model_path}")

    # Determine gradient checkpointing setting
    if use_gradient_checkpointing is None and model_name:
        mem_cfg = MEMORY_CONFIGS.get(model_name, {})
        use_gradient_checkpointing = mem_cfg.get('use_gradient_checkpointing', False)
    elif use_gradient_checkpointing is None:
        use_gradient_checkpointing = False

    if use_gradient_checkpointing:
        print("  Gradient checkpointing: ENABLED (memory optimization)")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    return model


def create_lora_config(lora_r: int = 64, lora_alpha: int = 128,
                       include_embeddings: bool = False, use_rslora: bool = False):
    """
    Create LoRA configuration

    Args:
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        include_embeddings: Include embed_tokens and lm_head (for continued pretraining)
                           These go to modules_to_save (trained directly, not via LoRA)
        use_rslora: Use rank-stabilized LoRA

    Returns:
        LoraConfig instance
    """
    target_modules = LORA_TARGET_MODULES.copy()

    # Embeddings must use modules_to_save (not target_modules) for 8-bit quantized models
    # modules_to_save keeps these layers as full-precision trainable parameters
    modules_to_save = ["embed_tokens", "lm_head"] if include_embeddings else None

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        lora_dropout=TRAINING_DEFAULTS['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    if use_rslora:
        config.use_rslora = True

    return config


def setup_model_with_lora(model_path: str, lora_r: int = 64, lora_alpha: int = 128,
                          include_embeddings: bool = None, use_rslora: bool = False,
                          device: str = "cuda:0", model_name: str = None):
    """
    Complete model setup: load model + add LoRA

    Args:
        model_path: Path to model or HuggingFace model ID
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        include_embeddings: Include embeddings in LoRA (None = use MEMORY_CONFIGS)
        use_rslora: Use rank-stabilized LoRA
        device: Target device
        model_name: Model name for memory config lookup (optional)

    Returns:
        (model, tokenizer) ready for training
    """
    # Determine include_embeddings from config if not specified
    if include_embeddings is None and model_name:
        mem_cfg = MEMORY_CONFIGS.get(model_name, {})
        include_embeddings = mem_cfg.get('train_embeddings', False)
        if not include_embeddings:
            print(f"  Embedding training: DISABLED (memory optimization for {model_name})")
    elif include_embeddings is None:
        include_embeddings = False

    # Load tokenizer
    tokenizer = load_tokenizer(model_path)

    # Load model
    model = load_model_8bit(model_path, device, model_name=model_name)

    # Add LoRA
    lora_config = create_lora_config(lora_r, lora_alpha, include_embeddings, use_rslora)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def create_training_args(output_dir: str, num_epochs: int, batch_size: int,
                        grad_accum: int, learning_rate: float, max_length: int,
                        save_strategy: str = "epoch", logging_steps: int = 10):
    """
    Create standard training arguments
    
    Args:
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        learning_rate: Learning rate
        max_length: Maximum sequence length
        save_strategy: When to save ("epoch" or "steps")
        logging_steps: Log every N steps
    
    Returns:
        SFTConfig instance
    """
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=TRAINING_DEFAULTS['weight_decay'],
        warmup_ratio=TRAINING_DEFAULTS['warmup_ratio'],
        lr_scheduler_type=TRAINING_DEFAULTS['lr_scheduler_type'],
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=TRAINING_DEFAULTS['save_total_limit'],
        fp16=False,
        bf16=False,
        max_length=max_length,
        gradient_checkpointing=False,
        optim=TRAINING_DEFAULTS['optim'],
        max_grad_norm=TRAINING_DEFAULTS['max_grad_norm'],
        report_to="none",
        dataloader_num_workers=0,
    )


def run_training(model, tokenizer, train_data, training_args, output_dir: Path):
    """
    Run training and save model
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_data: Training dataset
        training_args: Training arguments
        output_dir: Output directory
    
    Returns:
        Path to saved model
    """
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    print(f"\n✓ Training complete! Model saved to: {final_dir}")
    return final_dir


def save_training_info(output_dir: Path, info: dict):
    """
    Save training metadata
    
    Args:
        output_dir: Output directory
        info: Dictionary with training info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)


def train_script_wrapper(script_name: str, data_dir: Path, default_output_dir: Path,
                         train_fn=None, format_fn=None):
    """
    Common wrapper for training scripts
    
    Args:
        script_name: Name of the training script (for logging)
        data_dir: Data directory
        default_output_dir: Default output directory
        train_fn: Optional custom training function(args, cfg, model, tokenizer, train_data)
        format_fn: Optional function to format data (dataset) -> dataset with 'text' field
    
    Returns:
        Main function for the script
    """
    def main():
        parser = create_base_parser(f"Training: {script_name}")
        args = parser.parse_args()
        
        cfg = MODEL_CONFIGS[args.model]
        output_dir = Path(args.output) if args.output else default_output_dir / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print(f"Training: {script_name}")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Output: {output_dir}")
        
        # Load data
        train_data, val_data = load_jsonl_data(data_dir, max_samples=args.max_samples)
        if len(train_data) == 0:
            print("❌ No training data found!")
            return 1
        
        # Format data if needed
        if format_fn:
            print("Formatting data...")
            train_data = train_data.map(format_fn, batched=False)
            if len(val_data) > 0:
                val_data = val_data.map(format_fn, batched=False)
        
        # Setup model
        model_path = args.base_model if args.base_model else cfg['path']
        model, tokenizer = setup_model_with_lora(
            model_path,
            lora_r=cfg['lora_r'],
            lora_alpha=cfg['lora_alpha'],
            include_embeddings=None,  # Use MEMORY_CONFIGS
            device=args.device,
            model_name=args.model
        )
        
        # Training
        if train_fn:
            # Custom training function
            final_dir = train_fn(args, cfg, model, tokenizer, train_data, output_dir)
        else:
            # Standard training
            training_args = create_training_args(
                output_dir=str(output_dir),
                num_epochs=args.epochs,
                batch_size=cfg['batch'],
                grad_accum=cfg['grad_accum'],
                learning_rate=cfg['lr'],
                max_length=cfg['max_length']
            )
            final_dir = run_training(model, tokenizer, train_data, training_args, output_dir)
        
        # Save info
        save_training_info(output_dir, {
            "script": script_name,
            "model": args.model,
            "base_model": model_path,
            "epochs": args.epochs,
            "train_samples": len(train_data),
            "val_samples": len(val_data)
        })
        
        return 0
    
    return main
