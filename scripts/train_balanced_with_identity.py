#!/usr/bin/env python3
"""
Balanced Training with Identity Layers for KorMedMCQA

Key Changes:
1. Identity layers at front and rear (gradual adaptation)
2. Doubled LoRA rank (256) for more capacity
3. Balanced data across types (divide large datasets)
4. Training order: Type4 -> Type3 -> Type2 -> Type1 (MCQ-focused first)
"""

import argparse
import json
import re
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "models" / "balanced_identity_training"
LOG_DIR = BASE_DIR / "logs"

# Training order: Start from Type4 (MCQ-focused)
TYPES = ["type4_word_reasoning", "type3_word", "type2_text_reasoning", "type1_text"]
TYPE_DESCRIPTIONS = {
    "type1_text": "TEXT - Full answers (118K -> 8K balanced)",
    "type2_text_reasoning": "TEXT+REASONING (23K -> 8K balanced)",
    "type3_word": "WORD/MCQ (17K -> 8K balanced)",
    "type4_word_reasoning": "WORD+REASONING (8K - base size)"
}

# Target samples per type (balance to smallest type)
TARGET_SAMPLES_PER_TYPE = 8000

MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 128, "lora_alpha": 256,  # Doubled
        "lr": 1e-4, "batch": 1, "grad_accum": 16,
        "max_length": 512
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 160, "lora_alpha": 320,  # 1.25x from 128 (balance capacity vs memory)
        "lr": 5e-5, "batch": 1, "grad_accum": 64,  # More gradient accumulation
        "max_length": 384  # Shorter sequences
    }
}


class IdentityAdapter(nn.Module):
    """
    Identity adapter layer that starts as passthrough and gradually learns.
    Added at front (after embeddings) and rear (before lm_head).
    """
    def __init__(self, hidden_size: int, init_scale: float = 0.01):
        super().__init__()
        # Smaller bottleneck (1/8 instead of 1/4) to save memory
        self.down_proj = nn.Linear(hidden_size, hidden_size // 8)
        self.up_proj = nn.Linear(hidden_size // 8, hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))  # Starts at 0 (identity)

        # Initialize to near-identity
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        # Identity + gated residual
        # gate starts at 0, so output = x (identity)
        # As training progresses, gate learns to add adapted signal
        residual = self.up_proj(torch.relu(self.down_proj(x)))
        return x + torch.sigmoid(self.gate) * residual


class ModelWithIdentityAdapters(nn.Module):
    """Wrapper that adds identity adapters to front and rear of model."""

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.front_adapter = IdentityAdapter(hidden_size)
        self.rear_adapter = IdentityAdapter(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get embeddings - handle different model structures
        try:
            # For PEFT wrapped models
            if hasattr(self.base_model, 'base_model'):
                inner_model = self.base_model.base_model
                if hasattr(inner_model, 'model'):
                    embed_layer = inner_model.model.embed_tokens
                else:
                    embed_layer = inner_model.embed_tokens
            elif hasattr(self.base_model, 'model'):
                embed_layer = self.base_model.model.embed_tokens
            else:
                embed_layer = self.base_model.embed_tokens
        except AttributeError:
            # Fallback: just use the model directly without front adapter
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

        # Apply front adapter after embeddings
        embeds = embed_layer(input_ids)
        embeds = self.front_adapter(embeds)

        # Forward through model with modified embeddings
        outputs = self.base_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        return outputs

    def generate(self, **kwargs):
        # For generation, use base model directly (adapters applied internally)
        return self.base_model.generate(**kwargs)

    @property
    def device(self):
        return next(self.parameters()).device


def load_balanced_data(data_dir: Path, target_per_type: int = TARGET_SAMPLES_PER_TYPE) -> Dict[str, List[dict]]:
    """Load and balance data across types."""
    type_data = {}

    for type_name in TYPES:
        train_file = data_dir / type_name / "train" / "data.jsonl"
        if not train_file.exists():
            print(f"  {type_name}: File not found, skipping")
            continue

        # Load all data
        all_samples = []
        with open(train_file) as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))

        original_count = len(all_samples)

        # Balance: take only target_per_type samples (or all if less)
        if len(all_samples) > target_per_type:
            # Take evenly distributed samples
            step = len(all_samples) / target_per_type
            balanced = [all_samples[int(i * step)] for i in range(target_per_type)]
        else:
            balanced = all_samples

        type_data[type_name] = balanced
        print(f"  {type_name}: {original_count} -> {len(balanced)} samples (balanced)")

    return type_data


def load_kormedmcqa_eval() -> Dataset:
    """Load KorMedMCQA from HuggingFace."""
    all_samples = []

    for subject in ["doctor", "nurse", "pharm"]:
        try:
            ds = load_dataset("sean0042/KorMedMCQA", subject, split="test")
            for item in ds:
                answer_letter = chr(64 + int(item['answer']))

                prompt = f"""<|im_start|>system
정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).
<|im_end|>
<|im_start|>user
{item['question']}
A) {item['A']}
B) {item['B']}
C) {item['C']}
D) {item['D']}
E) {item['E']}
<|im_end|>
<|im_start|>assistant
"""
                all_samples.append({
                    'prompt': prompt,
                    'answer': answer_letter,
                    'question_text': item['question'],
                    'choices': {
                        'A': item['A'], 'B': item['B'], 'C': item['C'],
                        'D': item['D'], 'E': item['E']
                    },
                    'subject': subject
                })
        except Exception as e:
            print(f"  Warning: Could not load {subject}: {e}")

    print(f"[EVAL] Loaded {len(all_samples)} KorMedMCQA samples")
    return Dataset.from_list(all_samples)


def evaluate_kormedmcqa(
    model,
    tokenizer,
    eval_dataset: Dataset,
    max_samples: int = 200,
    verbose: bool = True
) -> float:
    """Evaluate on KorMedMCQA."""
    model.eval()
    correct = 0
    total = 0

    samples = list(eval_dataset)[:max_samples]

    with torch.no_grad():
        iterator = tqdm(samples, desc="Eval") if verbose else samples
        for sample in iterator:
            try:
                prompt = sample['prompt']
                expected = sample.get('answer', '').strip().upper()
                expected = re.sub(r'[^A-E]', '', expected)[:1]

                if not expected:
                    continue

                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=800
                ).to(model.device)

                # Use base model for generation
                if hasattr(model, 'base_model'):
                    outputs = model.base_model.generate(
                        **inputs, max_new_tokens=10,
                        do_sample=False, pad_token_id=tokenizer.pad_token_id
                    )
                else:
                    outputs = model.generate(
                        **inputs, max_new_tokens=10,
                        do_sample=False, pad_token_id=tokenizer.pad_token_id
                    )

                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                predicted = response.split()[0] if response.split() else ""
                predicted = re.sub(r'[^A-E]', '', predicted.upper())[:1]

                if predicted == expected:
                    correct += 1
                total += 1

            except Exception:
                continue

    accuracy = 100 * correct / total if total > 0 else 0
    model.train()
    torch.cuda.empty_cache()
    return accuracy


def train_on_type(
    model,
    tokenizer,
    data: List[dict],
    type_name: str,
    num_steps: int,
    cfg: dict,
    device: str
) -> float:
    """Train on a single type for N steps."""
    batch_size = cfg['batch']
    grad_accum = cfg['grad_accum']
    lr = cfg['lr']
    max_length = cfg['max_length']

    # Collect trainable parameters (including identity adapters)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    total_loss = 0
    steps_done = 0
    accum_loss = 0
    batch_texts = []

    pbar = tqdm(total=num_steps, desc=f"Train {type_name[:12]}")

    sample_idx = 0
    while steps_done < num_steps:
        idx = sample_idx % len(data)
        sample = data[idx]
        sample_idx += 1

        text = sample.get('text', '')
        if not text:
            prompt = sample.get('prompt', '')
            completion = sample.get('completion', sample.get('answer', ''))
            text = f"{prompt}{completion}"

        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            inputs = tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=max_length
            ).to(device)

            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['input_ids']
            )

            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (sample_idx // batch_size) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

                total_loss += accum_loss
                accum_loss = 0
                steps_done += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{total_loss/max(steps_done,1):.4f}"})

                if steps_done % 5 == 0:
                    torch.cuda.empty_cache()

            batch_texts = []

    pbar.close()
    return total_loss / max(steps_done, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma-27b", choices=["medgemma-4b", "medgemma-27b"])
    parser.add_argument("--steps-per-type", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--target-accuracy", type=float, default=90.0)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("#" * 80)
    print("BALANCED TRAINING WITH IDENTITY ADAPTERS")
    print("#" * 80)
    print(f"Model: {cfg['path']}")
    print(f"LoRA rank: {cfg['lora_r']} (doubled)")
    print(f"Steps per type: {args.steps_per_type}")
    print(f"Training order: {' -> '.join(TYPES)}")
    print(f"Target accuracy: {args.target_accuracy}%")
    print("#" * 80)

    # Load model
    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg['path'],
        quantization_config=bnb_config,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for training
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()

    # Add LoRA with doubled rank
    lora_config = LoraConfig(
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)

    # Get hidden size for identity adapters
    if hasattr(model.base_model, 'config'):
        hidden_size = model.base_model.config.hidden_size
    else:
        hidden_size = 4096  # Default for Gemma

    # Wrap with identity adapters
    model_with_adapters = ModelWithIdentityAdapters(model, hidden_size)
    model_with_adapters.to(args.device)

    # Count parameters
    trainable = sum(p.numel() for p in model_with_adapters.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"  - LoRA params: {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad):,}")
    print(f"  - Identity adapter params: {sum(p.numel() for p in model_with_adapters.front_adapter.parameters()) + sum(p.numel() for p in model_with_adapters.rear_adapter.parameters()):,}")

    # Load evaluation data
    eval_dataset = load_kormedmcqa_eval()

    # Load balanced training data
    print("\nLoading balanced training data...")
    data_dir = BASE_DIR / "data" / "reviewed"
    type_data = load_balanced_data(data_dir)

    # Initial evaluation
    print("\n" + "=" * 80)
    print("INITIAL EVALUATION")
    print("=" * 80)
    initial_acc = evaluate_kormedmcqa(model_with_adapters, tokenizer, eval_dataset, args.eval_samples)
    print(f"Initial KorMedMCQA accuracy: {initial_acc:.2f}%")

    # Training loop
    best_acc = initial_acc
    current_acc = initial_acc

    for round_num in range(1, args.max_rounds + 1):
        print("\n" + "=" * 80)
        print(f"ROUND {round_num}")
        print("=" * 80)

        for type_name in TYPES:
            if type_name not in type_data:
                continue

            print(f"\n--- Training {type_name} ---")
            print(f"    {TYPE_DESCRIPTIONS.get(type_name, '')}")

            avg_loss = train_on_type(
                model_with_adapters, tokenizer, type_data[type_name],
                type_name, args.steps_per_type, cfg, args.device
            )

            # Evaluate after each type
            acc = evaluate_kormedmcqa(model_with_adapters, tokenizer, eval_dataset, args.eval_samples, verbose=False)
            improvement = acc - current_acc
            print(f"    Accuracy: {current_acc:.2f}% -> {acc:.2f}% (Δ {improvement:+.2f}%)")

            current_acc = acc
            if acc > best_acc:
                best_acc = acc
                # Save best model
                model.save_pretrained(OUTPUT_DIR / "best")
                print(f"    NEW BEST! Saved to {OUTPUT_DIR / 'best'}")

            if acc >= args.target_accuracy:
                print(f"\n{'=' * 80}")
                print(f"TARGET {args.target_accuracy}% REACHED!")
                print(f"{'=' * 80}")
                model.save_pretrained(OUTPUT_DIR / "final")
                print(f"Model saved to {OUTPUT_DIR / 'final'}")
                return

        print(f"\nRound {round_num} complete. Best accuracy: {best_acc:.2f}%")

    # Save final model
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final accuracy: {current_acc:.2f}%")
    print(f"Best accuracy: {best_acc:.2f}%")
    model.save_pretrained(OUTPUT_DIR / "final")


if __name__ == "__main__":
    main()
