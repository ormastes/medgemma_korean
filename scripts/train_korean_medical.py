#!/usr/bin/env python3
"""
Korean Medical Training - Unified Script

Supports:
- Multiple data sources: raw, refined, reviewed
- Multiple data types: mcq, qa, instruction, reasoning, combined
- Multiple base models: MedGemma 4B, MedGemma 27B, Custom
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


# =============================================================================
# Configuration
# =============================================================================
MODELS = {
    "medgemma-4b": "google/medgemma-4b-it",
    "medgemma-27b": "google/medgemma-27b-text-it",
    "stage5": str(BASE_DIR / "models" / "staged_training" / "stage5_harmonization"),
    "stage6": str(BASE_DIR / "models" / "staged_training" / "stage6_hybrid_expansion"),
}

DATA_SOURCES = {
    "raw": DATA_DIR / "raw" / "by_source",
    "refined": DATA_DIR / "refined",
    "reviewed": DATA_DIR / "reviewed",
    "processed": DATA_DIR / "processed",
}

DATA_TYPES = ["mcq", "qa", "instruction", "reasoning", "combined"]

# Default hyperparameters by model size
HYPERPARAMS = {
    "4b": {
        "batch_size": 2,
        "grad_accum": 16,
        "lr": 1e-4,
        "lora_r": 64,
        "lora_alpha": 128,
        "max_seq_length": 1024,
    },
    "27b": {
        "batch_size": 1,
        "grad_accum": 32,
        "lr": 2e-5,
        "lora_r": 128,
        "lora_alpha": 256,
        "max_seq_length": 512,
    },
}


class AccuracyCallback(TrainerCallback):
    """Callback for MCQ accuracy evaluation."""

    def __init__(self, model, tokenizer, eval_dataset, target_acc=90.0, eval_steps=500, output_dir=None):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.target_acc = target_acc
        self.eval_steps = eval_steps
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.best_acc = 0.0
        self.history = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            acc = self._evaluate()
            self.history.append({"step": state.global_step, "accuracy": acc, "time": str(datetime.now())})
            print(f"\n[Step {state.global_step}] Accuracy: {acc:.2f}%")

            if acc > self.best_acc:
                self.best_acc = acc
                self._save_best()

            self._save_history()

            if acc >= self.target_acc:
                print(f"\n🎉 Target {self.target_acc}% reached!")
                control.should_training_stop = True

        return control

    def _evaluate(self, max_samples=300):
        self.model.eval()
        correct = 0
        total = 0

        indices = list(range(min(len(self.eval_dataset), max_samples)))

        for idx in tqdm(indices, desc="Eval", leave=False):
            try:
                item = self.eval_dataset[idx]
                text = item.get("text", "")
                expected = item.get("correct_answer", "").upper().strip()

                if not expected or expected not in "ABCDE":
                    continue

                # Extract question part and add MCQ instruction
                if "<|im_start|>user" in text:
                    user_start = text.find("<|im_start|>user") + len("<|im_start|>user\n")
                    user_end = text.find("<|im_end|>", user_start)
                    user_content = text[user_start:user_end].strip()

                    prompt = f"""<|im_start|>system
당신은 한국어 의료 전문 AI입니다.
<|im_end|>
<|im_start|>user
{user_content}

정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).
<|im_end|>
<|im_start|>assistant
"""
                else:
                    continue

                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)

                response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                response = response.strip().upper()

                predicted = None
                for letter in "ABCDE":
                    if letter in response[:5]:
                        predicted = letter
                        break

                if predicted == expected:
                    correct += 1
                total += 1

            except Exception:
                continue

            if total % 50 == 0:
                torch.cuda.empty_cache()

        self.model.train()
        return (correct / total * 100) if total > 0 else 0

    def _save_best(self):
        best_path = self.output_dir / "best_checkpoint"
        best_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(best_path))
        self.tokenizer.save_pretrained(str(best_path))

    def _save_history(self):
        with open(self.output_dir / "accuracy_history.json", "w") as f:
            json.dump({"best": self.best_acc, "target": self.target_acc, "history": self.history}, f, indent=2)


def load_data(source, data_type, split="train"):
    """Load data from specified source and type."""
    source_path = DATA_SOURCES.get(source)
    if not source_path:
        raise ValueError(f"Unknown source: {source}")

    if data_type == "combined":
        # Load all types and combine
        all_data = []
        for dtype in ["mcq", "qa", "instruction", "reasoning"]:
            type_path = source_path / dtype / split
            if type_path.exists():
                ds = load_from_disk(str(type_path))
                all_data.append(ds)
                print(f"  {dtype}: {len(ds)} samples")

        if all_data:
            return concatenate_datasets(all_data)
        else:
            raise FileNotFoundError(f"No data found in {source_path}")
    else:
        data_path = source_path / data_type / split
        if data_path.exists():
            return load_from_disk(str(data_path))
        else:
            raise FileNotFoundError(f"Data not found: {data_path}")


def main():
    parser = argparse.ArgumentParser(description="Korean Medical Training")

    # Data options
    parser.add_argument("--source", type=str, default="refined", choices=list(DATA_SOURCES.keys()),
                        help="Data source: raw, refined, reviewed, processed")
    parser.add_argument("--type", type=str, default="mcq", choices=DATA_TYPES,
                        help="Data type: mcq, qa, instruction, reasoning, combined")

    # Model options
    parser.add_argument("--model", type=str, default="medgemma-4b", choices=list(MODELS.keys()),
                        help="Base model")

    # Training options
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--target-accuracy", type=float, default=90.0)
    parser.add_argument("--output-dir", type=str, default=None)

    # Override hyperparameters
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=None)

    args = parser.parse_args()

    # Determine model size for default hyperparams
    model_size = "27b" if "27b" in args.model else "4b"
    hp = HYPERPARAMS[model_size].copy()

    # Override with args
    if args.batch_size:
        hp["batch_size"] = args.batch_size
    if args.lr:
        hp["lr"] = args.lr
    if args.lora_r:
        hp["lora_r"] = args.lora_r

    # Output directory
    output_dir = args.output_dir or str(
        BASE_DIR / "models" / f"korean_{args.type}_{args.source}_{args.model.replace('-', '_')}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Korean Medical Training")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Type: {args.type}")
    print(f"Model: {args.model} ({MODELS[args.model]})")
    print(f"Output: {output_dir}")
    print(f"Hyperparams: {hp}")

    # Load data
    print("\nLoading data...")
    train_ds = load_data(args.source, args.type, "train")
    print(f"Train: {len(train_ds)} samples")

    try:
        eval_ds = load_data(args.source, args.type, "validation")
        print(f"Eval: {len(eval_ds)} samples")
    except:
        eval_ds = train_ds.select(range(min(500, len(train_ds))))
        print(f"Using {len(eval_ds)} train samples for eval")

    # Load model
    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model_path = MODELS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # Add LoRA
    lora_config = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    sft_config = SFTConfig(
        output_dir=str(Path(output_dir) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        learning_rate=hp["lr"],
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=model_size == "27b",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        max_length=hp["max_seq_length"],
        dataset_text_field="text",
    )

    # Callback for MCQ accuracy
    callbacks = []
    if args.type == "mcq":
        acc_callback = AccuracyCallback(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_ds,
            target_acc=args.target_accuracy,
            eval_steps=500,
            output_dir=output_dir,
        )
        callbacks.append(acc_callback)

    # Train
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    # Save final
    print("\nSaving final model...")
    model.save_pretrained(str(Path(output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(output_dir) / "final"))

    # Save config
    config = {
        "source": args.source,
        "type": args.type,
        "model": args.model,
        "hyperparams": hp,
        "train_samples": len(train_ds),
        "completed": str(datetime.now()),
    }
    with open(Path(output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
