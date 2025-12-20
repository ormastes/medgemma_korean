#!/usr/bin/env python3
"""
Train on KorMedMCQA ONLY with verbose iteration logging
Shows accuracy verification during training
"""

import argparse
import json
import re
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "models" / "kormedmcqa_only"

MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 64, "lora_alpha": 128,
        "lr": 1e-4, "batch": 4, "grad_accum": 8,
        "max_length": 512, "grad_ckpt": False
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 384, "lora_alpha": 768,
        "lr": 5e-5, "batch": 4, "grad_accum": 8,
        "max_length": 1024, "grad_ckpt": True
    }
}


class VerboseAccuracyCallback(TrainerCallback):
    """Verbose callback showing iteration details and accuracy"""

    def __init__(self, eval_dataset, tokenizer, model, eval_steps=100,
                 target_accuracy=90.0, output_dir=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.eval_steps = eval_steps
        self.target = target_accuracy
        self.output_dir = output_dir
        self.best_acc = 0.0
        self.current_acc = 0.0
        self.best_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log every iteration with detailed info"""
        if logs and state.global_step > 0:
            logs['current_acc'] = self.current_acc
            logs['best_acc'] = self.best_acc
            
            # Verbose logging every step
            sep = "=" * 70
            print(f"\n{sep}")
            print(f"ITERATION {state.global_step}")
            print(f"{sep}")
            print(f"  Loss: {logs.get('loss', 0):.4f}")
            print(f"  Learning Rate: {logs.get('learning_rate', 0):.2e}")
            print(f"  Current Val Accuracy: {self.current_acc:.2f}%")
            print(f"  Best Accuracy: {self.best_acc:.2f}% (step {self.best_step})")
            print(f"  Target: {self.target}%")
            print(f"  Progress: {self.current_acc/self.target*100:.1f}% to target")
            print(f"{sep}\n")

    def on_step_end(self, args, state, control, **kwargs):
        # Evaluate every eval_steps
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            acc = self._evaluate(state.global_step)
            self.current_acc = acc

            if acc > self.best_acc:
                self.best_acc = acc
                self.best_step = state.global_step
                self._save_checkpoint(state.global_step)

            if acc >= self.target:
                print(f"\n🎯 TARGET ACCURACY {self.target}% REACHED! 🎯")
                print(f"Stopping training at step {state.global_step}")
                control.should_training_stop = True

    def _evaluate(self, step: int):
        hsep = "#" * 70
        print(f"\n{hsep}")
        print(f"VALIDATION EVALUATION - Step {step}")
        print(f"{hsep}")

        self.model.eval()
        correct = 0
        total = 0
        
        # Show sample predictions
        samples = list(self.eval_dataset)[:200]
        show_examples = 5  # Show first 5 predictions

        with torch.no_grad():
            for idx, sample in enumerate(samples):
                try:
                    prompt = sample['prompt']
                    expected = sample['answer'].strip().upper()

                    inputs = self.tokenizer(prompt, return_tensors="pt",
                                           truncation=True, max_length=800).to(self.model.device)

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()

                    # Extract answer
                    predicted = response.split()[0] if response.split() else ""
                    predicted = re.sub(r'[^A-E]', '', predicted.upper())
                    predicted = predicted[0] if predicted else ""

                    is_correct = (predicted == expected)
                    if is_correct:
                        correct += 1
                    total += 1

                    # Show first few examples
                    if idx < show_examples:
                        print(f"\n  Example {idx+1}:")
                        print(f"    Expected: {expected}")
                        print(f"    Predicted: {predicted}")
                        print(f"    Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
                        print(f"    Full response: {response[:100]}")

                except Exception as e:
                    print(f"  Error on sample {idx}: {e}")
                    continue

        acc = 100 * correct / total if total > 0 else 0

        sep = "=" * 70
        print(f"\n{sep}")
        print(f"VALIDATION RESULTS")
        print(f"{sep}")
        print(f"  Total Samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Wrong: {total - correct}")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Best So Far: {self.best_acc:.2f}% (step {self.best_step})")
        print(f"  Target: {self.target}%")
        print(f"  Gap to Target: {self.target - acc:.2f}%")
        print(f"{sep}\n")

        self.model.train()
        return acc

    def _save_checkpoint(self, step: int):
        if self.output_dir:
            ckpt = Path(self.output_dir) / "best_checkpoint"
            ckpt.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(ckpt)
            self.tokenizer.save_pretrained(ckpt)
            with open(ckpt / "info.json", 'w') as f:
                json.dump({"step": step, "accuracy": self.best_acc}, f, indent=2)
            print(f"💾 Saved checkpoint at step {step} with accuracy {self.best_acc:.2f}%")


def load_kormedmcqa_only(data_dir: Path):
    """Load only KorMedMCQA samples from type3 data"""
    train, val = [], []
    
    for split, samples in [("train", train), ("validation", val)]:
        f = data_dir / split / "data.jsonl"
        if f.exists():
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    # Filter only kormedmcqa source
                    if data.get('source', '').lower() == 'kormedmcqa':
                        samples.append(data)
    
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"KORMEDMCQA DATA LOADED")
    print(f"{sep}")
    print(f"  Train samples: {len(train)}")
    print(f"  Validation samples: {len(val)}")
    print(f"{sep}\n")
    
    return Dataset.from_list(train), Dataset.from_list(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma-27b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--target-accuracy", type=float, default=90.0)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--base-model", type=str, default=None, help="Path to base model (default: stage6 or HF)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    
    # Load base model
    if args.base_model:
        model_path = args.base_model
    else:
        # Try stage6 model first, fallback to HF
        stage6 = BASE_DIR / "models" / "staged_training" / "stage6"
        model_path = str(stage6) if stage6.exists() else cfg["path"]

    hsep = "#" * 70
    print(f"\n{hsep}")
    print(f"KORMEDMCQA-ONLY TRAINING")
    print(f"{hsep}")
    print(f"  Base Model: {model_path}")
    print(f"  Target Accuracy: {args.target_accuracy}%")
    print(f"  Epochs: {args.epochs}")
    print(f"  Eval Steps: {args.eval_steps}")
    print(f"{hsep}\n")

    # Load data - KorMedMCQA only
    type3_dir = BASE_DIR / "data" / "reviewed" / "type3_word"
    train_ds, val_ds = load_kormedmcqa_only(type3_dir)

    # Load model
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    print(f"Trainable params: {model.num_parameters(only_trainable=True):,}")

    # Training config
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_steps=50,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=cfg["grad_ckpt"],
        dataset_text_field="text",
        packing=False
    )

    # Callback
    callback = VerboseAccuracyCallback(
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        model=model,
        eval_steps=args.eval_steps,
        target_accuracy=args.target_accuracy,
        output_dir=OUTPUT_DIR
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[callback]
    )

    print("\n🚀 Starting training...\n")
    trainer.train()

    # Save final model
    final_dir = OUTPUT_DIR / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"\n✅ Training complete!")
    print(f"Best accuracy: {callback.best_acc:.2f}% at step {callback.best_step}")
    print(f"Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()
