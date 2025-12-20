#!/usr/bin/env python3
"""
Train Type 2: TEXT_REASONING
- Full text answer WITH <R>reasoning<R/> tokens
- Evaluation: Perplexity (lower is better)
- Target: Perplexity < 3.0
"""

import argparse
import json
import math
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "refined" / "type2_text_reasoning"
OUTPUT_DIR = BASE_DIR / "models" / "type2_text_reasoning"

# Special tokens
R_START = "<R>"
R_END = "<R/>"

MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 64, "lora_alpha": 128,
        "lr": 2e-5, "batch": 2, "grad_accum": 16,
        "max_length": 1024, "grad_ckpt": False
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 128, "lora_alpha": 256,
        "lr": 2e-5, "batch": 1, "grad_accum": 32,
        "max_length": 512, "grad_ckpt": True
    }
}


class PerplexityWithReasoningCallback(TrainerCallback):
    """Evaluate perplexity, check reasoning tokens used correctly"""

    def __init__(self, eval_dataset, tokenizer, model, eval_steps=500,
                 target_perplexity=3.0, output_dir=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.eval_steps = eval_steps
        self.target = target_perplexity
        self.output_dir = output_dir
        self.best_ppl = float('inf')
        self.best_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            ppl, reason_pct = self._evaluate(state.global_step)

            if ppl < self.best_ppl:
                self.best_ppl = ppl
                self.best_step = state.global_step
                self._save_checkpoint(state.global_step)

            if ppl <= self.target and reason_pct >= 90:
                print(f"\nTarget reached! PPL={ppl:.2f}, Reasoning={reason_pct}%")
                control.should_training_stop = True

    def _evaluate(self, step: int):
        print(f"\n{'='*50}")
        print(f"Type 2 (TEXT_REASONING) Evaluation - Step {step}")
        print(f"{'='*50}")

        self.model.eval()
        total_loss = 0
        count = 0
        has_reasoning = 0

        samples = list(self.eval_dataset)[:100]

        with torch.no_grad():
            for sample in samples:
                try:
                    prompt = sample['prompt']
                    inputs = self.tokenizer(prompt, return_tensors="pt",
                                           truncation=True, max_length=256).to(self.model.device)

                    # Generate
                    outputs = self.model.generate(**inputs, max_new_tokens=200,
                                                   do_sample=False,
                                                   pad_token_id=self.tokenizer.pad_token_id)
                    response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                                     skip_special_tokens=False)

                    # Check reasoning tokens
                    if R_START in response and R_END in response:
                        has_reasoning += 1

                    # Calculate loss
                    text = sample['text']
                    full_inputs = self.tokenizer(text, return_tensors="pt",
                                                truncation=True, max_length=512).to(self.model.device)
                    loss_outputs = self.model(**full_inputs, labels=full_inputs['input_ids'])
                    total_loss += loss_outputs.loss.item()
                    count += 1

                except:
                    continue

        avg_loss = total_loss / count if count > 0 else float('inf')
        ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        reason_pct = 100 * has_reasoning / count if count > 0 else 0

        print(f"  Samples: {count}")
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Using {R_START}...{R_END}: {has_reasoning}/{count} ({reason_pct:.1f}%)")
        print(f"  Best: {self.best_ppl:.2f} (step {self.best_step})")

        self.model.train()
        return ppl, reason_pct

    def _save_checkpoint(self, step: int):
        if self.output_dir:
            ckpt = Path(self.output_dir) / "best_checkpoint"
            ckpt.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(ckpt)
            self.tokenizer.save_pretrained(ckpt)
            with open(ckpt / "info.json", 'w') as f:
                json.dump({"step": step, "perplexity": self.best_ppl}, f)


def load_data(data_dir: Path):
    train, val = [], []
    for split, samples in [("train", train), ("validation", val)]:
        f = data_dir / split / "data.jsonl"
        if f.exists():
            with open(f, 'r', encoding='utf-8') as file:
                samples.extend([json.loads(l) for l in file])
    print(f"Loaded {len(train)} train, {len(val)} val")
    return Dataset.from_list(train), Dataset.from_list(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma-4b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--target-perplexity", type=float, default=3.0)
    parser.add_argument("--eval-steps", type=int, default=500)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    output_dir = str(OUTPUT_DIR / args.model)

    print("=" * 60)
    print(f"Type 2: TEXT_REASONING Training (With {R_START}...{R_END})")
    print("=" * 60)

    train_data, eval_data = load_data(DATA_DIR)
    if len(train_data) == 0:
        print("No data! Run refine_4types.py first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    tokenizer.add_special_tokens({"additional_special_tokens": [R_START, R_END]})

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg['path'], quantization_config=bnb,
                                                  device_map="auto", trust_remote_code=True,
                                                  attn_implementation="eager")
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=cfg['batch'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'], weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type="cosine", logging_steps=10,
        save_strategy="steps", save_steps=1000,
        bf16=True, max_length=cfg['max_length'],
        gradient_checkpointing=cfg['grad_ckpt'],
        optim="paged_adamw_8bit", max_grad_norm=0.3, report_to="none"
    )

    callback = PerplexityWithReasoningCallback(eval_data, tokenizer, model,
                                                args.eval_steps, args.target_perplexity, output_dir)

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_data,
                         processing_class=tokenizer, callbacks=[callback])
    trainer.train()

    trainer.save_model(output_dir + "/final")
    tokenizer.save_pretrained(output_dir + "/final")
    print(f"\nDone! Best perplexity: {callback.best_ppl:.2f}")


if __name__ == "__main__":
    main()
