#!/usr/bin/env python3
"""
Train Korean Medical MCQ from Stage 6 until 90% accuracy

This script:
1. Loads Stage 5 base model with Stage 6 LoRA adapter
2. Continues training on filtered MCQ data
3. Evaluates and loops until 90% accuracy achieved
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# GPU setup
from config.gpu_utils import setup_gpu, print_memory_usage

# =============================================================================
# Configuration
# =============================================================================
TARGET_ACCURACY = 90.0
MAX_EPOCHS = 20
EVAL_EVERY_N_STEPS = 500

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "korean_medical_mcq_filtered"
STAGE5_MODEL = BASE_DIR / "models" / "staged_training" / "stage5_harmonization"
STAGE6_ADAPTER = BASE_DIR / "models" / "staged_training" / "stage6_hybrid_expansion"
OUTPUT_DIR = BASE_DIR / "models" / "korean_mcq_90_accuracy"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1024
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

print("=" * 60)
print("Korean Medical MCQ Training - From Stage 6")
print("Target: 90% Accuracy on Filtered MCQ Data")
print("=" * 60)
print(f"Stage 5 Base Model: {STAGE5_MODEL}")
print(f"Stage 6 Adapter: {STAGE6_ADAPTER}")
print(f"Data: {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")


# =============================================================================
# Accuracy Evaluation Callback
# =============================================================================
class AccuracyEvaluationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, verification_dataset, target_accuracy, eval_every_n_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.verification_dataset = verification_dataset
        self.target_accuracy = target_accuracy
        self.eval_every_n_steps = eval_every_n_steps
        self.best_accuracy = 0.0
        self.accuracy_history = []
        self.reached_target = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0 and state.global_step > 0:
            accuracy = self.evaluate_accuracy()
            self.accuracy_history.append({
                "step": state.global_step,
                "accuracy": accuracy,
                "timestamp": str(datetime.now())
            })
            print(f"\n[Step {state.global_step}] Verification Accuracy: {accuracy:.2f}%")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f"  New best accuracy! Saving checkpoint...")
                self.model.save_pretrained(str(OUTPUT_DIR / "best_checkpoint"))
                self.tokenizer.save_pretrained(str(OUTPUT_DIR / "best_checkpoint"))

            if accuracy >= self.target_accuracy:
                print(f"\n🎉 Target accuracy {self.target_accuracy}% reached!")
                self.reached_target = True
                control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        accuracy = self.evaluate_accuracy()
        self.accuracy_history.append({
            "epoch": state.epoch,
            "step": state.global_step,
            "accuracy": accuracy,
            "timestamp": str(datetime.now())
        })
        print(f"\n[Epoch {state.epoch}] Verification Accuracy: {accuracy:.2f}%")

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"  New best accuracy! Saving checkpoint...")
            self.model.save_pretrained(str(OUTPUT_DIR / "best_checkpoint"))
            self.tokenizer.save_pretrained(str(OUTPUT_DIR / "best_checkpoint"))

        if accuracy >= self.target_accuracy:
            print(f"\n🎉 Target accuracy {self.target_accuracy}% reached!")
            self.reached_target = True
            control.should_training_stop = True

        # Save accuracy history
        with open(OUTPUT_DIR / "accuracy_history.json", "w") as f:
            json.dump({
                "target_accuracy": self.target_accuracy,
                "best_accuracy": self.best_accuracy,
                "history": self.accuracy_history
            }, f, indent=2)
        return control

    def evaluate_accuracy(self, max_samples=500):
        """Evaluate accuracy on verification set"""
        self.model.eval()
        correct = 0
        total = 0

        indices = list(range(len(self.verification_dataset)))
        if len(indices) > max_samples:
            import random
            random.shuffle(indices)
            indices = indices[:max_samples]

        for idx in tqdm(indices, desc="Evaluating", leave=False):
            example = self.verification_dataset[idx]
            try:
                text = example["text"]
                correct_answer = example.get("correct_answer", "")

                # Extract user question from ChatML
                if "<|im_start|>user" in text and "<|im_end|>" in text:
                    user_start = text.find("<|im_start|>user") + len("<|im_start|>user\n")
                    user_end = text.find("<|im_end|>", user_start)
                    user_content = text[user_start:user_end].strip()

                    prompt = f"""<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다. 정확하고 도움이 되는 의료 정보를 제공하세요.
<|im_end|>
<|im_start|>user
{user_content}

정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).
<|im_end|>
<|im_start|>assistant
"""
                else:
                    continue

                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().upper()

                # Extract predicted letter
                predicted = None
                for letter in ['A', 'B', 'C', 'D', 'E']:
                    if letter in response[:5]:
                        predicted = letter
                        break

                # Compare with expected (already clean A-E letter from filtered data)
                expected = correct_answer.upper().strip()

                if predicted and predicted == expected:
                    correct += 1
                total += 1

            except Exception as e:
                continue

            if total % 50 == 0:
                torch.cuda.empty_cache()

        self.model.train()
        accuracy = (correct / total * 100) if total > 0 else 0
        return accuracy


# =============================================================================
# Load Data
# =============================================================================
print("\n" + "=" * 60)
print("Loading Data")
print("=" * 60)

train_dataset = load_from_disk(str(DATA_DIR / "train"))
verification_dataset = load_from_disk(str(DATA_DIR / "verification"))

print(f"Training samples: {len(train_dataset)}")
print(f"Verification samples: {len(verification_dataset)}")


# =============================================================================
# Load Model
# =============================================================================
print("\n" + "=" * 60)
print("Loading Stage 5 Model + Stage 6 Adapter")
print("=" * 60)

device = setup_gpu("config/gpu_config.json")
print_memory_usage()

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load Stage 5 base model
print("Loading Stage 5 base model...")
model = AutoModelForCausalLM.from_pretrained(
    str(STAGE5_MODEL),
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

# Load tokenizer from Stage 6 (has same vocab)
tokenizer = AutoTokenizer.from_pretrained(str(STAGE6_ADAPTER))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print_memory_usage()

# Load Stage 6 LoRA adapter
print("Loading Stage 6 LoRA adapter...")
model = PeftModel.from_pretrained(model, str(STAGE6_ADAPTER))
print(f"Stage 6 adapter loaded!")

# Prepare for continued training
model = prepare_model_for_kbit_training(model)

# Make adapter trainable
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

model.print_trainable_parameters()
print_memory_usage()


# =============================================================================
# Training Setup
# =============================================================================
print("\n" + "=" * 60)
print("Setting Up Training")
print("=" * 60)

sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=MAX_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=False,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
    dataloader_num_workers=4,
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
)

accuracy_callback = AccuracyEvaluationCallback(
    model=model,
    tokenizer=tokenizer,
    verification_dataset=verification_dataset,
    target_accuracy=TARGET_ACCURACY,
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
)


# =============================================================================
# Training
# =============================================================================
print("\n" + "=" * 60)
print("Starting Training")
print("=" * 60)
print(f"Target: {TARGET_ACCURACY}% accuracy")
print(f"Evaluate every {EVAL_EVERY_N_STEPS} steps + epoch end")

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    callbacks=[accuracy_callback],
)

# Initial evaluation
print("\nInitial evaluation before training:")
initial_accuracy = accuracy_callback.evaluate_accuracy()
print(f"Initial accuracy: {initial_accuracy:.2f}%")

accuracy_callback.accuracy_history.append({
    "step": 0,
    "accuracy": initial_accuracy,
    "timestamp": str(datetime.now()),
    "note": "initial"
})

# Train
trainer.train()

# Save final model
print("\n" + "=" * 60)
print("Saving Final Model")
print("=" * 60)

model.save_pretrained(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

# Save final results
final_results = {
    "target_accuracy": TARGET_ACCURACY,
    "best_accuracy": accuracy_callback.best_accuracy,
    "reached_target": accuracy_callback.reached_target,
    "history": accuracy_callback.accuracy_history,
    "training_completed": str(datetime.now()),
}

with open(OUTPUT_DIR / "training_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\nTraining Complete!")
print(f"Best accuracy: {accuracy_callback.best_accuracy:.2f}%")
print(f"Target reached: {accuracy_callback.reached_target}")
print(f"Results saved to: {OUTPUT_DIR}")
