#!/usr/bin/env python3
"""
Train Korean Medical MCQ with MedGemma 27B

This script:
1. Loads MedGemma 27B text model (87.7% MedQA baseline)
2. Adds Korean vocabulary expansion
3. Trains with bigger LoRA for 90% Korean accuracy
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# GPU setup
from config.gpu_utils import setup_gpu, print_memory_usage

# =============================================================================
# Configuration
# =============================================================================
TARGET_ACCURACY = 90.0
MAX_EPOCHS = 10
EVAL_EVERY_N_STEPS = 200

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "korean_medical_mcq_filtered"
OUTPUT_DIR = BASE_DIR / "models" / "medgemma_27b_korean"

# Model
MODEL_ID = "google/medgemma-27b-text-it"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters - adjusted for 27B model on 48GB GPU
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 32  # Effective batch = 32
LEARNING_RATE = 2e-5  # Lower LR for larger model
MAX_SEQ_LENGTH = 512  # Shorter to save memory
WARMUP_RATIO = 0.05

# Bigger LoRA for 27B model
LORA_R = 128  # Much bigger rank
LORA_ALPHA = 256  # 2x rank
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

print("=" * 60)
print("MedGemma 27B Korean Medical MCQ Training")
print("Target: 90% Accuracy")
print("=" * 60)
print(f"Model: {MODEL_ID}")
print(f"Data: {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")


# =============================================================================
# Accuracy Evaluation Callback
# =============================================================================
class AccuracyEvaluationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, verification_dataset, target_accuracy, eval_every_n_steps, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.verification_dataset = verification_dataset
        self.target_accuracy = target_accuracy
        self.eval_every_n_steps = eval_every_n_steps
        self.output_dir = output_dir
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
                self.model.save_pretrained(str(self.output_dir / "best_checkpoint"))
                self.tokenizer.save_pretrained(str(self.output_dir / "best_checkpoint"))

            # Save history after each eval
            self._save_history()

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
            self.model.save_pretrained(str(self.output_dir / "best_checkpoint"))
            self.tokenizer.save_pretrained(str(self.output_dir / "best_checkpoint"))

        if accuracy >= self.target_accuracy:
            print(f"\n🎉 Target accuracy {self.target_accuracy}% reached!")
            self.reached_target = True
            control.should_training_stop = True

        self._save_history()
        return control

    def _save_history(self):
        with open(self.output_dir / "accuracy_history.json", "w") as f:
            json.dump({
                "target_accuracy": self.target_accuracy,
                "best_accuracy": self.best_accuracy,
                "history": self.accuracy_history
            }, f, indent=2)

    def evaluate_accuracy(self, max_samples=300):
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

                    # Use Gemma chat format
                    prompt = f"""<start_of_turn>user
{user_content}

정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).
<end_of_turn>
<start_of_turn>model
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

                # Compare with expected
                expected = correct_answer.upper().strip()

                if predicted and predicted == expected:
                    correct += 1
                total += 1

            except Exception as e:
                continue

            if total % 30 == 0:
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
print(f"Loading {MODEL_ID}")
print("=" * 60)

device = setup_gpu("config/gpu_config.json")
print_memory_usage()

# 4-bit quantization for 27B model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {MODEL_ID} with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print_memory_usage()

# Prepare for LoRA training
model = prepare_model_for_kbit_training(model)

# Add bigger LoRA
print(f"\nAdding LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
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
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    logging_steps=25,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,  # Enable for memory savings
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
    dataloader_num_workers=2,
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
)

accuracy_callback = AccuracyEvaluationCallback(
    model=model,
    tokenizer=tokenizer,
    verification_dataset=verification_dataset,
    target_accuracy=TARGET_ACCURACY,
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    output_dir=OUTPUT_DIR,
)


# =============================================================================
# Training
# =============================================================================
print("\n" + "=" * 60)
print("Starting Training")
print("=" * 60)
print(f"Target: {TARGET_ACCURACY}% accuracy")
print(f"Evaluate every {EVAL_EVERY_N_STEPS} steps + epoch end")
print(f"LoRA params: r={LORA_R}, alpha={LORA_ALPHA}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

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
accuracy_callback._save_history()

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
    "model": MODEL_ID,
    "target_accuracy": TARGET_ACCURACY,
    "best_accuracy": accuracy_callback.best_accuracy,
    "reached_target": accuracy_callback.reached_target,
    "lora_config": {
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
    },
    "history": accuracy_callback.accuracy_history,
    "training_completed": str(datetime.now()),
}

with open(OUTPUT_DIR / "training_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\nTraining Complete!")
print(f"Best accuracy: {accuracy_callback.best_accuracy:.2f}%")
print(f"Target reached: {accuracy_callback.reached_target}")
print(f"Results saved to: {OUTPUT_DIR}")
