#!/usr/bin/env python3
"""
Train Korean Medical Model Until 90% Verification Accuracy

This script:
1. Loads the 75/25 split Korean medical exam data
2. Trains using QLoRA instruction tuning
3. Evaluates on verification set after each epoch
4. Continues training until 90% accuracy is achieved on verification set
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
TARGET_ACCURACY = 90.0  # Target 90% accuracy
MAX_EPOCHS = 20  # Maximum epochs to try
EVAL_EVERY_N_STEPS = 500  # Evaluate every N steps

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "korean_medical_mcq_filtered"  # Filtered MCQ only data
VERIFICATION_DIR = BASE_DIR / "data" / "processed" / "korean_medical_mcq_filtered"  # Same filtered data
MODEL_DIR = BASE_DIR / "models" / "staged_training" / "stage5_harmonization"  # Stage 5 model
OUTPUT_DIR = BASE_DIR / "models" / "korean_medical_90_accuracy_v2"  # New output dir
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1024
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

print("=" * 60)
print("Korean Medical MCQ Training - Target 90% Accuracy")
print("Using Stage 5 Model + Filtered MCQ Data")
print("=" * 60)
print(f"Target accuracy: {TARGET_ACCURACY}%")
print(f"Max epochs: {MAX_EPOCHS}")
print(f"Data directory: {DATA_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Output directory: {OUTPUT_DIR}")


# =============================================================================
# Custom Callback for Accuracy Evaluation
# =============================================================================
class AccuracyEvaluationCallback(TrainerCallback):
    """Callback to evaluate accuracy on verification set and stop when target is reached"""

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
        # Evaluate periodically
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
                # Save best model
                self.model.save_pretrained(str(OUTPUT_DIR / "best_checkpoint"))
                self.tokenizer.save_pretrained(str(OUTPUT_DIR / "best_checkpoint"))

            if accuracy >= self.target_accuracy:
                print(f"\n🎉 Target accuracy {self.target_accuracy}% reached!")
                self.reached_target = True
                control.should_training_stop = True

        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        # Always evaluate at end of epoch
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

        # Sample subset for faster evaluation during training
        indices = list(range(len(self.verification_dataset)))
        if len(indices) > max_samples:
            import random
            random.shuffle(indices)
            indices = indices[:max_samples]

        for idx in tqdm(indices, desc="Evaluating", leave=False):
            example = self.verification_dataset[idx]
            try:
                # Extract question and expected answer from text
                text = example["text"]
                correct_answer = example.get("correct_answer", "")

                # Create prompt (extract user part from ChatML)
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
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().upper()

                # Extract predicted answer
                predicted = None
                for letter in ['A', 'B', 'C', 'D', 'E']:
                    if letter in response[:5]:
                        predicted = letter
                        break

                # Check if correct (handle various answer formats)
                expected = correct_answer.upper().strip()
                if expected.startswith("정답은"):
                    for letter in ['A', 'B', 'C', 'D', 'E']:
                        if letter in expected[:20]:
                            expected = letter
                            break

                if predicted and predicted == expected[:1]:
                    correct += 1
                total += 1

            except Exception as e:
                continue

            # Clear cache periodically
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

if not DATA_DIR.exists():
    print(f"Data not found at {DATA_DIR}")
    print("Please run filter_mcq_data.py first")
    sys.exit(1)

# Load train and verification datasets (separate directories for filtered data)
train_dataset = load_from_disk(str(DATA_DIR / "train"))
verification_dataset = load_from_disk(str(DATA_DIR / "verification"))

print(f"Training samples: {len(train_dataset)}")
print(f"Verification samples: {len(verification_dataset)}")

# Preview sample
print("\nSample training data:")
sample = train_dataset[0]
print(f"Text: {sample['text'][:200]}...")


# =============================================================================
# Load Model and Tokenizer
# =============================================================================
print("\n" + "=" * 60)
print("Loading Model")
print("=" * 60)

device = setup_gpu("config/gpu_config.json")
print_memory_usage()

# Use Stage 5 model (after harmonization training)
base_model_path = MODEL_DIR
print(f"Loading from Stage 5 model: {base_model_path}")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    str(base_model_path),
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded!")
print_memory_usage()

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =============================================================================
# Training Arguments (using SFTConfig for TRL 0.26+)
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
    gradient_checkpointing=False,  # Disabled for PEFT compatibility
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
    dataloader_num_workers=4,
    # SFT-specific settings
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
)

# Create accuracy callback
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
print(f"Target: {TARGET_ACCURACY}% accuracy on verification set")
print(f"Will evaluate every {EVAL_EVERY_N_STEPS} steps and at each epoch end")

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

# Train
trainer.train()

# Final evaluation
print("\n" + "=" * 60)
print("Training Complete - Final Evaluation")
print("=" * 60)

final_accuracy = accuracy_callback.evaluate_accuracy(max_samples=len(verification_dataset))
print(f"Final accuracy: {final_accuracy:.2f}%")
print(f"Best accuracy achieved: {accuracy_callback.best_accuracy:.2f}%")

# Save final model
print("\nSaving final model...")
model.save_pretrained(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

# Save training results
results = {
    "target_accuracy": TARGET_ACCURACY,
    "final_accuracy": final_accuracy,
    "best_accuracy": accuracy_callback.best_accuracy,
    "reached_target": accuracy_callback.reached_target,
    "accuracy_history": accuracy_callback.accuracy_history,
    "training_config": {
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_epochs": MAX_EPOCHS,
    },
    "data_stats": {
        "train_samples": len(train_dataset),
        "verification_samples": len(verification_dataset),
    },
    "timestamp": str(datetime.now()),
}

with open(OUTPUT_DIR / "training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR / 'training_results.json'}")

if accuracy_callback.reached_target:
    print(f"\n✅ SUCCESS: Achieved {accuracy_callback.best_accuracy:.2f}% accuracy (target: {TARGET_ACCURACY}%)")
else:
    print(f"\n⚠️ Did not reach target accuracy of {TARGET_ACCURACY}%")
    print(f"   Best accuracy achieved: {accuracy_callback.best_accuracy:.2f}%")
    print(f"   Consider continuing training with higher max_epochs")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
