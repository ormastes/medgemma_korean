#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 01 with Train 00 Quality Monitoring

This script trains on medical dictionary (train_01) while periodically
checking plain text quality (train_00). When train_00 quality drops below
a threshold, it automatically switches back to train_00 training until
quality recovers.

Strategy:
1. Train on train_01 data for N steps
2. Check train_00 validation loss (lower = better Korean fluency)
3. If loss > upper_bound: switch to train_00 training
4. Train on train_00 until loss < lower_bound
5. Switch back to train_01
6. Repeat until train_01 epochs complete

Directory Structure:
    Input:  model/00_trained/{model}/ (from train_00)
    Training: model/01_training/{model}/ (checkpoints)
    Output: model/01_trained/{model}/ (final model)
    Merged: model/01_another_lora_added/{model}/ (merged + new LoRA)

Usage:
    python train_01_with_00_monitor.py --model medgemma-27b --epochs 3
    python train_01_with_00_monitor.py --model medgemma-27b --upper-bound 2.5 --lower-bound 2.0
"""

import sys
import json
import random
import torch
import gc
import traceback
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from training_utils import (
    create_base_parser, load_tokenizer, load_jsonl_data,
    create_training_args, save_training_info
)
from training_config import MODEL_CONFIGS, MEMORY_CONFIGS
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

BASE_DIR = Path(__file__).parent.parent

# Data paths
TRAIN_00_DATA_DIR = BASE_DIR / "data" / "02_refined" / "00_plain_text"
TRAIN_01_MEDICAL_DICT = BASE_DIR / "data" / "02_refined" / "01_medical_dict.json"
TRAIN_01_CHAR_DICT = BASE_DIR / "data" / "02_refined" / "02_char_dict.json"
VALIDATION_DATA_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"

# Model paths
INPUT_DIR = BASE_DIR / "model" / "00_trained"
TRAINING_DIR = BASE_DIR / "model" / "01_training"
OUTPUT_DIR = BASE_DIR / "model" / "01_trained"
MERGED_DIR = BASE_DIR / "model" / "01_another_lora_added"

# Log file
LOG_FILE = TRAINING_DIR / "train_01_monitor_debug.log"

# Default thresholds for train_00 quality (perplexity-based)
DEFAULT_UPPER_BOUND = 3.0   # Switch to train_00 when loss > this
DEFAULT_LOWER_BOUND = 2.0   # Switch back to train_01 when loss < this

# Training prompt templates
TRAIN_01_TEMPLATE = """<start_of_turn>user
Meaning of word {term}:<end_of_turn>
<start_of_turn>model
{definition}<end_of_turn>"""

VALIDATION_PROMPT_TEMPLATE = """<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

{question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}

<end_of_turn>
<start_of_turn>model
"""


def log(msg: str, level: str = "INFO"):
    """Write debug log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    print(log_msg)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    except:
        pass


def clear_gpu_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_input_model_path(model_name: str) -> str:
    """Get input model path from 00_trained/"""
    input_path = INPUT_DIR / model_name

    if input_path.exists() and (input_path / "adapter_config.json").exists():
        log(f"Found trained model: {input_path}", "INFO")
        return str(input_path)

    raise ValueError(f"Model not found: {input_path}\nRun train_00 first!")


def load_train_00_data(max_samples: int = None) -> list:
    """Load train_00 plain text data."""
    train_data, _ = load_jsonl_data(TRAIN_00_DATA_DIR, max_samples=max_samples)
    log(f"Loaded {len(train_data)} train_00 samples", "INFO")
    return train_data


def load_train_01_data(max_samples: int = None) -> list:
    """Load train_01 medical dictionary data."""
    data = []

    if TRAIN_01_MEDICAL_DICT.exists():
        with open(TRAIN_01_MEDICAL_DICT, 'r', encoding='utf-8') as f:
            medical_dict = json.load(f)
            log(f"Loaded {len(medical_dict)} medical terms", "INFO")
            data.extend(medical_dict)

    if TRAIN_01_CHAR_DICT.exists():
        with open(TRAIN_01_CHAR_DICT, 'r', encoding='utf-8') as f:
            char_dict = json.load(f)
            log(f"Loaded {len(char_dict)} symbol terms", "INFO")
            data.extend(char_dict)

    random.shuffle(data)

    if max_samples and len(data) > max_samples:
        data = data[:max_samples]

    log(f"Total train_01 entries: {len(data)}", "INFO")
    return data


def load_kormedmcqa_data(max_samples: int = None) -> list:
    """Load KorMedMCQA test data for validation."""
    data = []
    test_file = VALIDATION_DATA_DIR / "test.jsonl"

    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line))
        log(f"Loaded {len(data)} KorMedMCQA samples", "INFO")

    return data


def format_train_01_entry(entry: dict) -> dict:
    """Format train_01 dictionary entry."""
    text = TRAIN_01_TEMPLATE.format(
        term=entry['term'],
        definition=entry['definition']
    )
    return {"text": text}


def compute_train_00_loss(model, tokenizer, val_samples, device: str, num_samples: int = 50) -> float:
    """
    Compute average loss on train_00 validation samples.
    Lower loss = better Korean language quality.

    Args:
        val_samples: Can be a list or a HuggingFace Dataset
    """
    model.eval()
    total_loss = 0.0
    count = 0

    # Sample random validation texts - handle both list and Dataset
    sample_size = min(num_samples, len(val_samples))
    if hasattr(val_samples, 'shuffle'):
        # HuggingFace Dataset - use shuffle and select
        samples = val_samples.shuffle(seed=random.randint(0, 10000)).select(range(sample_size))
    else:
        # Regular list
        samples = random.sample(val_samples, sample_size)

    with torch.no_grad():
        for sample in samples:
            text = sample.get('text', '')
            if not text or len(text) < 10:
                continue

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Add labels (same as input for language modeling)
            inputs['labels'] = inputs['input_ids'].clone()

            try:
                outputs = model(**inputs)
                loss = outputs.loss.item()
                total_loss += loss
                count += 1
            except Exception as e:
                log(f"Loss computation error: {e}", "WARNING")
                continue

    model.train()

    if count == 0:
        return float('inf')

    avg_loss = total_loss / count
    return avg_loss


def evaluate_kormedmcqa(model, tokenizer, test_data, device: str, num_samples: int = 20) -> float:
    """Evaluate on KorMedMCQA and return accuracy.

    Args:
        test_data: Can be a list or a HuggingFace Dataset
    """
    model.eval()
    correct = 0
    total = 0

    # Sample random test samples - handle both list and Dataset
    sample_size = min(num_samples, len(test_data))
    if hasattr(test_data, 'shuffle'):
        # HuggingFace Dataset
        samples = test_data.shuffle(seed=random.randint(0, 10000)).select(range(sample_size))
    else:
        # Regular list
        samples = random.sample(test_data, sample_size)

    # Get end_of_turn token
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    terminators = [tokenizer.eos_token_id]
    if end_of_turn_id != tokenizer.unk_token_id:
        terminators.append(end_of_turn_id)

    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    with torch.no_grad():
        for sample in samples:
            prompt = VALIDATION_PROMPT_TEMPLATE.format(
                question=sample['question'],
                A=sample['A'], B=sample['B'], C=sample['C'],
                D=sample['D'], E=sample['E']
            )
            expected = sample['answer'].upper()

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=terminators,
                )

                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                predicted = ""
                for char in response.strip().upper():
                    if char in 'ABCDE':
                        predicted = char
                        break

                if predicted == expected:
                    correct += 1
                total += 1

            except Exception as e:
                log(f"Evaluation error: {e}", "WARNING")
                continue

    tokenizer.padding_side = original_padding
    model.train()

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy


class MonitoredTrainer:
    """
    Custom trainer that monitors train_00 quality while training on train_01.
    Automatically switches between datasets based on quality thresholds.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_00_data: list,
        train_01_data: list,
        kormedmcqa_data: list,
        cfg: dict,
        args,
        training_dir: Path,
        upper_bound: float = DEFAULT_UPPER_BOUND,
        lower_bound: float = DEFAULT_LOWER_BOUND,
        check_interval: int = 100,  # Check every N steps
        train_00_recovery_steps: int = 50,  # Steps to train on train_00 when recovering
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_00_data = train_00_data
        self.train_01_data = train_01_data
        self.kormedmcqa_data = kormedmcqa_data
        self.cfg = cfg
        self.args = args
        self.training_dir = training_dir
        self.device = args.device

        # Quality thresholds
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.check_interval = check_interval
        self.train_00_recovery_steps = train_00_recovery_steps

        # Tracking
        self.global_step = 0
        self.train_01_steps = 0
        self.train_00_recovery_count = 0
        self.history = []

        # Current mode
        self.current_mode = "train_01"  # "train_01" or "train_00_recovery"

        log(f"Initialized MonitoredTrainer", "INFO")
        log(f"  Upper bound (switch to train_00): {upper_bound}", "INFO")
        log(f"  Lower bound (switch to train_01): {lower_bound}", "INFO")
        log(f"  Check interval: {check_interval} steps", "INFO")

    def format_train_00_batch(self, samples: list) -> Dataset:
        """Format train_00 samples as dataset."""
        formatted = [{"text": s.get('text', '')} for s in samples if s.get('text')]
        return Dataset.from_list(formatted)

    def format_train_01_batch(self, samples: list) -> Dataset:
        """Format train_01 samples as dataset."""
        formatted = [format_train_01_entry(s) for s in samples]
        return Dataset.from_list(formatted)

    def train_steps(self, dataset: Dataset, num_steps: int, mode: str):
        """Train for a specific number of steps on given dataset."""
        log(f"Training {num_steps} steps on {mode} data...", "INFO")

        # Create training arguments for this batch
        # Use bf16 instead of fp16 for compatibility with bfloat16 models
        training_args = TrainingArguments(
            output_dir=str(self.training_dir / "temp"),
            num_train_epochs=1,
            per_device_train_batch_size=self.cfg['batch'],
            gradient_accumulation_steps=self.cfg['grad_accum'],
            learning_rate=self.cfg['lr'],
            max_steps=num_steps,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            bf16=True,
            dataloader_pin_memory=False,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        trainer.train()
        self.global_step += num_steps

        if mode == "train_01":
            self.train_01_steps += num_steps
        else:
            self.train_00_recovery_count += 1

    def check_and_report(self):
        """Check train_00 quality and report status."""
        log(f"\n{'='*60}", "INFO")
        log(f"[Step {self.global_step}] Quality Check", "INFO")

        # Compute train_00 loss
        train_00_loss = compute_train_00_loss(
            self.model, self.tokenizer,
            self.train_00_data, self.device,
            num_samples=30
        )

        # Compute KorMedMCQA accuracy
        kormedmcqa_acc = 0.0
        if self.kormedmcqa_data:
            kormedmcqa_acc = evaluate_kormedmcqa(
                self.model, self.tokenizer,
                self.kormedmcqa_data, self.device,
                num_samples=20
            )

        log(f"  Train_00 Loss: {train_00_loss:.4f} (threshold: {self.lower_bound} - {self.upper_bound})", "INFO")
        log(f"  KorMedMCQA Acc: {kormedmcqa_acc:.1f}%", "INFO")
        log(f"  Current Mode: {self.current_mode}", "INFO")
        log(f"  Train_01 Steps: {self.train_01_steps}", "INFO")
        log(f"  Train_00 Recovery Count: {self.train_00_recovery_count}", "INFO")

        # Record history
        self.history.append({
            'step': self.global_step,
            'train_00_loss': train_00_loss,
            'kormedmcqa_acc': kormedmcqa_acc,
            'mode': self.current_mode,
            'train_01_steps': self.train_01_steps,
        })

        log(f"{'='*60}\n", "INFO")

        return train_00_loss

    def train(self, total_train_01_steps: int):
        """
        Main training loop with quality monitoring.

        Args:
            total_train_01_steps: Target number of steps on train_01 data
        """
        log(f"Starting monitored training (target: {total_train_01_steps} train_01 steps)", "INFO")

        # Initial quality check
        initial_loss = self.check_and_report()
        log(f"Initial train_00 loss: {initial_loss:.4f}", "INFO")

        # Prepare datasets
        train_01_dataset = self.format_train_01_batch(self.train_01_data)
        train_00_dataset = self.format_train_00_batch(self.train_00_data)

        while self.train_01_steps < total_train_01_steps:
            if self.current_mode == "train_01":
                # Train on train_01 for check_interval steps
                steps_remaining = total_train_01_steps - self.train_01_steps
                steps_to_train = min(self.check_interval, steps_remaining)

                self.train_steps(train_01_dataset, steps_to_train, "train_01")

                # Check quality
                current_loss = self.check_and_report()

                # Should we switch to train_00 recovery?
                if current_loss > self.upper_bound:
                    log(f"⚠️ Train_00 loss ({current_loss:.4f}) > upper_bound ({self.upper_bound})", "WARNING")
                    log(f"Switching to train_00 recovery mode...", "INFO")
                    self.current_mode = "train_00_recovery"

            else:  # train_00_recovery mode
                # Train on train_00 to recover Korean quality
                self.train_steps(train_00_dataset, self.train_00_recovery_steps, "train_00")

                # Check if recovered
                current_loss = self.check_and_report()

                if current_loss < self.lower_bound:
                    log(f"✓ Train_00 loss ({current_loss:.4f}) < lower_bound ({self.lower_bound})", "INFO")
                    log(f"Recovered! Switching back to train_01...", "INFO")
                    self.current_mode = "train_01"
                else:
                    log(f"Still recovering... (loss: {current_loss:.4f}, target: < {self.lower_bound})", "INFO")

        log(f"Training complete!", "INFO")
        log(f"  Total steps: {self.global_step}", "INFO")
        log(f"  Train_01 steps: {self.train_01_steps}", "INFO")
        log(f"  Train_00 recovery count: {self.train_00_recovery_count}", "INFO")

        return self.history


def merge_and_add_new_lora(model_path: str, output_path: str, cfg: dict):
    """Merge current LoRA into base and add a new LoRA adapter."""
    log(f"Merging LoRA and adding new adapter...", "INFO")

    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_path = peft_config.base_model_name_or_path

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()

    new_lora_config = LoraConfig(
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, new_lora_config)
    model.print_trainable_parameters()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))

    tokenizer = load_tokenizer(model_path)
    tokenizer.save_pretrained(str(output_path))

    log(f"Merged model with new LoRA saved to: {output_path}", "INFO")


def main():
    parser = create_base_parser("Train 01 with Train 00 Quality Monitoring")
    parser.add_argument("--upper-bound", type=float, default=DEFAULT_UPPER_BOUND,
                       help=f"Switch to train_00 when loss > this (default: {DEFAULT_UPPER_BOUND})")
    parser.add_argument("--lower-bound", type=float, default=DEFAULT_LOWER_BOUND,
                       help=f"Switch back to train_01 when loss < this (default: {DEFAULT_LOWER_BOUND})")
    parser.add_argument("--check-interval", type=int, default=100,
                       help="Check train_00 quality every N steps (default: 100)")
    parser.add_argument("--recovery-steps", type=int, default=50,
                       help="Steps to train on train_00 when recovering (default: 50)")
    parser.add_argument("--skip-merge", action="store_true",
                       help="Skip creating merged model with new LoRA")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    # Setup directories
    training_dir = TRAINING_DIR / args.model
    output_dir = OUTPUT_DIR / args.model
    merged_dir = MERGED_DIR / args.model

    training_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70, "INFO")
    log("Train 01 with Train 00 Quality Monitoring", "INFO")
    log("=" * 70, "INFO")
    log(f"Model: {args.model}", "INFO")
    log(f"Input from: {INPUT_DIR / args.model}", "INFO")
    log(f"Training checkpoints: {training_dir}", "INFO")
    log(f"Final output: {output_dir}", "INFO")
    log(f"Upper bound: {args.upper_bound}", "INFO")
    log(f"Lower bound: {args.lower_bound}", "INFO")
    log(f"Epochs: {args.epochs}", "INFO")

    # Get input model
    if args.base_model:
        model_path = args.base_model
        log(f"Using provided base model: {model_path}", "INFO")
    else:
        model_path = get_input_model_path(args.model)

    # Load all data
    train_00_data = load_train_00_data(max_samples=args.max_samples)
    train_01_data = load_train_01_data(max_samples=args.max_samples)
    kormedmcqa_data = load_kormedmcqa_data(max_samples=100)

    if len(train_01_data) == 0:
        log("No train_01 data found!", "ERROR")
        return 1

    # Load model
    log("Loading model from 00_trained...", "INFO")
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_path = peft_config.base_model_name_or_path
    log(f"Base model: {base_model_path}", "DEBUG")

    tokenizer = load_tokenizer(model_path)

    # Check memory config for gradient checkpointing
    from peft import prepare_model_for_kbit_training
    mem_cfg = MEMORY_CONFIGS.get(args.model, {})
    use_gradient_checkpointing = mem_cfg.get('use_gradient_checkpointing', False)
    if use_gradient_checkpointing:
        log("Gradient checkpointing: ENABLED (memory optimization)", "INFO")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    # Prepare for k-bit training (applies gradient checkpointing if enabled)
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=use_gradient_checkpointing
    )

    # Resize embeddings if tokenizer has more tokens (extended tokenizer)
    base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    if tokenizer_vocab_size > base_vocab_size:
        log(f"Resizing embeddings: {base_vocab_size} -> {tokenizer_vocab_size}", "INFO")
        base_model.resize_token_embeddings(tokenizer_vocab_size)

    model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
    model.print_trainable_parameters()

    # Calculate total steps
    steps_per_epoch = len(train_01_data) // (cfg['batch'] * cfg['grad_accum'])
    total_steps = steps_per_epoch * args.epochs
    log(f"Total train_01 steps target: {total_steps}", "INFO")

    # Create monitored trainer
    trainer = MonitoredTrainer(
        model=model,
        tokenizer=tokenizer,
        train_00_data=train_00_data,
        train_01_data=train_01_data,
        kormedmcqa_data=kormedmcqa_data,
        cfg=cfg,
        args=args,
        training_dir=training_dir,
        upper_bound=args.upper_bound,
        lower_bound=args.lower_bound,
        check_interval=args.check_interval,
        train_00_recovery_steps=args.recovery_steps,
    )

    # Train with monitoring
    history = trainer.train(total_steps)

    # Save final model
    log(f"Saving final model to: {output_dir}", "INFO")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training info
    save_training_info(output_dir, {
        "script": "train_01_with_00_monitor",
        "model": args.model,
        "base_model": model_path,
        "epochs": args.epochs,
        "train_01_samples": len(train_01_data),
        "train_00_samples": len(train_00_data),
        "upper_bound": args.upper_bound,
        "lower_bound": args.lower_bound,
        "check_interval": args.check_interval,
        "recovery_steps": args.recovery_steps,
        "total_steps": trainer.global_step,
        "train_01_steps": trainer.train_01_steps,
        "train_00_recovery_count": trainer.train_00_recovery_count,
        "history": history,
        "final_train_00_loss": history[-1]['train_00_loss'] if history else None,
        "final_kormedmcqa_acc": history[-1]['kormedmcqa_acc'] if history else None,
    })

    # Create merged model
    if not args.skip_merge:
        try:
            clear_gpu_memory()
            merge_and_add_new_lora(str(output_dir), str(merged_dir), cfg)
        except Exception as e:
            log(f"Warning: Failed to create merged model: {e}", "WARNING")
            traceback.print_exc()

    # Print summary
    log("=" * 70, "INFO")
    log("TRAINING COMPLETE", "INFO")
    log("=" * 70, "INFO")
    log(f"Final model: {output_dir}", "INFO")
    log(f"Merged model: {merged_dir}", "INFO")
    log(f"Total steps: {trainer.global_step}", "INFO")
    log(f"Train_01 steps: {trainer.train_01_steps}", "INFO")
    log(f"Train_00 recovery count: {trainer.train_00_recovery_count}", "INFO")

    if history:
        log(f"Final train_00 loss: {history[-1]['train_00_loss']:.4f}", "INFO")
        log(f"Final KorMedMCQA accuracy: {history[-1]['kormedmcqa_acc']:.1f}%", "INFO")

    # Print history summary
    log("\nTraining History:", "INFO")
    log(f"{'Step':>8} | {'Mode':>15} | {'T00 Loss':>10} | {'MCQ Acc':>10}", "INFO")
    log("-" * 55, "INFO")
    for h in history:
        log(f"{h['step']:>8} | {h['mode']:>15} | {h['train_00_loss']:>10.4f} | {h['kormedmcqa_acc']:>9.1f}%", "INFO")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}", "ERROR")
        traceback.print_exc()
        exit(1)
