#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 01 Mixed: Translation (en↔ko) + KorMedMCQA Combined Training

Trains both English-Korean translation and KorMedMCQA simultaneously.
Uses existing LoRA from train_00 (no new LoRA added).

Features:
- Bidirectional translation (en→ko and ko→en)
- KorMedMCQA training with format validation
- Separate validation log files for tracking
- Template-aware cache invalidation

Directory Structure:
    Input:  model/00_trained/{model}/ (from train_00)
    Training: model/01_mixed/{model}/training/ (checkpoints)
    Output: model/01_mixed/{model}/final/ (final model)

Usage:
    python train_01_mixed.py --model medgemma-4b --epochs 1
    python train_01_mixed.py --model medgemma-4b --max-translation 10000 --max-mcq 2000
"""

import sys
import json
import random
import hashlib
import torch
import gc
import traceback
from pathlib import Path
from datetime import datetime
from datasets import Dataset, concatenate_datasets

sys.path.insert(0, str(Path(__file__).parent))
from training_utils import (
    create_base_parser, load_tokenizer,
    create_training_args, save_training_info
)
from training_config import MODEL_CONFIGS, MEMORY_CONFIGS
from _train_text_format import (
    TRANSLATION_EN_TO_KO_TEMPLATE,
    TRANSLATION_KO_TO_EN_TEMPLATE,
    MCQ_TRAIN_TEMPLATE,
    MCQ_VALIDATE_TEMPLATE,
)
from _validation import (
    validate_mcq_output,
    validate_mcq_fields,
    validate_translation_output,
)
from trl import SFTTrainer
from transformers import TrainerCallback, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training

BASE_DIR = Path(__file__).parent.parent.parent

# Data paths
TRANSLATION_DIR = BASE_DIR / "data" / "02_refined" / "01_english_korean"
MCQ_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"

# Model paths
INPUT_DIR = BASE_DIR / "model" / "00_trained"     # Input from train_00
OUTPUT_BASE = BASE_DIR / "model" / "01_mixed"     # Output directory

# Log files
LOG_DIR = BASE_DIR / "logs" / "train_01_mixed"

DEFAULT_MAX_LENGTH = 512


def get_template_hash() -> str:
    """Generate hash of templates for cache invalidation."""
    templates = [
        TRANSLATION_EN_TO_KO_TEMPLATE,
        TRANSLATION_KO_TO_EN_TEMPLATE,
        MCQ_TRAIN_TEMPLATE,
    ]
    combined = "".join(templates)
    return hashlib.md5(combined.encode()).hexdigest()[:8]


class ValidationLogger:
    """Separate logger for validation metrics."""

    def __init__(self, model_name: str):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.main_log = LOG_DIR / f"{model_name}_main_{timestamp}.log"
        self.val_log = LOG_DIR / f"{model_name}_validation_{timestamp}.log"
        self.mcq_log = LOG_DIR / f"{model_name}_mcq_{timestamp}.log"
        self.trans_log = LOG_DIR / f"{model_name}_translation_{timestamp}.log"

    def log(self, msg: str, level: str = "INFO", to_file: str = "main"):
        """Write log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {msg}"
        print(log_msg)

        log_file = {
            "main": self.main_log,
            "val": self.val_log,
            "mcq": self.mcq_log,
            "trans": self.trans_log,
        }.get(to_file, self.main_log)

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + "\n")
        except:
            pass

    def log_mcq_result(self, step: int, sample_id: int, expected: str,
                       produced: str, result: dict):
        """Log MCQ validation result."""
        msg = (f"step={step} id={sample_id} expected={expected} "
               f"extracted={result['extracted_answer']} "
               f"format={result['format_valid']} "
               f"correct={result['is_correct']} "
               f"score={result['total_score']:.3f}")
        self.log(msg, "MCQ", "mcq")

    def log_trans_result(self, step: int, sample_id: int, direction: str,
                         result: dict):
        """Log translation validation result."""
        msg = (f"step={step} id={sample_id} dir={direction} "
               f"overlap={result['overlap_score']:.3f} "
               f"f1={result['f1_score']:.3f} "
               f"matched={result['matched_count']}/{result['produced_token_count']}")
        self.log(msg, "TRANS", "trans")


# Global logger instance
logger: ValidationLogger = None


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_input_lora_path(model_name: str) -> str:
    """Get LoRA adapter path from 00_trained/."""
    # Check lora_adapter subdirectory first
    lora_path = INPUT_DIR / model_name / "lora_adapter"
    if lora_path.exists() and (lora_path / "adapter_config.json").exists():
        return str(lora_path)

    # Check root directory
    root_path = INPUT_DIR / model_name
    if root_path.exists() and (root_path / "adapter_config.json").exists():
        return str(root_path)

    raise ValueError(f"LoRA not found in: {INPUT_DIR / model_name}\nRun train_00 first!")


def load_translation_data(max_samples: int = None) -> list:
    """Load translation data with both directions."""
    data = []
    train_file = TRANSLATION_DIR / "train.jsonl"

    if not train_file.exists():
        logger.log(f"Translation data not found: {train_file}", "WARNING")
        return []

    count = 0
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and count >= max_samples:
                break

            item = json.loads(line)
            en = item.get('english', '').strip()
            ko = item.get('korean', '').strip()

            if not en or not ko or len(en) < 5 or len(ko) < 3:
                continue

            # English to Korean
            text_en_ko = TRANSLATION_EN_TO_KO_TEMPLATE.format(english=en, korean=ko)
            data.append({"text": text_en_ko, "type": "trans_en_ko"})

            # Korean to English
            text_ko_en = TRANSLATION_KO_TO_EN_TEMPLATE.format(english=en, korean=ko)
            data.append({"text": text_ko_en, "type": "trans_ko_en"})

            count += 1

    logger.log(f"Loaded {len(data)} translation samples ({count} pairs)", "INFO")
    return data


def load_mcq_data(max_samples: int = None) -> tuple:
    """Load MCQ training and test data."""
    train_data = []
    test_data = []

    train_file = MCQ_DIR / "train.jsonl"
    test_file = MCQ_DIR / "test.jsonl"

    # Load training data
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                item = json.loads(line)

                # Format with template - includes {answer} field now
                text = MCQ_TRAIN_TEMPLATE.format(
                    question=item['question'],
                    A=item['A'], B=item['B'], C=item['C'], D=item['D'], E=item['E'],
                    translate_question="",
                    translate_A="", translate_B="", translate_C="",
                    translate_D="", translate_E="",
                    answer=item['answer']  # Template now ends with {answer}<end_of_turn>
                )
                train_data.append({
                    "text": text,
                    "type": "mcq",
                    "answer": item['answer']
                })

    # Load test data for validation
    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))

    logger.log(f"Loaded {len(train_data)} MCQ train, {len(test_data)} MCQ test", "INFO")
    return train_data, test_data


def load_validation_translation(max_samples: int = 50) -> list:
    """Load translation validation data."""
    data = []
    val_file = TRANSLATION_DIR / "validation.jsonl"

    if not val_file.exists():
        return []

    with open(val_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data.append(json.loads(line))

    return data


class MixedValidationCallback(TrainerCallback):
    """Callback for validating both MCQ and translation during training."""

    def __init__(self, tokenizer, mcq_test_data, trans_val_data, device,
                 eval_interval=100, mcq_samples=10, trans_samples=5):
        self.tokenizer = tokenizer
        self.mcq_test_data = mcq_test_data
        self.trans_val_data = trans_val_data
        self.device = device
        self.eval_interval = eval_interval
        self.mcq_samples = min(mcq_samples, len(mcq_test_data)) if mcq_test_data else 0
        self.trans_samples = min(trans_samples, len(trans_val_data)) if trans_val_data else 0

        self.history = []

        # Setup terminators
        self.end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if self.end_of_turn_id == tokenizer.unk_token_id:
            self.end_of_turn_id = None

        self.terminators = [tokenizer.eos_token_id]
        if self.end_of_turn_id is not None:
            self.terminators.append(self.end_of_turn_id)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_interval != 0 or model is None:
            return control

        step = state.global_step
        model.eval()

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        mcq_results = []
        trans_results = []

        try:
            # MCQ validation
            if self.mcq_samples > 0:
                logger.log(f"\n[Step {step}] MCQ Validation ({self.mcq_samples} samples)", "INFO", "val")

                indices = random.sample(range(len(self.mcq_test_data)), self.mcq_samples)

                for idx in indices:
                    sample = self.mcq_test_data[idx]
                    expected = sample['answer']

                    prompt = MCQ_VALIDATE_TEMPLATE.format(
                        question=sample['question'],
                        A=sample['A'], B=sample['B'], C=sample['C'],
                        D=sample['D'], E=sample['E']
                    )

                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_new_tokens=300, do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.terminators,
                        )

                    input_len = inputs['input_ids'].shape[1]
                    response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)

                    if "<end_of_turn>" in response:
                        response = response.split("<end_of_turn>")[0]

                    # Validate output
                    result = validate_mcq_output(response, expected)
                    field_result = validate_mcq_fields(response)

                    mcq_results.append(result)
                    logger.log_mcq_result(step, idx, expected, response[:100], result)

                # Summary
                correct = sum(1 for r in mcq_results if r['is_correct'])
                format_valid = sum(1 for r in mcq_results if r['format_valid'])
                avg_score = sum(r['total_score'] for r in mcq_results) / len(mcq_results)

                logger.log(f"MCQ: {correct}/{len(mcq_results)} correct, "
                          f"{format_valid}/{len(mcq_results)} format valid, "
                          f"avg_score={avg_score:.3f}", "INFO", "val")

            # Translation validation (ko→en)
            if self.trans_samples > 0:
                logger.log(f"\n[Step {step}] Translation Validation ({self.trans_samples} samples)", "INFO", "val")

                indices = random.sample(range(len(self.trans_val_data)), self.trans_samples)

                for idx in indices:
                    sample = self.trans_val_data[idx]
                    ko = sample.get('korean', '')
                    en_expected = sample.get('english', '')

                    prompt = f"""<start_of_turn>user
korean translate question:
{ko}
korean translate:
<end_of_turn>
<start_of_turn>model
"""

                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_new_tokens=128, do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.terminators,
                        )

                    input_len = inputs['input_ids'].shape[1]
                    response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)

                    if "<end_of_turn>" in response:
                        response = response.split("<end_of_turn>")[0].strip()

                    # Validate translation
                    result = validate_translation_output(en_expected, response)
                    trans_results.append(result)
                    logger.log_trans_result(step, idx, "ko→en", result)

                # Summary
                avg_overlap = sum(r['overlap_score'] for r in trans_results) / len(trans_results)
                avg_f1 = sum(r['f1_score'] for r in trans_results) / len(trans_results)

                logger.log(f"Translation: avg_overlap={avg_overlap:.3f}, avg_f1={avg_f1:.3f}", "INFO", "val")

        except Exception as e:
            logger.log(f"Validation error: {e}", "ERROR", "val")

        finally:
            self.tokenizer.padding_side = original_padding_side

        # Save to history
        self.history.append({
            'step': step,
            'mcq_correct': sum(1 for r in mcq_results if r['is_correct']) if mcq_results else 0,
            'mcq_total': len(mcq_results),
            'mcq_format_valid': sum(1 for r in mcq_results if r['format_valid']) if mcq_results else 0,
            'trans_avg_f1': sum(r['f1_score'] for r in trans_results) / len(trans_results) if trans_results else 0,
        })

        model.train()
        return control


class LossLoggingCallback(TrainerCallback):
    """Callback to log training loss to separate file."""

    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            loss = logs['loss']
            step = state.global_step
            self.losses.append({'step': step, 'loss': loss})
            logger.log(f"step={step} loss={loss:.4f}", "LOSS", "val")
        return control


def main():
    global logger

    parser = create_base_parser("Train 01 Mixed: Translation + MCQ")
    parser.add_argument("--max-translation", type=int, default=None,
                       help="Max translation pairs (default: all)")
    parser.add_argument("--max-mcq", type=int, default=None,
                       help="Max MCQ samples (default: all)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                       help=f"Max token length (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--eval-interval", type=int, default=100,
                       help="Validation interval in steps")
    parser.add_argument("--mcq-val-samples", type=int, default=10,
                       help="MCQ validation samples per eval")
    parser.add_argument("--trans-val-samples", type=int, default=5,
                       help="Translation validation samples per eval")
    parser.add_argument("--translation-ratio", type=float, default=0.7,
                       help="Ratio of translation vs MCQ data (default: 0.7)")
    args = parser.parse_args()

    # Initialize logger
    logger = ValidationLogger(args.model)

    cfg = MODEL_CONFIGS[args.model]
    cfg = {**cfg, 'max_length': args.max_length}

    # Setup directories
    output_base = OUTPUT_BASE / args.model
    training_dir = output_base / "training"
    final_dir = output_base / "final"

    training_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.log("=" * 70, "INFO")
    logger.log("Train 01 Mixed: Translation + KorMedMCQA", "INFO")
    logger.log("=" * 70, "INFO")
    logger.log(f"Model: {args.model}", "INFO")
    logger.log(f"Input LoRA: {INPUT_DIR / args.model}", "INFO")
    logger.log(f"Output: {final_dir}", "INFO")
    logger.log(f"Template hash: {get_template_hash()}", "INFO")
    logger.log(f"Validation logs: {LOG_DIR}", "INFO")

    # Get existing LoRA path (from train_00)
    if args.base_model:
        lora_path = args.base_model
    else:
        lora_path = get_input_lora_path(args.model)
    logger.log(f"Using LoRA from: {lora_path}", "INFO")

    # Load data
    logger.log("\nLoading data...", "INFO")
    trans_data = load_translation_data(args.max_translation)
    mcq_train_data, mcq_test_data = load_mcq_data(args.max_mcq)
    trans_val_data = load_validation_translation()

    if len(trans_data) == 0 and len(mcq_train_data) == 0:
        logger.log("No training data found!", "ERROR")
        return 1

    # Balance data based on ratio
    if trans_data and mcq_train_data:
        # Adjust MCQ samples to match ratio
        target_mcq = int(len(trans_data) * (1 - args.translation_ratio) / args.translation_ratio)
        if len(mcq_train_data) > target_mcq:
            mcq_train_data = random.sample(mcq_train_data, target_mcq)

        logger.log(f"Balanced: {len(trans_data)} translation, {len(mcq_train_data)} MCQ", "INFO")

    # Combine datasets
    all_data = trans_data + mcq_train_data
    random.shuffle(all_data)

    train_dataset = Dataset.from_list(all_data)
    logger.log(f"Total training samples: {len(train_dataset)}", "INFO")

    # Load model with existing LoRA (NO new LoRA)
    logger.log("\nLoading model with existing LoRA...", "INFO")

    peft_config = PeftConfig.from_pretrained(lora_path)
    base_model_path = peft_config.base_model_name_or_path

    # Load tokenizer
    tokenizer = load_tokenizer(lora_path)

    # Load base model
    mem_cfg = MEMORY_CONFIGS.get(args.model, {})
    use_gradient_checkpointing = mem_cfg.get('use_gradient_checkpointing', False)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=use_gradient_checkpointing
    )

    # Resize embeddings if needed
    base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    if tokenizer_vocab_size > base_vocab_size:
        logger.log(f"Resizing embeddings: {base_vocab_size} -> {tokenizer_vocab_size}", "INFO")
        base_model.resize_token_embeddings(tokenizer_vocab_size)

    # Load existing LoRA (trainable)
    model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
    logger.log("✓ Loaded existing LoRA (trainable)", "INFO")
    model.print_trainable_parameters()

    # Training arguments
    training_args = create_training_args(
        output_dir=str(training_dir),
        num_epochs=args.epochs,
        batch_size=cfg['batch'],
        grad_accum=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        max_length=cfg['max_length'],
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
    )

    # Callbacks
    callbacks = [
        MixedValidationCallback(
            tokenizer=tokenizer,
            mcq_test_data=mcq_test_data,
            trans_val_data=trans_val_data,
            device=args.device,
            eval_interval=args.eval_interval,
            mcq_samples=args.mcq_val_samples,
            trans_samples=args.trans_val_samples,
        ),
        LossLoggingCallback(log_interval=10),
    ]

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.log("\nStarting training...", "INFO")
    trainer.train()

    # Save final model
    logger.log(f"\nSaving final model to: {final_dir}", "INFO")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training info
    save_training_info(final_dir, {
        "script": "train_01_mixed",
        "model": args.model,
        "base_lora": lora_path,
        "epochs": args.epochs,
        "train_samples": len(train_dataset),
        "translation_samples": len(trans_data),
        "mcq_samples": len(mcq_train_data),
        "translation_ratio": args.translation_ratio,
        "template_hash": get_template_hash(),
        "validation_history": callbacks[0].history if callbacks else [],
        "log_files": {
            "main": str(logger.main_log),
            "validation": str(logger.val_log),
            "mcq": str(logger.mcq_log),
            "translation": str(logger.trans_log),
        }
    })

    logger.log("=" * 70, "INFO")
    logger.log("TRAIN 01 MIXED COMPLETE", "INFO")
    logger.log("=" * 70, "INFO")
    logger.log(f"Final model: {final_dir}", "INFO")
    logger.log(f"Validation logs: {LOG_DIR}", "INFO")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        if logger:
            logger.log(f"FATAL ERROR: {type(e).__name__}: {e}", "ERROR")
        traceback.print_exc()
        exit(1)
