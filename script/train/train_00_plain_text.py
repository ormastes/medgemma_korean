#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 00: Plain Text Pre-training with KorMedMCQA Validation

Learn general Korean language on plain text corpus.
Validates on KorMedMCQA after each epoch to track medical knowledge changes.

Directory Structure:
    Input:  model/raw/ (HuggingFace models)
    Training: model/00_training/{model}/ (checkpoints)
    Output: model/00_trained/{model}/ (final model)

Usage:
    python train_00_plain_text.py --model medgemma-27b --epochs 3
    python train_00_plain_text.py --model medgemma-4b --epochs 3 --val-samples 100
"""

import sys
import json
import torch
import gc
import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import TrainerCallback

sys.path.insert(0, str(Path(__file__).parent))
from training_utils import (
    create_base_parser, load_jsonl_data, setup_model_with_lora,
    create_training_args, save_training_info, load_tokenizer
)
from training_config import MODEL_CONFIGS
from data_validation import validate_and_report
from trl import SFTTrainer

# Default max length
DEFAULT_MAX_LENGTH = 512

BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data" / "02_refined" / "00_plain_text"
TEST_FILE = BASE_DIR / "data" / "02_refined" / "02_kor_med_test" / "test.jsonl"

# Model paths (NEW STRUCTURE)
RAW_MODEL_DIR = BASE_DIR / "model" / "raw"
RAW_LORA_DIR = BASE_DIR / "model" / "raw_lora_added"  # Input: base + LoRA initialized
TRAINING_DIR = BASE_DIR / "model" / "00_training"
TRAINED_DIR = BASE_DIR / "model" / "00_trained"

# Log file
LOG_FILE = TRAINING_DIR / "train_00_debug.log"


def log(msg: str, level: str = "INFO"):
    """Write debug log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    print(log_msg)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    except Exception as e:
        pass


def check_gpu_status():
    """Check and log GPU status"""
    log("=" * 50, "DEBUG")
    log("GPU Status Check", "DEBUG")

    try:
        log(f"PyTorch version: {torch.__version__}", "DEBUG")
        log(f"CUDA available: {torch.cuda.is_available()}", "DEBUG")

        if torch.cuda.is_available():
            log(f"CUDA version: {torch.version.cuda}", "DEBUG")
            log(f"GPU count: {torch.cuda.device_count()}", "DEBUG")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)

                log(f"GPU {i}: {props.name}", "DEBUG")
                log(f"  Total: {total_mem:.2f} GB, Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB", "DEBUG")
    except Exception as e:
        log(f"Error checking GPU: {e}", "ERROR")


def clear_gpu_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_raw_lora_model_path(model_name: str) -> str:
    """Get model path from raw_lora_added/ (base + LoRA initialized)"""
    lora_model_path = RAW_LORA_DIR / model_name

    if lora_model_path.exists():
        # Check if it has adapter files
        if (lora_model_path / "adapter_config.json").exists():
            log(f"Found initialized LoRA model: {lora_model_path}", "INFO")
            return str(lora_model_path)

    # If not found, fall back to raw HuggingFace model
    log(f"raw_lora_added/{model_name} not found, falling back to HuggingFace model", "WARNING")
    log(f"Run: python script/init_lora_on_raw.py --model {model_name}", "WARNING")

    config_file = RAW_MODEL_DIR / "model_paths.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        if model_name in config:
            return config[model_name]["huggingface_id"]

    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]['path']

    raise ValueError(f"Unknown model: {model_name}")


def load_test_data(filepath: Path, max_samples: int = None) -> list:
    """Load KorMedMCQA test data."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def format_prompt(sample: dict) -> str:
    """Format MCQ question as prompt."""
    question = sample['question']
    choices = f"A) {sample['A']}\nB) {sample['B']}\nC) {sample['C']}\nD) {sample['D']}\nE) {sample['E']}"

    prompt = f"""<start_of_turn>user
{question}

{choices}

정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).<end_of_turn>
<start_of_turn>model
"""
    return prompt


def extract_answer(response: str) -> str:
    """Extract answer letter from model response."""
    response = response.strip().upper()
    for char in response:
        if char in 'ABCDE':
            return char
    return response[:1] if response else ""


def evaluate_kormedmcqa(model, tokenizer, test_data: list, device: str) -> dict:
    """Evaluate model on KorMedMCQA test data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in tqdm(test_data, desc="Evaluating KorMedMCQA", leave=False):
            prompt = format_prompt(sample)
            expected = sample['answer']

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted = extract_answer(response)

            if predicted == expected:
                correct += 1
            total += 1

    model.train()
    accuracy = correct / total * 100 if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


class KorMedMCQAValidationCallback(TrainerCallback):
    """Callback to validate on KorMedMCQA after each epoch."""

    def __init__(self, tokenizer, test_data, device, results_list):
        self.tokenizer = tokenizer
        self.test_data = test_data
        self.device = device
        self.results_list = results_list
        self.current_epoch = 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.current_epoch += 1
        log(f"Epoch {self.current_epoch} completed - Running KorMedMCQA validation...", "INFO")

        result = evaluate_kormedmcqa(model, self.tokenizer, self.test_data, self.device)
        result["epoch"] = self.current_epoch
        result["step"] = state.global_step
        self.results_list.append(result)

        log(f"KorMedMCQA Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})", "INFO")

        if len(self.results_list) > 1:
            prev = self.results_list[-2]["accuracy"]
            change = result["accuracy"] - prev
            symbol = "+" if change >= 0 else ""
            log(f"Change from previous: {symbol}{change:.2f}%", "INFO")

        return control


def train_with_validation(args, cfg, model, tokenizer, train_data, training_dir, output_dir):
    """Custom training function with KorMedMCQA validation."""

    # Load test data
    if not TEST_FILE.exists():
        log(f"Warning: Test file not found: {TEST_FILE}", "WARNING")
        test_data = []
    else:
        val_samples = getattr(args, 'val_samples', None)
        test_data = load_test_data(TEST_FILE, max_samples=val_samples)
        log(f"Loaded {len(test_data)} KorMedMCQA test samples for validation", "INFO")

    # Results tracking
    validation_results = []

    # Baseline evaluation (before training)
    if test_data:
        log("BASELINE: Evaluating before training...", "INFO")
        baseline = evaluate_kormedmcqa(model, tokenizer, test_data, args.device)
        baseline["epoch"] = 0
        baseline["step"] = 0
        validation_results.append(baseline)
        log(f"Baseline KorMedMCQA Accuracy: {baseline['accuracy']:.2f}% ({baseline['correct']}/{baseline['total']})", "INFO")

    # Create training arguments - checkpoints go to training_dir
    training_args = create_training_args(
        output_dir=str(training_dir),  # Intermediate checkpoints here
        num_epochs=args.epochs,
        batch_size=cfg['batch'],
        grad_accum=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        max_length=cfg['max_length']
    )

    # Create callback for validation
    callbacks = []
    if test_data:
        val_callback = KorMedMCQAValidationCallback(
            tokenizer=tokenizer,
            test_data=test_data,
            device=args.device,
            results_list=validation_results
        )
        callbacks.append(val_callback)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    log("Starting training...", "INFO")
    trainer.train()

    # Save final model to output_dir (model/00_trained/{model}/)
    log(f"Saving final model to: {output_dir}", "INFO")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Print accuracy change summary
    if validation_results:
        log("=" * 60, "INFO")
        log("ACCURACY CHANGE SUMMARY", "INFO")
        log("=" * 60, "INFO")

        for i, result in enumerate(validation_results):
            if i == 0:
                change_str = "baseline"
            else:
                change = result["accuracy"] - validation_results[i-1]["accuracy"]
                change_str = f"{'+' if change >= 0 else ''}{change:.2f}%"

            log(f"Epoch {result['epoch']}: {result['accuracy']:.2f}% ({change_str})", "INFO")

        if len(validation_results) > 1:
            total_change = validation_results[-1]["accuracy"] - validation_results[0]["accuracy"]
            log(f"TOTAL CHANGE: {'+' if total_change >= 0 else ''}{total_change:.2f}%", "INFO")

        # Save validation results
        results_file = output_dir / "kormedmcqa_validation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "validation_results": validation_results,
                "baseline_accuracy": validation_results[0]["accuracy"],
                "final_accuracy": validation_results[-1]["accuracy"],
                "total_change": validation_results[-1]["accuracy"] - validation_results[0]["accuracy"],
            }, f, indent=2, ensure_ascii=False)

    log(f"Training complete! Final model saved to: {output_dir}", "INFO")
    return output_dir


def main():
    parser = create_base_parser("Train 00: Plain Text Pre-training with Validation")
    parser.add_argument("--val-samples", type=int, default=None,
                       help="Limit validation samples (default: use all 604)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                       help=f"Maximum token length (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip data validation at startup")
    args = parser.parse_args()

    # Setup directories
    training_dir = TRAINING_DIR / args.model
    output_dir = TRAINED_DIR / args.model

    training_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70, "INFO")
    log("Train 00: Plain Text Pre-training", "INFO")
    log("=" * 70, "INFO")
    log(f"Model: {args.model}", "INFO")
    log(f"Raw model from: {RAW_MODEL_DIR}", "INFO")
    log(f"Training checkpoints: {training_dir}", "INFO")
    log(f"Final output: {output_dir}", "INFO")
    log(f"Max length: {args.max_length}", "INFO")
    log(f"Epochs: {args.epochs}", "INFO")
    log(f"Device: {args.device}", "INFO")

    # Check GPU
    check_gpu_status()

    # Get model path from raw_lora_added/ (or fallback to HuggingFace)
    if args.base_model:
        model_path = args.base_model
        log(f"Using provided base model: {model_path}", "INFO")
    else:
        model_path = get_raw_lora_model_path(args.model)
        log(f"Using model: {model_path}", "INFO")

    cfg = MODEL_CONFIGS[args.model]
    # Override max_length from command line
    cfg = {**cfg, 'max_length': args.max_length}
    log(f"Model config: {cfg}", "DEBUG")

    # Load training data
    train_data, val_data = load_jsonl_data(DATA_DIR, max_samples=args.max_samples)
    if len(train_data) == 0:
        log(f"No training data found in: {DATA_DIR}", "ERROR")
        return 1

    log(f"Loaded {len(train_data)} training samples", "INFO")

    # ==========================================================================
    # DATA VALIDATION AT STARTUP
    # ==========================================================================
    if not args.skip_validation:
        log("\n" + "=" * 70, "INFO")
        log("STARTUP VALIDATION", "INFO")
        log("=" * 70, "INFO")

        # Load tokenizer for validation
        try:
            tokenizer = load_tokenizer(model_path)
        except:
            # Fallback for HuggingFace models
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg.get('path', model_path), trust_remote_code=True)

        # Validate training data (convert Dataset slice to list of dicts)
        sample_data = [train_data[i] for i in range(min(100, len(train_data)))]
        validate_and_report(
            sample_data,
            tokenizer,
            args.max_length,
            "Plain text training data (first 100)",
            log_fn=lambda msg: log(msg, "INFO")
        )

        log("=" * 70 + "\n", "INFO")

    # Load model - check if it's from raw_lora_added (already has LoRA)
    is_lora_model = Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists()

    if is_lora_model:
        # Load existing LoRA model (from raw_lora_added/)
        log("Loading pre-initialized LoRA model...", "INFO")
        from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from training_config import MEMORY_CONFIGS

        # Load tokenizer
        tokenizer = load_tokenizer(model_path)

        # Get base model path from adapter config
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path

        log(f"Base model: {base_model_path}", "DEBUG")

        # Check memory config for gradient checkpointing
        mem_cfg = MEMORY_CONFIGS.get(args.model, {})
        use_gradient_checkpointing = mem_cfg.get('use_gradient_checkpointing', False)
        if use_gradient_checkpointing:
            log("Gradient checkpointing: ENABLED (memory optimization)", "INFO")

        # Load base model with 8-bit
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

        # Load LoRA adapter (is_trainable=True to enable training)
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
        model.print_trainable_parameters()
    else:
        # Setup new model with LoRA (fallback for HuggingFace models)
        log("Loading model and adding new LoRA...", "INFO")
        model, tokenizer = setup_model_with_lora(
            model_path,
            lora_r=cfg['lora_r'],
            lora_alpha=cfg['lora_alpha'],
            include_embeddings=True,  # Include embeddings for continued pretraining
            use_rslora=True,  # Use rank-stabilized LoRA
            device=args.device
        )

    # Train with validation
    final_dir = train_with_validation(args, cfg, model, tokenizer, train_data, training_dir, output_dir)

    # Save training info
    save_training_info(output_dir, {
        "script": "train_00_plain_text",
        "model": args.model,
        "base_model": model_path,
        "epochs": args.epochs,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "training_dir": str(training_dir),
        "output_dir": str(output_dir),
        "with_kormedmcqa_validation": True
    })

    # Clear GPU memory
    clear_gpu_memory()

    log("=" * 70, "INFO")
    log("TRAIN 00 COMPLETE", "INFO")
    log("=" * 70, "INFO")
    log(f"Final model: {output_dir}", "INFO")
    log(f"Checkpoints: {training_dir}", "INFO")
    log(f"Debug log: {LOG_FILE}", "INFO")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}", "ERROR")
        traceback.print_exc()
        exit(1)
