#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 01-EK: English-Korean Translation
Learn English-Korean translation using parallel corpus

Directory Structure:
    Input:  model/00_trained/{model}/ (from train_00)
    Training: model/01_ek_training/{model}/ (checkpoints)
    Output: model/01_ek_trained/{model}/ (final model)

Key features:
- Bidirectional training (en→ko and ko→en)
- Uses proven parallel corpus (TED Talks, Tatoeba, etc.)
- Research-backed hyperparameters (see research/01__train_translate.md)

Research-Backed Training Configuration:
========================================
Based on: "How Much Data is Enough Data?" (arXiv 2024), EMNLP 2024, ICLR 2024

Data Size Thresholds:
- Minimum: 5,000 samples (below this = performance degradation)
- Recommended: 20,000 samples (good balance)
- Optimal: 100,000+ samples (Korean benefits 130% vs 46% average)

Hyperparameters:
- Learning Rate: 1e-4 to 2e-4 for LoRA (10x higher than full fine-tuning)
- LoRA Rank: 32-64 for translation tasks
- LoRA Alpha: 2x rank (empirically optimal)
- Epochs: 1-2 for 20k+ data, 2-3 for <20k data
- Early stopping: patience=3, min_delta=0.01

Korean-Specific:
- Korean shows 130% COMET improvement (vs 46% average for other languages)
- Bidirectional training doubles effective data
- Extended Korean tokenizer (23,699 new tokens) improves efficiency

References:
- How Much Data: https://arxiv.org/abs/2409.03454
- Fine-Tuning LLMs to Translate (EMNLP 2024): https://aclanthology.org/2024.emnlp-main.24.pdf
- When Scaling Meets LLM Finetuning (ICLR 2024): https://openreview.net/pdf?id=5HCnKDeTws
- CCMatrix: https://arxiv.org/abs/1911.04944
- OPUS-MT: https://arxiv.org/abs/2212.01936
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

sys.path.insert(0, str(Path(__file__).parent))
from training_utils import (
    create_base_parser, load_tokenizer,
    create_training_args, save_training_info, create_lora_config
)
from training_config import MODEL_CONFIGS, LORA_TARGET_MODULES
from _train_text_format import (
    TRANSLATION_EN_TO_KO_TEMPLATE,
    TRANSLATION_KO_TO_EN_TEMPLATE,
    TRANSLATION_VALIDATE_EN_TO_KO_TEMPLATE,
)
from trl import SFTTrainer
from transformers import TrainerCallback, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model

BASE_DIR = Path(__file__).parent.parent.parent

# Data paths
PARALLEL_DATA_DIR = BASE_DIR / "data" / "02_refined" / "01_english_korean"

# Model paths
INPUT_DIR = BASE_DIR / "model" / "00_trained"         # Input from train_00
TRAINING_DIR = BASE_DIR / "model" / "01_ek_training"  # Checkpoints
OUTPUT_DIR = BASE_DIR / "model" / "01_ek_trained"     # Final output

# Log file
LOG_FILE = TRAINING_DIR / "train_01_ek_debug.log"

# Default max length (translation pairs tend to be shorter)
DEFAULT_MAX_LENGTH = 256

# =============================================================================
# RESEARCH-BACKED TRAINING CONFIGURATION
# Based on: arXiv:2409.03454, EMNLP 2024, ICLR 2024
# See: research/01__train_translate.md
# =============================================================================

# Data size thresholds (from "How Much Data is Enough Data?")
DATA_SIZE_MINIMUM = 5_000       # Below this = performance degradation
DATA_SIZE_RECOMMENDED = 20_000  # Good balance of quality/time
DATA_SIZE_OPTIMAL = 100_000     # Korean benefits most (130% vs 46% avg)

# Epochs based on data size
def get_recommended_epochs(data_size: int) -> int:
    """Get recommended epochs based on dataset size (research-backed)."""
    if data_size < DATA_SIZE_MINIMUM:
        return 5  # Need multiple passes for tiny datasets
    elif data_size < DATA_SIZE_RECOMMENDED:
        return 3  # Standard for small datasets
    elif data_size < DATA_SIZE_OPTIMAL:
        return 2  # Avoid overfitting with medium datasets
    else:
        return 1  # Single pass sufficient for large datasets

# Learning rate recommendations (from Unsloth & Databricks guides)
TRANSLATION_LR_CONFIG = {
    "lora_standard": 2e-4,      # Starting point for LoRA
    "lora_conservative": 1e-4,  # More stable, recommended for translation
    "lora_high_rank": 5e-5,     # For rank 128+
    "full_finetune": 2e-5,      # 10x lower than LoRA
}

# LoRA configuration for translation (empirically optimal: alpha = 2 * rank)
TRANSLATION_LORA_CONFIG = {
    "quick": {"rank": 16, "alpha": 32},      # Fast experiments
    "standard": {"rank": 32, "alpha": 64},   # Recommended
    "optimal": {"rank": 64, "alpha": 128},   # Best quality
    "max": {"rank": 128, "alpha": 256},      # Near full fine-tuning
}

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.01

# Checkpoint and evaluation intervals
def get_eval_interval(total_steps: int) -> int:
    """Get evaluation interval based on total steps."""
    if total_steps < 500:
        return 50
    elif total_steps < 2000:
        return 100
    else:
        return 200

def get_checkpoint_interval(total_steps: int) -> int:
    """Get checkpoint interval based on total steps."""
    if total_steps < 500:
        return 100
    elif total_steps < 2000:
        return 250
    else:
        return 500


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


def get_input_lora_path(model_name: str) -> str:
    """Get LoRA adapter path from 00_trained/ (LoRA_0)"""
    lora_path = INPUT_DIR / model_name / "lora_adapter"

    if lora_path.exists() and (lora_path / "adapter_config.json").exists():
        log(f"Found LoRA_0 adapter: {lora_path}", "INFO")
        return str(lora_path)

    root_path = INPUT_DIR / model_name
    if root_path.exists() and (root_path / "adapter_config.json").exists():
        log(f"Found LoRA_0 adapter (old structure): {root_path}", "INFO")
        return str(root_path)

    raise ValueError(f"LoRA adapter not found in: {lora_path} or {root_path}\nRun train_00 first!")


def load_parallel_data(max_samples: int = None, bidirectional: bool = True) -> list:
    """
    Load English-Korean parallel data.

    Args:
        max_samples: Maximum number of samples
        bidirectional: If True, create both en→ko and ko→en pairs

    Returns:
        List of formatted training examples
    """
    data = []
    train_file = PARALLEL_DATA_DIR / "train.jsonl"

    if not train_file.exists():
        log(f"Training data not found: {train_file}", "ERROR")
        log("Run: python script/merge_korean_english_parallel.py", "ERROR")
        return []

    log(f"Loading parallel data from: {train_file}", "INFO")

    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            en = item.get('english', '').strip()
            ko = item.get('korean', '').strip()

            if en and ko:
                # English to Korean
                text_en_ko = TRANSLATION_EN_TO_KO_TEMPLATE.format(
                    english=en,
                    korean=ko
                )
                data.append({
                    "text": text_en_ko,
                    "direction": "en→ko",
                    "source": item.get('source', 'unknown')
                })

                # Korean to English (if bidirectional)
                if bidirectional:
                    text_ko_en = TRANSLATION_KO_TO_EN_TEMPLATE.format(
                        english=en,
                        korean=ko
                    )
                    data.append({
                        "text": text_ko_en,
                        "direction": "ko→en",
                        "source": item.get('source', 'unknown')
                    })

    random.shuffle(data)
    log(f"Total training samples: {len(data):,} (bidirectional={bidirectional})", "INFO")

    # Log source distribution
    source_counts = {}
    for item in data:
        src = item.get('source', 'unknown')
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        log(f"  {src}: {cnt:,}", "INFO")

    return data


def load_validation_data(max_samples: int = 100) -> list:
    """Load validation parallel data."""
    data = []
    val_file = PARALLEL_DATA_DIR / "validation.jsonl"

    if not val_file.exists():
        log(f"Validation file not found: {val_file}", "WARNING")
        return []

    with open(val_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            data.append(item)

    log(f"Loaded {len(data)} validation samples", "INFO")
    return data


class TranslationValidationCallback(TrainerCallback):
    """Callback to evaluate translation quality during training."""

    def __init__(self, validation_data, tokenizer, device, eval_interval=100, eval_samples=5):
        self.validation_data = validation_data
        self.tokenizer = tokenizer
        self.device = device
        self.eval_interval = eval_interval
        self.eval_samples = min(eval_samples, len(validation_data)) if validation_data else 0
        self.history = []

        self.end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if self.end_of_turn_id == tokenizer.unk_token_id:
            self.end_of_turn_id = None

        self.terminators = [tokenizer.eos_token_id]
        if self.end_of_turn_id is not None:
            self.terminators.append(self.end_of_turn_id)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_interval != 0 or model is None:
            return control

        if not self.validation_data or self.eval_samples == 0:
            return control

        model.eval()

        indices = random.sample(range(len(self.validation_data)), self.eval_samples)

        log(f"\n[Step {state.global_step}] Translation Validation ({self.eval_samples} samples)", "INFO")

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            for idx in indices:
                sample = self.validation_data[idx]
                en = sample.get('english', '')
                ko_expected = sample.get('korean', '')

                prompt = TRANSLATION_VALIDATE_EN_TO_KO_TEMPLATE.format(english=en)

                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=128, do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.terminators,
                    )

                input_len = inputs['input_ids'].shape[1]
                generated = outputs[0][input_len:]
                response = self.tokenizer.decode(generated, skip_special_tokens=False)

                if "<end_of_turn>" in response:
                    response = response.split("<end_of_turn>")[0].strip()

                # Log sample
                log(f"  EN: {en[:60]}...", "INFO")
                log(f"  Expected KO: {ko_expected[:60]}...", "INFO")
                log(f"  Generated:   {response[:60]}...", "INFO")
                log("", "INFO")

        except Exception as e:
            log(f"Validation error: {e}", "ERROR")

        finally:
            self.tokenizer.padding_side = original_padding_side

        self.history.append({
            'step': state.global_step,
            'samples_shown': self.eval_samples
        })

        model.train()
        return control


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE,
                 min_delta: float = EARLY_STOPPING_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        current_loss = metrics.get('eval_loss', float('inf'))

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            log(f"Early stopping: New best loss {current_loss:.4f}", "INFO")
        else:
            self.counter += 1
            log(f"Early stopping: No improvement for {self.counter}/{self.patience} evals", "INFO")

        if self.counter >= self.patience:
            log(f"Early stopping triggered after {self.patience} evals without improvement", "WARNING")
            control.should_training_stop = True

        return control


def main():
    parser = create_base_parser("Train 01-EK: English-Korean Translation")
    parser.add_argument("--show-samples-every", type=int, default=100,
                       help="Show translation samples every N steps")
    parser.add_argument("--eval-samples", type=int, default=5,
                       help="Number of validation samples to show")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                       help=f"Maximum token length (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--unidirectional", action="store_true",
                       help="Only train en→ko (default: bidirectional)")
    parser.add_argument("--lora-preset", type=str, default="optimal",
                       choices=["quick", "standard", "optimal", "max"],
                       help="LoRA configuration preset (default: optimal)")
    parser.add_argument("--lr-preset", type=str, default="lora_conservative",
                       choices=list(TRANSLATION_LR_CONFIG.keys()),
                       help="Learning rate preset (default: lora_conservative)")
    parser.add_argument("--auto-epochs", action="store_true",
                       help="Auto-select epochs based on data size (research-backed)")
    parser.add_argument("--early-stopping", action="store_true", default=True,
                       help="Enable early stopping (default: True)")
    parser.add_argument("--no-early-stopping", action="store_false", dest="early_stopping",
                       help="Disable early stopping")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    # Apply LoRA preset (research-backed configuration)
    lora_preset = TRANSLATION_LORA_CONFIG[args.lora_preset]
    cfg = {
        **cfg,
        'max_length': args.max_length,
        'lora_r': lora_preset['rank'],
        'lora_alpha': lora_preset['alpha'],
        'lr': TRANSLATION_LR_CONFIG[args.lr_preset]
    }

    # Setup directories
    training_dir = TRAINING_DIR / args.model
    output_dir = OUTPUT_DIR / args.model

    training_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70, "INFO")
    log("Train 01-EK: English-Korean Translation", "INFO")
    log("=" * 70, "INFO")
    log(f"Model: {args.model}", "INFO")
    log(f"Input from: {INPUT_DIR / args.model}", "INFO")
    log(f"Training checkpoints: {training_dir}", "INFO")
    log(f"Final output: {output_dir}", "INFO")
    log(f"Bidirectional: {not args.unidirectional}", "INFO")

    # Research-backed configuration
    log("", "INFO")
    log("Research-Backed Configuration:", "INFO")
    log(f"  LoRA Preset: {args.lora_preset} (rank={lora_preset['rank']}, alpha={lora_preset['alpha']})", "INFO")
    log(f"  Learning Rate: {cfg['lr']} ({args.lr_preset})", "INFO")
    log(f"  Early Stopping: {args.early_stopping} (patience={EARLY_STOPPING_PATIENCE})", "INFO")

    # Get LoRA_0 adapter
    if args.base_model:
        lora_0_path = args.base_model
        log(f"Using provided LoRA_0 path: {lora_0_path}", "INFO")
    else:
        lora_0_path = get_input_lora_path(args.model)

    # Load parallel data
    train_data = load_parallel_data(
        max_samples=args.max_samples,
        bidirectional=not args.unidirectional
    )
    if len(train_data) == 0:
        log("No training data found!", "ERROR")
        return 1

    # Data size analysis (research-backed recommendations)
    data_size = len(train_data)
    log("", "INFO")
    log("Data Size Analysis (Research-Backed):", "INFO")
    log(f"  Current: {data_size:,} samples", "INFO")

    if data_size < DATA_SIZE_MINIMUM:
        log(f"  WARNING: Data size below minimum threshold ({DATA_SIZE_MINIMUM:,})", "WARNING")
        log(f"  Research shows performance DEGRADATION with <5k samples!", "WARNING")
        log(f"  Consider: Collecting more parallel data or using data augmentation", "WARNING")
    elif data_size < DATA_SIZE_RECOMMENDED:
        log(f"  Status: Below recommended ({DATA_SIZE_RECOMMENDED:,}), but above minimum", "INFO")
        log(f"  Tip: More data will improve quality significantly", "INFO")
    elif data_size < DATA_SIZE_OPTIMAL:
        log(f"  Status: Good size, approaching optimal", "INFO")
    else:
        log(f"  Status: OPTIMAL size for Korean translation!", "INFO")
        log(f"  Korean shows 130% COMET improvement at this scale (vs 46% avg)", "INFO")

    # Auto-epoch selection based on research
    if args.auto_epochs:
        recommended_epochs = get_recommended_epochs(data_size)
        log(f"  Auto-epochs: {recommended_epochs} (based on data size)", "INFO")
        args.epochs = recommended_epochs
    else:
        recommended_epochs = get_recommended_epochs(data_size)
        if args.epochs != recommended_epochs:
            log(f"  Note: Research suggests {recommended_epochs} epochs for this data size", "INFO")

    log(f"  Final epochs: {args.epochs}", "INFO")

    # Calculate total steps for interval configuration
    effective_batch = cfg['batch'] * cfg['grad_accum']
    total_steps = (data_size * args.epochs) // effective_batch
    eval_interval = get_eval_interval(total_steps)
    checkpoint_interval = get_checkpoint_interval(total_steps)

    log("", "INFO")
    log("Training Schedule:", "INFO")
    log(f"  Total steps: ~{total_steps:,}", "INFO")
    log(f"  Eval interval: every {eval_interval} steps", "INFO")
    log(f"  Checkpoint interval: every {checkpoint_interval} steps", "INFO")

    # Load validation data
    validation_data = load_validation_data()

    # Create dataset
    train_dataset = Dataset.from_list(train_data)

    # Get base model path from LoRA config
    log("Reading LoRA_0 config...", "INFO")
    peft_config = PeftConfig.from_pretrained(lora_0_path)
    base_model_path = peft_config.base_model_name_or_path
    log(f"Base model from config: {base_model_path}", "DEBUG")

    # Load tokenizer
    tokenizer = load_tokenizer(lora_0_path)

    # ========================================================================
    # PROGRESSIVE LORA: Merge LoRA_0 and add NEW LoRA_1
    # ========================================================================
    log("=" * 70, "INFO")
    log("PROGRESSIVE LORA TRAINING", "INFO")
    log("=" * 70, "INFO")

    # Load base model
    log("Loading base model...", "INFO")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    log(f"✓ Base model loaded: {base_model_path}", "INFO")

    # Load and merge LoRA_0
    log(f"Loading LoRA_0 from: {lora_0_path}", "INFO")
    model = PeftModel.from_pretrained(
        base_model,
        lora_0_path,
        is_trainable=False
    )
    log("✓ LoRA_0 loaded", "INFO")

    log("Merging LoRA_0 into base model...", "INFO")
    model = model.merge_and_unload()
    log("✓ LoRA_0 merged", "INFO")

    # Add NEW trainable LoRA_1
    log("Adding NEW LoRA_1 adapter...", "INFO")
    lora_config = create_lora_config(
        rank=cfg.get('lora_r', 64),
        alpha=cfg.get('lora_alpha', 128),
        use_rslora=True,
        include_embeddings=False
    )

    model = get_peft_model(model, lora_config)
    log("✓ LoRA_1 added", "INFO")

    log("\nTrainable parameters:", "INFO")
    model.print_trainable_parameters()
    log("=" * 70, "INFO")

    # Training arguments
    training_args = create_training_args(
        output_dir=str(training_dir),
        num_epochs=args.epochs,
        batch_size=cfg['batch'],
        grad_accum=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        max_length=cfg['max_length']
    )

    # Callbacks
    callbacks = []

    # Validation callback
    validation_callback = TranslationValidationCallback(
        validation_data=validation_data,
        tokenizer=tokenizer,
        device=args.device,
        eval_interval=args.show_samples_every if args.show_samples_every else eval_interval,
        eval_samples=args.eval_samples
    )
    callbacks.append(validation_callback)

    # Early stopping callback (research-backed: prevent overfitting)
    if args.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA
        )
        callbacks.append(early_stopping_callback)
        log(f"Early stopping enabled: patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA}", "INFO")

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    log("Starting training...", "INFO")
    trainer.train()

    # Save final model
    log(f"Saving final model to: {output_dir}", "INFO")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training info with research-backed configuration
    save_training_info(output_dir, {
        "script": "train_01_english_korean",
        "model": args.model,
        "base_model": base_model_path,
        "previous_lora": lora_0_path,
        "training_method": "progressive_lora",
        "epochs": args.epochs,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_data),
        "bidirectional": not args.unidirectional,
        "data_sources": ["korean_parallel_corpora", "ted_talks", "opus_tatoeba"],
        # Research-backed configuration
        "research_config": {
            "lora_preset": args.lora_preset,
            "lora_rank": cfg['lora_r'],
            "lora_alpha": cfg['lora_alpha'],
            "lr_preset": args.lr_preset,
            "learning_rate": cfg['lr'],
            "early_stopping": args.early_stopping,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE if args.early_stopping else None,
            "auto_epochs": args.auto_epochs,
            "data_size_category": (
                "optimal" if data_size >= DATA_SIZE_OPTIMAL else
                "recommended" if data_size >= DATA_SIZE_RECOMMENDED else
                "minimum" if data_size >= DATA_SIZE_MINIMUM else
                "below_minimum"
            ),
        },
        "references": [
            "How Much Data is Enough Data? (arXiv 2024): https://arxiv.org/abs/2409.03454",
            "Fine-Tuning LLMs to Translate (EMNLP 2024): https://aclanthology.org/2024.emnlp-main.24.pdf",
            "When Scaling Meets LLM Finetuning (ICLR 2024): https://openreview.net/pdf?id=5HCnKDeTws",
            "CCMatrix (ACL 2021): https://arxiv.org/abs/1911.04944",
            "OPUS-MT (2022): https://arxiv.org/abs/2212.01936",
        ],
        "research_doc": "research/01__train_translate.md"
    })

    log("=" * 70, "INFO")
    log("TRAIN 01-EK COMPLETE", "INFO")
    log("=" * 70, "INFO")
    log(f"Final model: {output_dir}", "INFO")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}", "ERROR")
        traceback.print_exc()
        exit(1)
