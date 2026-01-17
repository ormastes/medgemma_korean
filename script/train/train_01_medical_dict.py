#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 01: Medical Dictionary + Symbol Dictionary
Learn Korean-English medical terminology and special symbols

Directory Structure:
    Input:  model/00_trained/{model}/ (from train_00)
    Training: model/01_training/{model}/ (checkpoints)
    Output: model/01_trained/{model}/ (final model)
    Merged: model/01_another_lora_added/{model}/ (merged + new LoRA)

Key features:
- Left padding for batch generation
- <end_of_turn> as termination token
- Generate response, then check with loss function
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
    create_training_args, save_training_info
)
from training_config import MODEL_CONFIGS
from data_validation import validate_and_report, check_prompt_templates
from trl import SFTTrainer
from transformers import TrainerCallback, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

BASE_DIR = Path(__file__).parent.parent

# Data paths
MEDICAL_DICT_FILE = BASE_DIR / "data" / "02_refined" / "01_medical_dict.json"
CHAR_DICT_FILE = BASE_DIR / "data" / "02_refined" / "02_char_dict.json"
VALIDATION_DATA_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"

# Model paths (NEW STRUCTURE)
INPUT_DIR = BASE_DIR / "model" / "00_trained"       # Input from train_00
TRAINING_DIR = BASE_DIR / "model" / "01_training"   # Checkpoints
OUTPUT_DIR = BASE_DIR / "model" / "01_trained"      # Final output
MERGED_DIR = BASE_DIR / "model" / "01_another_lora_added"  # Merged + new LoRA

# Log file
LOG_FILE = TRAINING_DIR / "train_01_debug.log"

# Default max length
DEFAULT_MAX_LENGTH = 256

# Training prompt template (includes expected response for SFT)
PROMPT_TEMPLATE = """<start_of_turn>user
Meaning of word {term}:<end_of_turn>
<start_of_turn>model
{definition}<end_of_turn>"""

# Validation prompt template (KorMedMCQA)
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


def get_input_lora_path(model_name: str) -> str:
    """Get LoRA adapter path from 00_trained/ (LoRA_0)"""
    # Check for LoRA adapter directory
    lora_path = INPUT_DIR / model_name / "lora_adapter"

    if lora_path.exists() and (lora_path / "adapter_config.json").exists():
        log(f"Found LoRA_0 adapter: {lora_path}", "INFO")
        return str(lora_path)

    # Fallback: check root directory (old structure)
    root_path = INPUT_DIR / model_name
    if root_path.exists() and (root_path / "adapter_config.json").exists():
        log(f"Found LoRA_0 adapter (old structure): {root_path}", "INFO")
        return str(root_path)

    raise ValueError(f"LoRA adapter not found in: {lora_path} or {root_path}\nRun train_00 first!")


def load_dictionary_data(max_samples: int = None) -> list:
    """Load medical dict and char dict, combine them."""
    data = []

    if MEDICAL_DICT_FILE.exists():
        with open(MEDICAL_DICT_FILE, 'r', encoding='utf-8') as f:
            medical_dict = json.load(f)
            log(f"Loaded {len(medical_dict)} medical terms", "INFO")
            data.extend(medical_dict)

    if CHAR_DICT_FILE.exists():
        with open(CHAR_DICT_FILE, 'r', encoding='utf-8') as f:
            char_dict = json.load(f)
            log(f"Loaded {len(char_dict)} symbol terms", "INFO")
            data.extend(char_dict)

    random.shuffle(data)

    if max_samples and len(data) > max_samples:
        data = data[:max_samples]

    log(f"Total dictionary entries: {len(data)}", "INFO")
    return data


def load_validation_data(max_samples: int = None) -> list:
    """Load KorMedMCQA test data for validation."""
    data = []
    test_file = VALIDATION_DATA_DIR / "test.jsonl"

    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line))
        log(f"Loaded {len(data)} validation samples (KorMedMCQA)", "INFO")
    else:
        log(f"Warning: Validation file not found: {test_file}", "WARNING")

    return data


def format_dict_entry(entry: dict) -> dict:
    """Format dictionary entry as training text."""
    text = PROMPT_TEMPLATE.format(
        term=entry['term'],
        definition=entry['definition']
    )
    return {"text": text, "term": entry['term'], "definition": entry['definition']}


def truncate_at_end_of_turn(response: str) -> str:
    """Truncate response at first <end_of_turn> token."""
    if "<end_of_turn>" in response:
        return response.split("<end_of_turn>")[0].strip()
    return response.strip()


def check_correctness(response: str, expected_answer: str) -> tuple:
    """Check if the predicted answer matches the expected answer."""
    response = truncate_at_end_of_turn(response)
    predicted = ""

    if "</reasoning>" in response:
        after_reasoning = response.split("</reasoning>")[-1].strip()
        for char in after_reasoning:
            if char.upper() in 'ABCDE':
                predicted = char.upper()
                break
    else:
        clean_response = response.strip()
        if clean_response:
            for char in reversed(clean_response):
                if char.upper() in 'ABCDE':
                    predicted = char.upper()
                    break

    is_correct = predicted == expected_answer.upper()
    return is_correct, predicted


def calc_score(response: str, expected_answer: str) -> dict:
    """Calculate score based on correctness."""
    is_correct, predicted = check_correctness(response, expected_answer)

    return {
        'total_score': 1.0 if is_correct else 0.0,
        'is_correct': is_correct,
        'predicted': predicted,
        'expected': expected_answer
    }


class KorMedMCQAValidationCallback(TrainerCallback):
    """Callback to evaluate on KorMedMCQA during training."""

    def __init__(self, validation_data, tokenizer, device, eval_interval=50, eval_samples=10):
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
        correct = 0
        total = 0

        indices = random.sample(range(len(self.validation_data)), self.eval_samples)

        log(f"[Step {state.global_step}] KorMedMCQA Validation ({self.eval_samples} samples)", "INFO")

        prompts = []
        expected_answers = []
        for idx in indices:
            sample = self.validation_data[idx]
            prompt = VALIDATION_PROMPT_TEMPLATE.format(
                question=sample['question'],
                A=sample['A'], B=sample['B'], C=sample['C'],
                D=sample['D'], E=sample['E']
            )
            prompts.append(prompt)
            expected_answers.append(sample['answer'])

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=300, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.terminators,
                )

            for i, (output, expected) in enumerate(zip(outputs, expected_answers)):
                input_len = inputs['input_ids'].shape[1]
                generated = output[input_len:]
                response = self.tokenizer.decode(generated, skip_special_tokens=False)
                response = truncate_at_end_of_turn(response)

                score = calc_score(response, expected)
                if score['is_correct']:
                    correct += 1
                total += 1

        except Exception as e:
            log(f"Validation error: {e}", "ERROR")

        finally:
            self.tokenizer.padding_side = original_padding_side

        accuracy = (correct / total * 100) if total > 0 else 0
        self.history.append({
            'step': state.global_step,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        })

        log(f"Accuracy: {accuracy:.1f}% ({correct}/{total})", "INFO")
        model.train()
        return control


def merge_and_add_new_lora(model_path: str, output_path: str, cfg: dict):
    """Merge current LoRA into base and add a new LoRA adapter."""
    log(f"Merging LoRA and adding new adapter...", "INFO")
    log(f"Input: {model_path}", "DEBUG")
    log(f"Output: {output_path}", "DEBUG")

    # Load the trained model
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_path = peft_config.base_model_name_or_path

    # Load base model
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)

    # Merge LoRA into base
    log("Merging LoRA weights...", "INFO")
    model = model.merge_and_unload()

    # Add new LoRA
    log("Adding new LoRA adapter...", "INFO")
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

    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))

    # Save tokenizer
    tokenizer = load_tokenizer(model_path)
    tokenizer.save_pretrained(str(output_path))

    log(f"Merged model with new LoRA saved to: {output_path}", "INFO")


def main():
    parser = create_base_parser("Train 01: Medical Dictionary + Symbol Dictionary")
    parser.add_argument("--show-samples-every", type=int, default=50,
                       help="Evaluate on KorMedMCQA every N steps")
    parser.add_argument("--eval-samples", type=int, default=10,
                       help="Number of KorMedMCQA samples for validation")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                       help=f"Maximum token length (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--skip-merge", action="store_true",
                       help="Skip creating merged model with new LoRA")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip data validation at startup")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    # Override max_length from command line
    cfg = {**cfg, 'max_length': args.max_length}

    # Setup directories
    training_dir = TRAINING_DIR / args.model
    output_dir = OUTPUT_DIR / args.model
    merged_dir = MERGED_DIR / args.model

    training_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70, "INFO")
    log("Train 01: Medical Dictionary + Symbol Dictionary", "INFO")
    log("=" * 70, "INFO")
    log(f"Model: {args.model}", "INFO")
    log(f"Input from: {INPUT_DIR / args.model}", "INFO")
    log(f"Training checkpoints: {training_dir}", "INFO")
    log(f"Final output: {output_dir}", "INFO")
    log(f"Merged output: {merged_dir}", "INFO")
    log(f"Max length: {args.max_length}", "INFO")
    log(f"Epochs: {args.epochs}", "INFO")

    # Get LoRA_0 adapter (from 00_trained/)
    if args.base_model:
        lora_0_path = args.base_model
        log(f"Using provided LoRA_0 path: {lora_0_path}", "INFO")
    else:
        lora_0_path = get_input_lora_path(args.model)

    # Load dictionary data
    dict_data = load_dictionary_data(max_samples=args.max_samples)
    if len(dict_data) == 0:
        log("No dictionary data found!", "ERROR")
        return 1

    # Load validation data
    validation_data = load_validation_data()

    # Format training data
    formatted_data = [format_dict_entry(entry) for entry in dict_data]
    train_dataset = Dataset.from_list(formatted_data)

    # Get base model path from LoRA config
    log("Reading LoRA_0 config...", "INFO")
    peft_config = PeftConfig.from_pretrained(lora_0_path)
    base_model_path = peft_config.base_model_name_or_path
    log(f"Base model from config: {base_model_path}", "DEBUG")

    # Load tokenizer from LoRA_0
    tokenizer = load_tokenizer(lora_0_path)

    # ==========================================================================
    # DATA VALIDATION AT STARTUP
    # ==========================================================================
    if not args.skip_validation:
        log("\n" + "=" * 70, "INFO")
        log("STARTUP VALIDATION", "INFO")
        log("=" * 70, "INFO")

        # Check prompt template
        if dict_data:
            sample_entry = dict_data[0]
            check_prompt_templates(
                {"DICT_PROMPT": PROMPT_TEMPLATE},
                tokenizer,
                args.max_length,
                sample_entry,
                log_fn=lambda msg: log(msg, "INFO")
            )

        # Validate formatted data
        validate_and_report(
            formatted_data[:100],
            tokenizer,
            args.max_length,
            "Dictionary training data (first 100)",
            log_fn=lambda msg: log(msg, "INFO")
        )

        log("=" * 70 + "\n", "INFO")

    # ========================================================================
    # PROGRESSIVE LORA: Merge LoRA_0 and add NEW LoRA_1
    # ========================================================================
    log("=" * 70, "INFO")
    log("PROGRESSIVE LORA TRAINING (Phase 1)", "INFO")
    log("=" * 70, "INFO")
    log("Step 1: Load base model in 8-bit", "INFO")
    log("Step 2: Load and merge LoRA_0 (from train_00)", "INFO")
    log("Step 3: Add NEW trainable LoRA_1", "INFO")
    log("Step 4: Train only LoRA_1 (previous knowledge frozen)", "INFO")
    log("=" * 70, "INFO")

    # Step 1: Load base model with 8-bit
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

    # Step 2: Load and merge LoRA_0
    log(f"Loading LoRA_0 from: {lora_0_path}", "INFO")
    model = PeftModel.from_pretrained(
        base_model,
        lora_0_path,
        is_trainable=False  # Load in inference mode
    )
    log("✓ LoRA_0 loaded", "INFO")

    log("Merging LoRA_0 into base model...", "INFO")
    model = model.merge_and_unload()
    log("✓ LoRA_0 merged (Phase 0 knowledge is now frozen)", "INFO")

    # Step 3: Add NEW trainable LoRA_1
    log("Adding NEW LoRA_1 adapter...", "INFO")
    from training_utils import create_lora_config
    lora_config = create_lora_config(
        rank=cfg.get('lora_r', 64),
        alpha=cfg.get('lora_alpha', 128),
        use_rslora=True,
        include_embeddings=False  # Embeddings already extended in Phase 0
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

    # Validation callback
    validation_callback = KorMedMCQAValidationCallback(
        validation_data=validation_data,
        tokenizer=tokenizer,
        device=args.device,
        eval_interval=args.show_samples_every,
        eval_samples=args.eval_samples
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[validation_callback],
    )

    log("Starting training...", "INFO")
    trainer.train()

    # Save final model
    log(f"Saving final model to: {output_dir}", "INFO")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training info
    save_training_info(output_dir, {
        "script": "train_01_medical_dict",
        "model": args.model,
        "base_model": base_model_path,
        "previous_lora": lora_0_path,
        "training_method": "progressive_lora",
        "lora_phase": 1,
        "epochs": args.epochs,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_data),
        "validation_history": validation_callback.history,
        "final_accuracy": validation_callback.history[-1]['accuracy'] if validation_callback.history else 0
    })

    # Create merged model with new LoRA (for train_02)
    if not args.skip_merge:
        try:
            clear_gpu_memory()
            merge_and_add_new_lora(str(output_dir), str(merged_dir), cfg)
        except Exception as e:
            log(f"Warning: Failed to create merged model: {e}", "WARNING")
            traceback.print_exc()

    log("=" * 70, "INFO")
    log("TRAIN 01 COMPLETE", "INFO")
    log("=" * 70, "INFO")
    log(f"Final model: {output_dir}", "INFO")
    log(f"Merged model: {merged_dir}", "INFO")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}", "ERROR")
        traceback.print_exc()
        exit(1)
