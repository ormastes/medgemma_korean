#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 02: Korean Medical Test (MCQ) with Reasoning

Two Training Modes:
1. FULL MODE: Uses detailed prompt with example (DETAILED_PROMPT_TEMPLATE)
   - Teaches the model the reasoning format
   - Uses fewer samples (can specify --full-samples)
   - Continues until reasoning_score >= threshold

2. NORMAL MODE: Uses simple prompt (SIMPLE_PROMPT_TEMPLATE)
   - Standard MCQ training
   - Uses all samples
   - Starts after FULL MODE completes

Auto-switch: When reasoning_score >= --reasoning-threshold (default 0.7),
             automatically switches from FULL to NORMAL mode.

Directory Structure:
    Input:  model/01_another_lora_added/{model}/ (from train_01)
    Training: model/02_training/{model}/ (checkpoints)
    Output: model/02_trained/{model}/ (final model)

Usage:
    python train_02_kor_med_test.py --model medgemma-27b --epochs 5
    python train_02_kor_med_test.py --model medgemma-27b --full-samples 500 --reasoning-threshold 0.8
"""

import sys
import json
import re
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
from training_config import MODEL_CONFIGS, MEMORY_CONFIGS
from data_validation import validate_and_report, check_prompt_templates
from _train_text_format import (
    MCQ_TRAIN_TEMPLATE, MCQ_VALIDATE_TEMPLATE
)
from trl import SFTTrainer
from transformers import TrainerCallback, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel, PeftConfig

BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data" / "02_refined" / "02_kor_med_test"

# Model paths (NEW STRUCTURE)
INPUT_DIR = BASE_DIR / "model" / "01_another_lora_added"  # Input from train_01
TRAINING_DIR = BASE_DIR / "model" / "02_training"         # Checkpoints
OUTPUT_DIR = BASE_DIR / "model" / "02_trained"            # Final output

# Log file
LOG_FILE = TRAINING_DIR / "train_02_debug.log"

# Default configs
DEFAULT_MAX_LENGTH = 1024  # FULL prompt max=633, NORMAL max=485, +300 response = ~933 max
DEFAULT_FULL_SAMPLES = 500  # Fewer samples for full mode
DEFAULT_REASONING_THRESHOLD = 0.7  # Switch to normal when score >= this


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


def get_lora_paths(model_name: str) -> tuple:
    """Get LoRA_0 and LoRA_1 adapter paths for progressive training."""
    # LoRA_0 from Phase 0 (plain text)
    lora_0_dir = BASE_DIR / "model" / "00_trained" / model_name
    lora_0_path = lora_0_dir / "lora_adapter"

    if not lora_0_path.exists():
        lora_0_path = lora_0_dir  # Fallback to old structure

    if not (lora_0_path / "adapter_config.json").exists():
        raise ValueError(f"LoRA_0 not found: {lora_0_path}\nRun train_00 first!")

    # LoRA_1 from Phase 1 (medical dict)
    lora_1_dir = BASE_DIR / "model" / "01_trained" / model_name
    lora_1_path = lora_1_dir / "lora_adapter"

    if not lora_1_path.exists():
        lora_1_path = lora_1_dir  # Fallback to old structure

    if not (lora_1_path / "adapter_config.json").exists():
        raise ValueError(f"LoRA_1 not found: {lora_1_path}\nRun train_01 first!")

    log(f"Found LoRA_0: {lora_0_path}", "INFO")
    log(f"Found LoRA_1: {lora_1_path}", "INFO")

    return str(lora_0_path), str(lora_1_path)


# =============================================================================
# PROMPT TEMPLATES (imported from _train_text_format.py)
# =============================================================================

# Training template - includes example + actual question with translate fields
TRAINING_TEMPLATE = MCQ_TRAIN_TEMPLATE

# Validation template - includes example, no expected response
VALIDATION_TEMPLATE = MCQ_VALIDATE_TEMPLATE


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def truncate_at_end_of_turn(response: str) -> str:
    """Truncate response at first <end_of_turn> token."""
    if "<end_of_turn>" in response:
        return response.split("<end_of_turn>")[0].strip()
    return response.strip()


def check_reasoning_format(text: str) -> float:
    """
    Check if reasoning format is proper (NEW STRUCTURED FORMAT).
    Returns a score from 0.0 to 1.0 based on format compliance.

    Expected format:
    reasoning:
    facts:
    - ...
    candidates:
    - ...
    criteria:
    - ...
    analysis:
    ...
    evaluation:
    - 평가기준: ...
    - 점수표: ...
    - 근거요약: ...
    answer:
    X
    """
    score = 0.0
    total_checks = 0

    # Required keywords in order
    required_keywords = [
        'reasoning:',
        'facts:',
        'candidates:',
        'criteria:',
        'analysis:',
        'evaluation:',
        'answer:'
    ]

    # Check: All required keywords exist
    for keyword in required_keywords:
        total_checks += 1
        if keyword in text:
            score += 1.0

    # Check: evaluation has Korean sub-fields
    total_checks += 3
    if '평가기준:' in text:
        score += 1.0
    if '점수표:' in text:
        score += 1.0
    if '근거요약:' in text:
        score += 1.0

    # Check: answer is at the end and is a single letter
    total_checks += 1
    answer_match = re.search(r'answer:\s*([A-E])\s*$', text.strip(), re.MULTILINE)
    if answer_match:
        score += 1.0

    # Check: facts has bullet points
    total_checks += 1
    facts_section = re.search(r'facts:(.*?)(?:candidates:|$)', text, re.DOTALL)
    if facts_section and '-' in facts_section.group(1):
        score += 1.0

    # Check: candidates has A, B, C, D, E
    total_checks += 1
    candidates_section = re.search(r'candidates:(.*?)(?:criteria:|$)', text, re.DOTALL)
    if candidates_section:
        cand_text = candidates_section.group(1)
        if 'A' in cand_text and 'B' in cand_text and 'E' in cand_text:
            score += 1.0

    # Check: 점수표 has percentages
    total_checks += 1
    if '점수표:' in text and re.search(r'\d+%', text):
        score += 1.0

    return score / total_checks if total_checks > 0 else 0.0


def check_correctness(response: str, expected_answer: str) -> tuple:
    """Check if the predicted answer matches the expected answer (NEW FORMAT)."""
    response = truncate_at_end_of_turn(response)
    predicted = ""

    # Try to find answer after "answer:" keyword
    answer_match = re.search(r'answer:\s*([A-E])', response, re.IGNORECASE | re.MULTILINE)
    if answer_match:
        predicted = answer_match.group(1).upper()
    else:
        # Fallback: look for last occurrence of A-E letter
        clean_response = response.strip()
        if clean_response:
            for char in reversed(clean_response):
                if char.upper() in 'ABCDE':
                    predicted = char.upper()
                    break

    is_correct = predicted == expected_answer.upper()
    return is_correct, predicted


def calc_score(response: str, expected_answer: str) -> dict:
    """Calculate combined score."""
    is_correct, predicted = check_correctness(response, expected_answer)
    correctness_score = 1.0 if is_correct else 0.0
    reasoning_score = check_reasoning_format(response)

    if is_correct:
        total_score = (correctness_score * 2/3) + (reasoning_score * 1/3)
    else:
        total_score = reasoning_score * 1/4

    return {
        'total_score': total_score,
        'is_correct': is_correct,
        'predicted': predicted,
        'reasoning_score': reasoning_score,
        'correctness_score': correctness_score
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def get_translate_fields(sample: dict) -> dict:
    """Get translate fields from sample, with fallbacks if missing."""
    return {
        'translate_question': sample.get('translate_question', '(translation pending)'),
        'translate_A': sample.get('translate_A', sample.get('A', '')),
        'translate_B': sample.get('translate_B', sample.get('B', '')),
        'translate_C': sample.get('translate_C', sample.get('C', '')),
        'translate_D': sample.get('translate_D', sample.get('D', '')),
        'translate_E': sample.get('translate_E', sample.get('E', '')),
    }


def generate_simple_reasoning(sample: dict) -> str:
    """Generate simple reasoning for training (structured format)."""
    answer = sample['answer']
    choices = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'],
               'D': sample['D'], 'E': sample['E']}

    # Build evaluation scores
    score_parts = []
    reason_parts = []
    for choice in ['A', 'B', 'C', 'D', 'E']:
        if choice == answer:
            score_parts.append(f"{choice}=90%")
            reason_parts.append(f"{choice}(정답으로 판단됨)")
        else:
            score_parts.append(f"{choice}=2%")
            reason_parts.append(f"{choice}(오답)")

    reasoning = f"""facts:
- Medical knowledge required
candidates:
- A, B, C, D, E
criteria:
- Medical accuracy
- Evidence-based
analysis:
Based on medical evidence.
evaluation:
- 평가기준: 의학적 정확성
- 점수표: {', '.join(score_parts)}
- 근거요약: {'; '.join(reason_parts)}"""

    return reasoning


def generate_detailed_reasoning(sample: dict) -> str:
    """Generate detailed reasoning for FULL mode training (structured format)."""
    answer = sample['answer']
    question = sample['question']
    choices = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'],
               'D': sample['D'], 'E': sample['E']}

    # Build evaluation scores
    score_parts = []
    reason_parts = []
    for choice in ['A', 'B', 'C', 'D', 'E']:
        text = choices[choice]
        if choice == answer:
            score_parts.append(f"{choice}=85%")
            reason_parts.append(f"{choice}({text[:40]}... - 정답으로 판단됨)")
        else:
            score_parts.append(f"{choice}=4%")
            reason_parts.append(f"{choice}({text[:30]}... - 오답)")

    reasoning = f"""facts:
- {question[:100]}...
- Medical knowledge analysis required
candidates:
- A, B, C, D, E
criteria:
- Medical accuracy
- Clinical relevance
- Evidence-based guidelines
analysis:
This question requires medical knowledge. Each choice analyzed based on medical evidence and clinical guidelines.
evaluation:
- 평가기준: 의학적 정확성, 임상적 연관성, 가이드라인 부합
- 점수표: {', '.join(score_parts)}
- 근거요약: {'; '.join(reason_parts)}"""

    return reasoning


def generate_model_response(sample: dict, detailed: bool = False) -> str:
    """
    Generate expected model response for training.

    Format:
    {translate}
    reasoning:
    {reasoning}
    answer:
    {answer}
    """
    translate_fields = get_translate_fields(sample)
    reasoning = generate_detailed_reasoning(sample) if detailed else generate_simple_reasoning(sample)
    answer = sample['answer']

    # Build translate block
    translate_text = f"""{translate_fields['translate_question']}
A) {translate_fields['translate_A']}
B) {translate_fields['translate_B']}
C) {translate_fields['translate_C']}
D) {translate_fields['translate_D']}
E) {translate_fields['translate_E']}"""

    return f"""{translate_text}
reasoning:
{reasoning}
answer:
{answer}<end_of_turn>"""


def format_mcq_training(sample: dict, detailed: bool = False) -> dict:
    """
    Format MCQ for training using MCQ_TRAIN_TEMPLATE.

    The template includes an example with full reasoning, then the actual question.
    Template now ends with {answer}<end_of_turn> - no need to append response.
    """
    translate_fields = get_translate_fields(sample)

    # Format template - includes example + question + answer
    # Template already ends with {answer}<end_of_turn>
    text = TRAINING_TEMPLATE.format(
        question=sample['question'],
        A=sample['A'], B=sample['B'], C=sample['C'],
        D=sample['D'], E=sample['E'],
        translate_question=translate_fields['translate_question'],
        translate_A=translate_fields['translate_A'],
        translate_B=translate_fields['translate_B'],
        translate_C=translate_fields['translate_C'],
        translate_D=translate_fields['translate_D'],
        translate_E=translate_fields['translate_E'],
        answer=sample['answer'],  # Template ends with {answer}<end_of_turn>
    )

    return {"text": text, "answer": sample['answer'], "detailed": detailed}


def format_mcq_full_mode(sample: dict) -> dict:
    """Format MCQ for FULL mode (detailed reasoning)."""
    return format_mcq_training(sample, detailed=True)


def format_mcq_normal_mode(sample: dict) -> dict:
    """Format MCQ for NORMAL mode (simple reasoning)."""
    return format_mcq_training(sample, detailed=False)


def load_mcq_data(data_dir: Path, max_samples: int = None) -> list:
    """Load MCQ training data."""
    data = []
    train_file = data_dir / "train.jsonl"

    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line))

    log(f"Loaded {len(data)} MCQ samples", "INFO")
    return data


def load_validation_data(data_dir: Path, max_samples: int = None) -> list:
    """Load MCQ test data for validation."""
    data = []
    test_file = data_dir / "test.jsonl"

    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line))
        log(f"Loaded {len(data)} validation samples", "INFO")

    return data


# =============================================================================
# TRAINER WITH MODE SWITCHING
# =============================================================================

class ModeAwareTrainer:
    """
    Trainer that handles FULL → NORMAL mode transition.

    FULL MODE: Trains with detailed prompts until reasoning_score >= threshold
    NORMAL MODE: Trains with simple prompts for remaining epochs
    """

    def __init__(
        self,
        model,
        tokenizer,
        mcq_data: list,
        validation_data: list,
        cfg: dict,
        args,
        training_dir: Path,
        output_dir: Path,
        max_length: int,
        full_samples: int,
        reasoning_threshold: float,
        eval_interval: int = 50,
        eval_samples: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mcq_data = mcq_data
        self.validation_data = validation_data
        self.cfg = cfg
        self.args = args
        self.training_dir = training_dir
        self.output_dir = output_dir
        self.device = args.device
        self.max_length = max_length

        self.full_samples = min(full_samples, len(mcq_data))
        self.reasoning_threshold = reasoning_threshold
        self.eval_interval = eval_interval
        self.eval_samples = eval_samples

        # State
        self.current_mode = "full"  # "full" or "normal"
        self.global_step = 0
        self.full_mode_complete = False
        self.history = []

        # Terminators
        self.end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        self.terminators = [tokenizer.eos_token_id]
        if self.end_of_turn_id != tokenizer.unk_token_id:
            self.terminators.append(self.end_of_turn_id)

        log(f"Initialized ModeAwareTrainer", "INFO")
        log(f"  Full mode samples: {self.full_samples}", "INFO")
        log(f"  Reasoning threshold: {reasoning_threshold}", "INFO")
        log(f"  Total samples available: {len(mcq_data)}", "INFO")

    def evaluate(self) -> dict:
        """Evaluate on validation data and return scores."""
        self.model.eval()
        correct = 0
        total_reasoning_score = 0.0
        total_combined_score = 0.0
        total = 0

        samples = random.sample(
            self.validation_data,
            min(self.eval_samples, len(self.validation_data))
        )

        original_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        with torch.no_grad():
            for sample in samples:
                # Get translate fields (use placeholders if missing in validation data)
                translate_fields = get_translate_fields(sample)

                # Format prompt using VALIDATION_TEMPLATE (no expected response)
                prompt = VALIDATION_TEMPLATE.format(
                    question=sample['question'],
                    A=sample['A'], B=sample['B'], C=sample['C'],
                    D=sample['D'], E=sample['E'],
                )
                expected = sample['answer']

                inputs = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.terminators,
                    )

                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=False
                    )
                    response = truncate_at_end_of_turn(response)

                    score_result = calc_score(response, expected)
                    total_reasoning_score += score_result['reasoning_score']
                    total_combined_score += score_result['total_score']

                    if score_result['is_correct']:
                        correct += 1
                    total += 1

                except Exception as e:
                    log(f"Evaluation error: {e}", "WARNING")
                    continue

        self.tokenizer.padding_side = original_padding
        self.model.train()

        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'reasoning_score': total_reasoning_score / max(1, total),
            'combined_score': total_combined_score / max(1, total),
            'correct': correct,
            'total': total,
        }

    def train_mode(self, mode: str, num_epochs: int = 1, max_steps: int = None):
        """Train in specified mode."""
        log(f"\n{'='*60}", "INFO")
        log(f"Training in {mode.upper()} mode", "INFO")
        log(f"{'='*60}", "INFO")

        # Prepare data
        if mode == "full":
            samples = random.sample(self.mcq_data, self.full_samples)
            formatted = [format_mcq_full_mode(s) for s in samples]
            log(f"Using {len(formatted)} samples (FULL mode with detailed prompt)", "INFO")
        else:
            formatted = [format_mcq_normal_mode(s) for s in self.mcq_data]
            log(f"Using {len(formatted)} samples (NORMAL mode with simple prompt)", "INFO")

        dataset = Dataset.from_list(formatted)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.training_dir / f"{mode}_mode"),
            num_train_epochs=num_epochs,
            max_steps=max_steps if max_steps else -1,
            per_device_train_batch_size=self.cfg['batch'],
            gradient_accumulation_steps=self.cfg['grad_accum'],
            learning_rate=self.cfg['lr'],
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            report_to="none",
            bf16=True,
            dataloader_pin_memory=False,
            max_grad_norm=1.0,
        )

        # Custom callback for evaluation
        trainer_self = self

        class EvalCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % trainer_self.eval_interval == 0:
                    trainer_self.global_step = state.global_step
                    result = trainer_self.evaluate()

                    log(f"[Step {state.global_step}] Mode: {trainer_self.current_mode}", "INFO")
                    log(f"  Reasoning Score: {result['reasoning_score']:.2%}", "INFO")
                    log(f"  Combined Score: {result['combined_score']:.2%}", "INFO")
                    log(f"  Accuracy: {result['accuracy']:.1f}%", "INFO")

                    trainer_self.history.append({
                        'step': state.global_step,
                        'mode': trainer_self.current_mode,
                        **result
                    })

                    # Check for mode switch in FULL mode
                    if trainer_self.current_mode == "full":
                        if result['reasoning_score'] >= trainer_self.reasoning_threshold:
                            log(f"✓ Reasoning score ({result['reasoning_score']:.2%}) >= threshold ({trainer_self.reasoning_threshold})", "INFO")
                            log(f"Switching to NORMAL mode!", "INFO")
                            trainer_self.full_mode_complete = True
                            control.should_training_stop = True

                return control

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            callbacks=[EvalCallback()],
        )

        trainer.train()

    def train(self, total_epochs: int):
        """
        Main training loop.

        1. Start in FULL mode
        2. When reasoning_score >= threshold, switch to NORMAL mode
        3. Train remaining epochs in NORMAL mode
        """
        log(f"Starting training with {total_epochs} total epochs", "INFO")

        # Phase 1: FULL mode (until reasoning threshold met)
        self.current_mode = "full"
        full_mode_epochs = 0
        max_full_epochs = max(1, total_epochs // 2)  # At most half the epochs in full mode

        while not self.full_mode_complete and full_mode_epochs < max_full_epochs:
            log(f"\nFULL mode epoch {full_mode_epochs + 1}/{max_full_epochs}", "INFO")
            self.train_mode("full", num_epochs=1)
            full_mode_epochs += 1

        if not self.full_mode_complete:
            log(f"⚠️ Max FULL mode epochs reached without hitting threshold", "WARNING")
            log(f"Switching to NORMAL mode anyway...", "INFO")

        # Phase 2: NORMAL mode (remaining epochs)
        remaining_epochs = total_epochs - full_mode_epochs
        if remaining_epochs > 0:
            self.current_mode = "normal"
            log(f"\nNORMAL mode for {remaining_epochs} epochs", "INFO")
            self.train_mode("normal", num_epochs=remaining_epochs)

        log(f"\nTraining complete!", "INFO")
        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = create_base_parser("Train 02: Korean Medical Test with Reasoning")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                       help=f"Maximum token length (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--full-samples", type=int, default=DEFAULT_FULL_SAMPLES,
                       help=f"Number of samples for FULL mode (default: {DEFAULT_FULL_SAMPLES})")
    parser.add_argument("--reasoning-threshold", type=float, default=DEFAULT_REASONING_THRESHOLD,
                       help=f"Reasoning score threshold to switch modes (default: {DEFAULT_REASONING_THRESHOLD})")
    parser.add_argument("--eval-interval", type=int, default=50,
                       help="Evaluate every N steps (default: 50)")
    parser.add_argument("--eval-samples", type=int, default=10,
                       help="Number of samples per evaluation (default: 10)")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip data validation at startup")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    # Setup directories
    training_dir = TRAINING_DIR / args.model
    output_dir = OUTPUT_DIR / args.model

    training_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70, "INFO")
    log("Train 02: Korean Medical Test with Reasoning", "INFO")
    log("=" * 70, "INFO")
    log(f"Model: {args.model}", "INFO")
    log(f"Input from: {INPUT_DIR / args.model}", "INFO")
    log(f"Training dir: {training_dir}", "INFO")
    log(f"Output dir: {output_dir}", "INFO")
    log(f"Max length: {args.max_length}", "INFO")
    log(f"Full mode samples: {args.full_samples}", "INFO")
    log(f"Reasoning threshold: {args.reasoning_threshold}", "INFO")
    log(f"Epochs: {args.epochs}", "INFO")

    # Get previous LoRA adapters (LoRA_0 and LoRA_1)
    if args.base_model:
        # Custom path - assume it's LoRA_1 only
        lora_0_path = None
        lora_1_path = args.base_model
        log(f"Using provided LoRA_1 path: {lora_1_path}", "INFO")
    else:
        lora_0_path, lora_1_path = get_lora_paths(args.model)

    # Load data
    mcq_data = load_mcq_data(DATA_DIR, max_samples=args.max_samples)
    if len(mcq_data) == 0:
        log("No MCQ data found!", "ERROR")
        return 1

    validation_data = load_validation_data(DATA_DIR)

    # Load tokenizer first for validation
    log("Loading tokenizer...", "INFO")
    tokenizer = load_tokenizer(lora_1_path)

    # ==========================================================================
    # DATA VALIDATION AT STARTUP
    # ==========================================================================
    if not args.skip_validation:
        log("\n" + "=" * 70, "INFO")
        log("STARTUP VALIDATION", "INFO")
        log("=" * 70, "INFO")

        # Check prompt template lengths
        sample_data = mcq_data[0] if mcq_data else None
        if sample_data:
            translate_fields = get_translate_fields(sample_data)
            sample_with_translate = {
                **sample_data,
                **translate_fields,
            }

            check_prompt_templates(
                {
                    "TRAINING": TRAINING_TEMPLATE,
                    "VALIDATION": VALIDATION_TEMPLATE,
                },
                tokenizer,
                args.max_length,
                sample_with_translate,
                log_fn=lambda msg: log(msg, "INFO")
            )

        # Format samples and check data lengths
        log("Checking training data (detailed mode)...", "INFO")
        detailed_samples = [format_mcq_full_mode(s) for s in mcq_data[:100]]
        validate_and_report(
            detailed_samples, tokenizer, args.max_length,
            "Training samples - detailed (first 100)",
            log_fn=lambda msg: log(msg, "INFO")
        )

        log("Checking training data (simple mode)...", "INFO")
        simple_samples = [format_mcq_normal_mode(s) for s in mcq_data[:100]]
        validate_and_report(
            simple_samples, tokenizer, args.max_length,
            "Training samples - simple (first 100)",
            log_fn=lambda msg: log(msg, "INFO")
        )

        log("=" * 70 + "\n", "INFO")

    # Get base model path from LoRA config
    log("Reading LoRA config...", "INFO")
    peft_config = PeftConfig.from_pretrained(lora_1_path)
    base_model_path = peft_config.base_model_name_or_path
    log(f"Base model from config: {base_model_path}", "DEBUG")

    # Check memory config for gradient checkpointing
    from peft import prepare_model_for_kbit_training
    mem_cfg = MEMORY_CONFIGS.get(args.model, {})
    use_gradient_checkpointing = mem_cfg.get('use_gradient_checkpointing', False)
    if use_gradient_checkpointing:
        log("Gradient checkpointing: ENABLED (memory optimization)", "INFO")

    # ========================================================================
    # PROGRESSIVE LORA: Merge LoRA_0 + LoRA_1 and add NEW LoRA_2
    # ========================================================================
    log("=" * 70, "INFO")
    log("PROGRESSIVE LORA TRAINING (Phase 2)", "INFO")
    log("=" * 70, "INFO")
    log("Step 1: Load base model in 8-bit", "INFO")
    log("Step 2: Load and merge LoRA_0 (from train_00)", "INFO")
    log("Step 3: Load and merge LoRA_1 (from train_01)", "INFO")
    log("Step 4: Add NEW trainable LoRA_2", "INFO")
    log("Step 5: Train only LoRA_2 (all previous knowledge frozen)", "INFO")
    log("=" * 70, "INFO")

    # Step 1: Load base model
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

    # Step 2: Load and merge LoRA_0
    if lora_0_path:
        log(f"Loading LoRA_0 from: {lora_0_path}", "INFO")
        model = PeftModel.from_pretrained(
            base_model,
            lora_0_path,
            is_trainable=False
        )
        log("✓ LoRA_0 loaded", "INFO")

        log("Merging LoRA_0 into base model...", "INFO")
        model = model.merge_and_unload()
        log("✓ LoRA_0 merged (Phase 0 knowledge frozen)", "INFO")
    else:
        model = base_model
        log("Skipping LoRA_0 (custom base model)", "INFO")

    # Step 3: Load and merge LoRA_1
    log(f"Loading LoRA_1 from: {lora_1_path}", "INFO")
    model = PeftModel.from_pretrained(
        model,
        lora_1_path,
        is_trainable=False
    )
    log("✓ LoRA_1 loaded", "INFO")

    log("Merging LoRA_1 into base model...", "INFO")
    model = model.merge_and_unload()
    log("✓ LoRA_1 merged (Phase 0 + 1 knowledge frozen)", "INFO")

    # Step 4: Add NEW trainable LoRA_2
    log("Adding NEW LoRA_2 adapter...", "INFO")
    from training_utils import create_lora_config
    from peft import get_peft_model

    lora_config = create_lora_config(
        rank=cfg.get('lora_r', 64),
        alpha=cfg.get('lora_alpha', 128),
        use_rslora=True,
        include_embeddings=False  # Embeddings already extended in Phase 0
    )

    model = get_peft_model(model, lora_config)
    log("✓ LoRA_2 added", "INFO")

    log("\nTrainable parameters:", "INFO")
    model.print_trainable_parameters()
    log("=" * 70, "INFO")

    # Create trainer
    trainer = ModeAwareTrainer(
        model=model,
        tokenizer=tokenizer,
        mcq_data=mcq_data,
        validation_data=validation_data,
        cfg=cfg,
        args=args,
        training_dir=training_dir,
        output_dir=output_dir,
        max_length=args.max_length,
        full_samples=args.full_samples,
        reasoning_threshold=args.reasoning_threshold,
        eval_interval=args.eval_interval,
        eval_samples=args.eval_samples,
    )

    # Train
    history = trainer.train(args.epochs)

    # Save final model
    log(f"Saving final model to: {output_dir}", "INFO")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training info
    save_training_info(output_dir, {
        "script": "train_02_kor_med_test",
        "model": args.model,
        "base_model": base_model_path,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "full_samples": args.full_samples,
        "reasoning_threshold": args.reasoning_threshold,
        "train_samples": len(mcq_data),
        "validation_samples": len(validation_data),
        "history": history,
        "final_reasoning_score": history[-1]['reasoning_score'] if history else 0,
        "final_accuracy": history[-1]['accuracy'] if history else 0,
    })

    # Clear GPU
    clear_gpu_memory()

    # Summary
    log("=" * 70, "INFO")
    log("TRAIN 02 COMPLETE", "INFO")
    log("=" * 70, "INFO")
    log(f"Final model: {output_dir}", "INFO")

    if history:
        log(f"Final reasoning score: {history[-1]['reasoning_score']:.2%}", "INFO")
        log(f"Final accuracy: {history[-1]['accuracy']:.1f}%", "INFO")

    log("\nTraining History:", "INFO")
    log(f"{'Step':>8} | {'Mode':>8} | {'Reasoning':>10} | {'Accuracy':>10}", "INFO")
    log("-" * 50, "INFO")
    for h in history:
        log(f"{h['step']:>8} | {h['mode']:>8} | {h['reasoning_score']:>9.1%} | {h['accuracy']:>9.1f}%", "INFO")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}", "ERROR")
        traceback.print_exc()
        exit(1)
