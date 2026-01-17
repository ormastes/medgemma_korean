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
# PROMPT TEMPLATES
# =============================================================================

# SIMPLE/NORMAL prompt (for NORMAL mode - after model learns reasoning format)
SIMPLE_PROMPT_TEMPLATE = """<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

{question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}

<end_of_turn>
<start_of_turn>model
<reasoning>
{reasoning}
</reasoning>{answer}<end_of_turn>"""

# DETAILED/FULL prompt (for FULL mode - teaches reasoning format with example)
DETAILED_PROMPT_TEMPLATE = """<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

```format
질문
A) 선택지
B) 선택지
C) 선택지
D) 선택지
E) 선택지

<reasoning> 블록 안에서 다음 단계로 분석하세요:
첫째: 질문과 선택지를 영어로 다시 작성
둘째: 질문을 명확하게 재정리
셋째: 관련 의학 지식 나열
넷째: 각 선택지 분석 및 확률
A) 분석 xx%
B) 분석 yy%
C) 분석 zz%
D) 분석 qq%
E) 분석 rr%
</reasoning>직후 정답 알파벳
```

```example
12세 여아가 급성림프모구백혈병으로 진단받고, 항암화학요법 2일 째 소변양이 현저히
감소하였다. 혈액검사결과는 다음과 같다. 조치는?
혈색소 8.5 g/dL, 백혈구 185,000/mm^3, 혈소판 78,000/mm^3, 그물적혈구 0.8%,
칼슘 6.9 mg/dL (참고치, 8.8~10.8), 인 8.2 mg/dL (참고치, 3.8~6.5),
요산 15 mg/dL (참고치, <7), 나트륨 132 meq/L, 칼륨 6.1 meq/L, 염소 103 meq/L,
혈액요소질소 58 mg/dL,  크레아티닌 2.4 mg/dL,  젖산탈수소효소(LDH) 1,800 U/L
A) 혈액투석
B) 칼슘 투여
C) 적혈구 수혈
D) 혈소판 수혈
E) 항암화학요법 중단

<reasoning>
첫째: A 12-year-old girl diagnosed with acute lymphoblastic leukemia (ALL), day 2 of chemotherapy, with markedly decreased urine output. Labs: Hb 8.5, WBC 185,000, Plt 78,000, Ca 6.9 (low), P 8.2 (high), uric acid 15 (high), K 6.1 (high), BUN 58, Cr 2.4, LDH 1,800.
A) Hemodialysis B) Calcium administration C) RBC transfusion D) Platelet transfusion E) Stop chemotherapy

둘째: 항암치료 후 대량의 암세포 파괴로 인한 종양용해증후군(Tumor Lysis Syndrome, TLS)이 의심됨.

셋째: 종양용해증후군(TLS)은 항암치료나 방사선 치료 후 암세포가 급격히 파괴되면서 세포 내 물질이 혈중으로 대량 방출되어 발생하는 종양학적 응급상황이다.

넷째: 각 선택지 분석
A) 혈액투석 - TLS로 인한 급성신부전과 생명위협적 고칼륨혈증의 적응증 85%
B) 칼슘 투여 - 고인산혈증에서 칼슘-인 침착 유발하므로 금기 5%
C) 적혈구 수혈 - Hb 8.5는 경도 빈혈, 현재 우선순위 아님 4%
D) 혈소판 수혈 - 78,000은 출혈 없이 즉각 수혈 불필요 3%
E) 항암화학요법 중단 - TLS 이미 발생, 중단해도 현재 위기 해결 불가 3%
</reasoning>A
```

{question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}

<end_of_turn>
<start_of_turn>model
<reasoning>
{reasoning}
</reasoning>{answer}<end_of_turn>"""

# Validation prompt (no expected response)
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
    Check if reasoning format is proper.
    Returns a score from 0.0 to 1.0 based on format compliance.
    """
    score = 0.0
    total_checks = 0

    # Check: <reasoning> and </reasoning> exist and in correct order
    total_checks += 1
    reasoning_start = text.find('<reasoning>')
    reasoning_end = text.find('</reasoning>')

    if reasoning_start == -1 or reasoning_end == -1 or reasoning_start >= reasoning_end:
        return 0.0

    reasoning_content = text[reasoning_start + len('<reasoning>'):reasoning_end]
    score += 1.0

    # Keywords to check
    keyword_list = ['첫째:', '둘째:', '셋째:', '넷째:', 'A)', 'B)', 'C)', 'D)', 'E)']
    selection_list = ['A)', 'B)', 'C)', 'D)', 'E)']

    for keyword in keyword_list:
        total_checks += 1
        keyword_pos = reasoning_content.find(keyword)
        if keyword_pos == -1:
            continue

        content_start = keyword_pos + len(keyword)
        content_end = len(reasoning_content)

        for next_kw in keyword_list:
            next_pos = reasoning_content.find(next_kw, content_start)
            if next_pos != -1 and next_pos < content_end:
                content_end = next_pos

        content = reasoning_content[content_start:content_end].strip()
        words = content.split()

        if len(words) < 3:
            continue

        if keyword in selection_list:
            if not re.search(r'\d+%', content):
                continue

        score += 1.0

    return score / total_checks if total_checks > 0 else 0.0


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

def generate_simple_reasoning(sample: dict) -> str:
    """Generate simple reasoning for training."""
    answer = sample['answer']
    choices = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'],
               'D': sample['D'], 'E': sample['E']}

    lines = []
    for choice, text in choices.items():
        if choice == answer:
            lines.append(f"{choice}) {text[:50]}... - 정답 90%")
        else:
            lines.append(f"{choice}) {text[:30]}... - 5%")

    return "\n".join(lines)


def generate_detailed_reasoning(sample: dict) -> str:
    """Generate detailed reasoning for FULL mode training."""
    answer = sample['answer']
    question = sample['question']
    choices = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'],
               'D': sample['D'], 'E': sample['E']}

    reasoning = f"""첫째: {question[:100]}...
A) {choices['A'][:30]} B) {choices['B'][:30]} C) {choices['C'][:30]} D) {choices['D'][:30]} E) {choices['E'][:30]}

둘째: 이 문제는 의학적 지식을 묻는 문제입니다.

셋째: 관련 의학 지식을 바탕으로 각 선택지를 분석합니다.

넷째: 각 선택지 분석"""

    for choice, text in choices.items():
        if choice == answer:
            reasoning += f"\n{choice}) {text[:50]}... - 정답으로 판단됨 85%"
        else:
            reasoning += f"\n{choice}) {text[:30]}... - 오답 5%"

    return reasoning


def format_mcq_full_mode(sample: dict) -> dict:
    """Format MCQ for FULL mode (detailed prompt with example)."""
    reasoning = generate_detailed_reasoning(sample)
    text = DETAILED_PROMPT_TEMPLATE.format(
        question=sample['question'],
        A=sample['A'], B=sample['B'], C=sample['C'],
        D=sample['D'], E=sample['E'],
        reasoning=reasoning,
        answer=sample['answer']
    )
    return {"text": text, "answer": sample['answer'], "mode": "full"}


def format_mcq_normal_mode(sample: dict) -> dict:
    """Format MCQ for NORMAL mode (simple prompt)."""
    reasoning = generate_simple_reasoning(sample)
    text = SIMPLE_PROMPT_TEMPLATE.format(
        question=sample['question'],
        A=sample['A'], B=sample['B'], C=sample['C'],
        D=sample['D'], E=sample['E'],
        reasoning=reasoning,
        answer=sample['answer']
    )
    return {"text": text, "answer": sample['answer'], "mode": "normal"}


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
                prompt = VALIDATION_PROMPT_TEMPLATE.format(
                    question=sample['question'],
                    A=sample['A'], B=sample['B'], C=sample['C'],
                    D=sample['D'], E=sample['E']
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
    tokenizer = load_tokenizer(model_path)

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
            sample_with_reasoning = {
                **sample_data,
                'reasoning': generate_detailed_reasoning(sample_data)
            }

            check_prompt_templates(
                {
                    "DETAILED (FULL mode)": DETAILED_PROMPT_TEMPLATE,
                    "SIMPLE (NORMAL mode)": SIMPLE_PROMPT_TEMPLATE,
                    "VALIDATION": VALIDATION_PROMPT_TEMPLATE,
                },
                tokenizer,
                args.max_length,
                sample_with_reasoning,
                log_fn=lambda msg: log(msg, "INFO")
            )

        # Format samples and check data lengths
        log("Checking FULL mode data...", "INFO")
        full_samples = [format_mcq_full_mode(s) for s in mcq_data[:100]]
        validate_and_report(
            full_samples, tokenizer, args.max_length,
            "FULL mode samples (first 100)",
            log_fn=lambda msg: log(msg, "INFO")
        )

        log("Checking NORMAL mode data...", "INFO")
        normal_samples = [format_mcq_normal_mode(s) for s in mcq_data[:100]]
        validate_and_report(
            normal_samples, tokenizer, args.max_length,
            "NORMAL mode samples (first 100)",
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
        "base_model": model_path,
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
