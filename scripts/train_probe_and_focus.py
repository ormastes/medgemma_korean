#!/usr/bin/env python3
"""
Probe-and-Focus Training for KorMedMCQA

Strategy:
1. PROBE PHASE: Train N steps on each type, measure KorMedMCQA improvement
2. FOCUS PHASE: Focus on best type until improvement < threshold
3. ROTATE: Move to next best type, repeat
4. STOP: When all types show < threshold improvement

This gives a fair comparison of all types before focusing resources.
"""

import argparse
import json
import re
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from tqdm import tqdm
import copy

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "models" / "probe_focus_training"
LOG_DIR = BASE_DIR / "logs"

TYPES = ["type1_text", "type2_text_reasoning", "type3_word", "type4_word_reasoning"]
TYPE_DESCRIPTIONS = {
    "type1_text": "TEXT - Full answers without reasoning (118K samples)",
    "type2_text_reasoning": "TEXT+REASONING - Full answers with <R>...<R/> (23K samples)",
    "type3_word": "WORD - MCQ letter answers A/B/C/D/E (17K samples)",
    "type4_word_reasoning": "WORD+REASONING - Short answers with reasoning (8K samples)"
}

MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 64, "lora_alpha": 128,
        "lr": 1e-4, "batch": 4, "grad_accum": 4,
        "max_length": 512
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 128, "lora_alpha": 256,
        "lr": 5e-5, "batch": 1, "grad_accum": 16,
        "max_length": 512
    }
}


@dataclass
class FailedQuestion:
    """Track a failed question for analysis"""
    question_id: int
    question_text: str  # Full question text (extracted from prompt)
    prompt_preview: str  # First 200 chars of prompt
    expected: str
    predicted: str
    full_response: str
    expected_choice_text: str = ""  # What the correct answer said
    predicted_choice_text: str = ""  # What the wrong answer said
    all_choices: Dict[str, str] = None  # All choice options
    fail_count: int = 1  # How many times this question failed
    last_failed_step: int = 0


def extract_choices(prompt: str) -> Dict[str, str]:
    """Extract choice options from prompt text"""
    choices = {}
    # Pattern: A) text or A. text or (A) text
    # Match lines like "A) some text" or "A. some text"
    pattern = r'([A-E])\s*[).\]]\s*(.+?)(?=\n[A-E]\s*[).\]]|\n<|\Z)'
    matches = re.findall(pattern, prompt, re.DOTALL)
    for letter, text in matches:
        choices[letter] = text.strip()[:150]  # Limit to 150 chars
    return choices


def extract_question_text(prompt: str) -> str:
    """Extract the question text from prompt (between user tag and choices)"""
    # Find content between <|im_start|>user and the first choice (A))
    match = re.search(r'<\|im_start\|>user\n(.+?)(?=\n[A-E]\s*[).\]])', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try to get content after "user" marker
    match = re.search(r'user[:\n]\s*(.+?)(?=\n[A-E]\s*[).\]])', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt[:500]  # Fallback to first 500 chars


@dataclass
class ProbeResult:
    """Result from probing a single type"""
    type_name: str
    before_accuracy: float
    after_accuracy: float
    improvement: float
    steps_trained: int
    avg_loss: float


@dataclass
class TypeTracker:
    """Track training progress for each type"""
    name: str
    probe_improvement: float = 0.0
    total_steps: int = 0
    total_samples: int = 0
    current_position: int = 0
    accuracy_history: List[Tuple[int, float]] = field(default_factory=list)
    improvements: List[float] = field(default_factory=list)
    is_exhausted: bool = False
    focus_rounds: int = 0


def load_kormedmcqa_eval_hf() -> Dataset:
    """Load KorMedMCQA directly from HuggingFace with original structure"""
    from datasets import load_dataset

    all_samples = []

    # Load all test sets (doctor, nurse, pharm, dentist)
    for subject in ["doctor", "nurse", "pharm", "dentist"]:
        try:
            ds = load_dataset("sean0042/KorMedMCQA", subject, split="test")
            for item in ds:
                # Convert answer number (1-5) to letter (A-E)
                answer_letter = chr(64 + int(item['answer']))  # 1->A, 2->B, etc.

                # Build prompt in same format as training
                prompt = f"""<|im_start|>system
의료 객관식 문제입니다. 정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).
<|im_end|>
<|im_start|>user
{item['question']}

A) {item['A']}
B) {item['B']}
C) {item['C']}
D) {item['D']}
E) {item['E']}
<|im_end|>
<|im_start|>assistant
"""
                all_samples.append({
                    'prompt': prompt,
                    'answer': answer_letter,
                    'question_text': item['question'],
                    'choices': {
                        'A': item['A'],
                        'B': item['B'],
                        'C': item['C'],
                        'D': item['D'],
                        'E': item['E']
                    },
                    'subject': subject,
                    'year': item.get('year', ''),
                    'cot': item.get('cot', '')
                })
        except Exception as e:
            print(f"Warning: Could not load {subject}: {e}")

    print(f"[EVAL] Loaded {len(all_samples)} KorMedMCQA samples from HuggingFace")
    return Dataset.from_list(all_samples)


def load_kormedmcqa_eval(data_dir: Path) -> Dataset:
    """Load KorMedMCQA - try HuggingFace first, fallback to local"""
    try:
        return load_kormedmcqa_eval_hf()
    except Exception as e:
        print(f"Could not load from HuggingFace ({e}), using local data...")

    # Fallback to local data
    samples = []
    val_file = data_dir / "type3_word" / "validation" / "data.jsonl"

    if val_file.exists():
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get('source', '').lower() == 'kormedmcqa':
                    samples.append(data)

    print(f"[EVAL] Loaded {len(samples)} KorMedMCQA samples (local)")
    return Dataset.from_list(samples)


def load_type_data(data_dir: Path, type_name: str) -> List[dict]:
    """Load and return data as list for random access"""
    samples = []
    train_file = data_dir / type_name / "train" / "data.jsonl"

    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))

    return samples


# Global tracker for failed questions (keeps last 10 with fail counts)
FAILED_QUESTIONS: Dict[int, FailedQuestion] = {}
MAX_FAILED_TRACK = 10
CURRENT_STEP = 0


def update_failed_questions(
    question_id: int,
    prompt: str,
    expected: str,
    predicted: str,
    full_response: str,
    step: int,
    sample: dict = None  # Original sample with pre-split fields
):
    """Update failed questions tracker"""
    global FAILED_QUESTIONS

    # Use pre-split data if available, otherwise extract from prompt
    if sample and 'choices' in sample:
        choices = sample['choices']
        question_text = sample.get('question_text', '')
    else:
        choices = extract_choices(prompt)
        question_text = extract_question_text(prompt)

    expected_text = choices.get(expected, "")
    predicted_text = choices.get(predicted, "")

    if question_id in FAILED_QUESTIONS:
        # Increment fail count for repeat failures
        FAILED_QUESTIONS[question_id].fail_count += 1
        FAILED_QUESTIONS[question_id].predicted = predicted
        FAILED_QUESTIONS[question_id].full_response = full_response
        FAILED_QUESTIONS[question_id].predicted_choice_text = predicted_text
        FAILED_QUESTIONS[question_id].last_failed_step = step
    else:
        # Add new failed question
        FAILED_QUESTIONS[question_id] = FailedQuestion(
            question_id=question_id,
            question_text=question_text,
            prompt_preview=prompt[:200].replace('\n', ' '),
            expected=expected,
            predicted=predicted,
            full_response=full_response[:100],
            expected_choice_text=expected_text,
            predicted_choice_text=predicted_text,
            all_choices=choices,
            fail_count=1,
            last_failed_step=step
        )

    # Keep only top 10 by fail_count (most frequently failing)
    if len(FAILED_QUESTIONS) > MAX_FAILED_TRACK:
        sorted_fails = sorted(
            FAILED_QUESTIONS.items(),
            key=lambda x: (-x[1].fail_count, -x[1].last_failed_step)
        )
        FAILED_QUESTIONS = dict(sorted_fails[:MAX_FAILED_TRACK])


def print_failed_analysis():
    """Print analysis of persistently failing questions"""
    if not FAILED_QUESTIONS:
        return

    print(f"\n{'='*80}")
    print("PERSISTENT FAILURE ANALYSIS (Top 10 Most Failing Questions)")
    print(f"{'='*80}")

    sorted_fails = sorted(
        FAILED_QUESTIONS.values(),
        key=lambda x: -x.fail_count
    )

    for i, fq in enumerate(sorted_fails, 1):
        print(f"\n  [{i}] Question #{fq.question_id} - Failed {fq.fail_count}x (last: step {fq.last_failed_step})")
        print(f"      Expected: {fq.expected} | Got: {fq.predicted}")
        print(f"      ")
        print(f"      QUESTION:")
        # Print question text with proper indentation
        for line in fq.question_text[:500].split('\n'):
            print(f"        {line}")
        print(f"      ")
        print(f"      CHOICES:")
        if fq.all_choices:
            for letter in sorted(fq.all_choices.keys()):
                marker = "→" if letter == fq.predicted else ("✓" if letter == fq.expected else " ")
                print(f"        {marker} {letter}) {fq.all_choices[letter][:70]}...")
        print(f"      ")
        print(f"      CORRECT ({fq.expected}): {fq.expected_choice_text[:80]}")
        print(f"      WRONG   ({fq.predicted}): {fq.predicted_choice_text[:80]}")

    print(f"\n{'='*80}")


def evaluate_kormedmcqa(
    model,
    tokenizer,
    eval_dataset: Dataset,
    max_samples: int = 200,
    verbose: bool = True,
    step: int = 0
) -> Tuple[float, List[dict]]:
    """
    Evaluate on KorMedMCQA, return accuracy and list of failures
    Also updates global FAILED_QUESTIONS tracker
    """
    global CURRENT_STEP
    CURRENT_STEP = step

    model.eval()
    correct = 0
    total = 0
    current_failures = []

    samples = list(eval_dataset)[:max_samples]

    with torch.no_grad():
        iterator = tqdm(samples, desc="Eval") if verbose else samples
        for idx, sample in enumerate(iterator):
            try:
                prompt = sample['prompt']
                expected = sample.get('answer', sample.get('completion', '')).strip().upper()
                expected = re.sub(r'[^A-E]', '', expected)[:1]

                if not expected:
                    continue

                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=800
                ).to(model.device)

                outputs = model.generate(
                    **inputs, max_new_tokens=10,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id
                )

                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                predicted = response.split()[0] if response.split() else ""
                predicted = re.sub(r'[^A-E]', '', predicted.upper())[:1]

                if predicted == expected:
                    correct += 1
                    # Remove from failed tracker if it was there and now passes
                    if idx in FAILED_QUESTIONS:
                        del FAILED_QUESTIONS[idx]
                else:
                    # Track failure - pass sample for pre-split data
                    update_failed_questions(idx, prompt, expected, predicted, response, step, sample)
                    current_failures.append({
                        "id": idx,
                        "expected": expected,
                        "predicted": predicted,
                        "response": response[:100],
                        "question": sample.get('question_text', ''),
                        "choices": sample.get('choices', {})
                    })

                total += 1

            except Exception:
                continue

    accuracy = 100 * correct / total if total > 0 else 0
    model.train()

    # Clear memory after evaluation
    torch.cuda.empty_cache()

    return accuracy, current_failures


def train_steps(
    model,
    tokenizer,
    data: List[dict],
    tracker: TypeTracker,
    num_steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_length: int,
    device: str
) -> float:
    """Train for N steps on given data, return avg loss"""
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    total_loss = 0
    steps_done = 0
    accum_loss = 0
    batch_texts = []

    start_pos = tracker.current_position

    pbar = tqdm(total=num_steps, desc=f"Train {tracker.name[:8]}")

    sample_idx = 0
    while steps_done < num_steps:
        # Get sample
        idx = (start_pos + sample_idx) % len(data)
        sample = data[idx]
        sample_idx += 1

        text = sample.get('text', '')
        if not text:
            prompt = sample.get('prompt', '')
            completion = sample.get('completion', sample.get('answer', ''))
            text = f"{prompt}{completion}"

        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            inputs = tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=max_length
            ).to(device)

            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['input_ids']
            )

            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (sample_idx // batch_size) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

                total_loss += accum_loss
                accum_loss = 0
                steps_done += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{total_loss/max(steps_done,1):.4f}"})

                # Clear CUDA cache periodically to prevent fragmentation
                if steps_done % 5 == 0:
                    torch.cuda.empty_cache()

            batch_texts = []

    pbar.close()

    tracker.current_position = (start_pos + sample_idx) % len(data)
    tracker.total_steps += steps_done
    tracker.total_samples += sample_idx

    return total_loss / max(steps_done, 1)


def probe_all_types(
    model,
    tokenizer,
    type_data: Dict[str, List[dict]],
    eval_dataset: Dataset,
    trackers: Dict[str, TypeTracker],
    probe_steps: int,
    cfg: dict,
    device: str,
    eval_samples: int
) -> List[ProbeResult]:
    """Probe each type with N steps, measure improvement"""
    results = []

    print("\n" + "="*80)
    print("PROBE PHASE: Testing all types")
    print("="*80)

    # Get baseline accuracy
    print("\nBaseline evaluation...")
    baseline_acc, _ = evaluate_kormedmcqa(model, tokenizer, eval_dataset, eval_samples, step=0)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    step_counter = 0
    for type_name in TYPES:
        print(f"\n--- Probing {type_name} ---")
        print(f"    {TYPE_DESCRIPTIONS[type_name]}")

        before_acc, _ = evaluate_kormedmcqa(model, tokenizer, eval_dataset, eval_samples, verbose=False, step=step_counter)

        avg_loss = train_steps(
            model, tokenizer, type_data[type_name],
            trackers[type_name], probe_steps,
            cfg["batch"], cfg["grad_accum"], cfg["lr"],
            cfg["max_length"], device
        )
        step_counter += probe_steps

        after_acc, failures = evaluate_kormedmcqa(model, tokenizer, eval_dataset, eval_samples, verbose=False, step=step_counter)
        improvement = after_acc - before_acc

        trackers[type_name].probe_improvement = improvement
        trackers[type_name].improvements.append(improvement)
        trackers[type_name].accuracy_history.append((trackers[type_name].total_steps, after_acc))

        result = ProbeResult(
            type_name=type_name,
            before_accuracy=before_acc,
            after_accuracy=after_acc,
            improvement=improvement,
            steps_trained=probe_steps,
            avg_loss=avg_loss
        )
        results.append(result)

        print(f"    Before: {before_acc:.2f}% -> After: {after_acc:.2f}% (Δ {improvement:+.2f}%)")
        print(f"    Avg loss: {avg_loss:.4f}")

    # Print summary
    print("\n" + "="*80)
    print("PROBE RESULTS SUMMARY")
    print("="*80)
    print(f"{'Type':<25} {'Before':>10} {'After':>10} {'Improvement':>12} {'Loss':>10}")
    print("-"*80)

    for r in sorted(results, key=lambda x: -x.improvement):
        print(f"{r.type_name:<25} {r.before_accuracy:>9.2f}% {r.after_accuracy:>9.2f}% {r.improvement:>+11.2f}% {r.avg_loss:>10.4f}")

    print("="*80)

    # Show failure analysis after probing
    print_failed_analysis()

    return results


def focus_on_type(
    model,
    tokenizer,
    type_name: str,
    type_data: List[dict],
    eval_dataset: Dataset,
    tracker: TypeTracker,
    steps_per_round: int,
    min_improvement: float,
    max_focus_rounds: int,
    cfg: dict,
    device: str,
    eval_samples: int,
    global_best: float,
    global_step: int = 0
) -> Tuple[float, float, int]:
    """Focus training on one type until improvement drops below threshold
    Returns: (final_accuracy, best_accuracy, total_steps_done)
    """
    print(f"\n{'#'*80}")
    print(f"FOCUS PHASE: {type_name}")
    print(f"{'#'*80}")
    print(f"  {TYPE_DESCRIPTIONS[type_name]}")
    print(f"  Min improvement threshold: {min_improvement}%")
    print(f"  Max focus rounds: {max_focus_rounds}")

    rounds = 0
    best_acc = global_best
    consecutive_low = 0
    step = global_step

    while rounds < max_focus_rounds and consecutive_low < 2:
        rounds += 1
        tracker.focus_rounds += 1

        print(f"\n  --- Focus Round {rounds} ---")

        before_acc, _ = evaluate_kormedmcqa(model, tokenizer, eval_dataset, eval_samples, verbose=False, step=step)

        avg_loss = train_steps(
            model, tokenizer, type_data,
            tracker, steps_per_round,
            cfg["batch"], cfg["grad_accum"], cfg["lr"],
            cfg["max_length"], device
        )
        step += steps_per_round

        after_acc, failures = evaluate_kormedmcqa(model, tokenizer, eval_dataset, eval_samples, verbose=False, step=step)
        improvement = after_acc - before_acc

        tracker.improvements.append(improvement)
        tracker.accuracy_history.append((tracker.total_steps, after_acc))

        print(f"  Accuracy: {before_acc:.2f}% -> {after_acc:.2f}% (Δ {improvement:+.2f}%)")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Current failures: {len(failures)}")

        if after_acc > best_acc:
            best_acc = after_acc
            print(f"  NEW BEST: {best_acc:.2f}%")

        if improvement < min_improvement:
            consecutive_low += 1
            print(f"  Low improvement ({consecutive_low}/2)")
        else:
            consecutive_low = 0

    if consecutive_low >= 2:
        tracker.is_exhausted = True
        print(f"\n  Type {type_name} EXHAUSTED (2 consecutive low improvements)")

    # Print failure analysis after focus phase
    print_failed_analysis()

    return tracker.accuracy_history[-1][1] if tracker.accuracy_history else 0, best_acc, step


def select_next_type(trackers: Dict[str, TypeTracker]) -> Optional[str]:
    """Select next type to focus on (highest recent improvement, not exhausted)"""
    candidates = []

    for name, tracker in trackers.items():
        if tracker.is_exhausted:
            continue

        # Use recent average improvement
        recent = tracker.improvements[-3:] if tracker.improvements else [tracker.probe_improvement]
        avg_improvement = sum(recent) / len(recent)

        candidates.append((name, avg_improvement, tracker.total_steps))

    if not candidates:
        return None

    # Sort by improvement (desc), then by steps (asc)
    candidates.sort(key=lambda x: (-x[1], x[2]))
    return candidates[0][0]


def main():
    parser = argparse.ArgumentParser(description="Probe-and-Focus Training")
    parser.add_argument("--model", default="medgemma-27b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--probe-steps", type=int, default=30, help="Steps for initial probe per type")
    parser.add_argument("--focus-steps", type=int, default=50, help="Steps per focus round")
    parser.add_argument("--min-improvement", type=float, default=1.0, help="Min improvement % threshold")
    parser.add_argument("--max-focus-rounds", type=int, default=10, help="Max focus rounds per type")
    parser.add_argument("--target-accuracy", type=float, default=90.0, help="Target accuracy")
    parser.add_argument("--eval-samples", type=int, default=200, help="Eval samples")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"probe_focus_{timestamp}.log"

    cfg = MODEL_CONFIGS[args.model]

    # Model path
    if args.base_model:
        model_path = args.base_model
    else:
        stage6 = BASE_DIR / "models" / "staged_training" / "stage6"
        model_path = str(stage6) if stage6.exists() else cfg["path"]

    print(f"\n{'#'*80}")
    print("PROBE-AND-FOCUS TRAINING")
    print(f"{'#'*80}")
    print(f"Model: {model_path}")
    print(f"Probe steps: {args.probe_steps}")
    print(f"Focus steps: {args.focus_steps}")
    print(f"Min improvement: {args.min_improvement}%")
    print(f"Target: {args.target_accuracy}%")
    print(f"{'#'*80}\n")

    # Load model
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": args.device},
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    print(f"Trainable params: {model.num_parameters(only_trainable=True):,}")

    # Load data
    data_dir = BASE_DIR / "data" / "reviewed"
    eval_dataset = load_kormedmcqa_eval(data_dir)

    print("\nLoading training data...")
    type_data = {}
    for type_name in TYPES:
        type_data[type_name] = load_type_data(data_dir, type_name)
        print(f"  {type_name}: {len(type_data[type_name])} samples")

    # Initialize trackers
    trackers = {t: TypeTracker(name=t) for t in TYPES}

    # Initial evaluation
    print("\n" + "="*80)
    print("INITIAL EVALUATION")
    print("="*80)
    initial_acc, _ = evaluate_kormedmcqa(model, tokenizer, eval_dataset, args.eval_samples, step=0)
    print(f"Initial KorMedMCQA accuracy: {initial_acc:.2f}%")

    global_best = initial_acc
    global_step = 0

    # PHASE 1: Probe all types
    probe_results = probe_all_types(
        model, tokenizer, type_data, eval_dataset, trackers,
        args.probe_steps, cfg, args.device, args.eval_samples
    )

    # Update global best and step count after probing
    for tracker in trackers.values():
        if tracker.accuracy_history:
            acc = tracker.accuracy_history[-1][1]
            if acc > global_best:
                global_best = acc
        global_step += tracker.total_steps

    # Check if already at target
    if global_best >= args.target_accuracy:
        print(f"\nTarget {args.target_accuracy}% reached during probing!")
    else:
        # PHASE 2: Focus rounds
        focus_iteration = 0
        while True:
            focus_iteration += 1

            next_type = select_next_type(trackers)
            if next_type is None:
                print("\nAll types exhausted!")
                break

            print(f"\n{'='*80}")
            print(f"FOCUS ITERATION {focus_iteration}: Selected {next_type}")
            print(f"{'='*80}")

            current_acc, global_best, global_step = focus_on_type(
                model, tokenizer, next_type, type_data[next_type],
                eval_dataset, trackers[next_type],
                args.focus_steps, args.min_improvement, args.max_focus_rounds,
                cfg, args.device, args.eval_samples, global_best, global_step
            )

            if global_best >= args.target_accuracy:
                print(f"\nTarget {args.target_accuracy}% REACHED!")
                break

            # Check if all exhausted
            if all(t.is_exhausted for t in trackers.values()):
                print("\nAll types exhausted!")
                break

    # Save model
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Final report
    print(f"\n{'#'*80}")
    print("FINAL REPORT")
    print(f"{'#'*80}")

    print(f"\n{'Type':<25} {'Steps':>8} {'Samples':>10} {'Best Acc':>10} {'Avg Impr':>10} {'Focus':>8} {'Status':<10}")
    print("-"*90)

    for type_name in TYPES:
        t = trackers[type_name]
        best_acc = max([a[1] for a in t.accuracy_history]) if t.accuracy_history else 0
        avg_impr = sum(t.improvements) / len(t.improvements) if t.improvements else 0
        status = "EXHAUSTED" if t.is_exhausted else "ACTIVE"
        print(f"{type_name:<25} {t.total_steps:>8} {t.total_samples:>10} {best_acc:>9.2f}% {avg_impr:>+9.2f}% {t.focus_rounds:>8} {status:<10}")

    print("-"*90)
    print(f"Global Best: {global_best:.2f}%")
    print(f"{'#'*80}")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    sorted_types = sorted(
        trackers.items(),
        key=lambda x: sum(x[1].improvements)/len(x[1].improvements) if x[1].improvements else 0,
        reverse=True
    )

    for i, (name, tracker) in enumerate(sorted_types, 1):
        avg = sum(tracker.improvements) / len(tracker.improvements) if tracker.improvements else 0
        print(f"  {i}. {name}: avg improvement {avg:+.2f}%")
        print(f"     {TYPE_DESCRIPTIONS[name]}")

    # Final failure analysis
    print("\n" + "#"*80)
    print("FINAL FAILURE ANALYSIS")
    print("#"*80)
    print_failed_analysis()

    # Prepare failed questions for report
    failed_questions_report = []
    for fq in sorted(FAILED_QUESTIONS.values(), key=lambda x: -x.fail_count):
        failed_questions_report.append({
            "question_id": fq.question_id,
            "fail_count": fq.fail_count,
            "question_text": fq.question_text,
            "all_choices": fq.all_choices,
            "expected": fq.expected,
            "expected_choice_text": fq.expected_choice_text,
            "last_predicted": fq.predicted,
            "predicted_choice_text": fq.predicted_choice_text,
            "last_response": fq.full_response,
            "last_failed_step": fq.last_failed_step
        })

    # Save report
    report = {
        "final_best_accuracy": global_best,
        "initial_accuracy": initial_acc,
        "target_accuracy": args.target_accuracy,
        "total_steps": global_step,
        "probe_steps": args.probe_steps,
        "focus_steps": args.focus_steps,
        "type_results": {
            name: {
                "total_steps": t.total_steps,
                "total_samples": t.total_samples,
                "improvements": t.improvements,
                "accuracy_history": t.accuracy_history,
                "probe_improvement": t.probe_improvement,
                "focus_rounds": t.focus_rounds,
                "is_exhausted": t.is_exhausted
            }
            for name, t in trackers.items()
        },
        "recommendations": [
            {"rank": i+1, "type": name, "avg_improvement": sum(t.improvements)/len(t.improvements) if t.improvements else 0}
            for i, (name, t) in enumerate(sorted_types)
        ],
        "persistent_failures": failed_questions_report
    }

    with open(OUTPUT_DIR / "final_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    # Also save failures separately for easy analysis
    with open(OUTPUT_DIR / "failed_questions.json", 'w') as f:
        json.dump(failed_questions_report, f, indent=2, ensure_ascii=False)

    print(f"\nModel saved to: {final_dir}")
    print(f"Report saved to: {OUTPUT_DIR / 'final_report.json'}")
    print(f"Failed questions saved to: {OUTPUT_DIR / 'failed_questions.json'}")


if __name__ == "__main__":
    main()
