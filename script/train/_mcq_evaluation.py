#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared MCQ Evaluation Utilities

Consolidates MCQ evaluation logic from train_00, train_01, train_02.
"""

import json
import random
import torch
from pathlib import Path
from typing import List, Dict, Callable, Optional
from tqdm import tqdm

from _callbacks import get_terminators, truncate_at_end_of_turn, generate_response


def load_mcq_test_data(filepath: Path, max_samples: int = None) -> List[dict]:
    """
    Load KorMedMCQA test data.

    Args:
        filepath: Path to test.jsonl
        max_samples: Maximum samples to load

    Returns:
        List of test samples
    """
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def extract_answer_letter(response: str) -> str:
    """
    Extract answer letter (A-E) from model response.

    Args:
        response: Model response

    Returns:
        Answer letter or empty string
    """
    response = response.strip().upper()

    # Look for "answer:" keyword first
    import re
    answer_match = re.search(r'answer:\s*([A-E])', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Fallback: find first A-E letter
    for char in response:
        if char in 'ABCDE':
            return char

    return response[:1] if response else ""


def evaluate_mcq_batch(
    model,
    tokenizer,
    test_data: List[dict],
    prompt_template: str,
    device: str,
    max_new_tokens: int = 300,
    score_fn: Callable = None,
    log_fn: Callable = None,
    show_progress: bool = True
) -> Dict:
    """
    Evaluate model on MCQ test data.

    Args:
        model: Model instance
        tokenizer: Tokenizer
        test_data: List of test samples
        prompt_template: Template with {question}, {A}, {B}, {C}, {D}, {E}
        device: Device string
        max_new_tokens: Max tokens to generate
        score_fn: Optional custom scoring function(response, expected) -> dict
        log_fn: Optional logging function for per-sample results
        show_progress: Show progress bar

    Returns:
        Dict with accuracy, correct, total, and detailed results
    """
    model.eval()
    terminators = get_terminators(tokenizer)

    correct = 0
    total = 0
    results = []

    # Use left padding for generation
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        iterator = tqdm(test_data, desc="Evaluating MCQ", leave=False) if show_progress else test_data

        with torch.no_grad():
            for i, sample in enumerate(iterator):
                prompt = prompt_template.format(
                    question=sample['question'],
                    A=sample['A'], B=sample['B'], C=sample['C'],
                    D=sample['D'], E=sample['E']
                )
                expected = sample['answer']

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=terminators,
                )

                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=False
                )
                response = truncate_at_end_of_turn(response)

                # Score the response
                if score_fn:
                    result = score_fn(response, expected)
                else:
                    predicted = extract_answer_letter(response)
                    is_correct = predicted == expected.upper()
                    result = {
                        'is_correct': is_correct,
                        'predicted': predicted,
                        'expected': expected
                    }

                if result.get('is_correct', False):
                    correct += 1
                total += 1

                result['sample_id'] = i
                result['response_preview'] = response[:100]
                results.append(result)

                if log_fn:
                    log_fn(f"sample={i} expected={expected} predicted={result.get('predicted', '?')} "
                           f"correct={result.get('is_correct', False)}")

    finally:
        tokenizer.padding_side = original_padding
        model.train()

    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def evaluate_with_format_scoring(
    model,
    tokenizer,
    test_data: List[dict],
    prompt_template: str,
    device: str,
    format_checker: Callable = None,
    max_new_tokens: int = 300,
    log_fn: Callable = None,
    show_progress: bool = True
) -> Dict:
    """
    Evaluate with both correctness and format scoring.

    Args:
        model: Model instance
        tokenizer: Tokenizer
        test_data: List of test samples
        prompt_template: Template for formatting questions
        device: Device string
        format_checker: Function(response) -> float (0.0-1.0 format score)
        max_new_tokens: Max tokens to generate
        log_fn: Logging function
        show_progress: Show progress bar

    Returns:
        Dict with accuracy, format scores, and detailed results
    """
    def score_fn(response, expected):
        predicted = extract_answer_letter(response)
        is_correct = predicted == expected.upper()

        format_score = format_checker(response) if format_checker else 0.0

        # Combined scoring: 2/3 correctness + 1/3 format
        if is_correct:
            total_score = (1.0 * 2/3) + (format_score * 1/3)
        else:
            total_score = format_score * 1/4

        return {
            'is_correct': is_correct,
            'predicted': predicted,
            'expected': expected,
            'format_score': format_score,
            'total_score': total_score,
            'format_valid': format_score >= 0.5
        }

    eval_result = evaluate_mcq_batch(
        model, tokenizer, test_data, prompt_template, device,
        max_new_tokens=max_new_tokens,
        score_fn=score_fn,
        log_fn=log_fn,
        show_progress=show_progress
    )

    # Compute additional metrics
    results = eval_result['results']
    format_valid_count = sum(1 for r in results if r.get('format_valid', False))
    total_format_score = sum(r.get('format_score', 0) for r in results)
    total_score_sum = sum(r.get('total_score', 0) for r in results)

    total = eval_result['total']
    eval_result.update({
        'format_valid_count': format_valid_count,
        'format_valid_rate': (format_valid_count / total * 100) if total > 0 else 0,
        'avg_format_score': (total_format_score / total) if total > 0 else 0,
        'avg_total_score': (total_score_sum / total) if total > 0 else 0
    })

    return eval_result


def quick_evaluate(
    model,
    tokenizer,
    test_data: List[dict],
    prompt_template: str,
    device: str,
    num_samples: int = 10,
    log_fn: Callable = None
) -> Dict:
    """
    Quick evaluation on a random subset.

    Args:
        model: Model instance
        tokenizer: Tokenizer
        test_data: Full test data
        prompt_template: Template for formatting
        device: Device string
        num_samples: Number of samples to evaluate
        log_fn: Logging function

    Returns:
        Dict with accuracy, correct, total
    """
    samples = random.sample(test_data, min(num_samples, len(test_data)))

    return evaluate_mcq_batch(
        model, tokenizer, samples, prompt_template, device,
        max_new_tokens=300,
        log_fn=log_fn,
        show_progress=False
    )


class MCQEvaluator:
    """
    Reusable MCQ evaluator class.

    Usage:
        evaluator = MCQEvaluator(tokenizer, test_data, prompt_template, device)
        result = evaluator.evaluate(model)
        quick_result = evaluator.quick_evaluate(model, num_samples=10)
    """

    def __init__(
        self,
        tokenizer,
        test_data: List[dict],
        prompt_template: str,
        device: str,
        format_checker: Callable = None
    ):
        self.tokenizer = tokenizer
        self.test_data = test_data
        self.prompt_template = prompt_template
        self.device = device
        self.format_checker = format_checker
        self.history = []

    def evaluate(
        self,
        model,
        max_samples: int = None,
        log_fn: Callable = None,
        show_progress: bool = True
    ) -> Dict:
        """Run full evaluation."""
        data = self.test_data[:max_samples] if max_samples else self.test_data

        if self.format_checker:
            result = evaluate_with_format_scoring(
                model, self.tokenizer, data, self.prompt_template,
                self.device, self.format_checker,
                log_fn=log_fn, show_progress=show_progress
            )
        else:
            result = evaluate_mcq_batch(
                model, self.tokenizer, data, self.prompt_template,
                self.device, log_fn=log_fn, show_progress=show_progress
            )

        self.history.append(result)
        return result

    def quick_evaluate(self, model, num_samples: int = 10, log_fn: Callable = None) -> Dict:
        """Quick evaluation on random subset."""
        return quick_evaluate(
            model, self.tokenizer, self.test_data, self.prompt_template,
            self.device, num_samples=num_samples, log_fn=log_fn
        )

    def get_history(self) -> List[Dict]:
        """Get evaluation history."""
        return self.history.copy()
