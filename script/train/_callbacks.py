#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Training Callbacks

Provides reusable callbacks for training scripts.
"""

import random
import torch
from typing import List, Callable, Optional
from transformers import TrainerCallback


class LossLoggingCallback(TrainerCallback):
    """
    Callback to log training loss.

    Usage:
        callback = LossLoggingCallback(log_fn=logger.log_val)
        trainer = SFTTrainer(..., callbacks=[callback])
    """

    def __init__(self, log_fn: Callable = print, log_interval: int = 1):
        """
        Args:
            log_fn: Function to log messages
            log_interval: Log every N log events (default: 1 = all)
        """
        self.log_fn = log_fn
        self.log_interval = log_interval
        self.losses = []
        self.log_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            loss = logs['loss']
            step = state.global_step
            self.losses.append({'step': step, 'loss': loss})

            self.log_count += 1
            if self.log_count % self.log_interval == 0:
                self.log_fn(f"step={step} loss={loss:.4f}")

        return control

    def get_losses(self) -> List[dict]:
        """Get recorded losses."""
        return self.losses.copy()


class EpochEndCallback(TrainerCallback):
    """
    Callback to run a function at the end of each epoch.

    Usage:
        def on_epoch(epoch, step, model):
            print(f"Epoch {epoch} done!")

        callback = EpochEndCallback(on_epoch)
    """

    def __init__(self, callback_fn: Callable):
        """
        Args:
            callback_fn: Function(epoch, step, model) to call at epoch end
        """
        self.callback_fn = callback_fn
        self.current_epoch = 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.current_epoch += 1
        self.callback_fn(self.current_epoch, state.global_step, model)
        return control


class StepIntervalCallback(TrainerCallback):
    """
    Callback to run a function at regular step intervals.

    Usage:
        def evaluate(step, model):
            # Run evaluation
            pass

        callback = StepIntervalCallback(evaluate, interval=100)
    """

    def __init__(self, callback_fn: Callable, interval: int = 100):
        """
        Args:
            callback_fn: Function(step, model) to call
            interval: Call every N steps
        """
        self.callback_fn = callback_fn
        self.interval = interval

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.interval == 0:
            self.callback_fn(state.global_step, model)
        return control


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback for early stopping based on a metric.

    Usage:
        def get_metric(model):
            return accuracy

        callback = EarlyStoppingCallback(
            metric_fn=get_metric,
            threshold=0.9,
            mode="max"  # Stop when metric >= threshold
        )
    """

    def __init__(
        self,
        metric_fn: Callable,
        threshold: float,
        mode: str = "max",
        check_interval: int = 100,
        patience: int = 3
    ):
        """
        Args:
            metric_fn: Function(model) -> float that computes metric
            threshold: Target threshold
            mode: "max" (stop when >= threshold) or "min" (stop when <= threshold)
            check_interval: Check every N steps
            patience: Stop after N consecutive checks above/below threshold
        """
        self.metric_fn = metric_fn
        self.threshold = threshold
        self.mode = mode
        self.check_interval = check_interval
        self.patience = patience
        self.consecutive_hits = 0
        self.history = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.check_interval != 0:
            return control

        if model is None:
            return control

        metric = self.metric_fn(model)
        self.history.append({'step': state.global_step, 'metric': metric})

        hit = (self.mode == "max" and metric >= self.threshold) or \
              (self.mode == "min" and metric <= self.threshold)

        if hit:
            self.consecutive_hits += 1
            if self.consecutive_hits >= self.patience:
                print(f"Early stopping: metric {metric:.4f} {'≥' if self.mode == 'max' else '≤'} {self.threshold}")
                control.should_training_stop = True
        else:
            self.consecutive_hits = 0

        return control


def get_terminators(tokenizer) -> List[int]:
    """
    Get terminator token IDs for generation.

    Returns list containing:
    - eos_token_id
    - <end_of_turn> token id (if exists)

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        List of terminator token IDs
    """
    terminators = [tokenizer.eos_token_id]

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id != tokenizer.unk_token_id:
        terminators.append(end_of_turn_id)

    return terminators


def truncate_at_end_of_turn(response: str) -> str:
    """
    Truncate response at first <end_of_turn> token.

    Args:
        response: Model response string

    Returns:
        Truncated response
    """
    if "<end_of_turn>" in response:
        return response.split("<end_of_turn>")[0].strip()
    return response.strip()


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 300,
    do_sample: bool = False
) -> str:
    """
    Generate a response from the model.

    Args:
        model: Model instance
        tokenizer: Tokenizer
        prompt: Input prompt
        device: Device string
        max_new_tokens: Maximum tokens to generate
        do_sample: Use sampling (False = greedy)

    Returns:
        Generated response (truncated at end_of_turn)
    """
    terminators = get_terminators(tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=False
    )

    return truncate_at_end_of_turn(response)


class MCQValidationCallback(TrainerCallback):
    """
    Callback to validate on MCQ test data during training.

    Usage:
        callback = MCQValidationCallback(
            tokenizer=tokenizer,
            test_data=test_samples,
            device="cuda:0",
            eval_interval=100,
            eval_samples=10,
            log_fn=logger.log
        )
    """

    def __init__(
        self,
        tokenizer,
        test_data: List[dict],
        device: str,
        prompt_template: str,
        eval_interval: int = 100,
        eval_samples: int = 10,
        log_fn: Callable = print,
        score_fn: Callable = None
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            test_data: List of test samples (must have 'question', 'A'-'E', 'answer')
            device: Device string
            prompt_template: Template for formatting questions
            eval_interval: Evaluate every N steps
            eval_samples: Number of samples per evaluation
            log_fn: Logging function
            score_fn: Optional scoring function(response, expected) -> dict
        """
        self.tokenizer = tokenizer
        self.test_data = test_data
        self.device = device
        self.prompt_template = prompt_template
        self.eval_interval = eval_interval
        self.eval_samples = min(eval_samples, len(test_data))
        self.log_fn = log_fn
        self.score_fn = score_fn or self._default_score

        self.history = []
        self.terminators = get_terminators(tokenizer)

    def _default_score(self, response: str, expected: str) -> dict:
        """Default scoring: check if answer letter matches."""
        response = truncate_at_end_of_turn(response).strip().upper()

        # Extract answer letter
        predicted = ""
        for char in response:
            if char in 'ABCDE':
                predicted = char
                break

        is_correct = predicted == expected.upper()
        return {'is_correct': is_correct, 'predicted': predicted}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_interval != 0:
            return control

        if model is None:
            return control

        model.eval()
        original_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        correct = 0
        total = 0

        try:
            samples = random.sample(self.test_data, self.eval_samples)

            for sample in samples:
                prompt = self.prompt_template.format(
                    question=sample['question'],
                    A=sample['A'], B=sample['B'], C=sample['C'],
                    D=sample['D'], E=sample['E']
                )
                expected = sample['answer']

                response = generate_response(
                    model, self.tokenizer, prompt, self.device,
                    max_new_tokens=300
                )

                result = self.score_fn(response, expected)
                if result.get('is_correct', False):
                    correct += 1
                total += 1

        except Exception as e:
            self.log_fn(f"Validation error: {e}")

        finally:
            self.tokenizer.padding_side = original_padding
            model.train()

        accuracy = (correct / total * 100) if total > 0 else 0
        self.history.append({
            'step': state.global_step,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        })

        self.log_fn(f"[Step {state.global_step}] MCQ Accuracy: {accuracy:.1f}% ({correct}/{total})")

        return control

    def get_history(self) -> List[dict]:
        """Get validation history."""
        return self.history.copy()
