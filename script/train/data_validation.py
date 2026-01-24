#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Validation Utilities

Shared functions for validating training data doesn't overflow max_length.
"""

from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizer


def check_data_lengths(
    data: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_key: str = "text",
    sample_size: int = None
) -> Dict:
    """
    Check if training data fits within max_length.

    Args:
        data: List of training samples
        tokenizer: Tokenizer to use
        max_length: Maximum allowed token length
        text_key: Key to extract text from samples
        sample_size: If set, only check this many random samples

    Returns:
        Dict with validation results:
        - total_samples: Total number of samples checked
        - overflow_count: Number of samples exceeding max_length
        - overflow_ratio: Percentage of overflowing samples
        - max_tokens: Maximum token count found
        - avg_tokens: Average token count
        - overflow_samples: List of (index, token_count) for overflowing samples
        - valid: True if all samples fit
    """
    import random

    samples_to_check = data
    if sample_size and len(data) > sample_size:
        indices = random.sample(range(len(data)), sample_size)
        samples_to_check = [(i, data[i]) for i in indices]
    else:
        samples_to_check = list(enumerate(data))

    overflow_samples = []
    token_counts = []

    for idx, sample in samples_to_check:
        text = sample.get(text_key, "")
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)
        token_counts.append(token_count)

        if token_count > max_length:
            overflow_samples.append((idx, token_count))

    total = len(token_counts)
    overflow_count = len(overflow_samples)

    return {
        "total_samples": total,
        "overflow_count": overflow_count,
        "overflow_ratio": (overflow_count / total * 100) if total > 0 else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "avg_tokens": sum(token_counts) / total if total > 0 else 0,
        "overflow_samples": overflow_samples[:10],  # First 10 only
        "valid": overflow_count == 0,
        "max_length": max_length,
    }


def validate_and_report(
    data: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    data_name: str,
    text_key: str = "text",
    log_fn=print,
    strict: bool = False
) -> bool:
    """
    Validate data and print report.

    Args:
        data: Training data
        tokenizer: Tokenizer
        max_length: Max token length
        data_name: Name for logging
        text_key: Key to get text from samples
        log_fn: Logging function
        strict: If True, raise error on overflow

    Returns:
        True if valid, False if overflow detected
    """
    log_fn(f"\n{'='*60}")
    log_fn(f"Data Validation: {data_name}")
    log_fn(f"{'='*60}")

    result = check_data_lengths(data, tokenizer, max_length, text_key)

    log_fn(f"  Total samples: {result['total_samples']}")
    log_fn(f"  Max length config: {max_length}")
    log_fn(f"  Token stats:")
    log_fn(f"    - Min: {result['min_tokens']}")
    log_fn(f"    - Max: {result['max_tokens']}")
    log_fn(f"    - Avg: {result['avg_tokens']:.1f}")

    if result['overflow_count'] > 0:
        log_fn(f"  ⚠️ OVERFLOW DETECTED:")
        log_fn(f"    - Count: {result['overflow_count']} ({result['overflow_ratio']:.1f}%)")
        log_fn(f"    - Examples (idx, tokens):")
        for idx, count in result['overflow_samples'][:5]:
            log_fn(f"      Sample {idx}: {count} tokens (overflow: +{count - max_length})")

        if strict:
            raise ValueError(f"Data overflow in {data_name}: {result['overflow_count']} samples exceed max_length={max_length}")

        log_fn(f"  ⚠️ These samples will be TRUNCATED during training!")
    else:
        log_fn(f"  ✓ All samples fit within max_length")

    log_fn(f"{'='*60}\n")

    return result['valid']


def filter_overflow_samples(
    data: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_key: str = "text"
) -> Tuple[List[Dict], int]:
    """
    Filter out samples that exceed max_length.

    Returns:
        Tuple of (filtered_data, removed_count)
    """
    filtered = []
    removed = 0

    for sample in data:
        text = sample.get(text_key, "")
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) <= max_length:
            filtered.append(sample)
        else:
            removed += 1

    return filtered, removed


def check_prompt_templates(
    templates: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    sample_data: Dict = None,
    log_fn=print
) -> Dict[str, Dict]:
    """
    Check token lengths of prompt templates.

    Args:
        templates: Dict of {template_name: template_string}
        tokenizer: Tokenizer
        max_length: Max allowed length
        sample_data: Sample data to fill template placeholders
        log_fn: Logging function

    Returns:
        Dict of {template_name: {tokens, fits, overhead}}
    """
    log_fn(f"\n{'='*60}")
    log_fn(f"Prompt Template Validation")
    log_fn(f"{'='*60}")

    results = {}

    for name, template in templates.items():
        # Try to fill template with sample data or placeholder
        try:
            if sample_data:
                filled = template.format(**sample_data)
            else:
                # Use placeholder values
                import re
                placeholders = re.findall(r'\{(\w+)\}', template)
                placeholder_data = {p: f"[{p}]" for p in placeholders}
                filled = template.format(**placeholder_data)
        except Exception:
            filled = template

        tokens = tokenizer.encode(filled, add_special_tokens=True)
        token_count = len(tokens)
        fits = token_count <= max_length
        overhead = max_length - token_count

        results[name] = {
            "tokens": token_count,
            "fits": fits,
            "overhead": overhead,
            "max_length": max_length,
        }

        status = "✓" if fits else "⚠️ OVERFLOW"
        log_fn(f"  {name}:")
        log_fn(f"    Tokens: {token_count} / {max_length} ({status})")
        log_fn(f"    Overhead for content: {overhead} tokens")

    log_fn(f"{'='*60}\n")

    return results
