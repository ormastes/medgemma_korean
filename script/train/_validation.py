#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Validation Utilities

Shared functions for validating model outputs and training data formats.
Used by train_00, train_01, train_02, etc.

Key validations:
1. MCQ output format (answer:\nX pattern)
2. Required fields exist with content (>2 tokens)
3. English tokens in Korean text detection
"""

import re
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# FIELD DEFINITIONS
# =============================================================================

# Required fields for MCQ format (from _train_text_format.py)
MCQ_REQUIRED_FIELDS = [
    'translate:',       # Translation of question
    'reasoning:',       # Reasoning section
    'facts:',           # Facts list
    'candidates:',      # Candidate answers
    'criteria:',        # Evaluation criteria
    'analysis:',        # Analysis text
    'evaluation:',      # Evaluation with Korean sub-fields
    'answer:',          # Final answer
]

# Fields that MUST have content (>2 tokens after the field name)
MCQ_CONTENT_REQUIRED_FIELDS = [
    'translate:',
    'facts:',
    'analysis:',
    'answer:',
]

# Korean evaluation sub-fields
KOREAN_EVAL_FIELDS = [
    '평가기준:',
    '점수표:',
    '근거요약:',
]

# Valid MCQ answers
VALID_MCQ_ANSWERS = ['A', 'B', 'C', 'D', 'E']


# =============================================================================
# MCQ OUTPUT VALIDATION
# =============================================================================

def extract_answer_from_output(output: str) -> Tuple[Optional[str], bool]:
    """
    Extract answer from model output.

    Rules:
    - Look for 'answer:\\n' pattern (answer: followed by newline)
    - Extract the character after newline
    - If pattern not found, format is invalid

    Args:
        output: Model output string

    Returns:
        Tuple of (answer_letter, format_valid)
        - answer_letter: A/B/C/D/E or None if not found
        - format_valid: True if 'answer:\\n' pattern exists
    """
    # Pattern: answer: followed by newline, then capture A-E
    pattern = r'answer:\s*\n\s*([A-E])'
    match = re.search(pattern, output, re.IGNORECASE)

    if match:
        return match.group(1).upper(), True

    # Fallback: answer: followed by letter (no newline required)
    # But this is NOT proper format, so format_valid = False
    pattern_fallback = r'answer:\s*([A-E])\s*(?:<end_of_turn>|$)'
    match_fallback = re.search(pattern_fallback, output, re.IGNORECASE)

    if match_fallback:
        return match_fallback.group(1).upper(), False

    return None, False


def validate_mcq_output(output: str, expected_answer: str = None) -> Dict[str, Any]:
    """
    Validate MCQ output format and extract score.

    Scoring:
    - format_score: 0.0 if no 'answer:\\n' pattern, 1.0 if valid
    - answer_score: 1.0 if correct, 0.0 if wrong
    - total_score: format_score * (1/3) + answer_score * (2/3)

    Args:
        output: Model output string
        expected_answer: Expected answer (A/B/C/D/E), optional

    Returns:
        Dict with validation results
    """
    answer, format_valid = extract_answer_from_output(output)

    result = {
        'extracted_answer': answer,
        'format_valid': format_valid,
        'format_score': 1.0 if format_valid else 0.0,
        'answer_score': 0.0,
        'total_score': 0.0,
        'is_correct': False,
    }

    # If format invalid, total score = 0 (ignore answer)
    if not format_valid:
        result['total_score'] = 0.0
        return result

    # Check answer correctness if expected is provided
    if expected_answer:
        expected = expected_answer.upper().strip()
        if answer == expected:
            result['answer_score'] = 1.0
            result['is_correct'] = True

    # Total score: weighted combination (format 1/3, answer 2/3)
    result['total_score'] = result['format_score'] * (1/3) + result['answer_score'] * (2/3)

    return result


def validate_mcq_fields(output: str, min_tokens: int = 2) -> Dict[str, Any]:
    """
    Validate that required MCQ fields exist and have content.

    Args:
        output: Model output string
        min_tokens: Minimum tokens required for content fields

    Returns:
        Dict with field validation results
    """
    results = {
        'fields_found': [],
        'fields_missing': [],
        'fields_empty': [],  # Found but content < min_tokens
        'korean_eval_found': [],
        'korean_eval_missing': [],
        'field_score': 0.0,
        'korean_eval_score': 0.0,
        'total_field_score': 0.0,
    }

    # Check required fields
    for field in MCQ_REQUIRED_FIELDS:
        if field in output:
            results['fields_found'].append(field)

            # Check if content field has enough tokens
            if field in MCQ_CONTENT_REQUIRED_FIELDS:
                content = extract_field_content(output, field)
                tokens = content.split() if content else []
                if len(tokens) < min_tokens:
                    results['fields_empty'].append(field)
        else:
            results['fields_missing'].append(field)

    # Check Korean evaluation sub-fields
    for field in KOREAN_EVAL_FIELDS:
        if field in output:
            results['korean_eval_found'].append(field)
        else:
            results['korean_eval_missing'].append(field)

    # Calculate scores
    total_required = len(MCQ_REQUIRED_FIELDS)
    found_valid = len(results['fields_found']) - len(results['fields_empty'])
    results['field_score'] = found_valid / total_required if total_required > 0 else 0.0

    total_korean = len(KOREAN_EVAL_FIELDS)
    results['korean_eval_score'] = len(results['korean_eval_found']) / total_korean if total_korean > 0 else 0.0

    # Total: 70% field presence, 30% Korean eval
    results['total_field_score'] = results['field_score'] * 0.7 + results['korean_eval_score'] * 0.3

    return results


def extract_field_content(text: str, field_name: str) -> str:
    """
    Extract content after a field name until the next field or end.

    Args:
        text: Full text
        field_name: Field name to extract content for (e.g., 'translate:')

    Returns:
        Content string after the field name
    """
    # Find the field
    idx = text.find(field_name)
    if idx == -1:
        return ""

    # Start after field name
    start = idx + len(field_name)

    # Find next field (any of the known fields)
    all_fields = MCQ_REQUIRED_FIELDS + KOREAN_EVAL_FIELDS + ['<end_of_turn>']
    end = len(text)

    for other_field in all_fields:
        if other_field == field_name:
            continue
        other_idx = text.find(other_field, start)
        if other_idx != -1 and other_idx < end:
            end = other_idx

    content = text[start:end].strip()
    return content


# =============================================================================
# REASONING EXTRACTION (ignore reasoning, only check answer)
# =============================================================================

def extract_answer_ignore_reasoning(output: str) -> Tuple[Optional[str], bool]:
    """
    Extract answer, ignoring all reasoning content.

    The reasoning is everything BEFORE 'answer:\\n'.
    Only validates that 'answer:\\n' exists and extracts the letter.

    Args:
        output: Model output string

    Returns:
        Tuple of (answer_letter, format_valid)
    """
    # Split at 'answer:' - everything before is reasoning (ignored)
    if 'answer:' not in output.lower():
        return None, False

    # Find answer: and check for newline pattern
    return extract_answer_from_output(output)


# =============================================================================
# ENGLISH TOKEN DETECTION IN KOREAN TEXT
# =============================================================================

def detect_english_tokens(korean_text: str, min_length: int = 2) -> List[str]:
    """
    Detect English tokens (words) in Korean text that shouldn't be there.

    Args:
        korean_text: Text that should be primarily Korean
        min_length: Minimum length of English word to detect

    Returns:
        List of detected English tokens
    """
    # Pattern: sequences of ASCII letters (excluding numbers and symbols)
    english_pattern = r'\b[A-Za-z]{' + str(min_length) + r',}\b'

    # Find all English words
    matches = re.findall(english_pattern, korean_text)

    # Filter out common acceptable English words in Korean text
    acceptable = {
        'A', 'B', 'C', 'D', 'E',  # MCQ options
        'I', 'II', 'III', 'IV', 'V',  # Roman numerals
        'pH', 'DNA', 'RNA', 'CT', 'MRI', 'ECG', 'EKG',  # Medical abbreviations
        'mg', 'ml', 'kg', 'cm', 'mm',  # Units
        'vs', 'etc', 'e', 'g',  # Common Latin
    }

    # Return non-acceptable English tokens
    return [m for m in matches if m not in acceptable and m.upper() not in acceptable]


def calculate_english_token_ratio(korean_text: str) -> Dict[str, Any]:
    """
    Calculate ratio of English tokens in Korean text.

    Args:
        korean_text: Text to analyze

    Returns:
        Dict with analysis results
    """
    # Total words (split by whitespace)
    all_words = korean_text.split()
    total_words = len(all_words)

    # English tokens
    english_tokens = detect_english_tokens(korean_text)
    english_count = len(english_tokens)

    # Ratio
    ratio = english_count / total_words if total_words > 0 else 0.0

    return {
        'total_words': total_words,
        'english_tokens': english_tokens,
        'english_count': english_count,
        'english_ratio': ratio,
        'is_clean': english_count == 0,
    }


# =============================================================================
# TRANSLATION OUTPUT VALIDATION (Token Overlap)
# =============================================================================

def tokenize_text(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize text into words/tokens.

    Args:
        text: Input text
        lowercase: Whether to lowercase tokens

    Returns:
        List of tokens
    """
    if not text:
        return []

    # Simple whitespace + punctuation tokenization
    # Remove punctuation and split by whitespace
    text_clean = re.sub(r'[^\w\s]', ' ', text)
    tokens = text_clean.split()

    if lowercase:
        tokens = [t.lower() for t in tokens]

    return tokens


def validate_translation_output(
    expected: str,
    produced: str,
    lowercase: bool = True
) -> Dict[str, Any]:
    """
    Validate translation output by comparing token overlap.

    Compares expected English text with produced English text.
    Calculates how many tokens from produced text match expected text.

    Score = (matched tokens in produced) / (total produced tokens)

    Args:
        expected: Expected English text (ground truth)
        produced: Produced/generated English text (model output)
        lowercase: Whether to lowercase tokens for comparison

    Returns:
        Dict with validation results:
        - expected_tokens: List of tokens in expected text
        - produced_tokens: List of tokens in produced text
        - matched_tokens: Tokens that appear in both
        - unmatched_tokens: Tokens in produced but not in expected
        - overlap_score: matched / produced (0.0 to 1.0)
        - precision: matched / produced
        - recall: matched / expected
        - f1_score: harmonic mean of precision and recall
    """
    expected_tokens = tokenize_text(expected, lowercase)
    produced_tokens = tokenize_text(produced, lowercase)

    # Create sets for comparison
    expected_set = set(expected_tokens)
    produced_set = set(produced_tokens)

    # Count matches (by produced tokens, considering duplicates)
    matched_count = 0
    matched_tokens = []
    unmatched_tokens = []

    for token in produced_tokens:
        if token in expected_set:
            matched_count += 1
            matched_tokens.append(token)
        else:
            unmatched_tokens.append(token)

    # Calculate scores
    produced_count = len(produced_tokens)
    expected_count = len(expected_tokens)

    # Overlap score: matched / produced (how much of produced is correct)
    overlap_score = matched_count / produced_count if produced_count > 0 else 0.0

    # Precision: matched / produced
    precision = overlap_score

    # Recall: matched / expected (how much of expected was captured)
    # Use unique matches for recall
    unique_matched = len(expected_set & produced_set)
    recall = unique_matched / len(expected_set) if expected_set else 0.0

    # F1 score
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'expected_tokens': expected_tokens,
        'produced_tokens': produced_tokens,
        'expected_token_count': expected_count,
        'produced_token_count': produced_count,
        'matched_tokens': matched_tokens,
        'matched_count': matched_count,
        'unmatched_tokens': unmatched_tokens,
        'unmatched_count': len(unmatched_tokens),
        'overlap_score': overlap_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }


def validate_translation_batch_outputs(
    expected_list: List[str],
    produced_list: List[str],
    lowercase: bool = True
) -> Dict[str, Any]:
    """
    Validate a batch of translation outputs.

    Args:
        expected_list: List of expected English texts
        produced_list: List of produced English texts
        lowercase: Whether to lowercase tokens

    Returns:
        Dict with batch statistics
    """
    if len(expected_list) != len(produced_list):
        raise ValueError(f"Length mismatch: expected={len(expected_list)}, produced={len(produced_list)}")

    results = {
        'total': len(expected_list),
        'avg_overlap_score': 0.0,
        'avg_precision': 0.0,
        'avg_recall': 0.0,
        'avg_f1_score': 0.0,
        'individual_results': [],
    }

    total_overlap = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for expected, produced in zip(expected_list, produced_list):
        result = validate_translation_output(expected, produced, lowercase)
        results['individual_results'].append(result)

        total_overlap += result['overlap_score']
        total_precision += result['precision']
        total_recall += result['recall']
        total_f1 += result['f1_score']

    n = results['total']
    if n > 0:
        results['avg_overlap_score'] = total_overlap / n
        results['avg_precision'] = total_precision / n
        results['avg_recall'] = total_recall / n
        results['avg_f1_score'] = total_f1 / n

    return results


def print_translation_output_report(results: Dict, log_fn=print):
    """Print translation output validation report."""
    log_fn(f"\n{'='*60}")
    log_fn(f"Translation Output Validation Report")
    log_fn(f"{'='*60}")
    log_fn(f"  Total samples: {results['total']}")
    log_fn(f"  Avg Overlap Score: {results['avg_overlap_score']*100:.1f}%")
    log_fn(f"  Avg Precision: {results['avg_precision']*100:.1f}%")
    log_fn(f"  Avg Recall: {results['avg_recall']*100:.1f}%")
    log_fn(f"  Avg F1 Score: {results['avg_f1_score']*100:.1f}%")
    log_fn(f"{'='*60}\n")


# =============================================================================
# PLAIN TEXT VALIDATION
# =============================================================================

def validate_plain_text_sample(sample: Dict, text_key: str = 'text') -> Dict[str, Any]:
    """
    Validate a plain text training sample.

    Args:
        sample: Training sample dict
        text_key: Key to extract text

    Returns:
        Dict with validation results
    """
    text = sample.get(text_key, '')

    result = {
        'has_text': bool(text),
        'text_length': len(text) if text else 0,
        'word_count': len(text.split()) if text else 0,
        'is_korean': False,
        'korean_ratio': 0.0,
        'is_valid': False,
    }

    if not text:
        return result

    # Check Korean character ratio
    korean_chars = len(re.findall(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]', text))
    total_chars = len(re.sub(r'\s', '', text))  # Exclude whitespace

    result['korean_ratio'] = korean_chars / total_chars if total_chars > 0 else 0.0
    result['is_korean'] = result['korean_ratio'] > 0.3  # At least 30% Korean
    result['is_valid'] = result['has_text'] and result['word_count'] >= 3

    return result


def validate_translation_pair(
    english: str,
    korean: str,
    check_english_in_korean: bool = True
) -> Dict[str, Any]:
    """
    Validate an English-Korean translation pair.

    Args:
        english: English text
        korean: Korean text
        check_english_in_korean: Whether to check for English tokens in Korean

    Returns:
        Dict with validation results
    """
    result = {
        'english_valid': bool(english) and len(english.split()) >= 1,
        'korean_valid': bool(korean) and len(korean.split()) >= 1,
        'english_length': len(english.split()) if english else 0,
        'korean_length': len(korean.split()) if korean else 0,
        'english_in_korean': [],
        'english_in_korean_ratio': 0.0,
        'is_valid': False,
    }

    if check_english_in_korean and korean:
        analysis = calculate_english_token_ratio(korean)
        result['english_in_korean'] = analysis['english_tokens']
        result['english_in_korean_ratio'] = analysis['english_ratio']

    result['is_valid'] = result['english_valid'] and result['korean_valid']

    return result


# =============================================================================
# BATCH VALIDATION
# =============================================================================

def validate_mcq_batch(
    outputs: List[str],
    expected_answers: List[str] = None,
    log_fn=print
) -> Dict[str, Any]:
    """
    Validate a batch of MCQ outputs.

    Args:
        outputs: List of model output strings
        expected_answers: List of expected answers (optional)
        log_fn: Logging function

    Returns:
        Dict with batch statistics
    """
    results = {
        'total': len(outputs),
        'format_valid_count': 0,
        'correct_count': 0,
        'format_accuracy': 0.0,
        'answer_accuracy': 0.0,
        'total_score_avg': 0.0,
        'individual_results': [],
    }

    total_score = 0.0

    for i, output in enumerate(outputs):
        expected = expected_answers[i] if expected_answers and i < len(expected_answers) else None
        result = validate_mcq_output(output, expected)
        results['individual_results'].append(result)

        if result['format_valid']:
            results['format_valid_count'] += 1
        if result['is_correct']:
            results['correct_count'] += 1
        total_score += result['total_score']

    total = results['total']
    if total > 0:
        results['format_accuracy'] = results['format_valid_count'] / total * 100
        results['answer_accuracy'] = results['correct_count'] / total * 100
        results['total_score_avg'] = total_score / total

    return results


def validate_translation_batch(
    pairs: List[Dict],
    english_key: str = 'english',
    korean_key: str = 'korean',
    log_fn=print
) -> Dict[str, Any]:
    """
    Validate a batch of translation pairs.

    Args:
        pairs: List of translation pair dicts
        english_key: Key for English text
        korean_key: Key for Korean text
        log_fn: Logging function

    Returns:
        Dict with batch statistics
    """
    results = {
        'total': len(pairs),
        'valid_count': 0,
        'english_in_korean_count': 0,
        'english_tokens_found': [],
        'invalid_pairs': [],
    }

    for i, pair in enumerate(pairs):
        english = pair.get(english_key, '')
        korean = pair.get(korean_key, '')

        validation = validate_translation_pair(english, korean)

        if validation['is_valid']:
            results['valid_count'] += 1
        else:
            results['invalid_pairs'].append(i)

        if validation['english_in_korean']:
            results['english_in_korean_count'] += 1
            results['english_tokens_found'].extend(validation['english_in_korean'])

    total = results['total']
    if total > 0:
        results['valid_ratio'] = results['valid_count'] / total * 100
        results['english_contamination_ratio'] = results['english_in_korean_count'] / total * 100

    # Deduplicate and count English tokens
    from collections import Counter
    results['english_token_counts'] = Counter(results['english_tokens_found']).most_common(20)

    return results


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_mcq_validation_report(results: Dict, log_fn=print):
    """Print MCQ batch validation report."""
    log_fn(f"\n{'='*60}")
    log_fn(f"MCQ Validation Report")
    log_fn(f"{'='*60}")
    log_fn(f"  Total samples: {results['total']}")
    log_fn(f"  Format valid: {results['format_valid_count']} ({results['format_accuracy']:.1f}%)")
    log_fn(f"  Correct answers: {results['correct_count']} ({results['answer_accuracy']:.1f}%)")
    log_fn(f"  Average score: {results['total_score_avg']:.3f}")
    log_fn(f"{'='*60}\n")


def print_translation_validation_report(results: Dict, log_fn=print):
    """Print translation batch validation report."""
    log_fn(f"\n{'='*60}")
    log_fn(f"Translation Validation Report")
    log_fn(f"{'='*60}")
    log_fn(f"  Total pairs: {results['total']}")
    log_fn(f"  Valid pairs: {results['valid_count']} ({results.get('valid_ratio', 0):.1f}%)")
    log_fn(f"  English contamination: {results['english_in_korean_count']} ({results.get('english_contamination_ratio', 0):.1f}%)")

    if results['english_token_counts']:
        log_fn(f"  Top English tokens in Korean:")
        for token, count in results['english_token_counts'][:10]:
            log_fn(f"    '{token}': {count} occurrences")

    log_fn(f"{'='*60}\n")
