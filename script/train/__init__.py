#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Scripts Package

Shared utilities for training MedGemma Korean models.

Modules:
    _logging: Logging utilities (TrainingLogger, clear_gpu_memory, check_gpu_status)
    _paths: Path utilities (BASE_DIR, get_model_output_dir, find_lora_path)
    _callbacks: Training callbacks (LossLoggingCallback, MCQValidationCallback)
    _mcq_evaluation: MCQ evaluation (evaluate_mcq_batch, MCQEvaluator)
    _add_lora: LoRA loading (load_model_with_lora, load_for_train_00/01/02)
    _validation: Output validation (validate_mcq_output, validate_mcq_fields)
    _train_text_format: Prompt templates (MCQ_TRAIN_TEMPLATE, MCQ_VALIDATE_TEMPLATE)

    training_config: Model configurations (MODEL_CONFIGS, MEMORY_CONFIGS)
    training_utils: Training utilities (create_training_args, setup_model_with_lora)

Usage:
    from _logging import TrainingLogger, clear_gpu_memory
    from _paths import BASE_DIR, get_model_output_dir
    from _callbacks import LossLoggingCallback, get_terminators
    from _mcq_evaluation import MCQEvaluator, evaluate_mcq_batch
    from _add_lora import load_for_train_00, load_for_train_01, load_for_train_02
"""

# Version
__version__ = "0.2.0"

# Expose commonly used items
from ._logging import (
    TrainingLogger,
    create_log_function,
    clear_gpu_memory,
    check_gpu_status,
    report_memory_usage,
)

from ._paths import (
    BASE_DIR,
    DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    get_data_path,
    get_model_output_dir,
    get_checkpoint_dir,
    get_log_dir,
    has_lora_adapter,
    find_lora_path,
    get_raw_lora_model_path,
    get_previous_phase_lora,
    ensure_dirs,
)

from ._callbacks import (
    LossLoggingCallback,
    EpochEndCallback,
    StepIntervalCallback,
    EarlyStoppingCallback,
    MCQValidationCallback,
    get_terminators,
    truncate_at_end_of_turn,
    generate_response,
)

from ._mcq_evaluation import (
    load_mcq_test_data,
    extract_answer_letter,
    evaluate_mcq_batch,
    evaluate_with_format_scoring,
    quick_evaluate,
    MCQEvaluator,
)

# Note: _add_lora, _validation, _train_text_format imported directly by scripts
