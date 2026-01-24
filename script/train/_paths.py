#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Path Utilities for Training Scripts

Provides unified path resolution for models, data, and logs.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

# Base directories
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # medgemma_korean/

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
REFINED_DATA_DIR = DATA_DIR / "02_refined"

# Model directories
MODEL_DIR = BASE_DIR / "model"
RAW_MODEL_DIR = MODEL_DIR / "raw"
RAW_LORA_DIR = MODEL_DIR / "raw_lora_added"

# Training output directories (by phase)
TRAIN_00_DIR = MODEL_DIR / "00_trained"
TRAIN_01_DIR = MODEL_DIR / "01_mixed"
TRAIN_02_DIR = MODEL_DIR / "02_trained"

# Checkpoint directories
TRAIN_00_CHECKPOINT_DIR = MODEL_DIR / "00_training"
TRAIN_01_CHECKPOINT_DIR = MODEL_DIR / "01_training"
TRAIN_02_CHECKPOINT_DIR = MODEL_DIR / "02_training"

# Log directory
LOG_DIR = BASE_DIR / "logs"


def get_data_path(data_type: str) -> Path:
    """
    Get path to refined data directory.

    Args:
        data_type: Type of data ("00_plain_text", "01_english_korean", "02_kor_med_test")

    Returns:
        Path to data directory
    """
    return REFINED_DATA_DIR / data_type


def get_model_output_dir(phase: str, model_name: str, subdir: str = None) -> Path:
    """
    Get model output directory for a training phase.

    Args:
        phase: Training phase ("00", "01", "02")
        model_name: Model name (e.g., "medgemma-4b")
        subdir: Optional subdirectory (e.g., "final", "training")

    Returns:
        Path to output directory
    """
    phase_dirs = {
        "00": TRAIN_00_DIR,
        "01": TRAIN_01_DIR,
        "02": TRAIN_02_DIR,
    }

    base = phase_dirs.get(phase, MODEL_DIR / f"{phase}_trained")
    path = base / model_name

    if subdir:
        path = path / subdir

    return path


def get_checkpoint_dir(phase: str, model_name: str) -> Path:
    """
    Get checkpoint directory for a training phase.

    Args:
        phase: Training phase ("00", "01", "02")
        model_name: Model name

    Returns:
        Path to checkpoint directory
    """
    checkpoint_dirs = {
        "00": TRAIN_00_CHECKPOINT_DIR,
        "01": TRAIN_01_CHECKPOINT_DIR,
        "02": TRAIN_02_CHECKPOINT_DIR,
    }

    base = checkpoint_dirs.get(phase, MODEL_DIR / f"{phase}_training")
    return base / model_name


def get_log_dir(script_name: str) -> Path:
    """
    Get log directory for a script.

    Args:
        script_name: Script name (e.g., "train_00", "train_01")

    Returns:
        Path to log directory
    """
    return LOG_DIR / script_name


def has_lora_adapter(model_path: str) -> bool:
    """Check if model path contains a LoRA adapter."""
    path = Path(model_path)
    return (path / "adapter_config.json").exists()


def find_lora_path(base_dir: Path, model_name: str) -> Optional[str]:
    """
    Find LoRA adapter path in a model directory.

    Checks:
    1. base_dir/model_name/lora_adapter/
    2. base_dir/model_name/final/
    3. base_dir/model_name/

    Args:
        base_dir: Base directory to search
        model_name: Model name

    Returns:
        Path string if found, None otherwise
    """
    # Possible locations in priority order
    candidates = [
        base_dir / model_name / "lora_adapter",
        base_dir / model_name / "final",
        base_dir / model_name,
    ]

    for path in candidates:
        if path.exists() and has_lora_adapter(str(path)):
            return str(path)

    return None


def get_raw_lora_model_path(model_name: str, fallback_to_hf: bool = True) -> str:
    """
    Get model path from raw_lora_added/ directory.

    Args:
        model_name: Model name (e.g., "medgemma-4b")
        fallback_to_hf: Fall back to HuggingFace model if not found

    Returns:
        Path to model

    Raises:
        ValueError: If model not found and no fallback
    """
    from training_config import MODEL_CONFIGS

    lora_model_path = RAW_LORA_DIR / model_name

    if lora_model_path.exists() and has_lora_adapter(str(lora_model_path)):
        return str(lora_model_path)

    if not fallback_to_hf:
        raise ValueError(
            f"raw_lora_added/{model_name} not found. "
            f"Run: python script/init_lora_on_raw.py --model {model_name}"
        )

    # Fall back to HuggingFace model
    print(f"Warning: raw_lora_added/{model_name} not found, using HuggingFace model")

    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]['path']

    raise ValueError(f"Unknown model: {model_name}")


def get_previous_phase_lora(phase: str, model_name: str) -> str:
    """
    Get LoRA path from previous training phase.

    Args:
        phase: Current phase ("01" needs "00", "02" needs "01")
        model_name: Model name

    Returns:
        Path to previous phase LoRA

    Raises:
        ValueError: If previous phase LoRA not found
    """
    previous_phase = {
        "01": "00",
        "02": "01",
    }

    if phase not in previous_phase:
        raise ValueError(f"Phase {phase} has no previous phase")

    prev = previous_phase[phase]
    prev_dir = get_model_output_dir(prev, model_name)

    lora_path = find_lora_path(prev_dir.parent, model_name)
    if not lora_path:
        # Try alternate directory structure
        lora_path = find_lora_path(MODEL_DIR / f"{prev}_trained", model_name)

    if not lora_path:
        raise ValueError(
            f"LoRA from phase {prev} not found for {model_name}. "
            f"Run train_{prev}_*.py first!"
        )

    return lora_path


def get_lora_chain(phase: str, model_name: str) -> Tuple[Optional[str], ...]:
    """
    Get chain of LoRA paths for progressive training.

    Args:
        phase: Current phase ("02" needs both "00" and "01")
        model_name: Model name

    Returns:
        Tuple of LoRA paths (lora_0, lora_1, ...) or None for missing
    """
    lora_paths = []

    # Phase 02 needs 00 and 01
    if phase == "02":
        for prev_phase in ["00", "01"]:
            try:
                path = get_previous_phase_lora(prev_phase if prev_phase == "01" else "01", model_name)
                # Actually find the right phase
                prev_dir = MODEL_DIR / f"{prev_phase}_trained" / model_name
                if prev_phase == "01":
                    prev_dir = MODEL_DIR / "01_mixed" / model_name / "final"
                found = find_lora_path(prev_dir.parent if prev_phase != "01" else prev_dir.parent.parent, model_name)
                lora_paths.append(found)
            except ValueError:
                lora_paths.append(None)
    else:
        # Single previous phase
        try:
            lora_paths.append(get_previous_phase_lora(phase, model_name))
        except ValueError:
            lora_paths.append(None)

    return tuple(lora_paths)


def ensure_dirs(*paths: Path):
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
