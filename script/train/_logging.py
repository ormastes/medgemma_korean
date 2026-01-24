#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Logging Utilities for Training Scripts

Provides unified logging with file output for all training scripts.
"""

import gc
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable


class TrainingLogger:
    """
    Unified logger for training scripts.

    Usage:
        logger = TrainingLogger("train_00", log_dir)
        logger.log("Starting training...")
        logger.log_val("Epoch 1: 85% accuracy")
    """

    def __init__(self, script_name: str, log_dir: Path,
                 create_val_log: bool = True):
        """
        Initialize logger.

        Args:
            script_name: Name of training script (e.g., "train_00")
            log_dir: Directory for log files
            create_val_log: Create separate validation log file
        """
        self.script_name = script_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log = self.log_dir / f"{script_name}_{timestamp}.log"
        self.val_log = self.log_dir / f"{script_name}_validation_{timestamp}.log" if create_val_log else None

    def log(self, msg: str, level: str = "INFO", to_val: bool = False):
        """
        Write log message with timestamp.

        Args:
            msg: Message to log
            level: Log level (INFO, DEBUG, WARNING, ERROR)
            to_val: Also write to validation log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {msg}"
        print(log_msg)

        try:
            with open(self.main_log, 'a', encoding='utf-8') as f:
                f.write(log_msg + "\n")

            if to_val and self.val_log:
                with open(self.val_log, 'a', encoding='utf-8') as f:
                    f.write(log_msg + "\n")
        except Exception:
            pass

    def log_val(self, msg: str, level: str = "VAL"):
        """Write to validation log only."""
        if not self.val_log:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {msg}"

        try:
            with open(self.val_log, 'a', encoding='utf-8') as f:
                f.write(log_msg + "\n")
        except Exception:
            pass

    def section(self, title: str, char: str = "=", width: int = 70):
        """Log a section header."""
        self.log(char * width)
        self.log(title)
        self.log(char * width)


def create_log_function(log_file: Path, val_log_file: Path = None) -> tuple:
    """
    Create simple log functions (for backward compatibility).

    Returns:
        (log_fn, log_val_fn) tuple
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str, level: str = "INFO", to_val: bool = False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {msg}"
        print(log_msg)

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + "\n")

            if to_val and val_log_file:
                with open(val_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_msg + "\n")
        except Exception:
            pass

    def log_val(msg: str, level: str = "VAL"):
        if not val_log_file:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {msg}"

        try:
            with open(val_log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + "\n")
        except Exception:
            pass

    return log, log_val


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_gpu_status(log_fn: Callable = print):
    """
    Check and log GPU status.

    Args:
        log_fn: Logging function to use (default: print)
    """
    log_fn("=" * 50)
    log_fn("GPU Status Check")

    try:
        log_fn(f"PyTorch version: {torch.__version__}")
        log_fn(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            log_fn(f"CUDA version: {torch.version.cuda}")
            log_fn(f"GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)

                log_fn(f"GPU {i}: {props.name}")
                log_fn(f"  Total: {total_mem:.2f} GB, Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    except Exception as e:
        log_fn(f"Error checking GPU: {e}")


def report_memory_usage(log_fn: Callable = print, gpu_total_gb: float = 48.0):
    """
    Report current GPU memory usage.

    Args:
        log_fn: Logging function
        gpu_total_gb: Total GPU memory in GB (for headroom calculation)
    """
    if not torch.cuda.is_available():
        log_fn("CUDA not available")
        return

    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(0) / (1024**3)

    log_fn("=" * 60)
    log_fn("MEMORY USAGE")
    log_fn("=" * 60)
    log_fn(f"Current allocated: {allocated:.2f} GB")
    log_fn(f"Current reserved: {reserved:.2f} GB")
    log_fn(f"Peak allocated: {max_allocated:.2f} GB")
    log_fn(f"GPU total: {gpu_total_gb:.2f} GB")
    log_fn(f"Headroom: {gpu_total_gb - max_allocated:.2f} GB")
    log_fn("=" * 60)
