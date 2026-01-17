#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging Utilities for MedGemma Korean Training

Provides centralized logging to log/ directory.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime


# Base directories
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "log"


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    console: bool = True,
    file_mode: str = "a"
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.

    Args:
        name: Logger name (usually script name without .py)
        log_file: Log filename (default: {name}_{timestamp}.log)
        level: Logging level (default: INFO)
        console: Also log to console (default: True)
        file_mode: 'a' for append, 'w' for overwrite (default: 'a')

    Returns:
        Configured logger

    Example:
        logger = setup_logger("train_01_medical_dict")
        logger.info("Starting training...")
    """
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"

    log_path = LOG_DIR / log_file
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"Log file: {log_path}")

    return logger


def get_latest_log(name_prefix: str) -> Path:
    """
    Get the most recent log file matching a prefix.

    Args:
        name_prefix: Log file prefix (e.g., "train_01")

    Returns:
        Path to most recent log file, or None if not found
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logs = sorted(
        LOG_DIR.glob(f"{name_prefix}*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return logs[0] if logs else None


def list_logs(name_prefix: str = None, limit: int = 10) -> list:
    """
    List recent log files.

    Args:
        name_prefix: Filter by prefix (optional)
        limit: Maximum number of logs to return

    Returns:
        List of (path, size, mtime) tuples
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    pattern = f"{name_prefix}*.log" if name_prefix else "*.log"

    logs = sorted(
        LOG_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:limit]

    result = []
    for log in logs:
        stat = log.stat()
        size_kb = stat.st_size / 1024
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        result.append((log, size_kb, mtime))

    return result


def create_progress_file(name: str) -> Path:
    """
    Create a progress tracking file.

    Args:
        name: Script/process name

    Returns:
        Path to progress file
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    progress_file = LOG_DIR / f"progress_{name}.txt"

    with open(progress_file, "w", encoding="utf-8") as f:
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Process: {name}\n")
        f.write("-" * 40 + "\n")

    return progress_file


def update_progress(progress_file: Path, message: str):
    """
    Append a progress update.

    Args:
        progress_file: Path to progress file
        message: Progress message
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


class TrainingLogger:
    """
    Convenience class for training scripts.

    Example:
        tlog = TrainingLogger("train_01_medical_dict", model="medgemma-4b")
        tlog.info("Loading data...")
        tlog.step(100, loss=0.5, accuracy=75.0)
        tlog.epoch(1, val_loss=0.4, val_accuracy=80.0)
        tlog.finish(final_accuracy=85.0)
    """

    def __init__(self, script_name: str, model: str = None, resume: bool = False):
        """
        Initialize training logger.

        Args:
            script_name: Name of training script
            model: Model name (optional, added to log filename)
            resume: If True, append to existing log
        """
        self.script_name = script_name
        self.model = model
        self.start_time = datetime.now()

        # Create log filename
        if model:
            log_name = f"{script_name}_{model}"
        else:
            log_name = script_name

        if resume:
            # Find existing log
            existing = get_latest_log(log_name)
            if existing:
                log_file = existing.name
                file_mode = "a"
            else:
                timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
                log_file = f"{log_name}_{timestamp}.log"
                file_mode = "w"
        else:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            log_file = f"{log_name}_{timestamp}.log"
            file_mode = "w"

        self.logger = setup_logger(log_name, log_file=log_file, file_mode=file_mode)
        self.log_path = LOG_DIR / log_file

        # Progress file
        self.progress_file = create_progress_file(log_name)

        # Log start
        self.logger.info("=" * 60)
        self.logger.info(f"Training: {script_name}")
        if model:
            self.logger.info(f"Model: {model}")
        self.logger.info("=" * 60)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def step(self, step: int, **metrics):
        """
        Log training step metrics.

        Args:
            step: Current step number
            **metrics: Metric name=value pairs (e.g., loss=0.5, accuracy=75.0)
        """
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        self.logger.info(f"Step {step}: {metrics_str}")
        update_progress(self.progress_file, f"Step {step}: {metrics_str}")

    def epoch(self, epoch: int, **metrics):
        """
        Log epoch metrics.

        Args:
            epoch: Current epoch number
            **metrics: Metric name=value pairs
        """
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        self.logger.info("-" * 40)
        self.logger.info(f"Epoch {epoch}: {metrics_str}")
        self.logger.info("-" * 40)
        update_progress(self.progress_file, f"Epoch {epoch} complete: {metrics_str}")

    def validation(self, **metrics):
        """
        Log validation results.

        Args:
            **metrics: Metric name=value pairs
        """
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        self.logger.info(f"Validation: {metrics_str}")
        update_progress(self.progress_file, f"Validation: {metrics_str}")

    def finish(self, **final_metrics):
        """
        Log training completion.

        Args:
            **final_metrics: Final metric values
        """
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        self.logger.info("=" * 60)
        self.logger.info("Training Complete!")
        self.logger.info(f"Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        if final_metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in final_metrics.items())
            self.logger.info(f"Final: {metrics_str}")

        self.logger.info(f"Log file: {self.log_path}")
        self.logger.info("=" * 60)

        update_progress(self.progress_file, f"FINISHED - Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")


def main():
    """Demo / test logging utilities."""
    print("=" * 60)
    print("Logging Utilities Demo")
    print("=" * 60)

    # Demo 1: Basic logger
    print("\n1. Basic logger:")
    logger = setup_logger("demo_basic")
    logger.info("This is an info message")
    logger.warning("This is a warning")

    # Demo 2: Training logger
    print("\n2. Training logger:")
    tlog = TrainingLogger("demo_training", model="test-model")
    tlog.info("Loading data...")
    tlog.step(100, loss=0.5, accuracy=75.0)
    tlog.step(200, loss=0.3, accuracy=82.0)
    tlog.epoch(1, val_loss=0.4, val_accuracy=80.0)
    tlog.validation(kormedmcqa_accuracy=78.5)
    tlog.finish(final_accuracy=85.0)

    # Demo 3: List logs
    print("\n3. Recent logs:")
    for log_path, size_kb, mtime in list_logs(limit=5):
        print(f"  {log_path.name} ({size_kb:.1f} KB) - {mtime}")

    print("\n" + "=" * 60)
    print(f"Logs saved to: {LOG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
