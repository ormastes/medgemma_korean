#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 01 (Medical Dictionary) and 02 (Korean Medical Test) in Loop
Purpose: Train both datasets with equal amounts iteratively

Debug version with extensive logging
"""

import argparse
import json
import torch
import subprocess
import os
import sys
import traceback
import signal
import gc
from pathlib import Path
from datetime import datetime
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from training_config import MODEL_CONFIGS

BASE_DIR = Path(__file__).parent.parent
MEDICAL_DICT_FILE = BASE_DIR / "data" / "02_refined" / "01_medical_dict.json"
CHAR_DICT_FILE = BASE_DIR / "data" / "02_refined" / "02_char_dict.json"
KOR_MED_TEST_FILE = BASE_DIR / "data" / "02_refined" / "02_kor_med_test" / "train.jsonl"

# Debug log file
LOG_FILE = BASE_DIR / "models" / "loop_training" / "debug.log"


def log(msg: str, level: str = "INFO"):
    """Write debug log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    print(log_msg)

    # Also write to file
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")


def check_gpu_status():
    """Check and log GPU status"""
    log("=" * 50, "DEBUG")
    log("GPU Status Check", "DEBUG")
    log("=" * 50, "DEBUG")

    try:
        import torch
        log(f"PyTorch version: {torch.__version__}", "DEBUG")
        log(f"CUDA available: {torch.cuda.is_available()}", "DEBUG")

        if torch.cuda.is_available():
            log(f"CUDA version: {torch.version.cuda}", "DEBUG")
            log(f"GPU count: {torch.cuda.device_count()}", "DEBUG")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)

                # Get current memory usage
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                free = total_mem - reserved

                log(f"GPU {i}: {props.name}", "DEBUG")
                log(f"  Total: {total_mem:.2f} GB", "DEBUG")
                log(f"  Allocated: {allocated:.2f} GB", "DEBUG")
                log(f"  Reserved: {reserved:.2f} GB", "DEBUG")
                log(f"  Free (approx): {free:.2f} GB", "DEBUG")
        else:
            log("CUDA NOT AVAILABLE - Training will fail!", "ERROR")
    except Exception as e:
        log(f"Error checking GPU: {e}", "ERROR")
        traceback.print_exc()


def clear_gpu_memory():
    """Clear GPU memory"""
    log("Clearing GPU memory...", "DEBUG")
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log("GPU memory cleared", "DEBUG")
    except Exception as e:
        log(f"Error clearing GPU memory: {e}", "WARNING")


def count_dict_samples() -> int:
    """Count medical dictionary + char dictionary samples"""
    count = 0
    if MEDICAL_DICT_FILE.exists():
        with open(MEDICAL_DICT_FILE, 'r') as f:
            count += len(json.load(f))
    if CHAR_DICT_FILE.exists():
        with open(CHAR_DICT_FILE, 'r') as f:
            count += len(json.load(f))
    return count


def count_mcq_samples() -> int:
    """Count MCQ training samples"""
    if not KOR_MED_TEST_FILE.exists():
        return 0
    count = 0
    with open(KOR_MED_TEST_FILE, 'r') as f:
        for _ in f:
            count += 1
    return count


def get_balanced_sample_sizes(samples_per_epoch: int = None) -> tuple:
    """Get balanced sample sizes for both datasets"""
    count1 = count_dict_samples()
    count2 = count_mcq_samples()

    log(f"Dataset 01 (Medical Dict + Char Dict): {count1} samples", "INFO")
    log(f"Dataset 02 (Kor Med Test): {count2} samples", "INFO")

    if samples_per_epoch is None:
        samples_per_epoch = min(count1, count2)

    return samples_per_epoch, samples_per_epoch, samples_per_epoch * 2


def run_training(script_name: str, model: str, base_model_path: str,
                output_dir: str, max_samples: int, epochs: int = 1,
                device: str = "cuda:0") -> bool:
    """Run a training script with extensive logging"""

    log(f"=" * 60, "INFO")
    log(f"Starting training: {script_name}", "INFO")
    log(f"=" * 60, "INFO")
    log(f"  Model: {model}", "INFO")
    log(f"  Base model: {base_model_path}", "INFO")
    log(f"  Output: {output_dir}", "INFO")
    log(f"  Samples: {max_samples}", "INFO")
    log(f"  Epochs: {epochs}", "INFO")
    log(f"  Device: {device}", "INFO")

    # Check base model exists
    base_model_abs = Path(base_model_path).resolve()
    if not base_model_abs.exists():
        # Check if it's a HuggingFace model
        if not base_model_path.startswith("google/"):
            log(f"Base model path does not exist: {base_model_abs}", "ERROR")
            return False
        else:
            log(f"Using HuggingFace model: {base_model_path}", "INFO")
            base_model_abs = base_model_path
    else:
        log(f"Base model path exists: {base_model_abs}", "DEBUG")
        # Check contents
        if Path(base_model_abs).is_dir():
            contents = list(Path(base_model_abs).iterdir())
            log(f"Base model directory contents: {[p.name for p in contents[:10]]}", "DEBUG")

    output_dir_abs = str(Path(output_dir).resolve())

    # Clear GPU memory before training
    clear_gpu_memory()
    check_gpu_status()

    cmd = [
        "python3",
        str(Path(__file__).parent / script_name),
        "--model", model,
        "--epochs", str(epochs),
        "--max-samples", str(max_samples),
        "--base-model", str(base_model_abs),
        "--output", output_dir_abs,
        "--device", device
    ]

    log(f"Command: {' '.join(cmd)}", "DEBUG")

    # Set environment variables for better debugging
    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages
    env["TORCH_USE_CUDA_DSA"] = "1"    # Device-side assertions

    try:
        log("Starting subprocess...", "DEBUG")
        start_time = datetime.now()

        result = subprocess.run(
            cmd,
            check=True,
            env=env,
            capture_output=False  # Let output go to stdout
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        log(f"Training completed in {duration:.1f} seconds", "INFO")
        log(f"Return code: {result.returncode}", "DEBUG")

        # Check output exists
        final_dir = Path(output_dir_abs) / "final"
        if final_dir.exists():
            log(f"Output saved to: {final_dir}", "INFO")
            contents = list(final_dir.iterdir())
            log(f"Output contents: {[p.name for p in contents]}", "DEBUG")
        else:
            log(f"Warning: Expected output directory not found: {final_dir}", "WARNING")

        return result.returncode == 0

    except subprocess.CalledProcessError as e:
        log(f"Training failed with return code: {e.returncode}", "ERROR")
        log(f"Command: {e.cmd}", "ERROR")
        if e.stdout:
            log(f"Stdout: {e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout}", "ERROR")
        if e.stderr:
            log(f"Stderr: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}", "ERROR")
        return False

    except Exception as e:
        log(f"Unexpected error: {type(e).__name__}: {e}", "ERROR")
        traceback.print_exc()
        return False

    finally:
        # Always clear GPU memory after training
        clear_gpu_memory()


def run_validation(model_path: str, output_file: str, max_samples: int = 100) -> dict:
    """Run validation script"""
    log(f"Running validation: {model_path}", "INFO")
    log(f"  Samples: {max_samples}", "INFO")

    cmd = [
        "python3",
        str(Path(__file__).parent / "validation_kor_med_test.py"),
        "--model", model_path,
        "--output", output_file,
        "--max-samples", str(max_samples)
    ]

    try:
        subprocess.run(cmd, check=True)

        with open(output_file, 'r') as f:
            results = json.load(f)

        log(f"Validation results: {results}", "DEBUG")
        return results

    except Exception as e:
        log(f"Validation failed: {e}", "WARNING")
        return {}


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    log(f"Received signal {signum}, shutting down gracefully...", "WARNING")
    clear_gpu_memory()
    sys.exit(1)


def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Loop training for datasets 01 and 02")
    parser.add_argument("--model", default="medgemma-27b", choices=list(MODEL_CONFIGS.keys()),
                       help="Model to train (default: medgemma-27b)")
    parser.add_argument("--base-model", default=None,
                       help="Path to base model (default: use model config path)")
    parser.add_argument("--samples-per-epoch", type=int, default=None,
                       help="Samples per dataset per epoch (default: use smaller dataset)")
    parser.add_argument("--total-epochs", type=int, default=5, help="Total training epochs")
    parser.add_argument("--output-dir", default="models/loop_training", help="Base output directory")
    parser.add_argument("--validate-every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    args = parser.parse_args()

    # Initialize logging
    log("=" * 70, "INFO")
    log("LOOP TRAINING STARTED", "INFO")
    log("=" * 70, "INFO")
    log(f"Arguments: {vars(args)}", "DEBUG")
    log(f"Python: {sys.version}", "DEBUG")
    log(f"Working directory: {os.getcwd()}", "DEBUG")

    # Check environment
    check_gpu_status()

    # Determine base model
    cfg = MODEL_CONFIGS[args.model]
    if args.base_model:
        base_model = args.base_model
        log(f"Using provided base model: {base_model}", "INFO")
    else:
        base_model = cfg['path']
        log(f"Using default model from config: {base_model}", "INFO")

    # Verify base model
    if not base_model.startswith("google/"):
        base_path = Path(base_model)
        if not base_path.exists():
            log(f"ERROR: Base model path does not exist: {base_path}", "ERROR")
            return 1

    log(f"Model configuration: {cfg}", "DEBUG")

    # Get balanced sample sizes
    samples_01, samples_02, total_per_epoch = get_balanced_sample_sizes(
        args.samples_per_epoch
    )

    log(f"Training configuration:", "INFO")
    log(f"  Model: {args.model}", "INFO")
    log(f"  Base model: {base_model}", "INFO")
    log(f"  Samples per epoch (each dataset): {samples_01}", "INFO")
    log(f"  Total samples per epoch: {total_per_epoch}", "INFO")
    log(f"  Total epochs: {args.total_epochs}", "INFO")
    log(f"  Total training samples: {total_per_epoch * args.total_epochs}", "INFO")
    log(f"  Device: {args.device}", "INFO")

    # Setup directories
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    current_model = base_model
    best_accuracy = 0.0
    best_epoch = 0

    results_log = []

    # Training loop
    for epoch in range(1, args.total_epochs + 1):
        log(f"", "INFO")
        log(f"{'#'*70}", "INFO")
        log(f"# EPOCH {epoch}/{args.total_epochs}", "INFO")
        log(f"{'#'*70}", "INFO")

        epoch_dir = output_base / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)

        # ========== Train 01: Medical Dictionary ==========
        log(f"", "INFO")
        log(f"--- Training 01: Medical Dictionary ({samples_01} samples) ---", "INFO")
        output_01 = epoch_dir / "after_01"

        success = run_training(
            "train_01_medical_dict.py",
            args.model,
            current_model,
            str(output_01),
            samples_01,
            epochs=1,
            device=args.device
        )

        if not success:
            log(f"Training 01 FAILED at epoch {epoch}", "ERROR")
            log(f"Check the output above for errors", "ERROR")
            break

        model_01_final = str(output_01 / "final")

        # Verify output exists
        if not Path(model_01_final).exists():
            log(f"ERROR: Expected model output not found: {model_01_final}", "ERROR")
            break

        log(f"Training 01 completed. Model at: {model_01_final}", "INFO")

        # ========== Train 02: Korean Medical Test ==========
        log(f"", "INFO")
        log(f"--- Training 02: Korean Medical Test ({samples_02} samples) ---", "INFO")
        output_02 = epoch_dir / "after_02"

        success = run_training(
            "train_02_kor_med_test.py",
            args.model,
            model_01_final,
            str(output_02),
            samples_02,
            epochs=1,
            device=args.device
        )

        if not success:
            log(f"Training 02 FAILED at epoch {epoch}", "ERROR")
            log(f"Check the output above for errors", "ERROR")
            break

        current_model = str(output_02 / "final")

        # Verify output exists
        if not Path(current_model).exists():
            log(f"ERROR: Expected model output not found: {current_model}", "ERROR")
            break

        log(f"Training 02 completed. Model at: {current_model}", "INFO")

        # ========== Validation ==========
        if epoch % args.validate_every == 0:
            log(f"", "INFO")
            log(f"--- Validation after epoch {epoch} ---", "INFO")
            val_output = output_base / f"validation_epoch_{epoch}.json"
            val_results = run_validation(current_model, str(val_output))

            if val_results:
                accuracy = val_results.get("accuracy", 0.0)
                log(f"Validation accuracy: {accuracy:.2f}%", "INFO")

                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch

                    # Save best checkpoint
                    import shutil
                    best_dir = output_base / "best_checkpoint"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    shutil.copytree(current_model, best_dir)
                    log(f"New best model saved (accuracy: {accuracy:.2f}%)", "INFO")

                results_log.append({
                    "epoch": epoch,
                    "accuracy": accuracy,
                    "model_path": current_model
                })

        log(f"", "INFO")
        log(f"Epoch {epoch} completed. Model saved to: {current_model}", "INFO")

        # Clear memory between epochs
        clear_gpu_memory()

    # Final summary
    log(f"", "INFO")
    log("=" * 70, "INFO")
    log("LOOP TRAINING COMPLETE", "INFO")
    log("=" * 70, "INFO")
    log(f"Best accuracy: {best_accuracy:.2f}% (epoch {best_epoch})", "INFO")
    log(f"Best model: {output_base / 'best_checkpoint'}", "INFO")
    log(f"Final model: {current_model}", "INFO")

    # Save training log
    log_file = output_base / "training_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            "model": args.model,
            "base_model": base_model,
            "total_epochs": args.total_epochs,
            "samples_per_epoch": total_per_epoch,
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "results": results_log
        }, f, indent=2)

    log(f"Training log saved to: {log_file}", "INFO")
    log(f"Debug log saved to: {LOG_FILE}", "INFO")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code if exit_code else 0)
    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}", "ERROR")
        traceback.print_exc()
        exit(1)
