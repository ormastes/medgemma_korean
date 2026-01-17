#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 00: Plain Text Pre-training
Learn general Korean language on plain text corpus
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from training_utils import train_script_wrapper

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "01_refined" / "00_plain_text"
OUTPUT_DIR = BASE_DIR / "models" / "train_00_plain_text"

# Create main function using the wrapper
main = train_script_wrapper(
    script_name="train_00_plain_text",
    data_dir=DATA_DIR,
    default_output_dir=OUTPUT_DIR
)

if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
