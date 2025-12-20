#!/usr/bin/env python3
"""
Download Additional Korean Medical Datasets from HuggingFace

This script downloads Korean medical datasets from the ChuGyouk collection
and other sources to expand the training data.
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset, DatasetDict
import json

# Output directory
OUTPUT_DIR = Path("data/raw/korean_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Downloading Additional Korean Medical Datasets")
print("=" * 60)

# Datasets to download from ChuGyouk collection
DATASETS_TO_DOWNLOAD = [
    # Medical QA and instruction datasets
    ("ChuGyouk/KorMedConceptsQA", None, "Korean medical concepts QA - 73.2k samples"),
    ("ChuGyouk/Asan-AMC-Healthinfo", None, "Asan Medical Center health info - 19.2k samples"),
    ("ChuGyouk/HealthSearchQA-ko", None, "Health search QA Korean - 3.17k samples"),
    ("ChuGyouk/AI_healthcare_QA", None, "AI healthcare QA - 12.1k samples"),
    ("ChuGyouk/MedQA", None, "MedQA translated - 22.9k samples"),
    ("ChuGyouk/MedQA-Evol-Korean", None, "Evolved medical QA Korean - 51.8k samples"),
    ("ChuGyouk/KoMedInstruct-52k", None, "Korean medical instruction - 52k samples"),
    ("ChuGyouk/GenMedGPT-5k-ko", None, "GenMedGPT Korean - 5.45k samples"),
    ("ChuGyouk/ChainofDiagnosis-Ko", None, "Chain of diagnosis Korean - 39.1k samples"),
    ("ChuGyouk/medical_questions_pairs_ko", None, "Medical question pairs Korean - 3.05k samples"),
    ("ChuGyouk/PubMedQA-test-Ko", None, "PubMedQA test Korean - 500 samples"),
    ("ChuGyouk/MedExpQA-Kor", None, "Medical explanatory QA Korean - 497 samples"),

    # Additional medical reasoning
    ("ChuGyouk/medical-o1-reasoning-SFT-Ko", None, "Medical O1 reasoning SFT Korean - 25.7k samples"),

    # SNUH datasets
    ("snuh/KorMedLawQA", None, "Korean Medical Law QA - 13k samples"),
]

downloaded = []
failed = []

for dataset_name, config, description in DATASETS_TO_DOWNLOAD:
    # Create safe directory name
    safe_name = dataset_name.replace("/", "_").replace("-", "_").lower()
    output_path = OUTPUT_DIR / safe_name

    if output_path.exists():
        print(f"\n[SKIP] {dataset_name} already exists at {output_path}")
        downloaded.append((dataset_name, str(output_path), "already_exists"))
        continue

    print(f"\n[DOWNLOADING] {dataset_name}")
    print(f"  Description: {description}")

    try:
        if config:
            ds = load_dataset(dataset_name, config, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, trust_remote_code=True)

        # Save to disk
        ds.save_to_disk(str(output_path))

        # Count samples
        if isinstance(ds, DatasetDict):
            total = sum(len(ds[split]) for split in ds)
        else:
            total = len(ds)

        print(f"  [OK] Downloaded {total} samples to {output_path}")
        downloaded.append((dataset_name, str(output_path), total))

    except Exception as e:
        print(f"  [ERROR] Failed to download: {e}")
        failed.append((dataset_name, str(e)))

# Summary
print("\n" + "=" * 60)
print("Download Summary")
print("=" * 60)

print(f"\nSuccessfully downloaded/found: {len(downloaded)}")
for name, path, count in downloaded:
    print(f"  - {name}: {count}")

if failed:
    print(f"\nFailed: {len(failed)}")
    for name, error in failed:
        print(f"  - {name}: {error[:50]}...")

# Save summary
summary = {
    "downloaded": downloaded,
    "failed": failed,
}

with open(OUTPUT_DIR / "download_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\nSummary saved to {OUTPUT_DIR / 'download_summary.json'}")
