#!/usr/bin/env python3
"""
Download and organize Korean medical datasets by source and type.

Creates organized directory structure:
- data/by_source/{source_name}/
- data/by_type/{mcq|instruction|qa|reasoning}/
"""

import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Dataset configurations
DATASETS = {
    # MCQ Datasets
    "kormedmcqa": {
        "hf_id": "sean0042/KorMedMCQA",
        "type": "mcq",
        "source": "original_korean",
        "description": "Korean healthcare licensing exam MCQ"
    },
    "medqa_korean": {
        "hf_id": "ChuGyouk/MedQA",
        "type": "mcq",
        "source": "translation",
        "description": "MedQA translated to Korean"
    },
    "medqa_evol_korean": {
        "hf_id": "ChuGyouk/MedQA-Evol-Korean",
        "type": "mcq",
        "source": "translation",
        "description": "Evolved MedQA in Korean"
    },
    "kormedlawqa": {
        "hf_id": "snuh/KorMedLawQA",
        "type": "mcq",
        "source": "synthetic",
        "description": "Korean medical law MCQ (GPT-4o-mini)"
    },
    "medexpqa_korean": {
        "hf_id": "ChuGyouk/MedExpQA-Kor",
        "type": "mcq",
        "source": "translation",
        "description": "Medical explanation QA"
    },

    # Instruction Datasets
    "komedinstruct_52k": {
        "hf_id": "ChuGyouk/KoMedInstruct-52k",
        "type": "instruction",
        "source": "translation",
        "description": "Korean medical instruction 52K"
    },
    "genmedgpt_5k_ko": {
        "hf_id": "ChuGyouk/GenMedGPT-5k-ko",
        "type": "instruction",
        "source": "translation",
        "description": "General medical GPT Korean"
    },

    # QA Datasets
    "healthsearchqa_ko": {
        "hf_id": "ChuGyouk/HealthSearchQA-ko",
        "type": "qa",
        "source": "synthetic",
        "description": "Health search QA (GPT-4o)"
    },
    "ai_healthcare_qa": {
        "hf_id": "ChuGyouk/AI_healthcare_QA",
        "type": "qa",
        "source": "synthetic",
        "description": "AI healthcare QA (GPT-4o)"
    },
    "asan_amc_healthinfo": {
        "hf_id": "ChuGyouk/Asan-AMC-Healthinfo",
        "type": "qa",
        "source": "original_korean",
        "description": "Asan Medical Center health info"
    },
    "kormedconceptsqa": {
        "hf_id": "ChuGyouk/KorMedConceptsQA",
        "type": "qa",
        "source": "curated",
        "description": "Korean medical concepts QA"
    },
    "pubmedqa_test_ko": {
        "hf_id": "ChuGyouk/PubMedQA-test-Ko",
        "type": "qa",
        "source": "translation",
        "description": "PubMedQA test Korean"
    },

    # Reasoning Datasets
    "medical_o1_reasoning_ko": {
        "hf_id": "ChuGyouk/medical-o1-reasoning-SFT-Ko",
        "type": "reasoning",
        "source": "synthetic",
        "description": "Medical o1 reasoning Korean"
    },
    "medical_reasoning_kormedmcqa": {
        "hf_id": "ChuGyouk/medical-reasoning-train-kormedmcqa",
        "type": "reasoning",
        "source": "synthetic",
        "description": "Medical reasoning (Gemini)"
    },
    "chainofdiagnosis_ko": {
        "hf_id": "ChuGyouk/ChainofDiagnosis-Ko",
        "type": "reasoning",
        "source": "translation",
        "description": "Chain of diagnosis Korean"
    },
}


def download_and_organize():
    """Download all datasets and organize by source and type."""

    # Create directories
    by_source_dir = DATA_DIR / "by_source"
    by_type_dir = DATA_DIR / "by_type"

    by_source_dir.mkdir(parents=True, exist_ok=True)
    by_type_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    stats = {
        "by_source": {},
        "by_type": {},
        "datasets": {}
    }

    for name, config in tqdm(DATASETS.items(), desc="Processing datasets"):
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"HuggingFace: {config['hf_id']}")
        print(f"Type: {config['type']}, Source: {config['source']}")

        try:
            # Download dataset
            ds = load_dataset(config['hf_id'])

            # Get all splits
            for split_name in ds.keys():
                split_data = ds[split_name]
                num_samples = len(split_data)

                print(f"  {split_name}: {num_samples} samples")

                # Save by source
                source_path = by_source_dir / name / split_name
                source_path.mkdir(parents=True, exist_ok=True)
                split_data.save_to_disk(str(source_path))

                # Track stats
                if name not in stats["datasets"]:
                    stats["datasets"][name] = {
                        "hf_id": config["hf_id"],
                        "type": config["type"],
                        "source": config["source"],
                        "description": config["description"],
                        "splits": {}
                    }
                stats["datasets"][name]["splits"][split_name] = num_samples

            # Update source stats
            source_type = config["source"]
            if source_type not in stats["by_source"]:
                stats["by_source"][source_type] = []
            stats["by_source"][source_type].append(name)

            # Update type stats
            data_type = config["type"]
            if data_type not in stats["by_type"]:
                stats["by_type"][data_type] = []
            stats["by_type"][data_type].append(name)

            print(f"  Saved to: {by_source_dir / name}")

        except Exception as e:
            print(f"  ERROR: {e}")
            stats["datasets"][name] = {"error": str(e)}

    # Create type-combined datasets
    print(f"\n{'='*60}")
    print("Creating combined datasets by type...")

    for data_type in ["mcq", "instruction", "qa", "reasoning"]:
        type_datasets = stats["by_type"].get(data_type, [])
        if not type_datasets:
            continue

        print(f"\nCombining {data_type} datasets: {type_datasets}")

        combined_data = []
        for ds_name in type_datasets:
            source_path = by_source_dir / ds_name
            if source_path.exists():
                for split_dir in source_path.iterdir():
                    if split_dir.is_dir():
                        try:
                            from datasets import load_from_disk
                            split_ds = load_from_disk(str(split_dir))
                            # Add source info
                            for item in split_ds:
                                item_dict = dict(item)
                                item_dict["_source"] = ds_name
                                item_dict["_type"] = data_type
                                combined_data.append(item_dict)
                        except Exception as e:
                            print(f"  Warning: Could not load {split_dir}: {e}")

        if combined_data:
            # Save combined dataset
            type_path = by_type_dir / data_type
            type_path.mkdir(parents=True, exist_ok=True)

            # Note: Can't easily combine different schemas, so just save metadata
            with open(type_path / "combined_info.json", "w") as f:
                json.dump({
                    "type": data_type,
                    "sources": type_datasets,
                    "total_samples": len(combined_data)
                }, f, indent=2, ensure_ascii=False)

            print(f"  {data_type}: {len(combined_data)} total samples from {len(type_datasets)} sources")

    # Save summary
    summary_path = DATA_DIR / "organization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Summary saved to:", summary_path)
    print("\nDatasets by Source:")
    for source, datasets in stats["by_source"].items():
        print(f"  {source}: {len(datasets)} datasets")
    print("\nDatasets by Type:")
    for dtype, datasets in stats["by_type"].items():
        print(f"  {dtype}: {len(datasets)} datasets")


def create_symlinks_by_type():
    """Create symlinks to organize data by type (alternative to copying)."""

    by_source_dir = DATA_DIR / "by_source"
    by_type_dir = DATA_DIR / "by_type"

    for name, config in DATASETS.items():
        source_path = by_source_dir / name
        if not source_path.exists():
            continue

        data_type = config["type"]
        type_path = by_type_dir / data_type / name

        type_path.parent.mkdir(parents=True, exist_ok=True)

        if not type_path.exists():
            try:
                type_path.symlink_to(source_path.resolve())
                print(f"Created symlink: {type_path} -> {source_path}")
            except Exception as e:
                print(f"Could not create symlink for {name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organize Korean medical datasets")
    parser.add_argument("--download", action="store_true", help="Download and organize datasets")
    parser.add_argument("--symlinks", action="store_true", help="Create symlinks by type")
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("Available datasets:\n")
        for name, config in DATASETS.items():
            print(f"{name}:")
            print(f"  HuggingFace: {config['hf_id']}")
            print(f"  Type: {config['type']}")
            print(f"  Source: {config['source']}")
            print(f"  Description: {config['description']}")
            print()
    elif args.download:
        download_and_organize()
    elif args.symlinks:
        create_symlinks_by_type()
    else:
        parser.print_help()
