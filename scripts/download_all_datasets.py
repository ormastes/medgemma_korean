#!/usr/bin/env python3
"""
Download all Korean medical datasets to data/raw/by_source/
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "by_source"
CONFIG_PATH = DATA_DIR / "data_config.json"

def download_all():
    """Download all datasets from config."""

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    stats = {}

    for name, ds_config in tqdm(config["datasets"].items(), desc="Downloading"):
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"  HuggingFace: {ds_config['hf_id']}")
        print(f"  Type: {ds_config['type']}")
        print(f"  Training: {ds_config['training_method']}")

        # Skip disabled datasets
        if ds_config.get('skip', False):
            print(f"  SKIPPED: {ds_config.get('skip_reason', 'disabled')}")
            stats[name] = {"status": "skipped", "reason": ds_config.get('skip_reason', 'disabled')}
            continue

        output_dir = RAW_DIR / name

        if output_dir.exists():
            print(f"  Already exists, skipping...")
            stats[name] = {"status": "exists", "path": str(output_dir)}
            continue

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Handle datasets with multiple configs
            configs = ds_config.get('configs', [None])
            all_splits = {}

            for cfg in configs:
                if cfg:
                    print(f"  Config: {cfg}")
                    ds = load_dataset(ds_config['hf_id'], cfg)
                else:
                    ds = load_dataset(ds_config['hf_id'])

                for split_name in ds.keys():
                    split_data = ds[split_name]
                    num_samples = len(split_data)

                    # For multi-config, append config name to split
                    if cfg and len(configs) > 1:
                        save_name = f"{split_name}_{cfg}"
                    else:
                        save_name = split_name

                    split_path = output_dir / save_name
                    split_data.save_to_disk(str(split_path))

                    all_splits[save_name] = num_samples
                    print(f"    {save_name}: {num_samples} samples")

            split_info = all_splits

            # Save metadata
            meta = {
                "hf_id": ds_config['hf_id'],
                "type": ds_config['type'],
                "training_method": ds_config['training_method'],
                "source_quality": ds_config['source_quality'],
                "splits": split_info,
                "fields": ds_config.get('fields', {})
            }
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            stats[name] = {"status": "downloaded", "splits": split_info}

        except Exception as e:
            print(f"  ERROR: {e}")
            stats[name] = {"status": "error", "error": str(e)}
            # Clean up empty directory on error
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()

    # Save download summary
    with open(RAW_DIR / "download_summary.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"Summary saved to: {RAW_DIR / 'download_summary.json'}")


if __name__ == "__main__":
    download_all()
