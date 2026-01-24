#!/usr/bin/env python3
"""
Update all training scripts to use dataset caching
"""
import re
from pathlib import Path

TRAIN_DIR = Path(__file__).parent / "train"

# Files to update
train_scripts = [
    "train_01_medical_dict.py",
    "train_02_kor_med_test.py",
    "train_01_with_00_monitor.py",
    "train_01_02_loop.py",
]

for script_name in train_scripts:
    script_path = TRAIN_DIR / script_name
    if not script_path.exists():
        print(f"⊘ Skipping {script_name} - not found")
        continue

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already updated
    if 'load_or_create_cached_dataset' in content:
        print(f"✓ {script_name} - already updated")
        continue

    # Update imports
    if 'load_jsonl_data, setup_model_with_lora' in content:
        content = content.replace(
            'load_jsonl_data, setup_model_with_lora',
            'load_jsonl_data, load_or_create_cached_dataset, setup_model_with_lora'
        )
        print(f"✓ {script_name} - updated imports")
    else:
        print(f"⊘ {script_name} - could not find import pattern")
        continue

    # Find and replace load_jsonl_data calls with load_or_create_cached_dataset
    # This is more complex since we need to add tokenizer loading first

    # Find the section where load_jsonl_data is called
    match = re.search(
        r'(\s+)# Load training data\n\s+train_data, val_data = load_jsonl_data\(DATA_DIR.*?\)',
        content,
        re.DOTALL
    )

    if match:
        indent = match.group(1)
        old_section = match.group(0)

        # New section with tokenizer first
        new_section = f"""{indent}# Load tokenizer FIRST (needed for dataset caching)
{indent}try:
{indent}    if args.tokenizer_path:
{indent}        from transformers import AutoTokenizer
{indent}        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
{indent}    else:
{indent}        tokenizer = load_tokenizer(model_path)
{indent}except:
{indent}    from transformers import AutoTokenizer
{indent}    tokenizer = AutoTokenizer.from_pretrained(cfg.get('path', model_path), trust_remote_code=True)
{indent}
{indent}# Load training data (with caching)
{indent}train_data, val_data = load_or_create_cached_dataset(
{indent}    DATA_DIR, tokenizer, max_samples=args.max_samples, skip_cache=False
{indent})"""

        content = content.replace(old_section, new_section)
        print(f"✓ {script_name} - replaced data loading")
    else:
        print(f"⊘ {script_name} - could not find data loading pattern")
        continue

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ {script_name} - saved")

print("\nDone!")
