#!/usr/bin/env python3
"""
Organize division-added data into division-specific folders
Creates: data/division_added/{division_id}/train.jsonl and validation.jsonl
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from tqdm import tqdm


def organize_by_division(source_dir: str = 'data/division_added', 
                         output_base: str = 'data/division_added',
                         min_samples: int = 10):
    """
    Organize data by primary division into separate folders
    
    Args:
        source_dir: Source directory with type folders
        output_base: Base output directory for division folders
        min_samples: Minimum samples required to create division folder
    """
    
    print(f"Organizing data from: {source_dir}")
    print(f"Output to: {output_base}")
    print(f"Minimum samples per division: {min_samples}")
    
    # Collect all data by division
    division_data = defaultdict(lambda: {'train': [], 'validation': []})
    division_counts = Counter()
    
    types = ['type1_text', 'type2_text_reasoning', 'type3_word', 'type4_word_reasoning']
    
    for dtype in types:
        for split in ['train', 'validation']:
            input_file = os.path.join(source_dir, dtype, f'{split}.jsonl')
            
            if not os.path.exists(input_file):
                print(f"Skipping {input_file} (not found)")
                continue
            
            print(f"\nReading {input_file}...")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Processing {dtype}/{split}"):
                    if line.strip():
                        sample = json.loads(line)
                        
                        # Get primary division
                        primary_div = sample.get('primary_division', 'UNKNOWN')
                        
                        # Skip if UNKNOWN or error
                        if primary_div == 'UNKNOWN':
                            continue
                        
                        # Add to division data
                        division_data[primary_div][split].append(sample)
                        division_counts[primary_div] += 1
    
    print(f"\n{'='*60}")
    print("Division Statistics")
    print(f"{'='*60}")
    print(f"{'Division':<20} {'Train':>10} {'Val':>10} {'Total':>10}")
    print(f"{'-'*60}")
    
    # Filter divisions by minimum samples
    divisions_to_save = []
    for div_id in sorted(division_data.keys()):
        train_count = len(division_data[div_id]['train'])
        val_count = len(division_data[div_id]['validation'])
        total = train_count + val_count
        
        if total >= min_samples:
            divisions_to_save.append(div_id)
            print(f"{div_id:<20} {train_count:>10} {val_count:>10} {total:>10}")
        else:
            print(f"{div_id:<20} {train_count:>10} {val_count:>10} {total:>10} (skipped)")
    
    print(f"{'-'*60}")
    print(f"Total divisions: {len(division_data)}")
    print(f"Divisions to save: {len(divisions_to_save)}")
    
    # Save division-specific data
    print(f"\n{'='*60}")
    print("Saving division-specific data...")
    print(f"{'='*60}")
    
    for div_id in tqdm(divisions_to_save, desc="Saving divisions"):
        div_dir = os.path.join(output_base, div_id)
        os.makedirs(div_dir, exist_ok=True)
        
        # Save train
        if division_data[div_id]['train']:
            train_file = os.path.join(div_dir, 'train.jsonl')
            with open(train_file, 'w', encoding='utf-8') as f:
                for sample in division_data[div_id]['train']:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Save validation
        if division_data[div_id]['validation']:
            val_file = os.path.join(div_dir, 'validation.jsonl')
            with open(val_file, 'w', encoding='utf-8') as f:
                for sample in division_data[div_id]['validation']:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Save metadata
        metadata = {
            'division_id': div_id,
            'train_samples': len(division_data[div_id]['train']),
            'validation_samples': len(division_data[div_id]['validation']),
            'total_samples': len(division_data[div_id]['train']) + len(division_data[div_id]['validation'])
        }
        
        metadata_file = os.path.join(div_dir, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted!")
    print(f"Division folders created in: {output_base}")
    
    # Generate division index
    index_file = os.path.join(output_base, 'division_index.json')
    division_index = {}
    
    for div_id in divisions_to_save:
        division_index[div_id] = {
            'train_samples': len(division_data[div_id]['train']),
            'validation_samples': len(division_data[div_id]['validation']),
            'total_samples': len(division_data[div_id]['train']) + len(division_data[div_id]['validation']),
            'path': os.path.join(output_base, div_id)
        }
    
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(division_index, f, indent=2, ensure_ascii=False)
    
    print(f"Division index saved to: {index_file}")
    
    return division_index


def main():
    parser = argparse.ArgumentParser(description="Organize data by division")
    parser.add_argument('--source', type=str, default='data/division_added',
                       help='Source directory')
    parser.add_argument('--output', type=str, default='data/division_added',
                       help='Output base directory')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum samples per division')
    
    args = parser.parse_args()
    
    organize_by_division(
        source_dir=args.source,
        output_base=args.output,
        min_samples=args.min_samples
    )


if __name__ == "__main__":
    main()
