#!/usr/bin/env python3
"""
Extract KorMedMCQA test set for validation and exclusion from training
Creates: data/kormedmcqa_test/ with test samples
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
import argparse

def extract_kormedmcqa_test(output_dir: str = 'data/kormedmcqa_test'):
    """Extract KorMedMCQA test set"""
    
    print("="*70)
    print("Extracting KorMedMCQA Test Set")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load from HuggingFace
    print("\nLoading KorMedMCQA from HuggingFace...")
    
    try:
        # Try loading the dataset
        dataset = load_dataset("sean0042/KorMedMCQA")
        
        # Extract test splits
        test_splits = ['test_doctor', 'test_nurse', 'test_pharm', 'test_dentist']
        
        all_test_samples = []
        test_questions = set()
        
        for split in test_splits:
            if split in dataset:
                print(f"\n{split}:")
                split_data = dataset[split]
                print(f"  Samples: {len(split_data)}")
                
                # Save split-specific file
                split_file = os.path.join(output_dir, f'{split}.jsonl')
                with open(split_file, 'w', encoding='utf-8') as f:
                    for sample in split_data:
                        # Extract question text for duplicate detection
                        question_text = sample.get('question', '')
                        test_questions.add(question_text)
                        
                        # Save sample
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        all_test_samples.append(sample)
                
                print(f"  Saved to: {split_file}")
        
        # Save combined test set
        combined_file = os.path.join(output_dir, 'all_test.jsonl')
        with open(combined_file, 'w', encoding='utf-8') as f:
            for sample in all_test_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n{'='*70}")
        print(f"Total test samples: {len(all_test_samples)}")
        print(f"Unique questions: {len(test_questions)}")
        print(f"Combined file: {combined_file}")
        print(f"{'='*70}")
        
        # Save test question IDs for exclusion
        exclusion_file = os.path.join(output_dir, 'test_questions.txt')
        with open(exclusion_file, 'w', encoding='utf-8') as f:
            for q in test_questions:
                f.write(q + '\n')
        
        print(f"\nTest questions saved to: {exclusion_file}")
        print("Use this file to exclude test samples from training data")
        
        return test_questions, all_test_samples
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nTrying local files...")
        
        # Try local files
        local_dir = 'data/raw/by_source/kormedmcqa'
        if os.path.exists(local_dir):
            return extract_from_local(local_dir, output_dir)
        else:
            print(f"Local directory not found: {local_dir}")
            return set(), []

def extract_from_local(local_dir: str, output_dir: str):
    """Extract from local arrow files"""
    
    import pyarrow.parquet as pq
    
    test_splits = ['test_doctor', 'test_nurse', 'test_pharm', 'test_dentist']
    all_test_samples = []
    test_questions = set()
    
    for split in test_splits:
        split_dir = os.path.join(local_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        # Find arrow/parquet files
        data_files = list(Path(split_dir).glob('*.arrow')) + \
                     list(Path(split_dir).glob('*.parquet'))
        
        if not data_files:
            print(f"No data files found in {split_dir}")
            continue
        
        print(f"\n{split}:")
        split_samples = []
        
        for data_file in data_files:
            table = pq.read_table(data_file)
            data = table.to_pylist()
            split_samples.extend(data)
        
        print(f"  Samples: {len(split_samples)}")
        
        # Save
        split_file = os.path.join(output_dir, f'{split}.jsonl')
        with open(split_file, 'w', encoding='utf-8') as f:
            for sample in split_samples:
                question_text = sample.get('question', '')
                test_questions.add(question_text)
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                all_test_samples.append(sample)
        
        print(f"  Saved to: {split_file}")
    
    # Save combined
    combined_file = os.path.join(output_dir, 'all_test.jsonl')
    with open(combined_file, 'w', encoding='utf-8') as f:
        for sample in all_test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nTotal test samples: {len(all_test_samples)}")
    print(f"Combined file: {combined_file}")
    
    return test_questions, all_test_samples

def main():
    parser = argparse.ArgumentParser(description="Extract KorMedMCQA test set")
    parser.add_argument('--output', type=str, default='data/kormedmcqa_test',
                       help='Output directory')
    
    args = parser.parse_args()
    
    extract_kormedmcqa_test(args.output)

if __name__ == "__main__":
    main()
