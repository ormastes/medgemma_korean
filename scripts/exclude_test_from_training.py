#!/usr/bin/env python3
"""
Remove KorMedMCQA test samples from training data
Ensures no test contamination in division_added data
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Set

def load_test_questions(test_file: str = 'data/kormedmcqa_test/test_questions.txt') -> Set[str]:
    """Load test question texts for exclusion"""
    
    if not os.path.exists(test_file):
        print(f"Warning: Test questions file not found: {test_file}")
        print("Run: python3 scripts/extract_kormedmcqa_test.py first")
        return set()
    
    test_questions = set()
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_questions.add(line.strip())
    
    print(f"Loaded {len(test_questions)} test questions for exclusion")
    return test_questions

def extract_question_from_prompt(prompt: str) -> str:
    """Extract question text from prompt"""
    # Remove special tokens
    lines = prompt.split('\n')
    question_lines = []
    capture = False
    
    for line in lines:
        if '<|im_start|>user' in line:
            capture = True
            continue
        if '<|im_end|>' in line and capture:
            break
        if capture:
            question_lines.append(line)
    
    return '\n'.join(question_lines).strip()

def remove_test_samples(input_file: str, output_file: str, test_questions: Set[str]) -> dict:
    """Remove test samples from training data"""
    
    print(f"\nProcessing: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"  File not found, skipping")
        return {'total': 0, 'removed': 0, 'kept': 0}
    
    stats = {
        'total': 0,
        'removed': 0,
        'kept': 0,
        'removed_samples': []
    }
    
    kept_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Checking"):
            if not line.strip():
                continue
            
            stats['total'] += 1
            sample = json.loads(line)
            
            # Extract question
            prompt = sample.get('prompt', '')
            question = extract_question_from_prompt(prompt)
            
            # Check if in test set
            is_test = question in test_questions
            
            if is_test:
                stats['removed'] += 1
                stats['removed_samples'].append(question[:100])
            else:
                stats['kept'] += 1
                kept_samples.append(sample)
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in kept_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  Total: {stats['total']}")
    print(f"  Removed: {stats['removed']} test samples")
    print(f"  Kept: {stats['kept']}")
    print(f"  Output: {output_file}")
    
    return stats

def clean_division_added_data(test_questions: Set[str], 
                              source_dir: str = 'data/division_added',
                              output_dir: str = 'data/division_added_clean'):
    """Remove test samples from all division_added data"""
    
    print("="*70)
    print("Cleaning Division-Added Data (Test Exclusion)")
    print("="*70)
    
    types = ['type1_text', 'type2_text_reasoning', 'type3_word', 'type4_word_reasoning']
    
    all_stats = {}
    
    # Clean type folders
    for dtype in types:
        print(f"\n{dtype}:")
        
        for split in ['train', 'validation']:
            input_file = os.path.join(source_dir, dtype, f'{split}.jsonl')
            output_file = os.path.join(output_dir, dtype, f'{split}.jsonl')
            
            stats = remove_test_samples(input_file, output_file, test_questions)
            all_stats[f"{dtype}/{split}"] = stats
    
    # Clean division folders
    print(f"\nCleaning division-specific folders...")
    division_dirs = [d for d in Path(source_dir).iterdir() 
                    if d.is_dir() and d.name.isdigit()]
    
    for div_dir in division_dirs:
        div_id = div_dir.name
        print(f"\nDivision {div_id}:")
        
        for split in ['train', 'validation']:
            input_file = div_dir / f'{split}.jsonl'
            output_file = Path(output_dir) / div_id / f'{split}.jsonl'
            
            if input_file.exists():
                stats = remove_test_samples(str(input_file), str(output_file), test_questions)
                all_stats[f"division_{div_id}/{split}"] = stats
    
    # Summary
    print("\n" + "="*70)
    print("CLEANING SUMMARY")
    print("="*70)
    
    total_removed = sum(s['removed'] for s in all_stats.values())
    total_kept = sum(s['kept'] for s in all_stats.values())
    total_all = sum(s['total'] for s in all_stats.values())
    
    print(f"\nTotal samples processed: {total_all}")
    print(f"Test samples removed: {total_removed}")
    print(f"Samples kept: {total_kept}")
    print(f"Removal rate: {total_removed/total_all*100:.2f}%")
    
    print(f"\nCleaned data saved to: {output_dir}")
    print("="*70)
    
    # Save stats
    stats_file = os.path.join(output_dir, 'test_exclusion_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': total_all,
                'removed': total_removed,
                'kept': total_kept,
                'removal_rate': total_removed/total_all if total_all > 0 else 0
            },
            'details': all_stats
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nStats saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Remove test samples from training data")
    parser.add_argument('--test-file', type=str, 
                       default='data/kormedmcqa_test/test_questions.txt',
                       help='Test questions file')
    parser.add_argument('--source', type=str, default='data/division_added',
                       help='Source directory')
    parser.add_argument('--output', type=str, default='data/division_added_clean',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load test questions
    test_questions = load_test_questions(args.test_file)
    
    if not test_questions:
        print("\nNo test questions loaded. Exiting.")
        print("Run: python3 scripts/extract_kormedmcqa_test.py first")
        return
    
    # Clean data
    clean_division_added_data(test_questions, args.source, args.output)

if __name__ == "__main__":
    main()
