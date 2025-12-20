#!/usr/bin/env python3
"""
Korean Proficiency Validation Script
Validates model's Korean language quality in medical context
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter
import argparse

class KoreanProficiencyChecker:
    def __init__(self):
        self.korean_char_range = ('\uac00', '\ud7a3')  # Hangul syllables
        self.korean_jamo_range = ('\u1100', '\u11ff')  # Hangul Jamo
        
    def is_korean(self, text: str) -> bool:
        """Check if text contains Korean characters"""
        if not text:
            return False
        
        korean_chars = sum(1 for c in text 
                          if '\uac00' <= c <= '\ud7a3' or '\u1100' <= c <= '\u11ff')
        total_chars = len([c for c in text if c.strip()])
        
        return korean_chars / max(total_chars, 1) > 0.3  # >30% Korean
    
    def check_korean_quality(self, text: str) -> Dict:
        """Check Korean text quality metrics"""
        
        metrics = {
            'has_korean': self.is_korean(text),
            'korean_ratio': 0.0,
            'has_english': bool(re.search(r'[a-zA-Z]{3,}', text)),
            'has_medical_terms': False,
            'char_count': len(text),
            'word_count': len(text.split()),
            'issues': []
        }
        
        if not text:
            metrics['issues'].append('Empty text')
            return metrics
        
        # Calculate Korean ratio
        korean_chars = sum(1 for c in text 
                          if '\uac00' <= c <= '\ud7a3' or '\u1100' <= c <= '\u11ff')
        total_chars = len([c for c in text if c.strip()])
        metrics['korean_ratio'] = korean_chars / max(total_chars, 1)
        
        # Check for common medical terms in Korean
        medical_terms = [
            '환자', '진단', '치료', '증상', '질병', '약물', '검사',
            '수술', '병원', '의사', '간호', '처방', '투여', '감염'
        ]
        metrics['has_medical_terms'] = any(term in text for term in medical_terms)
        
        # Check for issues
        if metrics['korean_ratio'] < 0.5:
            metrics['issues'].append('Low Korean ratio')
        
        if not metrics['has_medical_terms'] and len(text) > 50:
            metrics['issues'].append('No medical terms found')
        
        return metrics
    
    def validate_dataset(self, file_path: str, sample_size: int = 100) -> Dict:
        """Validate Korean quality in a dataset"""
        
        print(f"\nValidating: {file_path}")
        
        if not os.path.exists(file_path):
            return {'error': 'File not found'}
        
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                if line.strip():
                    samples.append(json.loads(line))
        
        results = {
            'total_samples': len(samples),
            'korean_samples': 0,
            'non_korean_samples': 0,
            'korean_ratio_avg': 0.0,
            'with_medical_terms': 0,
            'issues_count': Counter(),
            'sample_checks': []
        }
        
        korean_ratios = []
        
        for sample in samples:
            # Check completion text
            text = sample.get('completion', '') or sample.get('answer', '') or sample.get('text', '')
            
            metrics = self.check_korean_quality(text)
            
            if metrics['has_korean']:
                results['korean_samples'] += 1
                korean_ratios.append(metrics['korean_ratio'])
            else:
                results['non_korean_samples'] += 1
            
            if metrics['has_medical_terms']:
                results['with_medical_terms'] += 1
            
            for issue in metrics['issues']:
                results['issues_count'][issue] += 1
            
            results['sample_checks'].append({
                'text_preview': text[:100],
                'metrics': metrics
            })
        
        if korean_ratios:
            results['korean_ratio_avg'] = sum(korean_ratios) / len(korean_ratios)
        
        return results
    
    def print_report(self, results: Dict, file_path: str):
        """Print validation report"""
        
        print(f"\n{'='*70}")
        print(f"Korean Proficiency Report: {Path(file_path).name}")
        print(f"{'='*70}")
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        total = results['total_samples']
        korean = results['korean_samples']
        non_korean = results['non_korean_samples']
        
        print(f"\nSample Statistics:")
        print(f"  Total samples checked: {total}")
        print(f"  Korean samples: {korean} ({korean/total*100:.1f}%)")
        print(f"  Non-Korean samples: {non_korean} ({non_korean/total*100:.1f}%)")
        print(f"  Average Korean ratio: {results['korean_ratio_avg']*100:.1f}%")
        print(f"  With medical terms: {results['with_medical_terms']} ({results['with_medical_terms']/total*100:.1f}%)")
        
        if results['issues_count']:
            print(f"\nIssues Found:")
            for issue, count in results['issues_count'].most_common():
                print(f"  {issue}: {count} samples")
        
        # Sample previews
        print(f"\nSample Previews (first 3):")
        for i, check in enumerate(results['sample_checks'][:3], 1):
            print(f"\n  Sample {i}:")
            print(f"    Text: {check['text_preview']}...")
            print(f"    Korean: {'✓' if check['metrics']['has_korean'] else '✗'}")
            print(f"    Korean ratio: {check['metrics']['korean_ratio']*100:.1f}%")
            print(f"    Medical terms: {'✓' if check['metrics']['has_medical_terms'] else '✗'}")
        
        print(f"\n{'='*70}\n")


def validate_all_reviewed_data(sample_size: int = 100):
    """Validate all reviewed data types"""
    
    checker = KoreanProficiencyChecker()
    
    types = ['type1_text', 'type2_text_reasoning', 'type3_word', 'type4_word_reasoning']
    
    print("="*70)
    print("KOREAN PROFICIENCY VALIDATION - ALL DATA TYPES")
    print("="*70)
    
    all_results = {}
    
    for dtype in types:
        for split in ['train', 'validation']:
            file_path = f'data/reviewed/{dtype}/{split}/data.jsonl'
            
            if not os.path.exists(file_path):
                continue
            
            results = checker.validate_dataset(file_path, sample_size)
            checker.print_report(results, file_path)
            
            all_results[f"{dtype}/{split}"] = results
    
    # Overall summary
    print("="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    total_korean = sum(r['korean_samples'] for r in all_results.values())
    total_samples = sum(r['total_samples'] for r in all_results.values())
    
    print(f"\nTotal samples checked: {total_samples}")
    print(f"Korean samples: {total_korean} ({total_korean/total_samples*100:.1f}%)")
    print(f"Non-Korean samples: {total_samples - total_korean} ({(total_samples-total_korean)/total_samples*100:.1f}%)")
    
    print(f"\nBy Type:")
    for dtype in types:
        train_key = f"{dtype}/train"
        if train_key in all_results:
            r = all_results[train_key]
            total = r['total_samples']
            korean = r['korean_samples']
            print(f"  {dtype:25} Korean: {korean}/{total} ({korean/total*100:5.1f}%)")
    
    print("="*70)


def check_benchmark_data():
    """Check Korean benchmark datasets"""
    
    print("\n" + "="*70)
    print("KOREAN BENCHMARK DATASETS")
    print("="*70)
    
    benchmarks = {
        'KorMedMCQA': 'data/raw/by_source/kormedmcqa/',
        'KMMLU-Medical': 'data/raw/korean_datasets/kmmlu_medical/',
        'Korean Medical Reasoning': 'data/raw/by_source/medical_reasoning_kormedmcqa/'
    }
    
    for name, path in benchmarks.items():
        if os.path.exists(path):
            files = list(Path(path).rglob('*.jsonl'))
            print(f"\n✓ {name}")
            print(f"  Path: {path}")
            print(f"  Files: {len(files)}")
            
            # Check first file
            if files:
                with open(files[0], 'r', encoding='utf-8') as f:
                    sample = json.loads(f.readline())
                    text = str(sample)[:200]
                    has_korean = any('\uac00' <= c <= '\ud7a3' for c in text)
                    print(f"  Korean: {'✓ Yes' if has_korean else '✗ No'}")
                    print(f"  Sample: {text[:100]}...")
        else:
            print(f"\n✗ {name}: Not found")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Validate Korean proficiency in medical data")
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of samples to check per file')
    parser.add_argument('--file', type=str, help='Validate specific file')
    parser.add_argument('--all', action='store_true', help='Validate all reviewed data')
    parser.add_argument('--benchmarks', action='store_true', help='Check benchmark datasets')
    
    args = parser.parse_args()
    
    if args.file:
        checker = KoreanProficiencyChecker()
        results = checker.validate_dataset(args.file, args.sample_size)
        checker.print_report(results, args.file)
    elif args.benchmarks:
        check_benchmark_data()
    elif args.all:
        validate_all_reviewed_data(args.sample_size)
    else:
        print("Please specify --file, --all, or --benchmarks")
        parser.print_help()


if __name__ == "__main__":
    main()
