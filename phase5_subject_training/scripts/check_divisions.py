#!/usr/bin/env python3
"""
Check division annotations quality and distribution
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import argparse
from tqdm import tqdm


class DivisionChecker:
    def __init__(self, taxonomy_file: str = 'med_division.json'):
        with open(taxonomy_file, 'r', encoding='utf-8') as f:
            self.taxonomy = json.load(f)
        
        self.valid_ids = self._build_valid_ids()
        self.division_names = self._build_division_names()
    
    def _build_valid_ids(self) -> set:
        valid = set()
        for node in self.taxonomy['nodes']:
            valid.add(node['id'])
            if 'children' in node:
                self._collect_ids(node['children'], valid)
        return valid
    
    def _build_division_names(self) -> dict:
        names = {}
        for node in self.taxonomy['nodes']:
            names[node['id']] = node['name']
            if 'children' in node:
                self._collect_names(node['children'], names)
        return names
    
    def _collect_ids(self, children, valid_set):
        for child in children:
            if isinstance(child, dict):
                valid_set.add(child['id'])
                if 'children' in child:
                    self._collect_ids(child['children'], valid_set)
    
    def _collect_names(self, children, names_dict):
        for child in children:
            if isinstance(child, dict):
                names_dict[child['id']] = child['name']
                if 'children' in child:
                    self._collect_names(child['children'], names_dict)
    
    def check_file(self, file_path: str) -> dict:
        """Check a single file and return statistics"""
        
        print(f"\nChecking: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return {}
        
        stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'unknown': 0,
            'missing_fields': 0,
            'primary_divisions': Counter(),
            'all_divisions': Counter(),
            'invalid_ids': Counter(),
            'errors': []
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Checking"), 1):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    stats['total'] += 1
                    
                    # Check required fields
                    if 'divisions' not in sample or 'primary_division' not in sample:
                        stats['missing_fields'] += 1
                        stats['errors'].append(f"Line {line_num}: Missing required fields")
                        continue
                    
                    primary = sample.get('primary_division', 'UNKNOWN')
                    divisions = sample.get('divisions', [])
                    
                    # Check UNKNOWN
                    if primary == 'UNKNOWN':
                        stats['unknown'] += 1
                        continue
                    
                    # Check if primary is valid
                    is_valid = True
                    if primary not in self.valid_ids:
                        stats['invalid_ids'][primary] += 1
                        is_valid = False
                    
                    # Check all divisions
                    for div_id in divisions:
                        if div_id not in self.valid_ids and div_id != 'UNKNOWN':
                            stats['invalid_ids'][div_id] += 1
                            is_valid = False
                        stats['all_divisions'][div_id] += 1
                    
                    if is_valid:
                        stats['valid'] += 1
                        stats['primary_divisions'][primary] += 1
                    else:
                        stats['invalid'] += 1
                
                except json.JSONDecodeError as e:
                    stats['errors'].append(f"Line {line_num}: JSON decode error - {e}")
        
        return stats
    
    def print_stats(self, stats: dict, file_path: str):
        """Print statistics in a readable format"""
        
        print(f"\n{'='*70}")
        print(f"Division Check Report: {file_path}")
        print(f"{'='*70}")
        
        total = stats['total']
        if total == 0:
            print("No data found")
            return
        
        print(f"\nOverall Statistics:")
        print(f"  Total samples: {total}")
        print(f"  Valid: {stats['valid']} ({stats['valid']/total*100:.2f}%)")
        print(f"  Invalid: {stats['invalid']} ({stats['invalid']/total*100:.2f}%)")
        print(f"  Unknown: {stats['unknown']} ({stats['unknown']/total*100:.2f}%)")
        print(f"  Missing fields: {stats['missing_fields']}")
        
        if stats['invalid_ids']:
            print(f"\nInvalid Division IDs:")
            for div_id, count in stats['invalid_ids'].most_common(10):
                print(f"  {div_id}: {count}")
        
        print(f"\nPrimary Division Distribution (Top 15):")
        print(f"  {'Division':<6} {'Name':<40} {'Count':>8} {'%':>6}")
        print(f"  {'-'*62}")
        for div_id, count in stats['primary_divisions'].most_common(15):
            name = self.division_names.get(div_id, 'Unknown')
            pct = count / total * 100
            print(f"  {div_id:<6} {name:<40} {count:>8} {pct:>5.1f}%")
        
        if stats['errors']:
            print(f"\nErrors (showing first 5):")
            for error in stats['errors'][:5]:
                print(f"  {error}")
        
        print(f"\n{'='*70}\n")
    
    def check_all_types(self, base_dir: str = 'data/division_added'):
        """Check all data types"""
        
        types = ['type1_text', 'type2_text_reasoning', 'type3_word', 'type4_word_reasoning']
        
        all_stats = {}
        
        for dtype in types:
            for split in ['train', 'validation']:
                file_path = os.path.join(base_dir, dtype, f'{split}.jsonl')
                
                if not os.path.exists(file_path):
                    continue
                
                stats = self.check_file(file_path)
                self.print_stats(stats, file_path)
                
                all_stats[f"{dtype}/{split}"] = stats
        
        # Print summary
        self._print_summary(all_stats)
        
        return all_stats
    
    def _print_summary(self, all_stats: dict):
        """Print overall summary"""
        
        print(f"\n{'='*70}")
        print("Overall Summary")
        print(f"{'='*70}")
        
        total_samples = sum(s['total'] for s in all_stats.values())
        total_valid = sum(s['valid'] for s in all_stats.values())
        total_invalid = sum(s['invalid'] for s in all_stats.values())
        total_unknown = sum(s['unknown'] for s in all_stats.values())
        
        print(f"Total samples: {total_samples}")
        print(f"Valid: {total_valid} ({total_valid/total_samples*100:.2f}%)")
        print(f"Invalid: {total_invalid} ({total_invalid/total_samples*100:.2f}%)")
        print(f"Unknown: {total_unknown} ({total_unknown/total_samples*100:.2f}%)")
        
        # Aggregate primary divisions
        all_primary = Counter()
        for stats in all_stats.values():
            all_primary.update(stats['primary_divisions'])
        
        print(f"\nTop 10 Primary Divisions (Overall):")
        print(f"  {'Division':<6} {'Name':<40} {'Count':>8}")
        print(f"  {'-'*56}")
        for div_id, count in all_primary.most_common(10):
            name = self.division_names.get(div_id, 'Unknown')
            print(f"  {div_id:<6} {name:<40} {count:>8}")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Check division annotations")
    parser.add_argument('--file', type=str, help='Single file to check')
    parser.add_argument('--all', action='store_true', help='Check all types')
    parser.add_argument('--base-dir', type=str, default='data/division_added',
                       help='Base directory for data')
    
    args = parser.parse_args()
    
    checker = DivisionChecker()
    
    if args.file:
        stats = checker.check_file(args.file)
        checker.print_stats(stats, args.file)
    elif args.all:
        checker.check_all_types(base_dir=args.base_dir)
    else:
        print("Please specify --file or --all")
        parser.print_help()


if __name__ == "__main__":
    main()
