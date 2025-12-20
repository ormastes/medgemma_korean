#!/usr/bin/env python3
"""
Verify and fix malformed division annotations
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

class DivisionValidator:
    def __init__(self, taxonomy_file: str = 'med_division.json'):
        """Initialize validator with taxonomy"""
        with open(taxonomy_file, 'r', encoding='utf-8') as f:
            self.taxonomy = json.load(f)
        
        self.valid_ids = self._build_valid_ids()
    
    def _build_valid_ids(self) -> set:
        """Build set of all valid division IDs"""
        valid = set()
        for node in self.taxonomy['nodes']:
            valid.add(node['id'])
            if 'children' in node:
                self._collect_ids(node['children'], valid)
        return valid
    
    def _collect_ids(self, children, valid_set):
        """Recursively collect IDs"""
        for child in children:
            if isinstance(child, dict):
                valid_set.add(child['id'])
                if 'children' in child:
                    self._collect_ids(child['children'], valid_set)
    
    def validate_sample(self, sample: Dict) -> Dict[str, Any]:
        """Validate a single sample and return issues"""
        issues = []
        fixes = []
        
        # Check if divisions field exists
        if 'divisions' not in sample:
            issues.append("Missing 'divisions' field")
            sample['divisions'] = ["UNKNOWN"]
            fixes.append("Added default divisions")
        
        # Check if primary_division field exists
        if 'primary_division' not in sample:
            issues.append("Missing 'primary_division' field")
            sample['primary_division'] = sample.get('divisions', ["UNKNOWN"])[0]
            fixes.append("Set primary from first division")
        
        # Validate divisions list
        if not isinstance(sample.get('divisions'), list):
            issues.append("divisions is not a list")
            sample['divisions'] = [str(sample['divisions'])]
            fixes.append("Converted divisions to list")
        
        # Check if divisions list is empty
        if len(sample.get('divisions', [])) == 0:
            issues.append("Empty divisions list")
            sample['divisions'] = ["UNKNOWN"]
            fixes.append("Added UNKNOWN division")
        
        # Validate each division ID
        invalid_divs = []
        for div_id in sample.get('divisions', []):
            if div_id not in self.valid_ids and div_id != "UNKNOWN":
                invalid_divs.append(div_id)
        
        if invalid_divs:
            issues.append(f"Invalid division IDs: {invalid_divs}")
            # Try to fix by keeping only valid ones
            sample['divisions'] = [d for d in sample['divisions'] if d in self.valid_ids or d == "UNKNOWN"]
            if not sample['divisions']:
                sample['divisions'] = ["UNKNOWN"]
            fixes.append("Removed invalid division IDs")
        
        # Validate primary is in divisions
        if sample.get('primary_division') not in sample.get('divisions', []):
            issues.append("Primary division not in divisions list")
            sample['primary_division'] = sample['divisions'][0]
            fixes.append("Set primary to first division")
        
        return {
            'sample': sample,
            'issues': issues,
            'fixes': fixes,
            'is_valid': len(issues) == 0
        }
    
    def validate_file(self, input_file: str, output_file: str = None, fix: bool = False):
        """Validate entire file and optionally fix issues"""
        
        print(f"Validating: {input_file}")
        
        # Read data
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Line {line_num}: JSON decode error - {e}")
        
        print(f"Total samples: {len(data)}")
        
        # Validate each sample
        valid_count = 0
        fixed_count = 0
        issue_types = Counter()
        
        validated_data = []
        
        for idx, sample in enumerate(data):
            result = self.validate_sample(sample)
            
            if result['is_valid']:
                valid_count += 1
            else:
                for issue in result['issues']:
                    issue_types[issue] += 1
                
                if fix:
                    fixed_count += 1
            
            validated_data.append(result['sample'])
        
        # Report results
        print(f"\n=== Validation Results ===")
        print(f"Valid samples: {valid_count}/{len(data)} ({valid_count/len(data)*100:.2f}%)")
        print(f"Samples with issues: {len(data) - valid_count}")
        
        if issue_types:
            print(f"\nIssue breakdown:")
            for issue_type, count in issue_types.most_common():
                print(f"  {issue_type}: {count}")
        
        if fix:
            print(f"\nFixed samples: {fixed_count}")
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in validated_data:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                print(f"Saved fixed data to: {output_file}")
        
        # Division statistics
        self._print_division_stats(validated_data)
        
        return validated_data
    
    def _print_division_stats(self, data: List[Dict]):
        """Print statistics about division distribution"""
        primary_divs = Counter()
        all_divs = Counter()
        
        for sample in data:
            primary = sample.get('primary_division', 'UNKNOWN')
            primary_divs[primary] += 1
            
            for div in sample.get('divisions', []):
                all_divs[div] += 1
        
        print(f"\n=== Division Statistics ===")
        print(f"Primary division distribution (top 10):")
        for div_id, count in primary_divs.most_common(10):
            print(f"  {div_id}: {count}")
        
        print(f"\nAll divisions (top 15):")
        for div_id, count in all_divs.most_common(15):
            print(f"  {div_id}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Validate division annotations")
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, help='Output file (if fixing)')
    parser.add_argument('--fix', action='store_true', help='Fix issues automatically')
    parser.add_argument('--taxonomy', type=str, default='med_division.json',
                       help='Taxonomy JSON file')
    
    args = parser.parse_args()
    
    validator = DivisionValidator(taxonomy_file=args.taxonomy)
    validator.validate_file(args.input, args.output, fix=args.fix)


if __name__ == "__main__":
    main()
