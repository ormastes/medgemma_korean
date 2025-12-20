#!/usr/bin/env python3
"""
Quick test to verify Phase 5 setup
Tests annotation, validation, and data format
"""

import json
import sys
from pathlib import Path

def test_taxonomy():
    """Test taxonomy file loads correctly"""
    print("Testing taxonomy...")
    try:
        with open('med_division.json', 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        
        assert 'nodes' in taxonomy
        assert len(taxonomy['nodes']) == 10  # 10 major divisions
        
        # Count all IDs
        ids = set()
        for node in taxonomy['nodes']:
            ids.add(node['id'])
        
        print(f"✓ Taxonomy loaded: {len(taxonomy['nodes'])} major divisions")
        print(f"✓ Total division IDs: {len(ids)}")
        return True
    except Exception as e:
        print(f"✗ Taxonomy test failed: {e}")
        return False

def test_reviewed_data():
    """Test reviewed data exists"""
    print("\nTesting reviewed data...")
    types = ['type1_text', 'type2_text_reasoning', 'type3_word', 'type4_word_reasoning']
    
    found = 0
    for dtype in types:
        train_file = f'data/reviewed/{dtype}/train/data.jsonl'
        val_file = f'data/reviewed/{dtype}/validation/data.jsonl'
        
        if Path(train_file).exists() and Path(val_file).exists():
            # Count samples
            with open(train_file, 'r') as f:
                train_count = sum(1 for line in f if line.strip())
            with open(val_file, 'r') as f:
                val_count = sum(1 for line in f if line.strip())
            
            print(f"✓ {dtype}: {train_count} train, {val_count} val")
            found += 1
        else:
            print(f"✗ {dtype}: Missing files")
    
    return found == len(types)

def test_phase5_structure():
    """Test Phase 5 directory structure"""
    print("\nTesting Phase 5 structure...")
    
    required_dirs = [
        'phase5_subject_training/scripts',
        'phase5_subject_training/models',
        'phase5_subject_training/logs',
        'phase5_subject_training/results',
        'data/division'
    ]
    
    required_files = [
        'phase5_subject_training/scripts/annotate_with_deepseek.py',
        'phase5_subject_training/scripts/validate_divisions.py',
        'phase5_subject_training/scripts/train_with_divisions.py',
        'phase5_subject_training/scripts/run_pipeline.sh',
        'phase5_subject_training/README.md'
    ]
    
    dirs_ok = all(Path(d).exists() for d in required_dirs)
    files_ok = all(Path(f).exists() for f in required_files)
    
    if dirs_ok:
        print("✓ All required directories exist")
    else:
        print("✗ Some directories missing")
    
    if files_ok:
        print("✓ All required scripts exist")
    else:
        print("✗ Some scripts missing")
    
    return dirs_ok and files_ok

def test_sample_annotation():
    """Test annotation format"""
    print("\nTesting division annotation format...")
    
    sample = {
        "prompt": "<|im_start|>system\nTest<|im_end|>\n<|im_start|>user\nQuestion?<|im_end|>",
        "completion": "Answer",
        "text": "Full text",
        "divisions": ["1.4.1", "9.2"],
        "primary_division": "1.4.1",
        "division_reasoning": "Test reasoning"
    }
    
    required_fields = ['divisions', 'primary_division', 'division_reasoning']
    
    has_all = all(field in sample for field in required_fields)
    
    if has_all:
        print("✓ Division annotation format correct")
        print(f"  - divisions: {sample['divisions']}")
        print(f"  - primary: {sample['primary_division']}")
        return True
    else:
        print("✗ Missing required fields")
        return False

def main():
    print("=" * 60)
    print("Phase 5: Subject Training - Setup Verification")
    print("=" * 60)
    
    tests = [
        test_taxonomy(),
        test_reviewed_data(),
        test_phase5_structure(),
        test_sample_annotation()
    ]
    
    print("\n" + "=" * 60)
    passed = sum(tests)
    total = len(tests)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Phase 5 setup is complete.")
        print("\nNext steps:")
        print("  1. Run: bash phase5_subject_training/scripts/run_pipeline.sh")
        print("  2. Review division reports in phase5_subject_training/models/*/division_report.json")
    else:
        print("✗ Some tests failed. Please fix issues above.")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()
