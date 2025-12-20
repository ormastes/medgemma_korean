#!/usr/bin/env python3
"""
Verify Phase 5 Division Organization Setup
"""

import json
import os
from pathlib import Path

def check_scripts():
    """Check all required scripts exist"""
    scripts = [
        'add_divisions_to_reviewed.py',
        'check_divisions.py',
        'organize_by_division.py',
        'run_division_pipeline.sh',
        'train_with_divisions.py'
    ]
    
    print("Checking scripts...")
    missing = []
    for script in scripts:
        path = f'phase5_subject_training/scripts/{script}'
        if os.path.exists(path):
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} MISSING")
            missing.append(script)
    
    return len(missing) == 0

def check_docs():
    """Check documentation files"""
    docs = [
        'phase5_subject_training/README.md',
        'phase5_subject_training/QUICK_START.md',
        'phase5_subject_training/DIVISION_GUIDE.md',
        'phase5_subject_training/scripts/README.md',
        'PHASE5_DIVISION_SUMMARY.md'
    ]
    
    print("\nChecking documentation...")
    missing = []
    for doc in docs:
        if os.path.exists(doc):
            print(f"  ✓ {doc}")
        else:
            print(f"  ✗ {doc} MISSING")
            missing.append(doc)
    
    return len(missing) == 0

def check_dirs():
    """Check directory structure"""
    dirs = [
        'phase5_subject_training/scripts',
        'phase5_subject_training/models',
        'phase5_subject_training/logs',
        'phase5_subject_training/results',
        'data/division_added'
    ]
    
    print("\nChecking directories...")
    missing = []
    for directory in dirs:
        if os.path.exists(directory):
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} MISSING")
            missing.append(directory)
    
    return len(missing) == 0

def check_reviewed_data():
    """Check reviewed data exists"""
    types = ['type1_text', 'type2_text_reasoning', 'type3_word', 'type4_word_reasoning']
    
    print("\nChecking reviewed data...")
    found = 0
    for dtype in types:
        train = f'data/reviewed/{dtype}/train/data.jsonl'
        val = f'data/reviewed/{dtype}/validation/data.jsonl'
        
        if os.path.exists(train) and os.path.exists(val):
            with open(train) as f:
                train_count = sum(1 for _ in f)
            with open(val) as f:
                val_count = sum(1 for _ in f)
            print(f"  ✓ {dtype}: {train_count:,} train, {val_count:,} val")
            found += 1
        else:
            print(f"  ✗ {dtype}: Missing")
    
    return found == len(types)

def check_taxonomy():
    """Check taxonomy file"""
    print("\nChecking taxonomy...")
    if os.path.exists('med_division.json'):
        with open('med_division.json') as f:
            taxonomy = json.load(f)
        
        if 'nodes' in taxonomy and len(taxonomy['nodes']) == 10:
            print(f"  ✓ med_division.json: {len(taxonomy['nodes'])} major divisions")
            return True
        else:
            print(f"  ✗ med_division.json: Invalid structure")
            return False
    else:
        print(f"  ✗ med_division.json: Missing")
        return False

def print_summary():
    """Print usage summary"""
    print("\n" + "="*70)
    print("Phase 5: Division Organization Setup")
    print("="*70)
    
    checks = [
        ("Scripts", check_scripts()),
        ("Documentation", check_docs()),
        ("Directories", check_dirs()),
        ("Reviewed Data", check_reviewed_data()),
        ("Taxonomy", check_taxonomy())
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} checks passed")
    print("="*70)
    
    if passed == total:
        print("\n✅ All checks passed! Phase 5 is ready.")
        print("\nNext steps:")
        print("  1. Run: bash phase5_subject_training/scripts/run_division_pipeline.sh")
        print("  2. Wait ~6 hours for completion")
        print("  3. Check: cat data/division_added/division_index.json")
        print("  4. Review division distribution")
        print("  5. Train division-specific models as needed")
        
        print("\nQuick commands:")
        print("  # Full pipeline")
        print("  bash phase5_subject_training/scripts/run_division_pipeline.sh")
        print()
        print("  # Check one type only")
        print("  python3 phase5_subject_training/scripts/add_divisions_to_reviewed.py --type type1_text")
        print()
        print("  # Verify quality")
        print("  python3 phase5_subject_training/scripts/check_divisions.py --all")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1
    
    print("="*70)
    return 0

if __name__ == "__main__":
    exit(print_summary())
