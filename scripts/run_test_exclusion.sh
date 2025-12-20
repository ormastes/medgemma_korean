#!/bin/bash
# Complete pipeline: Extract test set and clean training data
# Ensures no test contamination in training

set -e

echo "======================================================================"
echo "KorMedMCQA Test Exclusion Pipeline"
echo "======================================================================"

# Step 1: Extract KorMedMCQA test set
echo ""
echo "Step 1: Extracting KorMedMCQA test set..."
echo "----------------------------------------------------------------------"

python3 scripts/extract_kormedmcqa_test.py --output data/kormedmcqa_test

echo ""
echo "✓ Test set extracted to: data/kormedmcqa_test/"

# Step 2: Remove test samples from division_added data
echo ""
echo "Step 2: Removing test samples from training data..."
echo "----------------------------------------------------------------------"

python3 scripts/exclude_test_from_training.py \
    --test-file data/kormedmcqa_test/test_questions.txt \
    --source data/division_added \
    --output data/division_added_clean

echo ""
echo "✓ Cleaned data saved to: data/division_added_clean/"

# Step 3: Summary
echo ""
echo "======================================================================"
echo "Pipeline Completed!"
echo "======================================================================"
echo ""
echo "Output Structure:"
echo "  data/kormedmcqa_test/          (Test set for validation)"
echo "    ├── test_doctor.jsonl"
echo "    ├── test_nurse.jsonl"
echo "    ├── test_pharm.jsonl"
echo "    ├── test_dentist.jsonl"
echo "    ├── all_test.jsonl            (Combined 604 samples)"
echo "    └── test_questions.txt        (For exclusion)"
echo ""
echo "  data/division_added_clean/     (Training data without test)"
echo "    ├── type1_text/"
echo "    ├── type2_text_reasoning/"
echo "    ├── type3_word/"
echo "    ├── type4_word_reasoning/"
echo "    ├── 1/                        (Division folders)"
echo "    ├── 2/"
echo "    ├── ..."
echo "    └── test_exclusion_stats.json (Removal statistics)"
echo ""
echo "Next Steps:"
echo "  1. Review: cat data/division_added_clean/test_exclusion_stats.json"
echo "  2. Train on: data/division_added_clean/"
echo "  3. Evaluate on: data/kormedmcqa_test/all_test.jsonl"
echo ""
