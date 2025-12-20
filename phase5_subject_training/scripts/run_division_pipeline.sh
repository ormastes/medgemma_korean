#!/bin/bash
# Complete pipeline for division-based data organization
# 1. Add divisions to reviewed data
# 2. Check division quality
# 3. Organize by division into separate folders

set -e

echo "======================================================================="
echo "Phase 5: Division-Based Data Organization Pipeline"
echo "======================================================================="

# Check GPU
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Step 1: Add divisions to all reviewed data using DeepSeek
echo ""
echo "======================================================================="
echo "Step 1: Adding divisions to reviewed data (DeepSeek on TITAN RTX)"
echo "======================================================================="

python3 phase5_subject_training/scripts/add_divisions_to_reviewed.py \
    --type all \
    --device cuda:0

echo ""
echo "✓ Step 1 completed: Divisions added to all data types"

# Step 2: Check division quality
echo ""
echo "======================================================================="
echo "Step 2: Checking division annotation quality"
echo "======================================================================="

python3 phase5_subject_training/scripts/check_divisions.py \
    --all \
    --base-dir data/division_added

echo ""
echo "✓ Step 2 completed: Division quality checked"

# Step 3: Organize by division
echo ""
echo "======================================================================="
echo "Step 3: Organizing data by division into separate folders"
echo "======================================================================="

python3 phase5_subject_training/scripts/organize_by_division.py \
    --source data/division_added \
    --output data/division_added \
    --min-samples 10

echo ""
echo "✓ Step 3 completed: Data organized by division"

# Final summary
echo ""
echo "======================================================================="
echo "Pipeline Completed Successfully!"
echo "======================================================================="
echo ""
echo "Output Structure:"
echo "  data/division_added/"
echo "    ├── type1_text/          (annotated data)"
echo "    ├── type2_text_reasoning/"
echo "    ├── type3_word/"
echo "    ├── type4_word_reasoning/"
echo "    ├── 1/                   (Cardiovascular division)"
echo "    │   ├── train.jsonl"
echo "    │   ├── validation.jsonl"
echo "    │   └── metadata.json"
echo "    ├── 2/                   (Respiratory division)"
echo "    ├── 3/                   (Gastroenterology division)"
echo "    ├── ..."
echo "    └── division_index.json  (index of all divisions)"
echo ""
echo "Next Steps:"
echo "  1. Review division_index.json for division distribution"
echo "  2. Check weak divisions from previous training reports"
echo "  3. Train division-specific models:"
echo "     python3 phase5_subject_training/scripts/train_division.py --division 1"
echo ""
