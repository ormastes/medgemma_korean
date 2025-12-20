#!/bin/bash
# Run remaining types after Type 1 completes

echo "======================================================================"
echo "Division Annotation - Remaining Types"
echo "======================================================================"

# Type 2
echo ""
echo "Type 2: TEXT_REASONING (23,018 train + 2,558 val) - ~1 hour"
python3 phase5_subject_training/scripts/add_divisions_to_reviewed.py \
    --type type2_text_reasoning \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1

echo "✓ Type 2 completed"
echo ""

# Type 3
echo "Type 3: WORD (16,701 train + 1,846 val) - ~1 hour"
python3 phase5_subject_training/scripts/add_divisions_to_reviewed.py \
    --type type3_word \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1

echo "✓ Type 3 completed"
echo ""

# Type 4
echo "Type 4: WORD_REASONING (7,957 train + 885 val) - ~30 min"
python3 phase5_subject_training/scripts/add_divisions_to_reviewed.py \
    --type type4_word_reasoning \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1

echo "✓ Type 4 completed"
echo ""
echo "======================================================================"
echo "All types completed!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check quality: python3 phase5_subject_training/scripts/check_divisions.py --all"
echo "  2. Organize: python3 phase5_subject_training/scripts/organize_by_division.py"
echo "  3. Exclude test: bash scripts/run_test_exclusion.sh"
echo ""

