#!/bin/bash
# Start remaining division annotations: Type 3, 2, 1

echo "Starting Type 3, 2, and 1 annotations..."
echo ""

# Type 3 - WORD (MCQ)
echo "=== Starting Type 3: WORD (16,701 samples) ==="
python3 scripts/fast_division_annotation.py \
    --type type3_word \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1 \
    --batch-size 8 \
    > logs/type3_run.log 2>&1 &
PID3=$!
echo "Type 3 started. PID: $PID3"
echo "Monitor: tail -f logs/type3_run.log"
echo ""

# Type 2 - TEXT_REASONING
echo "=== Starting Type 2: TEXT_REASONING (23,018 samples) ==="
python3 scripts/fast_division_annotation.py \
    --type type2_text_reasoning \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1 \
    --batch-size 8 \
    > logs/type2_run.log 2>&1 &
PID2=$!
echo "Type 2 started. PID: $PID2"
echo "Monitor: tail -f logs/type2_run.log"
echo ""

# Type 1 - TEXT (clear old data first)
echo "=== Starting Type 1: TEXT (118,431 samples) ==="
rm -f data/division_added/type1_text/train.jsonl
python3 scripts/fast_division_annotation.py \
    --type type1_text \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1 \
    --batch-size 8 \
    > logs/type1_run.log 2>&1 &
PID1=$!
echo "Type 1 started. PID: $PID1"
echo "Monitor: tail -f logs/type1_run.log"
echo ""

echo "All types started!"
echo ""
echo "PIDs: Type3=$PID3, Type2=$PID2, Type1=$PID1"
echo ""
echo "Monitor progress:"
echo "  ./monitor_division_annotation.sh"
echo ""
echo "NOTE: All 3 types running in parallel (not sequential)"
echo "This is FASTER but uses more resources"
