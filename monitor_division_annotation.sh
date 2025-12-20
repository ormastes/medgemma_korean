#!/bin/bash
# Monitor division annotation progress on TITAN RTX

echo "======================================================================"
echo "Division Annotation Monitor - TITAN RTX (cuda:1)"
echo "Time: $(date)"
echo "======================================================================"

# Check if process is running
PROCESS=$(ps aux | grep "add_divisions_to_reviewed.py" | grep -v grep)

if [ -z "$PROCESS" ]; then
    echo "❌ No annotation process running"
    echo ""
    echo "Start with:"
    echo "  python3 phase5_subject_training/scripts/add_divisions_to_reviewed.py \\"
    echo "    --type type1_text \\"
    echo "    --model deepseek-ai/deepseek-llm-7b-chat \\"
    echo "    --device cuda:1 &"
    exit 1
fi

echo "✓ Annotation process is running:"
echo "$PROCESS" | head -1
echo ""

# Get PID
PID=$(echo "$PROCESS" | awk '{print $2}' | head -1)
echo "PID: $PID"
echo ""

# GPU status
echo "GPU Status (TITAN RTX - cuda:1):"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | grep "^1,"
echo ""

# Check output files
echo "======================================================================"
echo "Output Files:"
echo "======================================================================"

for TYPE in type1_text type2_text_reasoning type3_word type4_word_reasoning; do
    TRAIN_FILE="data/division_added/$TYPE/train.jsonl"
    VAL_FILE="data/division_added/$TYPE/validation.jsonl"
    
    if [ -f "$TRAIN_FILE" ]; then
        TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
        echo "✓ $TYPE/train.jsonl: $TRAIN_COUNT samples"
    fi
    
    if [ -f "$VAL_FILE" ]; then
        VAL_COUNT=$(wc -l < "$VAL_FILE")
        echo "✓ $TYPE/validation.jsonl: $VAL_COUNT samples"
    fi
done

echo ""
echo "======================================================================"
echo "Commands:"
echo "======================================================================"
echo "Monitor GPU:    watch -n 1 nvidia-smi"
echo "Stop process:   kill $PID"
echo "Check progress: tail -f data/division_added/type1_text/train.jsonl | wc -l"
echo ""
