#!/bin/bash
# Check Training Status - Quick Summary
# Usage: ./check_training_status.sh

MODEL="medgemma-4b"
LOG_FILE="logs/train_00_plain_text.log"
CHECKPOINT_DIR="model/00_training/$MODEL"

echo "========================================================================"
echo "Training Status Check - $(date)"
echo "========================================================================"
echo ""

# Check if training is running
PID=$(ps aux | grep "train_00_plain_text.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✅ Training is RUNNING (PID: $PID)"
    echo ""
    # Show process details
    ps -p $PID -o pid,cmd,etime,%cpu,%mem
else
    echo "❌ Training is NOT running"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "Latest Checkpoints:"
echo "------------------------------------------------------------------------"
if [ -d "$CHECKPOINT_DIR" ]; then
    ls -lth "$CHECKPOINT_DIR" | grep checkpoint | head -5

    # Get latest checkpoint info
    LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR" | grep checkpoint | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo ""
        echo "Latest: $LATEST_CKPT"
        CKPT_TIME=$(stat -c "%y" "$CHECKPOINT_DIR/$LATEST_CKPT" | cut -d'.' -f1)
        echo "Saved at: $CKPT_TIME"
    fi
else
    echo "No checkpoints found"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "Recent Log Activity (last 20 lines with loss/accuracy):"
echo "------------------------------------------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -100 "$LOG_FILE" | grep -E "loss|accuracy|epoch.*%" | tail -20

    echo ""
    echo "Last log timestamp:"
    tail -1 "$LOG_FILE" | head -c 200
else
    echo "❌ Log file not found: $LOG_FILE"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "GPU Status:"
echo "------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "========================================================================"
echo "To watch live updates: ./watch_training.sh"
echo "To view full log: tail -f $LOG_FILE"
echo "========================================================================"
