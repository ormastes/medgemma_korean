#!/bin/bash
# Resume Training from Last Checkpoint
# Usage: ./resume_training.sh [--background]

MODEL="medgemma-4b"
CHECKPOINT_DIR="model/00_training/$MODEL"
LOG_FILE="logs/train_00_plain_text.log"
SCRIPT="script/train/train_00_plain_text.py"

echo "========================================================================"
echo "Resume Training - $MODEL"
echo "========================================================================"
echo ""

# Check for existing checkpoints
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR" | grep checkpoint | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "✅ Found checkpoint: $LATEST_CKPT"
        CKPT_NUM=$(echo "$LATEST_CKPT" | cut -d'-' -f2)
        echo "   Step: $CKPT_NUM / 7813 ($(( CKPT_NUM * 100 / 7813 ))% complete)"
        echo ""
    else
        echo "⚠️  No checkpoints found - will start from beginning"
        echo ""
    fi
else
    echo "⚠️  No training directory found - will start fresh"
    echo ""
fi

# Check if training is already running
PID=$(ps aux | grep "$SCRIPT" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "❌ Training is already running (PID: $PID)"
    echo "   Use: kill $PID  # to stop it first"
    exit 1
fi

echo "Training will automatically resume from latest checkpoint."
echo "Log file: $LOG_FILE"
echo ""

# Run in background or foreground
if [ "$1" == "--background" ]; then
    echo "Starting training in BACKGROUND..."
    echo "Use './check_training_status.sh' to monitor progress"
    echo "Use './watch_training.sh' to watch live logs"
    echo ""

    nohup python3 "$SCRIPT" --model "$MODEL" --epochs 1 --resume > "$LOG_FILE" 2>&1 &
    NEW_PID=$!
    echo $NEW_PID > logs/train_00_plain_text.pid

    echo "✅ Training started (PID: $NEW_PID)"
    echo "   Log: $LOG_FILE"
    echo "   PID saved to: logs/train_00_plain_text.pid"
    echo ""
    echo "To stop: kill $NEW_PID"
else
    echo "Starting training in FOREGROUND..."
    echo "Press Ctrl+C to stop"
    echo ""
    echo "========================================================================"

    python3 "$SCRIPT" --model "$MODEL" --epochs 1 --resume 2>&1 | tee -a "$LOG_FILE"
fi
