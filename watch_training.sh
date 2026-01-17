#!/bin/bash
# Watch Training Progress - Live Log Viewer
# Usage: ./watch_training.sh [model_name]

MODEL=${1:-medgemma-4b}
LOG_FILE="logs/train_00_plain_text.log"

echo "========================================================================"
echo "Training Progress Monitor - $MODEL"
echo "========================================================================"
echo "Log file: $LOG_FILE"
echo ""
echo "Press Ctrl+C to stop watching"
echo "========================================================================"
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "Training may not be running."
    exit 1
fi

# Show last 50 lines first
echo "Recent log entries:"
echo "------------------------------------------------------------------------"
tail -50 "$LOG_FILE"
echo "------------------------------------------------------------------------"
echo ""
echo "Following new log entries (live updates)..."
echo ""

# Follow the log file with grep to highlight important lines
tail -f "$LOG_FILE" | grep --line-buffered -E "loss|epoch|step|accuracy|Error|complete|Saving"
