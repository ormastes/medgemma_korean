#!/bin/bash
# =============================================================================
# Monitor Training Progress
# =============================================================================
# Shows current training status, loss, and validation scores
#
# Usage:
#   ./monitor_training.sh --model medgemma-4b
#   ./monitor_training.sh --model medgemma-27b --follow
# =============================================================================

MODEL="medgemma-4b"
FOLLOW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --follow|-f)
            FOLLOW=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL   Model name (medgemma-4b or medgemma-27b)"
            echo "  --follow, -f    Follow log output (like tail -f)"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BASE_DIR/logs"

LOG_FILE="$LOG_DIR/pipeline_${MODEL}.log"
PROGRESS_FILE="$LOG_DIR/progress_${MODEL}.txt"
PID_FILE="$LOG_DIR/pipeline_${MODEL}.pid"

echo "============================================================"
echo "Training Monitor: $MODEL"
echo "============================================================"

# Check if running
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Status: RUNNING (PID: $PID)"
    else
        echo "Status: STOPPED (stale PID file)"
    fi
else
    echo "Status: NOT RUNNING"
fi

# Show progress
if [[ -f "$PROGRESS_FILE" ]]; then
    echo ""
    echo "Current Step:"
    echo "  $(cat "$PROGRESS_FILE")"
fi

echo ""
echo "============================================================"
echo "Recent Log Entries"
echo "============================================================"

if [[ -f "$LOG_FILE" ]]; then
    # Show last 20 lines with highlights
    tail -20 "$LOG_FILE" | while IFS= read -r line; do
        # Highlight important info
        if [[ "$line" == *"STEP"* ]]; then
            echo -e "\033[1;32m$line\033[0m"  # Green bold
        elif [[ "$line" == *"ERROR"* ]]; then
            echo -e "\033[1;31m$line\033[0m"  # Red bold
        elif [[ "$line" == *"Loss"* ]] || [[ "$line" == *"loss"* ]]; then
            echo -e "\033[1;33m$line\033[0m"  # Yellow bold
        elif [[ "$line" == *"Accuracy"* ]] || [[ "$line" == *"accuracy"* ]]; then
            echo -e "\033[1;36m$line\033[0m"  # Cyan bold
        elif [[ "$line" == *"Reasoning"* ]]; then
            echo -e "\033[1;35m$line\033[0m"  # Magenta bold
        else
            echo "$line"
        fi
    done
else
    echo "No log file found: $LOG_FILE"
fi

echo ""
echo "============================================================"
echo "Training Stats"
echo "============================================================"

if [[ -f "$LOG_FILE" ]]; then
    # Extract latest validation scores
    echo ""
    echo "Latest Validation Scores:"
    grep -E "(KorMedMCQA|Accuracy|Reasoning Score|Combined Score)" "$LOG_FILE" | tail -5

    # Count training steps
    echo ""
    echo "Step Counts:"
    echo "  Train_00 entries: $(grep -c "Train 00" "$LOG_FILE" 2>/dev/null || echo 0)"
    echo "  Train_01 entries: $(grep -c "Train 01" "$LOG_FILE" 2>/dev/null || echo 0)"
    echo "  Train_02 entries: $(grep -c "Train 02" "$LOG_FILE" 2>/dev/null || echo 0)"

    # Show any errors
    ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE" 2>/dev/null || echo 0)
    if [[ "$ERROR_COUNT" -gt 0 ]]; then
        echo ""
        echo -e "\033[1;31mErrors Found: $ERROR_COUNT\033[0m"
        grep "ERROR" "$LOG_FILE" | tail -3
    fi
fi

echo ""
echo "============================================================"
echo "Commands"
echo "============================================================"
echo "  Follow log:    tail -f $LOG_FILE"
echo "  Full log:      less $LOG_FILE"
echo "  Stop training: kill \$(cat $PID_FILE)"
echo ""

if [[ "$FOLLOW" == true ]]; then
    echo "Following log output (Ctrl+C to stop)..."
    echo ""
    tail -f "$LOG_FILE"
fi
