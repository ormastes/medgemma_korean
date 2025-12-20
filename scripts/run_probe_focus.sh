#!/bin/bash
# Run Probe-and-Focus Training for KorMedMCQA
#
# Strategy:
#   1. PROBE: Train few steps on each type, measure KorMedMCQA improvement
#   2. FOCUS: Focus on best type until improvement < 1%
#   3. ROTATE: Move to next best type
#   4. STOP: When all types exhausted
#
# Also tracks failed questions to analyze persistent failures

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

# Default parameters
MODEL="medgemma-27b"
PROBE_STEPS=30
FOCUS_STEPS=50
MIN_IMPROVEMENT=1.0
TARGET_ACCURACY=90.0
MAX_FOCUS_ROUNDS=10
EVAL_SAMPLES=200
DEVICE="cuda:0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --probe-steps)
            PROBE_STEPS="$2"
            shift 2
            ;;
        --focus-steps)
            FOCUS_STEPS="$2"
            shift 2
            ;;
        --min-improvement)
            MIN_IMPROVEMENT="$2"
            shift 2
            ;;
        --target)
            TARGET_ACCURACY="$2"
            shift 2
            ;;
        --max-focus-rounds)
            MAX_FOCUS_ROUNDS="$2"
            shift 2
            ;;
        --eval-samples)
            EVAL_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Probe-and-Focus Training:"
            echo "  1. Probe all 4 types with N steps each"
            echo "  2. Focus on best type until improvement < threshold"
            echo "  3. Rotate to next best type"
            echo "  4. Stop when all types exhausted or target reached"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model (medgemma-4b or medgemma-27b, default: medgemma-27b)"
            echo "  --probe-steps N        Steps for probing each type (default: 30)"
            echo "  --focus-steps N        Steps per focus round (default: 50)"
            echo "  --min-improvement PCT  Min improvement % to continue (default: 1.0)"
            echo "  --target PCT           Target KorMedMCQA accuracy (default: 90.0)"
            echo "  --max-focus-rounds N   Max rounds per type during focus (default: 10)"
            echo "  --eval-samples N       KorMedMCQA samples for eval (default: 200)"
            echo "  --device DEVICE        CUDA device (default: cuda:0)"
            echo ""
            echo "Output:"
            echo "  models/probe_focus_training/final/     - Trained model"
            echo "  models/probe_focus_training/final_report.json - Full report"
            echo "  models/probe_focus_training/failed_questions.json - Persistent failures"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/probe_focus_$TIMESTAMP.log"

echo "========================================"
echo "Probe-and-Focus Training"
echo "========================================"
echo "Model: $MODEL"
echo "Probe steps (per type): $PROBE_STEPS"
echo "Focus steps (per round): $FOCUS_STEPS"
echo "Min improvement: $MIN_IMPROVEMENT%"
echo "Target accuracy: $TARGET_ACCURACY%"
echo "Max focus rounds: $MAX_FOCUS_ROUNDS"
echo "Eval samples: $EVAL_SAMPLES"
echo "Device: $DEVICE"
echo "Log file: $LOG_FILE"
echo "========================================"
echo ""

python3 "$SCRIPT_DIR/train_probe_and_focus.py" \
    --model "$MODEL" \
    --probe-steps "$PROBE_STEPS" \
    --focus-steps "$FOCUS_STEPS" \
    --min-improvement "$MIN_IMPROVEMENT" \
    --target-accuracy "$TARGET_ACCURACY" \
    --max-focus-rounds "$MAX_FOCUS_ROUNDS" \
    --eval-samples "$EVAL_SAMPLES" \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Log: $LOG_FILE"
echo "Model: $BASE_DIR/models/probe_focus_training/final/"
echo "Report: $BASE_DIR/models/probe_focus_training/final_report.json"
echo "Failed questions: $BASE_DIR/models/probe_focus_training/failed_questions.json"
echo ""
echo "To analyze persistent failures:"
echo "  cat $BASE_DIR/models/probe_focus_training/failed_questions.json | python3 -m json.tool"
