#!/bin/bash
# Run adaptive multi-type training for KorMedMCQA
#
# This script trains on all 4 data types and finds which one
# most effectively improves KorMedMCQA accuracy.
#
# Strategy:
#   1. Train a few steps on each type
#   2. Evaluate on KorMedMCQA after each round
#   3. Focus on type with best improvement
#   4. When improvement < 1%, move to next type
#   5. Stop when all types exhausted or target reached

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

# Default parameters
MODEL="medgemma-27b"
STEPS_PER_ROUND=50
MIN_IMPROVEMENT=1.0
TARGET_ACCURACY=90.0
MAX_ROUNDS=100
EVAL_SAMPLES=200
DEVICE="cuda:0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --steps)
            STEPS_PER_ROUND="$2"
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
        --max-rounds)
            MAX_ROUNDS="$2"
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
            echo "Options:"
            echo "  --model MODEL          Model to use (medgemma-4b or medgemma-27b, default: medgemma-27b)"
            echo "  --steps N              Training steps per round (default: 50)"
            echo "  --min-improvement PCT  Min improvement % to continue type (default: 1.0)"
            echo "  --target PCT           Target KorMedMCQA accuracy (default: 90.0)"
            echo "  --max-rounds N         Maximum training rounds (default: 100)"
            echo "  --eval-samples N       KorMedMCQA samples for evaluation (default: 200)"
            echo "  --device DEVICE        CUDA device (default: cuda:0)"
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
LOG_FILE="$LOG_DIR/adaptive_training_$TIMESTAMP.log"

echo "========================================"
echo "Adaptive Multi-Type Training"
echo "========================================"
echo "Model: $MODEL"
echo "Steps per round: $STEPS_PER_ROUND"
echo "Min improvement: $MIN_IMPROVEMENT%"
echo "Target accuracy: $TARGET_ACCURACY%"
echo "Max rounds: $MAX_ROUNDS"
echo "Eval samples: $EVAL_SAMPLES"
echo "Device: $DEVICE"
echo "Log file: $LOG_FILE"
echo "========================================"
echo ""

python3 "$SCRIPT_DIR/train_adaptive_types.py" \
    --model "$MODEL" \
    --steps-per-round "$STEPS_PER_ROUND" \
    --min-improvement "$MIN_IMPROVEMENT" \
    --target-accuracy "$TARGET_ACCURACY" \
    --max-rounds "$MAX_ROUNDS" \
    --eval-samples "$EVAL_SAMPLES" \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete!"
echo "Log saved to: $LOG_FILE"
echo "Model saved to: $BASE_DIR/models/adaptive_training/"
