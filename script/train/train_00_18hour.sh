#!/bin/bash
# 18-Hour Continued Pretraining Session
# Optimized for A6000 GPU with medgemma_ded_med_normal tokenizer
#
# Target: ~18 hours training time
# Strategy: 70K samples × 2 epochs = ~112M tokens
# Expected: Perplexity <3.0, KorMedMCQA 30-40%

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

echo "=============================================="
echo "18-Hour Continued Pretraining Session"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model: medgemma-4b"
echo "  Tokenizer: medgemma_ded_med_normal (282,721 tokens)"
echo "  Samples: 70,000"
echo "  Epochs: 2"
echo "  Device: cuda:0 (A6000)"
echo "  Expected tokens: ~112M"
echo "  Expected time: ~18 hours"
echo ""

# Check tokenizer exists
TOKENIZER="model/tokenizer/medgemma_ded_med_normal"
if [ ! -d "$TOKENIZER" ]; then
    echo "Error: Tokenizer not found at $TOKENIZER"
    exit 1
fi

# Create directories
mkdir -p logs
mkdir -p model/00_training/medgemma-4b
mkdir -p model/00_trained/medgemma-4b

# Log file with timestamp
LOG_FILE="logs/train_00_18hour_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training..."
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

# Start training
python3 script/train/train_00_plain_text.py \
    --model medgemma-4b \
    --tokenizer-path "$TOKENIZER" \
    --max-samples 70000 \
    --epochs 2 \
    --device cuda:0 \
    --val-samples 100 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training Complete! ✓"
    echo "=============================================="
    echo ""
    echo "Output location:"
    echo "  Model: model/00_trained/medgemma-4b/"
    echo "  Log: $LOG_FILE"
    echo ""
    echo "Validation results:"
    cat model/00_trained/medgemma-4b/kormedmcqa_validation_results.json 2>/dev/null || echo "  (See log file)"
    echo ""
    echo "Next steps:"
    echo "  1. Check KorMedMCQA accuracy in validation results"
    echo "  2. Proceed to train_01 (medical dictionary SFT)"
    echo "  3. Then train_02 (MCQ with reasoning)"
else
    echo "Training Failed (exit code: $EXIT_CODE)"
    echo "=============================================="
    echo "Check log file: $LOG_FILE"
fi

exit $EXIT_CODE
