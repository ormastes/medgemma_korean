#!/bin/bash
# =============================================================================
# Full Training Pipeline
# =============================================================================
# Runs the complete training pipeline:
# 1. Train 00 (plain text)
# 2. Train 01 with 00 monitoring (medical dictionary)
# 3. Add second LoRA adapter
# 4. Train 02 (MCQ with reasoning)
#
# Prerequisites:
#   python init_lora_on_raw.py --model medgemma-4b [--extended-tokenizer]
#
# Usage:
#   ./run_full_pipeline.sh --model medgemma-4b --device cuda:0
#   ./run_full_pipeline.sh --model medgemma-27b --device cuda:0 --epochs 3
#   ./run_full_pipeline.sh --model medgemma-4b --background
#
# To monitor progress:
#   tail -f logs/pipeline_medgemma-4b.log
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Default values
# =============================================================================
MODEL="medgemma-4b"
DEVICE="cuda:0"
EPOCHS_00=1
EPOCHS_01=3
EPOCHS_02=5
BACKGROUND=false
SKIP_00=false
SKIP_01=false
SKIP_02=false
EXTENDED_TOKENIZER=false

# =============================================================================
# Parse arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --epochs-00)
            EPOCHS_00="$2"
            shift 2
            ;;
        --epochs-01)
            EPOCHS_01="$2"
            shift 2
            ;;
        --epochs-02)
            EPOCHS_02="$2"
            shift 2
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        --skip-00)
            SKIP_00=true
            shift
            ;;
        --skip-01)
            SKIP_01=true
            shift
            ;;
        --skip-02)
            SKIP_02=true
            shift
            ;;
        --extended-tokenizer)
            EXTENDED_TOKENIZER=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Prerequisites:"
            echo "  1. Run init_lora_on_raw.py first to initialize LoRA"
            echo "  2. (Optional) Run build_korean_tokenizer.py for extended tokenizer"
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model name (medgemma-4b or medgemma-27b)"
            echo "  --device DEVICE     CUDA device (default: cuda:0)"
            echo "  --epochs-00 N       Epochs for train_00 (default: 1)"
            echo "  --epochs-01 N       Epochs for train_01 (default: 3)"
            echo "  --epochs-02 N       Epochs for train_02 (default: 5)"
            echo "  --background        Run in background"
            echo "  --skip-00           Skip train_00"
            echo "  --skip-01           Skip train_01"
            echo "  --skip-02           Skip train_02"
            echo "  --extended-tokenizer Use extended Korean tokenizer"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Pipeline steps:"
            echo "  Step 1: Train 00 (Plain Text)"
            echo "  Step 2: Train 01 (Medical Dictionary)"
            echo "  Step 3: Add second LoRA adapter"
            echo "  Step 4: Train 02 (MCQ with Reasoning)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup paths
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/pipeline_${MODEL}.log"
PROGRESS_FILE="$LOG_DIR/progress_${MODEL}.txt"

# Max lengths from config
# train_02: FULL prompt max=633, NORMAL max=485, + ~300 response = ~933 max
if [[ "$MODEL" == "medgemma-4b" ]]; then
    MAX_LEN_00=512
    MAX_LEN_01=256
    MAX_LEN_02=1024    # Sufficient for FULL mode (633+300=933)
    FULL_SAMPLES=500
    CHECK_INTERVAL=100
else
    MAX_LEN_00=512
    MAX_LEN_01=256
    MAX_LEN_02=1024    # Memory optimized (36.1GB peak vs 41.4GB at 2048)
    FULL_SAMPLES=300
    CHECK_INTERVAL=50
fi

# =============================================================================
# Logging functions
# =============================================================================
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

update_progress() {
    echo "$1" > "$PROGRESS_FILE"
    log "PROGRESS: $1"
}

# =============================================================================
# Main pipeline function
# =============================================================================
run_pipeline() {
    log "============================================================"
    log "FULL TRAINING PIPELINE"
    log "============================================================"
    log "Model: $MODEL"
    log "Device: $DEVICE"
    log "Epochs: 00=$EPOCHS_00, 01=$EPOCHS_01, 02=$EPOCHS_02"
    log "Max lengths: 00=$MAX_LEN_00, 01=$MAX_LEN_01, 02=$MAX_LEN_02"
    log "Extended tokenizer: $EXTENDED_TOKENIZER"
    log "Log file: $LOG_FILE"
    log "============================================================"

    cd "$SCRIPT_DIR"

    # =========================================================================
    # Step 1: Train 00 (Plain Text)
    # =========================================================================
    if [[ "$SKIP_00" == false ]]; then
        update_progress "Step 1/4: Training on plain text (train_00)..."
        log ""
        log "============================================================"
        log "STEP 1: Train 00 - Plain Text"
        log "============================================================"

        python3 train/train_00_plain_text.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --epochs "$EPOCHS_00" \
            --max-length "$MAX_LEN_00" \
            2>&1 | tee -a "$LOG_FILE"

        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log "ERROR: train_00 failed!"
            exit 1
        fi
        log "Step 1 complete!"
    else
        log "Skipping train_00..."
    fi

    # =========================================================================
    # Step 2: Train 01 with 00 monitoring (Medical Dictionary)
    # =========================================================================
    if [[ "$SKIP_01" == false ]]; then
        update_progress "Step 2/4: Training on medical dictionary with 00 monitoring (train_01)..."
        log ""
        log "============================================================"
        log "STEP 2: Train 01 - Medical Dictionary (with 00 monitoring)"
        log "============================================================"

        python3 train/train_01_with_00_monitor.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --epochs "$EPOCHS_01" \
            --check-interval "$CHECK_INTERVAL" \
            2>&1 | tee -a "$LOG_FILE"

        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log "ERROR: train_01 failed!"
            exit 1
        fi
        log "Step 2 complete!"
    else
        log "Skipping train_01..."
    fi

    # =========================================================================
    # Step 3: Add second LoRA adapter (for train_02)
    # =========================================================================
    update_progress "Step 3/4: Adding second LoRA adapter..."
    log ""
    log "============================================================"
    log "STEP 3: Add Second LoRA Adapter"
    log "============================================================"

    python3 add_lora_adapter.py \
        --base-model "$BASE_DIR/model/01_trained/$MODEL" \
        --output "$BASE_DIR/model/01_another_lora_added/$MODEL" \
        --adapter-name "mcq_reasoning" \
        2>&1 | tee -a "$LOG_FILE"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log "ERROR: add_lora_adapter failed!"
        exit 1
    fi
    log "Step 3 complete!"

    # =========================================================================
    # Step 4: Train 02 (MCQ with Reasoning)
    # =========================================================================
    if [[ "$SKIP_02" == false ]]; then
        update_progress "Step 4/4: Training on MCQ with reasoning (train_02)..."
        log ""
        log "============================================================"
        log "STEP 4: Train 02 - MCQ with Reasoning"
        log "============================================================"

        python3 train/train_02_kor_med_test.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --epochs "$EPOCHS_02" \
            --max-length "$MAX_LEN_02" \
            --full-samples "$FULL_SAMPLES" \
            2>&1 | tee -a "$LOG_FILE"

        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log "ERROR: train_02 failed!"
            exit 1
        fi
        log "Step 4 complete!"
    else
        log "Skipping train_02..."
    fi

    # =========================================================================
    # Done
    # =========================================================================
    update_progress "COMPLETE: All training steps finished!"
    log ""
    log "============================================================"
    log "PIPELINE COMPLETE!"
    log "============================================================"
    log "Final models:"
    log "  - 00_trained: $BASE_DIR/model/00_trained/$MODEL"
    log "  - 01_trained: $BASE_DIR/model/01_trained/$MODEL"
    log "  - 02_trained: $BASE_DIR/model/02_trained/$MODEL"
    log "============================================================"
}

# =============================================================================
# Run pipeline
# =============================================================================
if [[ "$BACKGROUND" == true ]]; then
    log "Starting pipeline in background..."
    log "Monitor with: tail -f $LOG_FILE"

    nohup bash -c "$(declare -f log update_progress run_pipeline); \
        MODEL='$MODEL' DEVICE='$DEVICE' \
        EPOCHS_00='$EPOCHS_00' EPOCHS_01='$EPOCHS_01' EPOCHS_02='$EPOCHS_02' \
        MAX_LEN_00='$MAX_LEN_00' MAX_LEN_01='$MAX_LEN_01' MAX_LEN_02='$MAX_LEN_02' \
        FULL_SAMPLES='$FULL_SAMPLES' CHECK_INTERVAL='$CHECK_INTERVAL' \
        LOG_FILE='$LOG_FILE' PROGRESS_FILE='$PROGRESS_FILE' \
        SCRIPT_DIR='$SCRIPT_DIR' BASE_DIR='$BASE_DIR' \
        SKIP_00='$SKIP_00' SKIP_01='$SKIP_01' SKIP_02='$SKIP_02' \
        EXTENDED_TOKENIZER='$EXTENDED_TOKENIZER' \
        run_pipeline" >> "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$LOG_DIR/pipeline_${MODEL}.pid"

    echo ""
    echo "Pipeline started in background (PID: $PID)"
    echo ""
    echo "Commands:"
    echo "  Monitor:  tail -f $LOG_FILE"
    echo "  Progress: cat $PROGRESS_FILE"
    echo "  Stop:     kill $PID"
    echo ""
else
    run_pipeline
fi
