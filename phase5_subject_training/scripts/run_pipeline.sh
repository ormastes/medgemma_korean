#!/bin/bash
# Master pipeline for division-based training
# Phase 5: Subject Training with Medical Divisions

set -e

echo "================================================"
echo "Phase 5: Subject Training Pipeline"
echo "================================================"

# Step 1: Annotate reviewed data with DeepSeek
echo ""
echo "Step 1: Annotating data with medical divisions using DeepSeek on TITAN RTX..."
echo "------------------------------------------------"

for TYPE in type1_text type2_text_reasoning type3_word type4_word_reasoning; do
    echo "Processing $TYPE..."
    
    # Train set
    python3 phase5_subject_training/scripts/annotate_with_deepseek.py \
        --input data/reviewed/$TYPE/train/data.jsonl \
        --output data/division/$TYPE/train.jsonl \
        --device cuda:0
    
    # Validation set
    python3 phase5_subject_training/scripts/annotate_with_deepseek.py \
        --input data/reviewed/$TYPE/validation/data.jsonl \
        --output data/division/$TYPE/validation.jsonl \
        --device cuda:0
    
    echo "$TYPE completed!"
done

# Step 2: Validate and fix malformed annotations
echo ""
echo "Step 2: Validating division annotations..."
echo "------------------------------------------------"

for TYPE in type1_text type2_text_reasoning type3_word type4_word_reasoning; do
    echo "Validating $TYPE..."
    
    # Train set
    python3 phase5_subject_training/scripts/validate_divisions.py \
        --input data/division/$TYPE/train.jsonl \
        --output data/division/$TYPE/train_fixed.jsonl \
        --fix
    
    # Validation set
    python3 phase5_subject_training/scripts/validate_divisions.py \
        --input data/division/$TYPE/validation.jsonl \
        --output data/division/$TYPE/validation_fixed.jsonl \
        --fix
    
    # Replace original with fixed
    mv data/division/$TYPE/train_fixed.jsonl data/division/$TYPE/train.jsonl
    mv data/division/$TYPE/validation_fixed.jsonl data/division/$TYPE/validation.jsonl
    
    echo "$TYPE validated and fixed!"
done

# Step 3: Train with division tracking
echo ""
echo "Step 3: Training with division-aware evaluation..."
echo "------------------------------------------------"

for TYPE in type1_text type2_text_reasoning type3_word type4_word_reasoning; do
    echo "Training $TYPE..."
    
    python3 phase5_subject_training/scripts/train_with_divisions.py \
        --train-data data/division/$TYPE/train.jsonl \
        --val-data data/division/$TYPE/validation.jsonl \
        --model google/gemma-2-2b-it \
        --output-dir phase5_subject_training/models/$TYPE \
        --epochs 3 \
        --batch-size 4 \
        --device cuda:0
    
    echo "$TYPE training completed!"
    echo "Division report: phase5_subject_training/models/$TYPE/division_report.json"
done

echo ""
echo "================================================"
echo "Phase 5 Pipeline Completed!"
echo "================================================"
echo ""
echo "Results:"
echo "  - Annotated data: data/division/"
echo "  - Models: phase5_subject_training/models/"
echo "  - Division reports: phase5_subject_training/models/*/division_report.json"
echo ""
echo "Next: Review division reports to identify weak subject areas"
