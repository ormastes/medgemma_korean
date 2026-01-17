#!/bin/bash
# Example: Complete training pipeline

set -e  # Exit on error

echo "============================================"
echo "MedGemma Korean Training Pipeline"
echo "============================================"

# Configuration
MODEL="medgemma-4b"
BASE_OUTPUT="models/pipeline_$(date +%Y%m%d)"

# Step 1: Train on plain Korean text
echo -e "\n[Step 1/4] Training on plain Korean text..."
python3 script/train/train/train_00_plain_text.py \
  --model $MODEL \
  --epochs 3 \
  --output $BASE_OUTPUT/train_00

# Step 2: Add LoRA adapter for medical training
echo -e "\n[Step 2/4] Adding LoRA adapter..."
python3 script/add_lora_adapter.py \
  --base-model $BASE_OUTPUT/train_00/final \
  --output $BASE_OUTPUT/with_adapter \
  --adapter-name medical

# Step 3: Loop training on medical data
echo -e "\n[Step 3/4] Loop training (Medical Dict + MCQ)..."
python3 script/train/train/train_01_02_loop.py \
  --model $MODEL \
  --base-model $BASE_OUTPUT/with_adapter \
  --total-epochs 5 \
  --samples-per-epoch 1000 \
  --output-dir $BASE_OUTPUT/loop_training \
  --validate-every 1

# Step 4: Final validation
echo -e "\n[Step 4/4] Final validation..."
python3 script/validation_kor_med_test.py \
  --model-path $BASE_OUTPUT/loop_training/best_checkpoint \
  --output $BASE_OUTPUT/final_validation.json

echo -e "\n============================================"
echo "Training Complete!"
echo "============================================"
echo "Best model: $BASE_OUTPUT/loop_training/best_checkpoint"
echo "Results: $BASE_OUTPUT/final_validation.json"
