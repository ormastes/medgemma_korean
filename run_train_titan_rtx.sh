#!/bin/bash
# Run training pipeline on TITAN RTX (cuda:1)
# This uses reduced memory settings (no embeddings)

set -e

MODEL="medgemma-4b"
DEVICE="cuda:1"
EPOCHS=${1:-1}  # Default 1 epoch for testing

echo "============================================"
echo "Training Pipeline on TITAN RTX (cuda:1)"
echo "============================================"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS"
echo ""

# Ensure we're using no-embedding config for TITAN RTX
echo "Checking LoRA config for TITAN RTX compatibility..."
python3 -c "
import json
config_path = 'model/raw_lora_added/medgemma-4b/adapter_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)
if config.get('modules_to_save'):
    print('WARNING: Embeddings enabled. Disabling for TITAN RTX...')
    config['modules_to_save'] = None
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print('Config updated: modules_to_save = None')
else:
    print('Config OK: embeddings disabled')
"

echo ""
echo "============================================"
echo "Step 1/3: train_00 (Plain Text)"
echo "============================================"
python3 script/train/train_00_plain_text.py \
    --model $MODEL \
    --device $DEVICE \
    --epochs $EPOCHS \
    --max-samples 1000 \
    --skip-validation

echo ""
echo "============================================"
echo "Step 2/3: train_01_mixed (Translation + MCQ)"
echo "============================================"
python3 script/train/train_01_mixed.py \
    --model $MODEL \
    --device $DEVICE \
    --epochs $EPOCHS \
    --max-translation 500 \
    --max-mcq 500

echo ""
echo "============================================"
echo "Step 3/3: train_02 (MCQ with Reasoning)"
echo "============================================"
python3 script/train/train_02_kor_med_test.py \
    --model $MODEL \
    --device $DEVICE \
    --epochs $EPOCHS \
    --max-samples 500 \
    --full-samples 100

echo ""
echo "============================================"
echo "Training Complete!"
echo "============================================"
echo "Output directories:"
echo "  - model/00_trained/$MODEL/"
echo "  - model/01_mixed/$MODEL/final/"
echo "  - model/02_trained/$MODEL/"
