# Training Scripts (script)

Shared configuration and organized training pipeline.

## Scripts

### Configuration
- `training_config.py` - Shared MODEL_CONFIGS, LoRA settings, training defaults

### Training Pipeline

1. **`train/train_00_plain_text.py`** - Plain text pre-training
   - Purpose: Learn general Korean language
   - Input: Plain Korean corpus
   - Output: Base model with Korean proficiency

2. **`add_lora_adapter.py`** - Add LoRA adapter layer
   - Purpose: Prevent catastrophic forgetting
   - Run after train_00, before medical training
   - Adds new adapter for medical domain
   ```bash
   python add_lora_adapter.py \
     --base-model models/train_00/final \
     --output models/train_00_with_adapter \
     --adapter-name medical
   ```

3. **`train/train_01_medical_dict.py`** - Medical dictionary training
   - Input: Korean-English medical terminology
   - Purpose: Learn medical vocabulary

4. **`train/train_02_kor_med_test.py`** - Korean medical MCQ training
   - Input: Korean medical exam questions
   - Purpose: Learn medical knowledge and reasoning

5. **`train/train_01_02_loop.py`** - Loop training (01 + 02)
   - Purpose: Train 01 and 02 with equal data amounts
   - Alternates between medical dict and MCQ
   - Runs validation after each epoch
   ```bash
   python train/train_01_02_loop.py \
     --model medgemma-4b \
     --base-model models/train_00_with_adapter \
     --total-epochs 5 \
     --samples-per-epoch 1000
   ```

### Validation
- **`validation_kor_med_test.py`** - Validate on KorMedMCQA
  - Used for models from train_00, 01, 02
  - Target: ≥90% accuracy
  ```bash
  python validation_kor_med_test.py \
    --model-path models/loop_training/epoch_5/after_02 \
    --output results/validation.json
  ```

## Training Flow

```
Plain Korean Corpus
        ↓
train/train_00_plain_text.py → Base model with Korean
        ↓
add_lora_adapter.py → Add medical adapter
        ↓
train/train_01_02_loop.py → Loop: Medical dict + MCQ
        ↓
validation_kor_med_test.py → Evaluate
```

## Key Features

- ✓ Shared configuration (no duplication)
- ✓ Catastrophic forgetting prevention (LoRA adapter)
- ✓ Balanced training (equal data amounts)
- ✓ Regular validation (track progress)
- ✓ Best checkpoint saving
