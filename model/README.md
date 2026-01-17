# model/ Directory Structure

## Overview

```
model/
├── raw/                         # Base HuggingFace models (reference only)
│   └── model_paths.json         # Model ID mapping
├── raw_lora_added/              # Base + LoRA initialized (untrained)
│   ├── medgemma-4b/             # Input for train_00
│   └── medgemma-27b/
├── 00_training/                 # Checkpoints during train_00
│   ├── medgemma-4b/
│   └── medgemma-27b/
├── 00_trained/                  # Final models after train_00
│   ├── medgemma-4b/
│   └── medgemma-27b/
├── 01_training/                 # Checkpoints during train_01
│   ├── medgemma-4b/
│   └── medgemma-27b/
├── 01_trained/                  # Final models after train_01
│   ├── medgemma-4b/
│   └── medgemma-27b/
├── 01_another_lora_added/       # Merged LoRA + new LoRA (for train_02)
│   ├── medgemma-4b/
│   └── medgemma-27b/
├── 02_training/                 # Checkpoints during train_02
│   ├── medgemma-4b/
│   └── medgemma-27b/
└── 02_trained/                  # Final models after train_02
    ├── medgemma-4b/
    └── medgemma-27b/
```

## Training Pipeline with LoRA Chaining

```
HuggingFace Model (google/medgemma-27b-text-it)
    ↓
init_lora_on_raw.py  (add untrained LoRA)
    ↓
raw_lora_added/{model}/  (Base + LoRA initialized)
    ↓
train/train_00_plain_text.py  (train on Korean text)
    ↓ checkpoints → 00_training/{model}/
    ↓ final → 00_trained/{model}/
    ↓
train/train_01_medical_dict.py  (train on medical dictionary)
    ↓ checkpoints → 01_training/{model}/
    ↓ final → 01_trained/{model}/
    ↓ merge_and_unload() + new LoRA → 01_another_lora_added/{model}/
    ↓
train/train_02_kor_med_test.py  (train on KorMedMCQA)
    ↓ checkpoints → 02_training/{model}/
    ↓ final → 02_trained/{model}/
```

## LoRA Chaining Strategy

1. **init_lora_on_raw.py**: Initialize LoRA on base model (untrained)
2. **train_00**: Continue training the LoRA adapter on plain text
3. **train_01**: Continue training same LoRA on medical dictionary
   - After training: `merge_and_unload()` merges LoRA into base weights
   - Then adds a new fresh LoRA for train_02
4. **train_02**: Train the new LoRA on MCQ with reasoning

This approach:
- Avoids nested LoRA adapters (which cause warnings)
- Properly chains knowledge: Korean → Medical terms → MCQ reasoning
- Each phase builds on previous learning

## Usage

```bash
# Step 0: Initialize LoRA on raw model (one-time)
python script/init_lora_on_raw.py --model medgemma-27b

# Step 1: Train on plain Korean text
python script/train/train/train_00_plain_text.py --model medgemma-27b --epochs 1

# Step 2: Train on medical dictionary
python script/train/train/train_01_medical_dict.py --model medgemma-27b --epochs 3

# Step 3: Train on KorMedMCQA
python script/train/train/train_02_kor_med_test.py --model medgemma-27b --epochs 5
```

## Input/Output Summary

| Script | Input From | Output To | Also Creates |
|--------|------------|-----------|--------------|
| init_lora_on_raw.py | HuggingFace | raw_lora_added/ | - |
| train/train_00_plain_text.py | raw_lora_added/ | 00_trained/ | 00_training/ (checkpoints) |
| train/train_01_medical_dict.py | 00_trained/ | 01_trained/ | 01_training/, 01_another_lora_added/ |
| train/train_02_kor_med_test.py | 01_another_lora_added/ | 02_trained/ | 02_training/ (checkpoints) |

## Model Files

Each trained model directory contains:
```
{model}/
├── adapter_config.json           # LoRA configuration
├── adapter_model.safetensors     # LoRA weights
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── training_info.json            # Training metadata
└── kormedmcqa_validation_results.json  # Validation history
```

## Raw Models

`raw/model_paths.json`:
```json
{
  "medgemma-4b": {
    "huggingface_id": "google/medgemma-4b-it"
  },
  "medgemma-27b": {
    "huggingface_id": "google/medgemma-27b-text-it"
  }
}
```

These are HuggingFace model IDs - downloaded automatically when training starts.

## Debug Logs

Each training script writes debug logs to:
- `00_training/train_00_debug.log`
- `01_training/train_01_debug.log`
- `02_training/train_02_debug.log`
