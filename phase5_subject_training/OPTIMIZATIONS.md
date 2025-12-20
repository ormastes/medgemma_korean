# Phase 5 Optimizations Applied

## GPU Memory Optimization (A6000)

Phase 5 training scripts have been updated with the same optimizations as Phase 4:

### Model Configurations

#### MedGemma-27B (Primary)
- **Target:** 40GB GPU usage on A6000
- **LoRA rank:** 384 (3x baseline)
- **Batch size:** 4 (4x baseline)
- **Gradient accumulation:** 8
- **Max sequence length:** 1024 (2x baseline)
- **Trainable params:** 1.8B (6.3%)
- **Expected GPU usage:** ~40GB / 49GB (81%)

#### MedGemma-4B (Fast iteration)
- **LoRA rank:** 128
- **Batch size:** 8
- **Max sequence length:** 1024

#### Gemma-2-2B (Lightweight)
- **LoRA rank:** 64
- **Batch size:** 16
- **Max sequence length:** 1024

## Key Features

1. **4-bit Quantization:** NF4 quantization for memory efficiency
2. **LoRA:** Parameter-efficient fine-tuning
3. **Gradient Checkpointing:** For 27B model
4. **Mixed Precision:** BF16 training
5. **Paged AdamW 8-bit:** Memory-efficient optimizer

## Usage

### Training with Optimized Settings

```bash
# MedGemma-27B on A6000 (40GB usage)
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division/type1_text/train.jsonl \
    --val-data data/division/type1_text/validation.jsonl \
    --model medgemma-27b \
    --output-dir phase5_subject_training/models/type1_text \
    --epochs 3 \
    --device cuda:0 \
    --eval-steps 100

# MedGemma-4B (faster)
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --model medgemma-4b \
    ...

# Gemma-2-2B (lightweight)
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --model gemma-2-2b \
    ...
```

## Expected Performance

### A6000 GPU (49GB total)

| Model | Memory | Utilization | Speed | Trainable Params |
|-------|--------|-------------|-------|------------------|
| MedGemma-27B | 40GB | 100% | ~50-60s/step | 1.8B (6.3%) |
| MedGemma-4B | 18GB | 60% | ~20-30s/step | 536M (4.5%) |
| Gemma-2-2B | 12GB | 40% | ~10-15s/step | 268M (3.8%) |

## Changes from Original

### Before (Original Phase 5)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# No LoRA, no quantization
# ~45GB for 27B model (full precision)
```

### After (Optimized)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # 4-bit NF4
    device_map="auto",
    attn_implementation="eager"
)
model = get_peft_model(model, lora_config)  # LoRA adapters
# ~40GB for 27B model with larger LoRA rank
```

## Benefits

1. **3x Larger LoRA:** More trainable parameters (384 vs 128)
2. **4x Batch Size:** Faster convergence
3. **2x Sequence Length:** Better long-context handling
4. **100% GPU Utilization:** Maximum hardware usage
5. **Division Tracking:** Per-division performance metrics preserved

## Compatibility

- ✅ Division tracking fully preserved
- ✅ Division report generation unchanged
- ✅ All Phase 5 features maintained
- ✅ Compatible with existing data format
- ✅ Same CLI interface with added model selection

## Notes

- Default model is now `medgemma-27b` (was `google/gemma-2-2b-it`)
- Use `--model` flag to select configuration
- A6000 recommended for medgemma-27b
- A6000 sufficient for smaller models
